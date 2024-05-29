import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from argparse import ArgumentParser
from pathlib import Path
import math
import numpy as np

from torch import nn
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import wandb
from PIL import Image
from torchvision.transforms import ToTensor

from roma.benchmarks import MegadepthDenseBenchmark
from roma.benchmarks import Mega1500PoseLibBenchmark
from roma.datasets.megadepth import MegadepthBuilder
from roma.losses.robust_loss_tiny_roma import RobustLosses
from roma.benchmarks import MegaDepthPoseEstimationBenchmark, MegadepthDenseBenchmark, HpatchesHomogBenchmark
from roma.train.train import train_k_steps
from roma.checkpointing import CheckPoint

resolutions = {"low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6), "xfeat": (600,800)}

def kde(x, std = 0.1):
    # use a gaussian kernel to estimate density
    x = x.half() # Do it in half precision TODO: remove hardcoding
    scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

class BasicLayer(nn.Module):
    """
        Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, relu = True):
        super().__init__()
        self.layer = nn.Sequential(
                                        nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
                                        nn.BatchNorm2d(out_channels, affine=False),
                                        nn.ReLU(inplace = True) if relu else nn.Identity()
                                    )

    def forward(self, x):
        return self.layer(x)

class XFeatModel(nn.Module):
    """
        Implementation of architecture described in 
        "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self, xfeat = None, freeze_xfeat = True, sample_mode = "threshold_balanced", symmetric = True):
        super().__init__()
        if xfeat is None:
            xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096).net
            del xfeat.heatmap_head, xfeat.keypoint_head, xfeat.fine_matcher
        if freeze_xfeat:
            xfeat.train(False)
            self.xfeat = [xfeat]# hide params from ddp
        else:
            self.xfeat = nn.ModuleList([xfeat])
        self.freeze_xfeat = freeze_xfeat
        match_dim = 256
        self.coarse_matcher = nn.Sequential(
            BasicLayer(64+64+2, match_dim,),
            BasicLayer(match_dim, match_dim,), 
            BasicLayer(match_dim, match_dim,), 
            BasicLayer(match_dim, match_dim,), 
            nn.Conv2d(match_dim, 3, kernel_size=1, bias=True, padding=0))
        fine_match_dim = 64
        self.fine_matcher = nn.Sequential(
            BasicLayer(24+24+2, fine_match_dim,),
            BasicLayer(fine_match_dim, fine_match_dim,), 
            BasicLayer(fine_match_dim, fine_match_dim,), 
            BasicLayer(fine_match_dim, fine_match_dim,), 
            nn.Conv2d(fine_match_dim, 3, kernel_size=1, bias=True, padding=0),)
        self.sample_mode = sample_mode
        self.sample_thresh = 0.05
        self.symmetric = symmetric
    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        H, W = x.shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw        
    def forward_single(self, x):
        with torch.inference_mode(self.freeze_xfeat):
            xfeat = self.xfeat[0]
            with torch.no_grad():
                x = x.mean(dim=1, keepdim = True)
                x = xfeat.norm(x)

            #main backbone
            x1 = xfeat.block1(x)
            x2 = xfeat.block2(x1 + xfeat.skip1(x))
            x3 = xfeat.block3(x2)
            x4 = xfeat.block4(x3)
            x5 = xfeat.block5(x4)
            x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
            x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
            feats = xfeat.block_fusion( x3 + x4 + x5 )
        if self.freeze_xfeat:
            return x2.clone(), feats.clone()
        return x2, feats

    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
    
    def pos_embed(self, corr_volume: torch.Tensor):
        B, H1, W1, H0, W0 = corr_volume.shape 
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1+1/W1,1-1/W1, W1), 
                torch.linspace(-1+1/H1,1-1/H1, H1), 
                indexing = "xy"), 
            dim = -1).float().cuda().reshape(H1*W1, 2)
        corr_volume = corr_volume.reshape(B,H1*W1,H0,W0).softmax(dim=1) # B, HW, H, W
        pos_embeddings = torch.einsum('bchw,cd->bdhw', corr_volume, grid)
        return pos_embeddings
    
    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None, unnormalize = False):
        #assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from roma.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im
     
    def corr_volume(self, feat0, feat1):
        """
            input:
                feat0 -> torch.Tensor(B, C, H, W)
                feat1 -> torch.Tensor(B, C, H, W)
            return:
                corr_volume -> torch.Tensor(B, H, W, H, W)
        """
        B, C, H0, W0 = feat0.shape
        B, C, H1, W1 = feat1.shape
        feat0 = feat0.view(B, C, H0*W0)
        feat1 = feat1.view(B, C, H1*W1)
        corr_volume = torch.einsum('bci,bcj->bji', feat0, feat1).reshape(B, H1, W1, H0 , W0)/math.sqrt(C) #16*16*16
        #print(feat0.requires_grad, corr_volume.requires_grad)
        return corr_volume
    
    @torch.inference_mode()
    def match_from_path(self, im0_path, im1_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        im0 = ToTensor()(Image.open(im0_path))[None].to(device)
        im1 = ToTensor()(Image.open(im1_path))[None].to(device)
        return self.match(im0, im1, batched = False)
    
    @torch.inference_mode()
    def match(self, im0, im1, *args, batched = True):
        # stupid
        if isinstance(im0, (str, Path)):
            return self.match_from_path(im0, im1)
        B,C,H0,W0 = im0.shape
        B,C,H1,W1 = im1.shape
        self.train(False)
        corresps = self.forward({"im_A":im0, "im_B":im1})
        flow = F.interpolate(
            corresps[4]["flow"], 
            size = (H0, W0), 
            mode = "bilinear", align_corners = False).permute(0,2,3,1).reshape(B,H0,W0,2)
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1+1/W0,1-1/W0, W0), 
                torch.linspace(-1+1/H0,1-1/H0, H0), 
                indexing = "xy"), 
            dim = -1).float().to(flow.device).expand(B, H0, W0, 2)
        
        certainty = F.interpolate(corresps[4]["certainty"], size = (H0,W0), mode = "bilinear", align_corners = False)
        warp, cert = torch.cat((grid, flow), dim = -1), certainty[:,0].sigmoid()
        if batched:
            return warp, cert
        else:
            return warp[0], cert[0]

    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]
            
    def forward(self, batch):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:

        """
        im0 = batch["im_A"]
        im1 = batch["im_B"]
        corresps = {}
        im0, rh0, rw0 = self.preprocess_tensor(im0)
        im1, rh1, rw1 = self.preprocess_tensor(im1)
        B, C, H0, W0 = im0.shape
        B, C, H1, W1 = im1.shape
        to_normalized = torch.tensor((2/W1, 2/H1, 1)).to(im0.device)[None,:,None,None]
 
        if im0.shape[-2:] == im1.shape[-2:]:
            x = torch.cat([im0, im1], dim=0)
            x = self.forward_single(x)
            feats_x0_c, feats_x1_c = x[1].chunk(2)
            feats_x0_f, feats_x1_f = x[0].chunk(2)
        else:
            feats_x0_f, feats_x0_c = self.forward_single(im0)
            feats_x1_f, feats_x1_c = self.forward_single(im1)
        corr_volume = self.corr_volume(feats_x0_c, feats_x1_c)
        coarse_warp = self.pos_embed(corr_volume)
        coarse_matches = torch.cat((coarse_warp, torch.zeros_like(coarse_warp[:,-1:])), dim=1)
        feats_x1_c_warped = F.grid_sample(feats_x1_c, coarse_matches.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        coarse_matches_delta = self.coarse_matcher(torch.cat((feats_x0_c, feats_x1_c_warped, coarse_warp), dim=1))
        #print(f"{coarse_matches_delta=}")
        #corresps[8] = {"gm_flow": coarse_matches[:,:2], "gm_certainty": coarse_matches[:,2:]}
        coarse_matches = coarse_matches + coarse_matches_delta * to_normalized
        corresps[8] = {"flow": coarse_matches[:,:2], "certainty": coarse_matches[:,2:]}
        coarse_matches_up = F.interpolate(coarse_matches, size = feats_x0_f.shape[-2:], mode = "bilinear", align_corners = False)        
        coarse_matches_up_detach = coarse_matches_up.detach()#note the detach
        feats_x1_f_warped = F.grid_sample(feats_x1_f, coarse_matches_up_detach.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        fine_matches_delta = self.fine_matcher(torch.cat((feats_x0_f, feats_x1_f_warped, coarse_matches_up_detach[:,:2]), dim=1))
        fine_matches = coarse_matches_up_detach+fine_matches_delta * to_normalized
        corresps[4] = {"flow": fine_matches[:,:2], "certainty": fine_matches[:,2:]}
        return corresps
    




def train(args):
    rank = 0
    gpus = 1
    device_id = rank % torch.cuda.device_count()
    roma.LOCAL_RANK = 0
    torch.cuda.set_device(device_id)
        
    resolution = "xfeat"
    wandb_log = not args.dont_log_wandb
    experiment_name = Path(__file__).stem
    wandb_mode = "online" if wandb_log and rank == 0 else "disabled"
    wandb.init(project="roma", entity=args.wandb_entity, name=experiment_name, reinit=False, mode = wandb_mode)
    checkpoint_dir = "workspace/checkpoints/"
    h,w = resolutions[resolution]
    model = XFeatModel(freeze_xfeat = False).to(device_id)
    # Num steps
    global_step = 0
    batch_size = args.gpu_batch_size
    step_size = gpus*batch_size
    roma.STEP_SIZE = step_size
    
    N = (32 * 250000)  # 250k steps of batch size 32
    # checkpoint every
    k = 25000 // roma.STEP_SIZE

    # Data
    mega = MegadepthBuilder(data_root="data/megadepth", loftr_ignore=True, imc21_ignore = True)
    use_horizontal_flip_aug = True
    normalize = False # don't imgnet normalize
    rot_prob = 0
    depth_interpolation_mode = "bilinear"
    megadepth_train1 = mega.build_scenes(
        split="train_loftr", min_overlap=0.01, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
        ht=h,wt=w, normalize = normalize
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr", min_overlap=0.35, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
        ht=h,wt=w, normalize = normalize
    )
    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
    # Loss and optimizer
    depth_loss = RobustLosses(
        ce_weight=0.01, 
        local_dist={4:4},
        depth_interpolation_mode=depth_interpolation_mode,
        alpha = {4:0.25, 8:0.25},
        c = 1e-4,
        epe_mask_prob_th = 0.01,
        )
    parameters = [
        {"params": model.parameters(), "lr": roma.STEP_SIZE * 1e-4 / 8},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[(9*N/roma.STEP_SIZE)//10])
    #megadense_benchmark = MegadepthDenseBenchmark("data/megadepth", num_samples = 1000, h=h,w=w)
    mega1500_benchmark = Mega1500PoseLibBenchmark("data/megadepth", num_ransac_iter = 1, test_every = 30)

    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step)
    roma.GLOBAL_STEP = global_step
    grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
    grad_clip_norm = 0.01
    #megadense_benchmark.benchmark(model)
    for n in range(roma.GLOBAL_STEP, N, k * roma.STEP_SIZE):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples = batch_size * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size = batch_size,
                sampler = mega_sampler,
                num_workers = 16,
            )
        )
        train_k_steps(
            n, k, mega_dataloader, model, depth_loss, optimizer, lr_scheduler, grad_scaler, grad_clip_norm = grad_clip_norm,
        )
        checkpointer.save(model, optimizer, lr_scheduler, roma.GLOBAL_STEP)
        wandb.log(mega1500_benchmark.benchmark(model, model_name=experiment_name), step = roma.GLOBAL_STEP)

def test_mega_8_scenes(model, name):
    mega_8_scenes_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth",
                                                scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                    'mega_8_scenes_0025_0.1_0.3.npz',
                                                    'mega_8_scenes_0021_0.1_0.3.npz',
                                                    'mega_8_scenes_0008_0.1_0.3.npz',
                                                    'mega_8_scenes_0032_0.1_0.3.npz',
                                                    'mega_8_scenes_1589_0.1_0.3.npz',
                                                    'mega_8_scenes_0063_0.1_0.3.npz',
                                                    'mega_8_scenes_0024_0.1_0.3.npz',
                                                    'mega_8_scenes_0019_0.3_0.5.npz',
                                                    'mega_8_scenes_0025_0.3_0.5.npz',
                                                    'mega_8_scenes_0021_0.3_0.5.npz',
                                                    'mega_8_scenes_0008_0.3_0.5.npz',
                                                    'mega_8_scenes_0032_0.3_0.5.npz',
                                                    'mega_8_scenes_1589_0.3_0.5.npz',
                                                    'mega_8_scenes_0063_0.3_0.5.npz',
                                                    'mega_8_scenes_0024_0.3_0.5.npz'])
    mega_8_scenes_results = mega_8_scenes_benchmark.benchmark(model, model_name=name)
    print(mega_8_scenes_results)
    json.dump(mega_8_scenes_results, open(f"results/mega_8_scenes_{name}.json", "w"))

def test_mega1500(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_{name}.json", "w"))

def test_mega_dense(model, name):
    megadense_benchmark = MegadepthDenseBenchmark("data/megadepth", num_samples = 1000)
    megadense_results = megadense_benchmark.benchmark(model)
    json.dump(megadense_results, open(f"results/mega_dense_{name}.json", "w"))
    
def test_hpatches(model, name):
    hpatches_benchmark = HpatchesHomogBenchmark("data/hpatches")
    hpatches_results = hpatches_benchmark.benchmark(model)
    json.dump(hpatches_results, open(f"results/hpatches_{name}.json", "w"))

def test_mega1500_poselib(model, name):
    mega1500_benchmark = Mega1500PoseLibBenchmark("data/megadepth", num_ransac_iter = 5, test_every = 1)
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_{name}.json", "w"))

if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    import roma
    parser = ArgumentParser()
    parser.add_argument("--only_test", action='store_true')
    parser.add_argument("--debug_mode", action='store_true')
    parser.add_argument("--dont_log_wandb", action='store_true')
    parser.add_argument("--train_resolution", default='medium')
    parser.add_argument("--gpu_batch_size", default=8, type=int)
    parser.add_argument("--wandb_entity", required = False)

    args, _ = parser.parse_known_args()
    roma.DEBUG_MODE = args.debug_mode
    if not args.only_test:
        train(args)

    experiment_name = Path(__file__).stem
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XFeatModel(freeze_xfeat=False).to(device)
    
    model.load_state_dict(torch.load(f"{experiment_name}_latest.pth")["model"])
    #test_mega1500(model, experiment_name)
    test_mega1500_poselib(model, experiment_name)