import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from argparse import ArgumentParser
from pathlib import Path
import math

from torch import nn
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import wandb

from roma.benchmarks import MegadepthDenseBenchmark
from roma.datasets.megadepth import MegadepthBuilder
from roma.losses.robust_loss import RobustLosses
from roma.benchmarks import MegaDepthPoseEstimationBenchmark, MegadepthDenseBenchmark, HpatchesHomogBenchmark
from roma.train.train import train_k_steps
from roma.checkpointing import CheckPoint

resolutions = {"low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6)}

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

    def __init__(self, xfeat = None, freeze_xfeat = True):
        super().__init__()
        if xfeat is None:
            xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096).net
            if freeze_xfeat:
                xfeat.train(False)
                self.xfeat = [xfeat]# hide params from ddp
            else:
                raise NotImplementedError("Fine-tuning of XFeat is not supported yet, due to extra params which needs removing.")
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
         
    def forward_single(self, x):
        with torch.inference_mode(self.freeze_xfeat):
            xfeat = self.xfeat[0]
            with torch.no_grad():
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
    
    def pos_embed(self, corr_volume):
        H,W = corr_volume.shape[-2:] 
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1+1/W,1-1/W, W), 
                torch.linspace(-1+1/H,1-1/H, H), 
                indexing = "xy"), 
            dim = -1).float().cuda().reshape(H*W, 2)
        corr_volume = corr_volume.softmax(dim=1) # B, HW, H, W
        pos_embeddings = torch.einsum('bchw,cd->bdhw', corr_volume, grid)
        return pos_embeddings
     
    def corr_volume(self, feat0, feat1):
        """
            input:
                feat0 -> torch.Tensor(B, C, H, W)
                feat1 -> torch.Tensor(B, C, H, W)
            return:
                corr_volume -> torch.Tensor(B, H, W, H, W)
        """
        B, C, H, W = feat0.shape
        feat0 = feat0.view(B, C, H*W)
        feat1 = feat1.view(B, C, H*W)
        corr_volume = torch.einsum('bci,bcj->bji', feat0, feat1).reshape(B, (H*W), H , W)/math.sqrt(C) #16*16*16
        #print(feat0.requires_grad, corr_volume.requires_grad)
        return corr_volume
    
    @torch.inference_mode()
    def match(self, im0, im1, batched = True):
        B,C,H,W = im0.shape
        self.train(False)
        corresps = self.forward({"im_A":im0, "im_B":im1})
        flow = F.interpolate(corresps[4]["flow"], size = (H,W), mode = "bilinear", align_corners = False)
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1+1/W,1-1/W, W), 
                torch.linspace(-1+1/H,1-1/H, H), 
                indexing = "xy"), 
            dim = -1).float().to(flow.device).reshape(1, H, W, 2)
        
        certainty = F.interpolate(corresps[4]["certainty"], size = (H,W), mode = "bilinear", align_corners = False)
        return torch.cat((flow, certainty), dim = 1).permute(0,2,3,1), certainty[:,0]
        
    def forward(self, batch):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:

        """
        im0 = batch["im_A"]
        im1 = batch["im_B"]
        B, C, H, W = im0.shape
        corresps = {}
        x = torch.cat([im0, im1], dim=0)
        x = x.mean(dim=1, keepdim=True)#Same as in xfeat
        x = self.forward_single(x)
        feats_x0_c, feats_x1_c = x[1].chunk(2)
        
        corr_volume = self.corr_volume(feats_x0_c, feats_x1_c)
        coarse_warp = self.pos_embed(corr_volume)
        coarse_matches = torch.cat((coarse_warp, torch.zeros_like(coarse_warp[:,-1:])), dim=1)
        feats_x1_c_warped = F.grid_sample(feats_x1_c, coarse_matches.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        coarse_matches_delta = self.coarse_matcher(torch.cat((feats_x0_c, feats_x1_c_warped, coarse_warp), dim=1))
        #print(f"{coarse_matches_delta=}")
        corresps[8] = {"gm_flow": coarse_matches[:,:2], "gm_certainty": coarse_matches[:,2:]}
        coarse_matches = coarse_matches + coarse_matches_delta
        corresps[8] = {"flow": coarse_matches[:,:2], "certainty": coarse_matches[:,2:]}
        coarse_matches_up = F.interpolate(coarse_matches, size = x[0].shape[-2:], mode = "bilinear", align_corners = False)        
        feats_x0_f, feats_x1_f = x[0].chunk(2)
        coarse_matches_up_detach = coarse_matches_up.detach()#note the detach
        feats_x1_f_warped = F.grid_sample(feats_x1_f, coarse_matches_up_detach.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        fine_matches_delta = self.fine_matcher(torch.cat((feats_x0_f, feats_x1_f_warped, coarse_matches_up_detach[:,:2]), dim=1))
        fine_matches = coarse_matches_up_detach+fine_matches_delta
        corresps[4] = {"flow": fine_matches[:,:2], "certainty": fine_matches[:,2:]}
        return corresps
    




def train(args):
    dist.init_process_group('nccl')
    #torch._dynamo.config.verbose=True
    gpus = int(os.environ['WORLD_SIZE'])
    # create model and move it to GPU with id rank
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}")
    device_id = rank % torch.cuda.device_count()
    roma.LOCAL_RANK = device_id
    torch.cuda.set_device(device_id)
    
    resolution = args.train_resolution
    wandb_log = not args.dont_log_wandb
    # TODO: use Path.name instead
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    wandb_mode = "online" if wandb_log and rank == 0 else "disabled"
    wandb.init(project="roma", entity=args.wandb_entity, name=experiment_name, reinit=False, mode = wandb_mode)
    checkpoint_dir = "workspace/checkpoints/"
    h,w = resolutions[resolution]
    model = XFeatModel().to(device_id)
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
    rot_prob = 0
    depth_interpolation_mode = "bilinear"
    megadepth_train1 = mega.build_scenes(
        split="train_loftr", min_overlap=0.01, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
        ht=h,wt=w,
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr", min_overlap=0.35, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
        ht=h,wt=w,
    )
    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
    # Loss and optimizer
    depth_loss = RobustLosses(
        ce_weight=0.01, 
        local_dist={4:32},
        local_largest_scale=4,
        depth_interpolation_mode=depth_interpolation_mode,
        alpha = 0.5,
        c = 1e-4,)
    parameters = [
        {"params": model.parameters(), "lr": roma.STEP_SIZE * 1e-4 / 8},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[(9*N/roma.STEP_SIZE)//10])
    megadense_benchmark = MegadepthDenseBenchmark("data/megadepth", num_samples = 1000, h=h,w=w)
    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step)
    roma.GLOBAL_STEP = global_step
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters = False, gradient_as_bucket_view=True)
    grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
    grad_clip_norm = 0.01
    for n in range(roma.GLOBAL_STEP, N, k * roma.STEP_SIZE):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples = batch_size * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size = batch_size,
                sampler = mega_sampler,
                num_workers = 8,
            )
        )
        train_k_steps(
            n, k, mega_dataloader, ddp_model, depth_loss, optimizer, lr_scheduler, grad_scaler, grad_clip_norm = grad_clip_norm,
        )
        checkpointer.save(model, optimizer, lr_scheduler, roma.GLOBAL_STEP)
        wandb.log(megadense_benchmark.benchmark(model), step = roma.GLOBAL_STEP)

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
    parser.add_argument("--gpu_batch_size", default=4, type=int)
    parser.add_argument("--wandb_entity", required = False)

    args, _ = parser.parse_known_args()
    roma.DEBUG_MODE = args.debug_mode
    if not args.only_test:
        train(args)