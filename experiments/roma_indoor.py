import os
import torch
from argparse import ArgumentParser

from torch import nn
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import json
import wandb
from tqdm import tqdm

from roma.benchmarks import MegadepthDenseBenchmark
from roma.datasets.megadepth import MegadepthBuilder
from roma.datasets.scannet import ScanNetBuilder
from roma.losses.robust_loss import RobustLosses
from roma.benchmarks import MegadepthDenseBenchmark, ScanNetBenchmark
from roma.train.train import train_k_steps
from roma.models.matcher import *
from roma.models.transformer import Block, TransformerDecoder, MemEffAttention
from roma.models.encoders import *
from roma.checkpointing import CheckPoint

resolutions = {"low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6)}

def get_model(pretrained_backbone=True, resolution = "medium", **kwargs):
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]), 
        decoder_dim, 
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp = True,
        pos_enc = False,)
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True
    
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512+128+(2*7+1)**2,
                2 * 512+128+(2*7+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius = 7,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "8": ConvRefiner(
                2 * 512+64+(2*3+1)**2,
                2 * 512+64+(2*3+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius = 3,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "4": ConvRefiner(
                2 * 256+32+(2*2+1)**2,
                2 * 256+32+(2*2+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius = 2,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "2": ConvRefiner(
                2 * 64+16,
                128+16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks = hidden_blocks,
                displacement_emb = displacement_emb,
                displacement_emb_dim = 6,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict({
        "16": proj16,
        "8": proj8,
        "4": proj4,
        "2": proj2,
        "1": proj1,
        })
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(coordinate_decoder, 
                      gps, 
                      proj, 
                      conv_refiner, 
                      detach=True, 
                      scales=["16", "8", "4", "2", "1"], 
                      displacement_dropout_p = displacement_dropout_p,
                      gm_warp_dropout_p = gm_warp_dropout_p)
    h,w = resolutions[resolution]
    encoder = CNNandDinov2(
        cnn_kwargs = dict(
            pretrained=pretrained_backbone,
            amp = True),
        amp = True,
        use_vgg = True,
    )
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, alpha=1, beta=0,**kwargs)
    return matcher

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
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    wandb_mode = "online" if wandb_log and rank == 0 and False else "disabled"
    wandb.init(project="roma", entity=args.wandb_entity, name=experiment_name, reinit=False, mode = wandb_mode)
    checkpoint_dir = "workspace/checkpoints/"
    h,w = resolutions[resolution]
    model = get_model(pretrained_backbone=True, resolution=resolution, attenuate_cert = False).to(device_id)
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
    
    scannet = ScanNetBuilder(data_root="data/scannet")
    scannet_train = scannet.build_scenes(split="train", ht=h, wt=w, use_horizontal_flip_aug = use_horizontal_flip_aug)
    scannet_train = ConcatDataset(scannet_train)
    scannet_ws = scannet.weight_scenes(scannet_train, alpha=0.75)

    # Loss and optimizer
    depth_loss_scannet = RobustLosses(
        ce_weight=0.0, 
        local_dist={1:4, 2:4, 4:8, 8:8},
        local_largest_scale=8,
        depth_interpolation_mode=depth_interpolation_mode,
        alpha = 0.5,
        c = 1e-4,)
    # Loss and optimizer
    depth_loss_mega = RobustLosses(
        ce_weight=0.01, 
        local_dist={1:4, 2:4, 4:8, 8:8},
        local_largest_scale=8,
        depth_interpolation_mode=depth_interpolation_mode,
        alpha = 0.5,
        c = 1e-4,)
    parameters = [
        {"params": model.encoder.parameters(), "lr": roma.STEP_SIZE * 5e-6 / 8},
        {"params": model.decoder.parameters(), "lr": roma.STEP_SIZE * 1e-4 / 8},
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
        scannet_ws_sampler = torch.utils.data.WeightedRandomSampler(
            scannet_ws, num_samples=batch_size * k, replacement=False
        )
        scannet_dataloader = iter(
            torch.utils.data.DataLoader(
                scannet_train,
                batch_size=batch_size,
                sampler=scannet_ws_sampler,
                num_workers=gpus * 8,
            )
        )
        for n_k in tqdm(range(n, n + 2 * k, 2),disable = roma.RANK > 0):
            train_k_steps(
                n_k, 1, mega_dataloader, ddp_model, depth_loss_mega, optimizer, lr_scheduler, grad_scaler, grad_clip_norm = grad_clip_norm, progress_bar=False
            )
            train_k_steps(
                n_k + 1, 1, scannet_dataloader, ddp_model, depth_loss_scannet, optimizer, lr_scheduler, grad_scaler, grad_clip_norm = grad_clip_norm, progress_bar=False
            )
        checkpointer.save(model, optimizer, lr_scheduler, roma.GLOBAL_STEP)
        wandb.log(megadense_benchmark.benchmark(model), step = roma.GLOBAL_STEP)

def test_scannet(model, name, resolution, sample_mode):
    scannet_benchmark = ScanNetBenchmark("data/scannet")
    scannet_results = scannet_benchmark.benchmark(model)
    json.dump(scannet_results, open(f"results/scannet_{name}.json", "w"))

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    warnings.filterwarnings('ignore')#, category=UserWarning)#, message='WARNING batched routines are designed for small sizes.')
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    
    import roma
    parser = ArgumentParser()
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--debug_mode", action='store_true')
    parser.add_argument("--dont_log_wandb", action='store_true')
    parser.add_argument("--train_resolution", default='medium')
    parser.add_argument("--gpu_batch_size", default=4, type=int)
    parser.add_argument("--wandb_entity", required = False)

    args, _ = parser.parse_known_args()
    roma.DEBUG_MODE = args.debug_mode
    if not args.test:
        train(args)
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    checkpoint_dir = "workspace/"
    checkpoint_name = checkpoint_dir + experiment_name + ".pth"
    test_resolution = "medium"
    sample_mode = "threshold_balanced"
    symmetric = True
    upsample_preds = False
    attenuate_cert = True

    model = get_model(pretrained_backbone=False, resolution = test_resolution, sample_mode = sample_mode, upsample_preds = upsample_preds, symmetric=symmetric, name=experiment_name, attenuate_cert = attenuate_cert)
    model = model.cuda()
    states = torch.load(checkpoint_name)
    model.load_state_dict(states["model"])
    test_scannet(model, experiment_name, resolution = test_resolution, sample_mode = sample_mode)
