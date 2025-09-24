import sys
import warnings
from functools import partial

import torch
import torch.nn as nn
from loguru import logger

from romatch.models.encoders import CNNandDinov2
from romatch.models.matcher import (
    GP,
    ConvRefiner,
    CosKernel,
    Decoder,
    RegressionMatcher,
)
from romatch.models.tiny import TinyRoMa
from romatch.models.transformer import Block, MemEffAttention, TransformerDecoder


def tiny_roma_v1_model(
    weights=None, freeze_xfeat=False, exact_softmax=False, xfeat=None
):
    model = TinyRoMa(
        xfeat=xfeat, freeze_xfeat=freeze_xfeat, exact_softmax=exact_softmax
    )
    if weights is not None:
        model.load_state_dict(weights)
    return model


def roma_model(
    resolution,
    upsample_preds,
    device=None,
    weights=None,
    dinov2_weights=None,
    amp_dtype: torch.dtype = torch.float16,
    use_custom_corr=True,
    symmetric=True,
    upsample_res=None,
    sample_thresh=0.05,
    sample_mode="threshold_balanced",
    attenuate_cert = True,
    **kwargs,
):
    if sys.platform != "linux":
        use_custom_corr = False
        warnings.warn("Local correlation is not supported on non-Linux platforms, setting use_custom_corr to False")
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)

    if str(device) == "cpu":
        amp_dtype = torch.float32

    assert resolution[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert resolution[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

    logger.info(
        f"Using coarse resolution {resolution}, and upsample res {upsample_res}"
    )

    if sys.platform != "linux":
        use_custom_corr = False
        warnings.warn("Local correlation is not supported on non-Linux platforms, setting use_custom_corr to False")
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="TypedStorage is deprecated"
    )
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(
            *[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]
        ),
        decoder_dim,
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp=True,
        pos_enc=False,
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True
    partial_conv_refiner = partial(
        ConvRefiner,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        displacement_emb=displacement_emb,
        corr_in_other=True,
        amp=True,
        disable_local_corr_grad=disable_local_corr_grad,
        bn_momentum=0.01,
        use_custom_corr=use_custom_corr,
    )

    conv_refiner = nn.ModuleDict(
        {
            "16": partial_conv_refiner(
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 + 1,
                displacement_emb_dim=128,
                local_corr_radius=7,
            ),
            "8": partial_conv_refiner(
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 + 1,
                displacement_emb_dim=64,
                local_corr_radius=3,
            ),
            "4": partial_conv_refiner(
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 + 1,
                displacement_emb_dim=32,
                local_corr_radius=2,
            ),
            "2": partial_conv_refiner(
                2 * 64 + 16,
                128 + 16,
                2 + 1,
                displacement_emb_dim=16,
            ),
            "1": partial_conv_refiner(
                2 * 9 + 6,
                24,
                2 + 1,
                displacement_emb_dim=6,
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
    proj = nn.ModuleDict(
        {
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
        }
    )
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(
        coordinate_decoder,
        gps,
        proj,
        conv_refiner,
        detach=True,
        scales=["16", "8", "4", "2", "1"],
        displacement_dropout_p=displacement_dropout_p,
        gm_warp_dropout_p=gm_warp_dropout_p,
    )

    encoder = CNNandDinov2(
        cnn_kwargs=dict(pretrained=False, amp=True),
        amp=True,
        dinov2_weights=dinov2_weights,
        amp_dtype=amp_dtype,
    )
    h, w = resolution
    
    matcher = RegressionMatcher(
        encoder,
        decoder,
        h=h,
        w=w,
        upsample_preds=upsample_preds,
        upsample_res=upsample_res,
        symmetric=symmetric,
        attenuate_cert=attenuate_cert,
        sample_mode=sample_mode,
        sample_thresh=sample_thresh,
        **kwargs,
    ).to(device)
    matcher.load_state_dict(weights)
    return matcher
