import torch
import torch.nn as nn
import torch.nn.functional as F

from roma.utils.utils import get_grid
from .layers.block import Block
from .layers.attention import MemEffAttention
from .dinov2 import vit_large

class TransformerDecoder(nn.Module):
    def __init__(self, blocks, hidden_dim, out_dim, is_classifier = False, *args, 
                 amp = False, pos_enc = True, learned_embeddings = False, embedding_dim = None, amp_dtype = torch.float16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = blocks
        self.to_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._scales = [16]
        self.is_classifier = is_classifier
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.pos_enc = pos_enc
        self.learned_embeddings = learned_embeddings
        if self.learned_embeddings:
            self.learned_pos_embeddings = nn.Parameter(nn.init.kaiming_normal_(torch.empty((1, hidden_dim, embedding_dim, embedding_dim))))

    def scales(self):
        return self._scales.copy()

    def forward(self, gp_posterior, features, old_stuff, new_scale):
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):
            B,C,H,W = gp_posterior.shape
            x = torch.cat((gp_posterior, features), dim = 1)
            B,C,H,W = x.shape
            grid = get_grid(B, H, W, x.device).reshape(B,H*W,2)
            if self.learned_embeddings:
                pos_enc = F.interpolate(self.learned_pos_embeddings, size = (H,W), mode = 'bilinear', align_corners = False).permute(0,2,3,1).reshape(1,H*W,C)
            else:
                pos_enc = 0
            tokens = x.reshape(B,C,H*W).permute(0,2,1) + pos_enc
            z = self.blocks(tokens)
            out = self.to_out(z)
            out = out.permute(0,2,1).reshape(B, self.out_dim, H, W)
            warp, certainty = out[:, :-1], out[:, -1:]
            return warp, certainty, None


