import torch

def kde(x, std = 0.1):
    # use a gaussian kernel to estimate density
    x = x.half() # Do it in half precision TODO: remove hardcoding
    scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density


def approx_kde(x, std = 0.1, max_num_cmp = 30_000):
    # use a gaussian kernel to estimate density
    if len(x.shape) > 2:
        raise ValueError(f"Needs shape N, D got shape {x.shape}")
    x = x.half() # Do it in half precision TODO: remove hardcoding
    y = torch.multinomial(x, min(max_num_cmp,len(x.shape[-2])), replacement=False)
    scores = (-torch.cdist(x,y)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density
