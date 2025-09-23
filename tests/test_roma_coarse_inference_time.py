from romatch import roma_outdoor
import torch
from tqdm import tqdm
import time


def test_inference_time(model, name):
    T = 100
    im_A = torch.randn(8, 3, 560, 560).to(device)
    im_B = torch.randn(8, 3, 560, 560).to(device)
    # burn in
    for i in range(10):
        model.match(im_A, im_B, batched=True)
    start_time = time.time()
    for t in tqdm(range(T)):
        model.match(im_A, im_B, batched=True)
    end_time = time.time()
    return (end_time - start_time) / T


if __name__ == "__main__":
    device = "cuda"
    model = roma_outdoor(
        device=device,
        coarse_res=560,
        upsample_res=None,
        symmetric=True,
        upsample_preds=False,
        use_custom_corr=True,
    )
    experiment_name = "roma_latest"
    results = test_inference_time(model, experiment_name)