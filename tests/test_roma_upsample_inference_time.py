from romatch import roma_outdoor
import torch
from tqdm import tqdm
import time


def test_inference_time(model, name):
    T = 1000
    im_A = torch.randn(8, 3, 560, 560).to(device)
    im_B = torch.randn(8, 3, 560, 560).to(device)
    im_A_high_res = torch.randn(8, 3, 864, 864).to(device)
    im_B_high_res = torch.randn(8, 3, 864, 864).to(device)
    # burn in
    for i in range(10):
        model.match(
            im_A,
            im_B,
            im_A_high_res=im_A_high_res,
            im_B_high_res=im_B_high_res,
            batched=True,
        )
    start_time = time.time()
    for t in tqdm(range(T)):
        model.match(
            im_A,
            im_B,
            im_A_high_res=im_A_high_res,
            im_B_high_res=im_B_high_res,
            batched=True,
        )
    end_time = time.time()
    return (end_time - start_time) / T


if __name__ == "__main__":
    device = "cuda"
    model = roma_outdoor(
        device=device,
        coarse_res=560,
        upsample_res=864,
        symmetric=True,
        upsample_preds=True,
        use_custom_corr=True,
        amp_dtype=torch.bfloat16,
    )
    experiment_name = "roma_latest"
    results = test_inference_time(model, experiment_name)
