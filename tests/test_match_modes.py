import torch
from PIL import Image
from romatch.models.matcher import RegressionMatcher


def test_bs_one_tensor_inputs(model: RegressionMatcher, device, coarse_res: int, upsample_res: int):
    model.match(
        torch.randn(1, 3, coarse_res, coarse_res).to(device),
        torch.randn(1, 3, coarse_res, coarse_res).to(device),
        im_A_high_res=torch.randn(1, 3, upsample_res, upsample_res).to(device),
        im_B_high_res=torch.randn(1, 3, upsample_res, upsample_res).to(device),
    )


def test_bs_8_tensor_inputs(model: RegressionMatcher, device, coarse_res: int, upsample_res: int):
    model.match(
        torch.randn(8, 3, coarse_res, coarse_res).to(device),
        torch.randn(8, 3, coarse_res, coarse_res).to(device),
        im_A_high_res=torch.randn(8, 3, upsample_res, upsample_res).to(device),
        im_B_high_res=torch.randn(8, 3, upsample_res, upsample_res).to(device),
    )


def test_pil_inputs(model: RegressionMatcher):
    model.match(Image.open("assets/toronto_A.jpg"), Image.open("assets/toronto_B.jpg"))


def test_str_inputs(model: RegressionMatcher):
    model.match("assets/toronto_A.jpg", "assets/toronto_B.jpg")


if __name__ == "__main__":
    from romatch import roma_outdoor

    coarse_res = 560
    upsample_res = 1152
    for device in [torch.device("cuda")]:
        model = roma_outdoor(
            device=device,
            coarse_res=coarse_res,
            upsample_res=upsample_res,
            use_custom_corr=True,
            symmetric=True,
            upsample_preds=True,
        )
        for is_symmetric in [True, False]:
            for upsample_preds in [True, False]:
                for batched in [True, False]:
                    model.symmetric = is_symmetric
                    model.upsample_preds = upsample_preds
                    model.batched = batched
                    test_bs_one_tensor_inputs(model, device, coarse_res, upsample_res)
                    test_bs_8_tensor_inputs(model, device, coarse_res, upsample_res)
                    test_pil_inputs(model)
                    test_str_inputs(model)
                    print(f"Done with {is_symmetric=}, {upsample_preds=}, {batched=}, {device=}")

    for device in [torch.device("cpu")]:
        model = roma_outdoor(
            device=device,
            coarse_res=coarse_res,
            upsample_res=upsample_res,
            use_custom_corr=True,
            symmetric=True,
            upsample_preds=True,
        )
        model.symmetric = is_symmetric
        model.upsample_preds = upsample_preds
        model.batched = batched
        model.device = device
        test_bs_one_tensor_inputs(model, device, coarse_res, upsample_res)
        test_bs_8_tensor_inputs(model, device, coarse_res, upsample_res)
        test_pil_inputs(model)
        test_str_inputs(model)
        print(f"Done with {is_symmetric=}, {upsample_preds=}, {batched=}, {device=}")
