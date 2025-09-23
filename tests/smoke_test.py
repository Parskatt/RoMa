def test_smoke():
    import torch
    from romatch import roma_outdoor
    device = torch.device('cpu')
    model = roma_outdoor(device=device)
    assert model._get_device() == device
    assert model.w_resized == 560, f"Expected 560, got {model.w_resized}"
    assert model.h_resized == 560, f"Expected 560, got {model.h_resized}"
    assert model.upsample_res == (864, 864), f"Expected (864, 864), got {model.upsample_res}"

if __name__ == "__main__":
    test_smoke()