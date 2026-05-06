import torch


def gaussian_kernel_1d(length, sigma, device, dtype):
    x = torch.arange(length, device=device, dtype=torch.float32)
    x = x - (length - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.max().clamp_min(1e-6)
    return g.to(dtype=dtype).view(1, 1, length)


def last_vit_select(patch_tokens, k=1, sigma=64.0, eps=1e-6):
    """
    patch_tokens: [B, N, C]
    return:
        cls_token: [B, C]
        spatial_score: [B, N]
    """
    B, N, C = patch_tokens.shape

    x_detach = patch_tokens
    x_float = patch_tokens.float()

    x_fft = torch.fft.fft(x_float, dim=-1)
    g = gaussian_kernel_1d(C, sigma, x_float.device, x_float.dtype)

    x_fft = torch.fft.fftshift(x_fft, dim=-1)
    x_fft = x_fft * g
    x_fft = torch.fft.ifftshift(x_fft, dim=-1)

    x_low = torch.fft.ifft(x_fft, dim=-1).real

    score = x_float.abs() / (x_low - x_float).abs().clamp_min(eps)

    k = min(k, N)
    _, indices = torch.topk(score, k=k, dim=1, largest=True)

    selected = torch.gather(x_detach, dim=1, index=indices)
    cls_token = selected.mean(dim=1)

    spatial_score = score.mean(dim=-1)
    spatial_score = spatial_score - spatial_score.amin(dim=1, keepdim=True)
    spatial_score = spatial_score / spatial_score.amax(dim=1, keepdim=True).clamp_min(eps)

    return cls_token.to(patch_tokens.dtype), spatial_score.to(patch_tokens.dtype)
