import torch


def last_vit_select(patch_tokens: torch.Tensor, k: int = 1, sigma: float = 64.0):
    """
    Select a CLS-like token from patch tokens and return per-patch score.
    patch_tokens: [B, N, C]
    returns:
      cls_token: [B, C]
      last_score: [B, N]
    """
    if patch_tokens.ndim != 3:
        raise ValueError(f"patch_tokens must be [B, N, C], got {patch_tokens.shape}")

    score = patch_tokens.pow(2).mean(dim=-1)
    if sigma is not None and sigma > 0:
        score = score / float(sigma)
    score = torch.softmax(score, dim=-1)

    k = max(1, min(int(k), patch_tokens.shape[1]))
    if k == 1:
        idx = score.argmax(dim=-1, keepdim=True)
        cls_token = torch.gather(
            patch_tokens, 1, idx.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        ).squeeze(1)
    else:
        topk_score, topk_idx = torch.topk(score, k=k, dim=-1)
        topk_tokens = torch.gather(
            patch_tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        )
        weight = topk_score / (topk_score.sum(dim=-1, keepdim=True) + 1e-6)
        cls_token = (topk_tokens * weight.unsqueeze(-1)).sum(dim=1)

    return cls_token, score
