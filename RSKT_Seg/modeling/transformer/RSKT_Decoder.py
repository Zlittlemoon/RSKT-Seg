import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert
from .EfficientAggregator import EffAggregatorLayer
from .OriAggregator import OriAggregatorLayer
from .RSKT_Upsample import RSKT_Upsample
from .visualize_corr import visualize_corr


class RSKT_Decoder(nn.Module):
    def __init__(
        self,
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims=(64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        decoder_clip_guidance_dims=(256, 128),
        decoder_clip_guidance_proj_dims=(32, 16),
        decoder_dino_guidance_dims=(256, 128),
        decoder_dino_guidance_proj_dims=(32, 16),
        feat_dim=512,
        use_clip_corr=True,
        use_dino_corr=True,
        fusion_type="simple_separate",
        use_remote_clip=True,
        use_remote_dino=True,
        use_rotate=True,
        num_layers=4,
        use_efficient=True,
        nheads=4,
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type="linear",
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        self.use_remote_clip = use_remote_clip
        self.use_remote_dino = use_remote_dino
        self.use_rotate = use_rotate

        if use_efficient:
            self.layers = nn.ModuleList(
                [
                    EffAggregatorLayer(
                        hidden_dim=hidden_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    OriAggregatorLayer(
                        hidden_dim=hidden_dim,
                        text_guidance_dim=text_guidance_proj_dim,
                        appearance_guidance=appearance_guidance_proj_dim,
                        nheads=nheads,
                        input_resolution=feature_resolution,
                        pooling_size=pooling_size,
                        window_size=window_size,
                        attention_type=attention_type,
                        pad_len=pad_len,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.guidance_projection = (
            nn.Sequential(
                nn.Conv2d(
                    appearance_guidance_dim,
                    appearance_guidance_proj_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
            )
            if appearance_guidance_dim > 0
            else None
        )

        self.text_guidance_projection = (
            nn.Sequential(
                nn.Linear(text_guidance_dim, text_guidance_proj_dim),
                nn.ReLU(),
            )
            if text_guidance_dim > 0
            else None
        )

        self.CLIP_decoder_guidance_projection = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                    )
                    for d, dp in zip(decoder_guidance_dims, decoder_clip_guidance_proj_dims)
                ]
            )
            if decoder_clip_guidance_dims[0] > 0
            else None
        )

        if self.use_rotate:
            self.conv1 = nn.Conv2d(prompt_channel * 4, hidden_dim, kernel_size=7, stride=1, padding=3)
        else:
            self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        if self.use_remote_clip:
            self.CLIP_decoder_guidance_projection_remote = (
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                        )
                        for d, dp in zip(decoder_guidance_dims, decoder_clip_guidance_proj_dims)
                    ]
                )
                if decoder_clip_guidance_dims[0] > 0
                else None
            )
        else:
            self.CLIP_decoder_guidance_projection_remote = None

        if self.use_remote_dino:
            if fusion_type == "simple_concatenation":
                self.conv1 = nn.Conv2d(prompt_channel * 5, hidden_dim, kernel_size=7, stride=1, padding=3)
            elif fusion_type == "simple_mean":
                self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
            elif fusion_type == "simple_separate":
                self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
                self.fusion_corr = nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)

            self.DINO_decoder_guidance_projection = (
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                        )
                        for d, dp in zip(decoder_guidance_dims, decoder_dino_guidance_proj_dims)
                    ]
                )
                if decoder_dino_guidance_dims[0] > 0
                else None
            )
        else:
            self.conv2 = None
            self.fusion_corr = None
            self.fusion_feats = None
            self.DINO_decoder_guidance_projection = None

        self.Fusiondecoder1 = RSKT_Upsample(
            hidden_dim,
            decoder_dims[0],
            decoder_clip_guidance_proj_dims[0],
            decoder_dino_guidance_proj_dims[0],
            use_remote_clip=use_remote_clip,
            use_remote_dino=use_remote_dino,
        )

        self.Fusiondecoder2 = RSKT_Upsample(
            decoder_dims[0],
            decoder_dims[1],
            decoder_clip_guidance_proj_dims[1],
            decoder_dino_guidance_proj_dims[1],
            use_remote_clip=use_remote_clip,
            use_remote_dino=use_remote_dino,
        )

        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.pad_len = pad_len

        # =========================================================================
        # LAST-ViT spatial cost-map guidance branch
        # =========================================================================
        # 这一版不再使用：
        #   class-wise LAST token -> class_prior -> final logit calibration
        #
        # 而是使用：
        #   LAST spatial score + CLIP-text cost map
        #   -> class-wise spatial selection score
        #   -> refine CLIP cost map before RS-Fusion
        #
        # 这样可以保留 [B, T, H, W] 空间信息，避免压缩成 [B, T] 的全局类别 prior。
        # =========================================================================

        # 关闭原来的 class-level global prior。
        self.use_classwise_last_prior = False
        self.last_prior_alpha = None

        # 打开新的 spatial guidance。
        self.use_last_spatial_guidance = True

        # LAST score 只是弱稳定性辅助。
        # selection_score = norm(corr) + last_score_weight * norm(last_score)
        self.last_score_weight = 0.10

        # spatial selection score 注入 CLIP cost map 的强度。
        # 第一版建议 0.05，避免过强影响原 RSKT 主路径。
        self.last_corr_beta = 0.05

        # 是否 detach selection score。
        # 第一版建议 True，避免 selection 分支反向扰动 CLIP encoder。
        self.detach_last_selection = True

        self._debug_last_score_iter = 0
        self._debug_last_score_warmup = 3
        self._debug_last_score_every = 500

    # -------------------------------------------------------------------------
    # Basic shape helpers
    # -------------------------------------------------------------------------
    def _infer_hw_from_tokens(self, n_tokens: int):
        h = int(n_tokens ** 0.5)
        if h * h != n_tokens:
            raise ValueError(f"Cannot infer square spatial size from token number: {n_tokens}")
        return h, h

    def _to_feature_map(self, feat):
        """
        Convert CLIP dense output to [B, C, H, W].

        Supported:
            [B, C, H, W]
            [B, L, C], where L = H*W or 1 + H*W
        """
        if feat is None:
            return None

        if feat.dim() == 4:
            return feat

        if feat.dim() == 3:
            B, L, C = feat.shape

            # If there is a cls token, remove it.
            h = int((L - 1) ** 0.5)
            if h * h == L - 1:
                patch_feat = feat[:, 1:, :]
                H, W = h, h
            else:
                H, W = self._infer_hw_from_tokens(L)
                patch_feat = feat

            patch_feat = rearrange(patch_feat, "B (H W) C -> B C H W", H=H, W=W)
            return patch_feat

        raise ValueError(f"Unsupported feature shape: {feat.shape}")

    def _last_score_to_map(self, last_score, H, W):
        """
        Convert LAST spatial_score to [B, 1, H, W].

        Supported:
            [B, H*W]
            [B, 1+H*W]
            [B, H, W]
            [B, 1, H, W]
        """
        if last_score is None:
            return None

        if last_score.dim() == 4:
            score_map = last_score
            if score_map.shape[-2:] != (H, W):
                score_map = F.interpolate(score_map, size=(H, W), mode="bilinear", align_corners=False)
            return score_map

        if last_score.dim() == 3:
            # [B, H, W]
            if last_score.shape[-2:] == (H, W):
                return last_score.unsqueeze(1)

        if last_score.dim() == 2:
            B, L = last_score.shape

            if L == H * W:
                return last_score.view(B, 1, H, W)

            if L == H * W + 1:
                return last_score[:, 1:].view(B, 1, H, W)

            # fallback: infer from L
            h = int(L ** 0.5)
            if h * h == L:
                score_map = last_score.view(B, 1, h, h)
                if (h, h) != (H, W):
                    score_map = F.interpolate(score_map, size=(H, W), mode="bilinear", align_corners=False)
                return score_map

        raise ValueError(f"Unsupported last_score shape: {last_score.shape}, expected spatial size {(H, W)}")

    def _normalize_spatial_map(self, x, eps=1e-6):
        """
        Normalize spatial map to [0, 1] for each sample/class.

        x:
            [..., H, W]
        """
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min).clamp_min(eps)

    # -------------------------------------------------------------------------
    # Correlation
    # -------------------------------------------------------------------------
    def correlation(self, img_feats, text_feats, last_score=None):
        """
        img_feats:
            [B, C, H, W] or [B, L, C]

        text_feats:
            [B, T, P, C]

        return:
            corr: [B, P, T, H, W]
        """
        img_feats = self._to_feature_map(img_feats)
        img_feats = F.normalize(img_feats, dim=1)
        text_feats = F.normalize(text_feats, dim=-1)

        corr = torch.einsum("bchw,btpc->bpthw", img_feats, text_feats)
        return corr

    def correlation_rotate(self, img_feats, text_feats, last_score=None):
        """
        img_feats:
            list of four CLIP dense outputs.
            each one can be [B, C, H, W] or [B, L, C]

        text_feats:
            [B, T, P, C]

        return:
            corr: [B, 4P, T, H, W]
        """
        img_feats0 = F.normalize(self._to_feature_map(img_feats[0]), dim=1)
        img_feats1 = F.normalize(self._to_feature_map(img_feats[1]), dim=1)
        img_feats2 = F.normalize(self._to_feature_map(img_feats[2]), dim=1)
        img_feats3 = F.normalize(self._to_feature_map(img_feats[3]), dim=1)

        # rotate back to original coordinate
        img_feats1 = torch.rot90(img_feats1, k=3, dims=(2, 3))
        img_feats2 = torch.rot90(img_feats2, k=2, dims=(2, 3))
        img_feats3 = torch.rot90(img_feats3, k=1, dims=(2, 3))

        img_feats = torch.cat(
            (
                img_feats0.unsqueeze(dim=1),
                img_feats1.unsqueeze(dim=1),
                img_feats2.unsqueeze(dim=1),
                img_feats3.unsqueeze(dim=1),
            ),
            dim=1,
        )
        # [B, 4, C, H, W]

        text_feats = F.normalize(text_feats, dim=-1)

        corr = torch.einsum("bnchw,btpc->bnpthw", img_feats, text_feats)
        corr = rearrange(corr, "B N P T H W -> B (N P) T H W")
        return corr

    # -------------------------------------------------------------------------
    # LAST spatial guidance
    # -------------------------------------------------------------------------
    def build_last_selection_score(self, img_feats, text_feats, corr, last_score):
        """
        Build class-wise LAST spatial guidance map.

        This replaces the previous:
            class-wise LAST token -> class_prior [B, T]

        with:
            selection_score [B, T, H, W]

        img_feats:
            [B, C, H, W] or [B, L, C]

        text_feats:
            [B, T, P, C]
            kept for interface consistency.

        corr:
            [B, P, T, H, W]

        last_score:
            [B, H*W] or equivalent

        return:
            selection_score: [B, T, H, W]
        """
        if img_feats is None or text_feats is None or corr is None or last_score is None:
            return None

        img_map = self._to_feature_map(img_feats)
        B, C, H, W = img_map.shape

        score_map = self._last_score_to_map(last_score, H, W)
        if score_map is None:
            return None

        score_map = self._normalize_spatial_map(score_map)
        # [B, 1, H, W]

        # corr: [B, P, T, H, W] -> [B, T, H, W]
        corr_for_select = corr.mean(dim=1)

        # Convert correlation to non-negative class relevance.
        corr_for_select = self._normalize_spatial_map(corr_for_select)
        # [B, T, H, W]

        # LAST score only acts as a weak stability auxiliary signal.
        # Main signal is still class-wise CLIP-text relevance.
        selection_score = corr_for_select + self.last_score_weight * score_map
        # [B, T, H, W]

        # Normalize again after fusion.
        selection_score = self._normalize_spatial_map(selection_score)

        # Remove class-wise spatial mean.
        # This keeps only spatial contrast and avoids degeneration into class-level global prior.
        selection_score = selection_score - selection_score.mean(dim=(-2, -1), keepdim=True)

        if self.detach_last_selection:
            selection_score = selection_score.detach()

        return selection_score

    def build_last_selection_score_from_clip(self, img_feats, text_feats, last_score):
        """
        First spatial-guidance version:
            If rotation is used, only use the original 0-degree CLIP branch
            to build LAST selection score.

        The original RSKT rotation cost map is still kept unchanged.
        The generated selection_score will be broadcast to all prompt/rotation channels.

        return:
            selection_score: [B, T, H, W] or None
        """
        if not self.use_last_spatial_guidance:
            return None

        if img_feats is None or last_score is None:
            return None

        if isinstance(img_feats, list):
            base_img_feats = img_feats[0]
            base_last_score = last_score[0] if isinstance(last_score, list) else last_score
        else:
            base_img_feats = img_feats
            base_last_score = last_score

        if base_img_feats is None or base_last_score is None:
            return None

        base_corr = self.correlation(base_img_feats, text_feats)

        selection_score = self.build_last_selection_score(
            img_feats=base_img_feats,
            text_feats=text_feats,
            corr=base_corr,
            last_score=base_last_score,
        )

        return selection_score

    def apply_last_spatial_guidance(self, corr, selection_score):
        """
        Inject LAST-guided spatial score into CLIP cost map.

        corr:
            [B, P, T, H, W]

        selection_score:
            [B, T, H, W]

        return:
            refined corr: [B, P, T, H, W]
        """
        if corr is None or selection_score is None:
            return corr

        return corr + self.last_corr_beta * selection_score.unsqueeze(1)

    # -------------------------------------------------------------------------
    # Original RSKT modules
    # -------------------------------------------------------------------------
    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, "B P T H W -> (B T) P H W")
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, "(B T) C H W -> B C T H W", B=B)
        return corr_embed

    def simple_separate_corr(self, clip_corr, dino_corr, files_name):
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()

        clip_corr = rearrange(clip_corr, "B P T H W -> (B T) P H W")
        dino_corr = rearrange(dino_corr, "B P T H W -> (B T) P H W")

        # visualize_corr(
        #     clip_corr.permute(1, 0, 2, 3)[0].unsqueeze(0),
        #     files_name[0],
        #     save_prefix="./vis_cost_clip_DLRSD/",
        # )
        # visualize_corr(
        #     dino_corr.permute(1, 0, 2, 3),
        #     files_name[0],
        #     save_prefix="./vis_cost_dino_DLRSD/",
        # )

        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)

        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)

        fused_corr = torch.cat([clip_embed_corr, dino_embed_corr], dim=1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, "(B T) C H W -> B C T H W", B=B)

        clip_embed_corr = rearrange(clip_embed_corr, "(B T) C H W -> B C T H W", B=B)
        dino_embed_corr = rearrange(dino_embed_corr, "(B T) C H W -> B C T H W", B=B)

        return fused_corr, clip_embed_corr, dino_embed_corr

    def simple_concatenation_corr(self, clip_corr, dino_corr):
        B = clip_corr.shape[0]

        clip_corr = rearrange(clip_corr, "B P T H W -> (B T) P H W")
        dino_corr = rearrange(dino_corr, "B P T H W -> (B T) P H W")

        fused_corr = self.conv1(torch.cat([clip_corr, dino_corr], dim=1))
        fused_corr = rearrange(fused_corr, "(B T) C H W -> B C T H W", B=B)

        return fused_corr

    def simple_mean_corr(self, clip_corr, dino_corr):
        B = clip_corr.shape[0]

        clip_corr = rearrange(clip_corr, "B P T H W -> (B T) P H W")
        dino_corr = rearrange(dino_corr, "B P T H W -> (B T) P H W")

        fused_corr = self.conv1(
            torch.mean(
                torch.cat([clip_corr, dino_corr], dim=1),
                dim=1,
                keepdim=True,
            )
        )
        fused_corr = rearrange(fused_corr, "(B T) C H W -> B C T H W", B=B)

        return fused_corr

    def corr_fusion_embed(self, clip_corr, dino_corr):
        B = clip_corr.shape[0]

        clip_corr = rearrange(clip_corr, "B P T H W -> (B T) P H W")
        dino_corr = rearrange(dino_corr, "B P T H W -> (B T) P H W")

        fused_corr = torch.cat([clip_corr, dino_corr], dim=1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, "(B T) C H W -> B C T H W", B=B)

        return fused_corr

    def Fusion_conv_decoer(self, x, clip_guidance, clip_guidance_remote, dino_guidance):
        B = x.shape[0]

        corr_embed = rearrange(x, "B C T H W -> (B T) C H W")
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0], clip_guidance_remote[0], dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1], clip_guidance_remote[1], dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, "(B T) () H W -> B T H W", B=B)

        return corr_embed

    def forward(
        self,
        files_name,
        img_feats,
        dino_feat,
        text_feats,
        appearance_guidance,
        appearance_guidance_remote,
        dino_guidance,
        last_score=None,
    ):
        """
        Arguments:
            img_feats:
                CLIP dense features.
                Can be [B, L, C], [B, C, H, W], or list of 4 branches when rotation is enabled.

            dino_feat:
                DINO dense feature, usually [B, C, H, W].

            text_feats:
                [B, T, P, C]

            appearance_guidance:
                CLIP decoder guidance features.

            appearance_guidance_remote:
                RemoteCLIP decoder guidance features.

            dino_guidance:
                DINO decoder guidance features.

            last_score:
                LAST-ViT spatial score.
                Can be [B, H*W] or list of 4 branches when rotation is enabled.
        """
        self._debug_last_score_iter += 1
        should_log_last_score = (
            self._debug_last_score_iter <= self._debug_last_score_warmup
            or self._debug_last_score_iter % self._debug_last_score_every == 0
        )

        if should_log_last_score:
            if isinstance(last_score, list):
                print("decoder last_score:", [None if x is None else x.shape for x in last_score])
            else:
                print("decoder last_score:", None if last_score is None else last_score.shape)

            print("use_last_spatial_guidance:", self.use_last_spatial_guidance)
            print("last_corr_beta:", float(self.last_corr_beta))
            print("last_score_weight:", float(self.last_score_weight))
            print("detach_last_selection:", self.detach_last_selection)

        # Build LAST spatial guidance map.
        # This keeps spatial information [B, T, H, W],
        # instead of compressing it into class prior [B, T].
        last_selection_score = None
        if self.use_last_spatial_guidance:
            last_selection_score = self.build_last_selection_score_from_clip(
                img_feats=img_feats,
                text_feats=text_feats,
                last_score=last_score,
            )

        if dino_feat is not None and img_feats is not None:
            if self.fusion_type == "simple_separate":
                if isinstance(img_feats, list):
                    corr = self.correlation_rotate(img_feats, text_feats)
                else:
                    corr = self.correlation(img_feats, text_feats)

                # Inject LAST spatial guidance into CLIP cost map.
                corr = self.apply_last_spatial_guidance(corr, last_selection_score)

                dino_corr = self.correlation(dino_feat, text_feats)

                fused_corr_embed, clip_embed_corr, dino_embed_corr = self.simple_separate_corr(
                    clip_corr=corr,
                    dino_corr=dino_corr,
                    files_name=files_name,
                )

                fused_corr_embed = fused_corr_embed + clip_embed_corr

            elif self.fusion_type == "simple_concatenation":
                if isinstance(img_feats, list):
                    corr = self.correlation_rotate(img_feats, text_feats)
                else:
                    corr = self.correlation(img_feats, text_feats)

                # Inject LAST spatial guidance into CLIP cost map.
                corr = self.apply_last_spatial_guidance(corr, last_selection_score)

                dino_corr = self.correlation(dino_feat, text_feats)
                fused_corr_embed = self.simple_concatenation_corr(corr, dino_corr)

            elif self.fusion_type == "simple_mean":
                if isinstance(img_feats, list):
                    corr = self.correlation_rotate(img_feats, text_feats)
                else:
                    corr = self.correlation(img_feats, text_feats)

                # Inject LAST spatial guidance into CLIP cost map.
                corr = self.apply_last_spatial_guidance(corr, last_selection_score)

                dino_corr = self.correlation(dino_feat, text_feats)
                fused_corr_embed = self.simple_mean_corr(corr, dino_corr)

            else:
                raise NotImplementedError(f"Unknown fusion_type: {self.fusion_type}")

        elif dino_feat is not None and img_feats is None:
            corr = self.correlation(dino_feat, text_feats)
            fused_corr_embed = self.corr_embed(corr)
            print("Only DINO feature is used.")

        elif dino_feat is None and img_feats is not None:
            if isinstance(img_feats, list):
                corr = self.correlation_rotate(img_feats, text_feats)
            else:
                corr = self.correlation(img_feats, text_feats)

            # Inject LAST spatial guidance into CLIP cost map.
            corr = self.apply_last_spatial_guidance(corr, last_selection_score)

            fused_corr_embed = self.corr_embed(corr)
            print("Only CLIP feature is used.")

        else:
            raise ValueError("Both img_feats and dino_feat are None.")

        projected_guidance, projected_text_guidance = None, None
        CLIP_projected_decoder_guidance = [None, None]
        CLIP_projected_decoder_guidance_remote = [None, None]
        DINO_projected_decoder_guidance = [None, None]

        if self.guidance_projection is not None and appearance_guidance is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.guidance_projection is not None and appearance_guidance is None and dino_feat is not None:
            projected_guidance = self.guidance_projection(dino_feat)

        if self.CLIP_decoder_guidance_projection is not None and appearance_guidance is not None:
            CLIP_projected_decoder_guidance = [
                proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])
            ]

        if self.CLIP_decoder_guidance_projection_remote is not None and appearance_guidance_remote is not None:
            CLIP_projected_decoder_guidance_remote = [
                proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection_remote, appearance_guidance_remote)
            ]

        if self.DINO_decoder_guidance_projection is not None and dino_guidance is not None:
            DINO_projected_decoder_guidance = [
                proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)
            ]

        if self.text_guidance_projection is not None:
            text_for_guidance = text_feats.mean(dim=-2)
            text_for_guidance = text_for_guidance / text_for_guidance.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_for_guidance)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)

        logit = self.Fusion_conv_decoer(
            fused_corr_embed,
            CLIP_projected_decoder_guidance,
            CLIP_projected_decoder_guidance_remote,
            DINO_projected_decoder_guidance,
        )

        # Do not apply class-level global prior in this experiment.
        # LAST information has already been injected into CLIP cost map spatially.
        return logit