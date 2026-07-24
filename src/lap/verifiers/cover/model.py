"""Project-owned two-view 7-D CoVer verifier."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from lap.verifiers.cover.w3_contracts import BACKBONE_ID
from lap.verifiers.cover.w3_contracts import BACKBONE_REVISION
from lap.verifiers.cover.w3_contracts import HISTORY_SHAPE


@dataclass(frozen=True)
class VerifierConfig:
    backbone_width: int = 1024
    embedding_width: int = 512
    visual_tokens: int = 576
    num_heads: int = 8
    pooling_layers: int = 4
    trajectory_layers: int = 4
    feed_forward_width: int = 1024
    history_length: int = 10
    action_width: int = 7
    use_wrist: bool = True

    @property
    def fusion_input_width(self) -> int:
        return self.embedding_width * (3 if self.use_wrist else 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone_width": self.backbone_width,
            "embedding_width": self.embedding_width,
            "visual_tokens": self.visual_tokens,
            "num_heads": self.num_heads,
            "pooling_layers": self.pooling_layers,
            "trajectory_layers": self.trajectory_layers,
            "feed_forward_width": self.feed_forward_width,
            "history_length": self.history_length,
            "action_width": self.action_width,
            "use_wrist": self.use_wrist,
            "backbone_id": BACKBONE_ID,
            "backbone_revision": BACKBONE_REVISION,
        }


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(value)))


class AttentionPoolBlock(nn.Module):
    def __init__(self, *, width: int, input_width: int, heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, kdim=input_width, vdim=input_width)
        self.layer_norm = nn.LayerNorm(width)
        self.q_layer_norm = nn.LayerNorm(width)
        self.mlp = FeedForward(width, width)

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(self.q_layer_norm(query), tokens, tokens, need_weights=False)
        query = self.layer_norm(query + attended)
        return self.layer_norm(query + self.mlp(query))


class AttentionPool(nn.Module):
    def __init__(self, *, width: int, input_width: int, heads: int, layers: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(
            [AttentionPoolBlock(width=width, input_width=input_width, heads=heads) for _ in range(layers)]
        )
        self.layer_norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("pooling tokens must have shape [sequence, batch, width]")
        query = self.query.expand(-1, tokens.shape[1], -1)
        for block in self.blocks:
            query = block(query, tokens)
        return self.layer_norm(query[0])


class TextAwareVisualExtraction(nn.Module):
    def __init__(self, *, width: int, tokens: int) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("pos_emb", torch.zeros(tokens, width), persistent=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("backbone tokens must have shape [batch, sequence, width]")
        position = self.pos_emb
        if tokens.shape[1] != position.shape[0]:
            position = F.interpolate(position.T.unsqueeze(0), size=tokens.shape[1], mode="linear", align_corners=False)[
                0
            ].T
        return tokens + position.to(device=tokens.device, dtype=tokens.dtype) * self.temperature.to(tokens.dtype)


class VerifierModel(nn.Module):
    """Scores semantic contexts against fixed-length UR5e action histories."""

    def __init__(self, config: VerifierConfig, backbone: nn.Module) -> None:
        super().__init__()
        if config.history_length != HISTORY_SHAPE[0] or config.action_width != HISTORY_SHAPE[1]:
            raise ValueError("W3 model requires float32[10, 7] histories")
        self.config = config
        self.backbone = backbone
        self.text_aware_visual_extraction = TextAwareVisualExtraction(
            width=config.backbone_width, tokens=config.visual_tokens
        )
        self.vision_poolings = AttentionPool(
            width=config.embedding_width,
            input_width=config.backbone_width,
            heads=config.num_heads,
            layers=config.pooling_layers,
        )
        self.text_pooling = AttentionPool(
            width=config.embedding_width,
            input_width=config.backbone_width,
            heads=config.num_heads,
            layers=config.pooling_layers,
        )
        self.semantic_fusion = nn.Linear(config.fusion_input_width, config.embedding_width)
        self.action_step_encoder = nn.Linear(config.action_width, config.embedding_width)
        position = torch.arange(config.history_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.embedding_width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / config.embedding_width)
        )
        trajectory_position = torch.zeros(config.history_length, config.embedding_width)
        trajectory_position[:, 0::2] = torch.sin(position * div_term)
        trajectory_position[:, 1::2] = torch.cos(position * div_term[: trajectory_position[:, 1::2].shape[1]])
        self.register_buffer("trajectory_position", trajectory_position, persistent=True)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_width,
            nhead=config.num_heads,
            dim_feedforward=config.feed_forward_width,
            activation="gelu",
            batch_first=False,
            dropout=0.1,
        )
        self.trajectory_encoder = nn.TransformerEncoder(layer, num_layers=config.trajectory_layers)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)  # noqa: FBT003

    def train(self, mode: bool = True) -> VerifierModel:  # noqa: FBT001,FBT002
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in self.named_parameters() if value.requires_grad}

    def encode_semantic(
        self, base_rgb: torch.Tensor, wrist_rgb: torch.Tensor | None, instructions: Sequence[str]
    ) -> torch.Tensor:
        if base_rgb.ndim != 4 or base_rgb.shape[1:] != (3, 384, 384):
            raise ValueError("base_rgb must have shape [B, 3, 384, 384]")
        if self.config.use_wrist and wrist_rgb is None:
            raise ValueError("two-view verifier requires wrist_rgb")
        with torch.no_grad():
            base_tokens = self.backbone.encode_image_tokens(base_rgb)
            wrist_tokens = self.backbone.encode_image_tokens(wrist_rgb) if wrist_rgb is not None else None
            text_tokens = self.backbone.encode_text_tokens(instructions)
        base_tokens = self.text_aware_visual_extraction(base_tokens)
        base_feature = self.vision_poolings(base_tokens.transpose(0, 1))
        features = [base_feature]
        if self.config.use_wrist:
            assert wrist_tokens is not None
            wrist_tokens = self.text_aware_visual_extraction(wrist_tokens)
            features.append(self.vision_poolings(wrist_tokens.transpose(0, 1)))
        text_feature = self.text_pooling(text_tokens.transpose(0, 1))
        features.append(text_feature)
        return F.normalize(self.semantic_fusion(torch.cat(features, dim=-1)), dim=-1)

    def encode_action(self, histories: torch.Tensor) -> torch.Tensor:
        if histories.ndim != 3 or tuple(histories.shape[1:]) != HISTORY_SHAPE or histories.dtype != torch.float32:
            raise ValueError("action_histories must be float32[B, 10, 7]")
        if not torch.isfinite(histories).all():
            raise ValueError("action_histories must be finite")
        padding = torch.all(histories == -5.0, dim=-1)
        real_seen = torch.zeros(histories.shape[0], dtype=torch.bool, device=histories.device)
        for step in range(histories.shape[1]):
            invalid = padding[:, step] & real_seen
            if invalid.any():
                raise ValueError("action history padding must be leading full rows")
            real_seen |= ~padding[:, step]
        if (~real_seen).any():
            raise ValueError("action history must contain at least one real row")
        tokens = self.action_step_encoder(histories)
        tokens = tokens + self.trajectory_position.to(tokens.device, tokens.dtype).unsqueeze(0)
        encoded = self.trajectory_encoder(tokens.transpose(0, 1), src_key_padding_mask=padding)
        weights = (~padding).to(encoded.dtype).T.unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)

    def forward(
        self,
        base_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor | None,
        instructions: Sequence[str],
        histories: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        semantic = self.encode_semantic(base_rgb, wrist_rgb, instructions)
        action = self.encode_action(histories)
        scale = self.logit_scale.clamp(0.0, math.log(100.0)).exp()
        semantic_to_action = scale * semantic @ action.T
        action_to_semantic = semantic_to_action.T
        return {
            "semantic_embedding": semantic,
            "action_embedding": action,
            "semantic_to_action_logits": semantic_to_action,
            "action_to_semantic_logits": action_to_semantic,
        }

    def contrastive_loss(self, outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        count = outputs["semantic_to_action_logits"].shape[0]
        labels = torch.arange(count, device=outputs["semantic_to_action_logits"].device)
        semantic_loss = F.cross_entropy(outputs["semantic_to_action_logits"], labels)
        action_loss = F.cross_entropy(outputs["action_to_semantic_logits"], labels)
        loss = (semantic_loss + action_loss) / 2.0
        with torch.no_grad():
            top1 = (outputs["semantic_to_action_logits"].argmax(dim=1) == labels).float().mean()
        return loss, {
            "loss": float(loss.detach()),
            "semantic_loss": float(semantic_loss.detach()),
            "action_loss": float(action_loss.detach()),
            "retrieval_top1": float(top1),
        }


class TinyFrozenBackbone(nn.Module):
    """Deterministic fixture backbone with the same public contract as SigLIP2."""

    def __init__(self, *, width: int = 32, tokens: int = 8) -> None:
        super().__init__()
        self.width = width
        self.tokens = tokens
        self.image_projection = nn.Linear(3, width, bias=False)
        self.text_projection = nn.Linear(2, width, bias=False)
        with torch.no_grad():
            self.image_projection.weight.fill_(0.1)
            self.text_projection.weight.fill_(0.2)
        for parameter in self.parameters():
            parameter.requires_grad_(False)  # noqa: FBT003

    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        token = self.image_projection(pooled).unsqueeze(1)
        return token.expand(-1, self.tokens, -1)

    def encode_text_tokens(self, instructions: Sequence[str]) -> torch.Tensor:
        values = []
        for text in instructions:
            digest = hashlib.sha256(text.encode()).digest()
            values.append([digest[0] / 255.0, digest[1] / 255.0])
        inputs = torch.tensor(values, dtype=torch.float32, device=self.image_projection.weight.device)
        return self.text_projection(inputs).unsqueeze(1)


class OpenClipSigLIP2Backbone(nn.Module):
    """Lazy OpenCLIP adapter; tests inject TinyFrozenBackbone instead."""

    def __init__(self, *, pretrained: str | None = None) -> None:
        super().__init__()
        try:
            import open_clip  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - environment-specific
            raise RuntimeError("W3 requires open_clip_torch in the LAP environment") from error
        model, _, preprocess = open_clip.create_model_and_transforms(BACKBONE_ID, pretrained=pretrained)
        self.model = model.eval()
        self.preprocess = preprocess
        self.backbone_revision = BACKBONE_REVISION
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)  # noqa: FBT003
        self.tokenizer = open_clip.get_tokenizer(BACKBONE_ID)

    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual
        tokens = visual.forward_features(images) if hasattr(visual, "forward_features") else visual(images)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        return tokens

    def encode_text_tokens(self, instructions: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(instructions)).to(next(self.model.parameters()).device)
        features = self.model.encode_text(tokens)
        return features.unsqueeze(1)
