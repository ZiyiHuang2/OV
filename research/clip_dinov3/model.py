"""Standalone frozen CLIP and DINOv3 patch encoders with a trainable projection."""

import math
from hashlib import sha256

import clip
from dinov3.hub.backbones import Weights, dinov3_vitb16
import torch
from torch import nn
from torch.nn import functional as F


CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
DINO_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
PROMPT = "a photo of a {}."
CLIP_WIDTH, DINO_WIDTH, PATCH_SIZE = 512, 768, 16


def verify_weights(path, expected):
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"Unexpected checkpoint SHA-256 for {path}: {actual}")
    return actual


class DualPatchSegmenter(nn.Module):
    def __init__(self, clip_path, dino_path, size):
        super().__init__()
        if size <= 0 or size % PATCH_SIZE:
            raise ValueError("Encoder input size must be a positive multiple of 16")
        self.size = size
        self.weight_hashes = {"clip": verify_weights(clip_path, CLIP_SHA256)}
        self.clip, _ = clip.load(str(clip_path), device="cpu", jit=False)
        self.clip.requires_grad_(False).eval()
        self.register_buffer("clip_mean", self.rgb_stat((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("clip_std", self.rgb_stat((0.26862954, 0.26130258, 0.27577711)))
        self.dino = None
        if dino_path is None:
            return
        self.weight_hashes["dinov3"] = verify_weights(dino_path, DINO_SHA256)
        self.dino = dinov3_vitb16(pretrained=False, weights=Weights.LVD1689M)
        self.dino.load_state_dict(torch.load(dino_path, map_location="cpu", weights_only=True), strict=True)
        self.dino.requires_grad_(False).eval()
        self.projection = nn.Linear(CLIP_WIDTH + DINO_WIDTH, CLIP_WIDTH)
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.weight[:, :CLIP_WIDTH].copy_(torch.eye(CLIP_WIDTH))
            self.projection.bias.zero_()
        self.register_buffer("dino_mean", self.rgb_stat((0.485, 0.456, 0.406)))
        self.register_buffer("dino_std", self.rgb_stat((0.229, 0.224, 0.225)))

    @staticmethod
    def rgb_stat(values):
        return torch.tensor(values).view(1, 3, 1, 1)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        if self.dino is not None:
            self.dino.eval()
        return self

    @torch.no_grad()
    def text_features(self, classes):
        tokens = clip.tokenize([PROMPT.format(name) for name in classes])
        return F.normalize(self.clip.encode_text(tokens.to(self.clip_mean.device)).float(), dim=-1)

    @torch.no_grad()
    def clip_tokens(self, rgb):
        """Return the class token followed by all row-major projected patch tokens."""
        visual = self.clip.visual
        x = visual.conv1((rgb - self.clip_mean) / self.clip_std)
        batch, width, height, grid_width = x.shape
        patch_tokens = x.flatten(2).transpose(1, 2)
        class_token = visual.class_embedding.view(1, 1, width).expand(batch, -1, -1)
        position = visual.positional_embedding
        native_grid = math.isqrt(position.shape[0] - 1)
        if (height, grid_width) != (native_grid, native_grid):
            patch_position = position[1:].T.reshape(1, width, native_grid, native_grid)
            patch_position = F.interpolate(
                patch_position, (height, grid_width), mode="bicubic", align_corners=False,
            ).flatten(2)[0].T
            position = torch.cat((position[:1], patch_position), dim=0)
        x = visual.ln_pre(torch.cat((class_token, patch_tokens), dim=1) + position)
        x = visual.transformer(x.transpose(0, 1)).transpose(0, 1)
        return visual.ln_post(x) @ visual.proj

    @torch.no_grad()
    def resize_rgb(self, images):
        return F.interpolate(
            images.float() / 255.0, (self.size, self.size), mode="bilinear",
            align_corners=False, antialias=True,
        )

    @torch.no_grad()
    def encode_patches(self, images):
        rgb = self.resize_rgb(images)
        clip_patches = self.clip_tokens(rgb)[:, 1:]
        dino_input = (rgb - self.dino_mean) / self.dino_std
        dino_patches = self.dino.forward_features(dino_input)["x_norm_patchtokens"].float()
        return clip_patches, dino_patches

    def pixel_logits(self, patches, text, output_size):
        similarities = F.normalize(patches, dim=-1) @ text.T
        grid = self.size // PATCH_SIZE
        logits = similarities.transpose(1, 2).reshape(patches.shape[0], len(text), grid, grid)
        logits = logits * self.clip.logit_scale.exp()
        return F.interpolate(logits, output_size, mode="bilinear", align_corners=False)

    def forward(self, images, text):
        if self.dino is None:
            patches = self.clip_tokens(self.resize_rgb(images))[:, 1:]
            return self.pixel_logits(patches, text, images.shape[-2:])
        clip_patches, dino_patches = self.encode_patches(images)
        embedding = self.projection(torch.cat((clip_patches, dino_patches), dim=-1))
        return self.pixel_logits(embedding, text, images.shape[-2:])

    @torch.no_grad()
    def audit_initialization(self, images, text):
        clip_patches, dino_patches = self.encode_patches(images)
        grid = self.size // PATCH_SIZE
        assert clip_patches.shape[1:] == (grid * grid, CLIP_WIDTH)
        assert dino_patches.shape[1:] == (grid * grid, DINO_WIDTH)
        baseline = self.pixel_logits(clip_patches, text, images.shape[-2:])
        projected = self.projection(torch.cat((clip_patches, dino_patches), dim=-1))
        fused = self.pixel_logits(projected, text, images.shape[-2:])
        torch.testing.assert_close(baseline, fused, atol=1e-6, rtol=1e-6)
        rgb = F.interpolate(images.float() / 255.0, (224, 224), mode="bilinear",
                            align_corners=False, antialias=True)
        native = self.clip.encode_image((rgb - self.clip_mean) / self.clip_std)
        extracted = self.clip_tokens(rgb)[:, 0]
        torch.testing.assert_close(native, extracted, atol=5e-6, rtol=1e-5)
        return {
            "grid": [grid, grid], "clip_patches": list(clip_patches.shape),
            "dino_patches": list(dino_patches.shape), "classes": len(text),
            "pixel_logits": list(fused.shape),
            "initial_clip_only_max_abs_error": (baseline - fused).abs().max().item(),
            "native_clip_class_token_max_abs_error": (native - extracted).abs().max().item(),
        }
