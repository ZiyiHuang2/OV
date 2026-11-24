"""Class-count-independent heads for the frozen CLIP/DINOv3 C-series.
Research adaptations, not full reproductions. See C_SERIES_DESIGN.md.
All visual tokens are row-major [B,N,D]; scores are [B,N,K].
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    variant: str
    affinity: str = "dual"
    neighbors: int = 32
    topk: int = 4
    rs: bool | None = None
    pixel: bool | None = None
    fusion: str = "learned"

    def __post_init__(self):
        if self.variant not in ("C0", "B0", "D0", "C1", "C2", "C3", "C4", "C5", "C6"):
            raise ValueError("Unknown variant")
        if self.affinity not in ("visual", "semantic", "dual") or self.fusion not in ("learned", "mean"):
            raise ValueError("Unknown interaction mode")
        if self.neighbors < 1 or self.topk < 1:
            raise ValueError("neighbors and topk must be positive")
        if (self.rs or self.pixel) and self.variant not in ("C4", "C5", "C6"):
            raise ValueError("Domain switches require C4/C5/C6")

    @property
    def dual_cost(self):
        return self.variant in ("C2", "C4", "C5", "C6")

    @property
    def relation(self):
        return self.variant in ("C3", "C4", "C5", "C6")

    @property
    def use_rs(self):
        return self.rs if self.rs is not None else self.variant in ("C5", "C6")

    @property
    def use_pixel(self):
        return self.pixel if self.pixel is not None else self.variant in ("C5", "C6")


def norm(x):
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def map_grid(x, grid):
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], *grid)


def identity_projection(in_dim, out_dim):
    layer = nn.Linear(in_dim, out_dim)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[:, :out_dim].copy_(torch.eye(out_dim))
        layer.bias.zero_()
    return layer


class WeightedLayers(nn.Module):
    """USE-inspired per-layer normalization/projection and learned layer scales."""
    def __init__(self, width, count, out):
        super().__init__()
        self.proj = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out)) for _ in range(count))
        self.scales = nn.Parameter(torch.zeros(count))

    def forward(self, xs):
        if len(xs) != len(self.proj):
            raise ValueError("Layer count mismatch")
        return sum(w * proj(x) for w, proj, x in zip(self.scales.softmax(0), self.proj, xs))


def pool_regions(dino, labels, grid, classes, ignore):
    """One semantic region per present image/class; adaptive area pooling at grid."""
    regions, ids = [], []
    for i, target in enumerate(labels):
        for c in target.unique().tolist():
            if c == ignore:
                continue
            if not 0 <= c < classes:
                raise ValueError("Nonlocal training label")
            w = F.adaptive_avg_pool2d((target == c).float()[None, None], grid).flatten()
            if w.sum() > 0:
                regions.append((w[:, None] * dino[i]).sum(0) / w.sum())
                ids.append(c)
    if not regions:
        return dino.new_empty((0, dino.shape[-1])), labels.new_empty((0,))
    return torch.stack(regions), labels.new_tensor(ids)


def region_contrastive(regions, ids, text, scale):
    """Region->class CE + multi-positive class->region NCE; repeated labels aren't negatives."""
    if not len(ids):
        return text.sum() * 0
    score = scale * (norm(regions) @ norm(text).T)
    row = F.cross_entropy(score, ids)
    present = ids.unique()
    col = score[:, present]
    positives = ids[:, None] == present[None, :]
    reverse = (col.logsumexp(0) - col.masked_fill(~positives, -torch.inf).logsumexp(0)).mean()
    return (row + reverse) / 2


class PrototypeMemory(nn.Module):
    """Seen-class memory; text queries attend to a bank, never index by test class ID."""
    def __init__(self, classes, width):
        super().__init__()
        self.register_buffer("bank", torch.zeros(classes, width))
        self.register_buffer("counts", torch.zeros(classes, dtype=torch.long))
        self.weight = nn.Linear(width, 1)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, text):
        valid = self.counts > 0
        if not valid.any():
            return text
        bank = self.bank[valid].detach().clone()
        global_bias = (self.weight(bank).relu().softmax(0) * bank).sum(0)
        update, _ = self.attn(text[None], bank[None], bank[None], need_weights=False)
        return text + self.gate.sigmoid() * (norm(update[0]) + norm(global_bias))

    @torch.no_grad()
    def update(self, regions, ids):
        if not self.training:
            raise RuntimeError("Prototype updates during evaluation are forbidden")
        if torch.distributed.is_initialized():
            raise RuntimeError("C-series prototype memory currently supports single process only")
        for c in ids.unique().tolist():
            feat = norm(regions[ids == c])
            if self.counts[c] > 0:
                weight = (1 - feat @ norm(self.bank[c])).clamp_min(1e-6)
                target = (feat * (weight / weight.sum())[:, None]).sum(0)
                self.bank[c].copy_(norm(.9 * self.bank[c] + .1 * target))
            else:
                self.bank[c].copy_(norm(feat.mean(0)))
            self.counts[c] += len(feat)


class RelationPropagation(nn.Module):
    """Joint visual/semantic affinities on patches; a single residual propagation."""
    def __init__(self, mode, neighbors):
        super().__init__()
        self.mode, self.neighbors = mode, neighbors
        if mode == "dual":
            self.mix = nn.Parameter(torch.zeros(()))
        self.strength = nn.Parameter(torch.tensor(-2.0))

    def weights(self, dino, reference):
        n = dino.shape[1]
        if n > 4096:
            raise ValueError("Dense relation grid exceeds 4096 tokens; use a sparse implementation")
        if self.mode != "semantic":
            v = norm(dino)
            av = v @ v.transpose(1, 2)
        if self.mode != "visual":
            s = norm(reference - reference.mean(-1, keepdim=True))
            ass = s @ s.transpose(1, 2)
        a = av if self.mode == "visual" else ass if self.mode == "semantic" else self.mix.sigmoid() * av + (1 - self.mix.sigmoid()) * ass
        # Include self explicitly; select neighbors using joint affinity, not two serial smoothers.
        eye = torch.eye(n, device=a.device, dtype=torch.bool)[None]
        a = torch.where(eye, torch.ones_like(a), a)
        idx = a.masked_fill(eye, 2.).topk(min(n, self.neighbors), dim=-1).indices
        selected = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, idx, True)
        return (a / .1).masked_fill(~selected, -torch.inf).softmax(-1)

    def forward(self, seed, dino, reference):
        a = self.weights(dino, reference)
        return seed + self.strength.sigmoid() * (a @ seed - seed)


class LinearClassMix(nn.Module):
    """Text-conditioned linear attention along classes on pooled spatial sites."""
    def __init__(self, dim, text_dim):
        super().__init__()
        self.text = nn.Linear(text_dim, dim)
        self.qk = nn.Linear(dim, 2 * dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, text, batch):
        _, d, h, w = x.shape
        k = len(text)
        ph, pw = min(6, h), min(6, w)
        p = F.adaptive_avg_pool2d(x, (ph, pw)).reshape(batch, k, d, ph * pw).permute(0, 3, 1, 2)
        q, key = self.qk(p + self.text(text)[None, None]).chunk(2, -1)
        q, key = F.elu(q.float()) + 1, F.elu(key.float()) + 1
        v = self.v(p).float()
        kv = torch.einsum("bpkd,bpke->bpde", key, v)
        den = torch.einsum("bpkd,bpd->bpk", q, key.sum(2)).unsqueeze(-1).clamp_min(1e-6)
        delta = self.out(torch.einsum("bpkd,bpde->bpke", q, kv) / den)
        delta = delta + self.ffn(p + delta)
        delta = delta.permute(0, 2, 3, 1).reshape(batch * k, d, ph, pw)
        return x + F.interpolate(delta, (h, w), mode="bilinear", align_corners=False)


class DecodeStage(nn.Module):
    """SED-inspired FAM + transposed-conv SFM with reduced visual guidance."""
    def __init__(self, din, dout, text_dim, cw, dw, factor):
        super().__init__()
        self.depth = nn.Conv2d(din, din, 7, padding=3, groups=din)
        self.mlp = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, 4 * din), nn.GELU(), nn.Linear(4 * din, din))
        self.mix = LinearClassMix(din, text_dim)
        self.up = nn.ConvTranspose2d(din, dout, 2, stride=2)
        # Same-resolution ViT layers; construct spatial scales explicitly.
        self.guide = nn.Sequential(nn.Conv2d(cw + dw, 8, 1), nn.GroupNorm(1, 8), nn.GELU(),
                                   nn.ConvTranspose2d(8, 8, factor, stride=factor))
        self.fuse = nn.Sequential(nn.Conv2d(dout + 10, dout, 3, padding=1), nn.GroupNorm(1, dout), nn.GELU(),
                                  nn.Conv2d(dout, dout, 3, padding=1), nn.GroupNorm(1, dout), nn.GELU())

    def forward(self, x, text, guide, coarse, batch):
        k = len(text)
        x = x + self.mlp(self.depth(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.up(self.mix(x, text, batch))
        g = self.guide(guide)
        if g.shape[-2:] != x.shape[-2:]:
            raise ValueError("Guide grid mismatch")
        g = g[:, None].expand(-1, k, -1, -1, -1).reshape(batch * k, 8, *g.shape[-2:])
        raw = F.interpolate(coarse, x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((x, g, raw), 1))


class GuidedDecoder(nn.Module):
    def __init__(self, text_dim, cw, dw):
        super().__init__()
        self.embed = nn.Conv2d(2, 32, 3, padding=1)
        self.stages = nn.ModuleList([DecodeStage(32, 16, text_dim, cw, dw, 2), DecodeStage(16, 8, text_dim, cw, dw, 4)])
        self.out = nn.Conv2d(8, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, seed, reference, text, guides, grid):
        b, _, k = seed.shape
        raw = torch.stack((map_grid(seed, grid), map_grid(reference, grid)), dim=2).reshape(b * k, 2, *grid)
        x = self.embed(raw)
        for stage, (c, d) in zip(self.stages, guides):
            x = stage(x, text, map_grid(torch.cat((c, d), -1), grid), raw, b)
        delta = self.out(x).reshape(b, k, *x.shape[-2:])
        return delta


class CombinationCore(nn.Module):
    def __init__(self, cfg, train_classes, clip_dim=512, dino_dim=768, clip_width=768, clip_layers=12, dino_layers=4):
        super().__init__()
        self.cfg = cfg
        self.base = identity_projection(clip_dim + (dino_dim if cfg.variant in ("B0", "C1") else 0), clip_dim)
        if cfg.variant == "C1":
            self.clip_merge = WeightedLayers(clip_width, clip_layers, 128)
            self.dino_merge = WeightedLayers(dino_dim, dino_layers, 128)
            self.merge = nn.Sequential(nn.Linear(256, clip_dim), nn.GELU(), nn.Linear(clip_dim, clip_dim))
            nn.init.zeros_(self.merge[-1].weight)
            nn.init.zeros_(self.merge[-1].bias)
        if cfg.dual_cost:
            self.text_map = nn.Sequential(nn.Linear(clip_dim, dino_dim), nn.Tanh(), nn.Linear(dino_dim, dino_dim))
            self.dino_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))
            if cfg.fusion == "learned":
                self.fusion = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
                nn.init.zeros_(self.fusion[-1].weight)
                nn.init.constant_(self.fusion[-1].bias, -1.5)
        if cfg.relation:
            self.relation = RelationPropagation(cfg.affinity, cfg.neighbors)
        if cfg.use_rs:
            self.memory = PrototypeMemory(train_classes, dino_dim)
        if cfg.variant not in ("C0", "B0"):
            self.decoder = GuidedDecoder(clip_dim, clip_width, dino_dim)
        if cfg.variant == "C6":
            self.selector = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))
            nn.init.zeros_(self.selector[-1].weight)
            nn.init.zeros_(self.selector[-1].bias)

    def forward(self, clip, dino, text, scale, grid, output_size, clip_layers=(), dino_layers=(), guides=(), labels=None, ignore=255, force_keep_all=False):
        cfg = self.cfg
        if clip.shape[1] != grid[0] * grid[1] or text.ndim != 2 or len(text) < 1:
            raise ValueError("Invalid token grid or vocabulary")
        if cfg.variant != "C0" and (dino is None or dino.shape[:2] != clip.shape[:2]):
            raise ValueError("DINO and CLIP grids differ")
        if not self.training and labels is not None:
            raise ValueError("Ground truth cannot enter inference")
        text = norm(text)
        native = norm(clip) @ text.T
        z = self.base(torch.cat((clip, dino), -1) if cfg.variant in ("B0", "C1") else clip)
        if cfg.variant == "C1":
            z = z + self.merge(torch.cat((self.clip_merge(clip_layers), self.dino_merge(dino_layers)), -1))
        cs = norm(z) @ text.T
        full_k = len(text)
        keep = torch.arange(full_k, device=text.device)
        result = {}
        if cfg.variant == "C6":
            coarse = map_grid((scale * cs).detach(), grid)
            aux = coarse + self.selector(coarse.reshape(-1, 1, *grid)).reshape_as(coarse)
            if self.training:
                result["aux_logits"] = F.interpolate(aux, output_size, mode="bilinear", align_corners=False)
            elif not force_keep_all:
                # Batch union is intentional; batch=1 for per-image latency comparisons.
                keep = aux.topk(min(cfg.topk, full_k), dim=1).indices.unique(sorted=True)
                text, cs = text[keep], cs[..., keep]
        seed = scale * cs
        if cfg.dual_cost:
            aligned = norm(self.text_map(text))
            if cfg.use_rs:
                aligned = norm(self.memory(aligned))
            ds = norm(dino) @ aligned.T
            ds_scale = self.dino_scale.exp().clamp(max=100)
            if cfg.fusion == "mean":
                weight = .5
            else:
                weight = self.fusion(torch.stack((cs, ds, ds - cs), -1)).squeeze(-1).sigmoid()
            seed = seed + weight * (ds_scale * ds - seed)
            if self.training and labels is not None:
                regions, ids = pool_regions(dino.detach(), labels, grid, full_k, ignore)
                result["alignment_loss"] = region_contrastive(regions, ids, aligned, ds_scale)
                result["regions"], result["region_ids"] = regions, ids
        if cfg.relation:
            # Stable full-vocabulary native CLIP profile, unchanged by C6 pruning.
            seed = self.relation(seed, dino, native.detach())
        if cfg.variant in ("C0", "B0"):
            logits = map_grid(seed, grid)
        else:
            if len(guides) != 2:
                raise ValueError("Two guide levels required")
            delta = self.decoder(seed, scale * native[..., keep], text, guides, grid)
            logits = F.interpolate(map_grid(seed, grid), output_size, mode="bilinear", align_corners=False)
            logits = logits + F.interpolate(delta, output_size, mode="bilinear", align_corners=False)
        if cfg.variant in ("C0", "B0"):
            logits = F.interpolate(logits, output_size, mode="bilinear", align_corners=False)
        if len(keep) != full_k:
            logits = logits.new_full((len(clip), full_k, *output_size), -1e4).index_copy(1, keep, logits)
        result["logits"], result["kept"] = logits, keep
        return result


def segmentation_loss(logits, labels, ignore=255, pixel=False, status=None):
    """Unknown/ignore pixels never become negatives. Status 2 needs explicit annotation.
    0=unknown, 1=known semantic label, 2=verified out-of-vocabulary negative.
    Status 2 is supervision only, not an extra inferred class at evaluation.
    """
    valid = labels != ignore
    if status is None:
        status = valid.long()
    if status.shape != labels.shape or not ((status >= 0) & (status <= 2)).all() or not torch.equal(status == 1, valid):
        raise ValueError("Pixel status must match verified semantic labels")
    if valid.any():
        loss = F.cross_entropy(logits, labels, ignore_index=ignore)
    else:
        loss = logits.sum() * 0
    if pixel and valid.any() and logits.shape[1] > 1:
        scores = logits.permute(0, 2, 3, 1)[valid]
        target = labels[valid].unsqueeze(1)
        pos = scores.gather(1, target).squeeze(1)
        neg = scores.scatter(1, target, -torch.inf).max(1).values
        loss = loss + .1 * (1. - pos + neg).clamp_min(0).mean()
    if pixel and (status == 2).any():
        loss = loss + .1 * F.softplus(logits.permute(0, 2, 3, 1)[status == 2]).mean()
    return loss
