"""Class-count-independent heads for the frozen CLIP/DINOv3 C-series.
Research adaptations, not full reproductions. See C_SERIES_DESIGN.md.
All visual tokens are row-major [B,N,D]; scores are [B,N,K].
"""
import math
import numpy as np
from scipy import ndimage
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ("C0", "B0", "D0", "C1", "C2", "C3", "C4", "C5", "C6", "E01", "E02")


@dataclass(frozen=True)
class Config:
    variant: str
    affinity: str = "dual"
    neighbors: int = 32
    topk: int = 4
    rs: bool | None = None
    pixel: bool | None = None
    fusion: str = "context"
    region_mode: str = "component"
    relation_source: str = "joint"
    relation_control: str = "conditional"
    router: str = "dual"
    route_train: bool = True
    pseudo: bool = True
    pseudo_conf: float = .9
    pseudo_margin: float = .2
    relation_chunk: int = 128
    decoder: str = "guided"

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError("Unknown variant")
        if self.affinity not in ("visual", "semantic", "dual") or self.fusion not in ("context", "separate", "learned", "mean") or self.decoder not in ("guided", "shared"):
            raise ValueError("Unknown interaction mode")
        if self.region_mode not in ("component", "class") or self.relation_source not in ("native", "fused", "joint") or self.relation_control not in ("global", "conditional") or self.router not in ("union", "dual"):
            raise ValueError("Unknown research control")
        if not .5 < self.pseudo_conf <= 1 or not 0 <= self.pseudo_margin <= 1:
            raise ValueError("Invalid pseudo-label confidence")
        if self.neighbors < 1 or self.topk < 1 or self.relation_chunk < 1:
            raise ValueError("neighbors and topk must be positive")
        if (self.rs or self.pixel) and self.variant not in ("C4", "C5", "C6"):
            raise ValueError("Domain switches require C4/C5/C6")
        if self.variant in ("E01", "E02") and self.decoder != "guided":
            raise ValueError("C1 factorial controls require the default decoder setting")

    @property
    def concat_seed(self):
        return self.variant in ("B0", "C1", "E01", "E02")

    @property
    def multilayer_residual(self):
        return self.variant in ("C1", "E01")

    @property
    def has_decoder(self):
        return self.variant not in ("C0", "B0", "E01")

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


def pool_regions(dino, labels, grid, classes, ignore, mode="component"):
    """Pool connected semantic components, not claimed instances. Pixel bin counts
    preserve small components even when they occupy less than one patch.
    CPU component labeling once per image/class; GPU projection in bounded chunks.
    """
    regions, ids = [], []
    h, w = labels.shape[-2:]
    gh, gw = grid
    bins = ((np.arange(h)[:, None] * gh // h) * gw + np.arange(w)[None, :] * gw // w).ravel()
    for i, target in enumerate(labels.detach().cpu().numpy()):
        for c in np.unique(target):
            if c == ignore:
                continue
            if not 0 <= c < classes:
                raise ValueError("Nonlocal training label")
            mask = target == c
            if mode == "class":
                components, count = mask.astype(np.int32), 1
            else:
                components, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
            comp = components.ravel()[mask.ravel()] - 1
            patch = bins[mask.ravel()]
            # Sort once, then use bounded component blocks rather than allocating R*N at once.
            order = np.argsort(comp, kind="stable")
            comp, patch = comp[order], patch[order]
            for begin in range(0, count, 128):
                end = min(count, begin + 128)
                lo, hi = np.searchsorted(comp, [begin, end])
                weights = np.bincount((comp[lo:hi]-begin)*gh*gw+patch[lo:hi], minlength=(end-begin)*gh*gw).reshape(end-begin, gh*gw)
                weights = torch.as_tensor(weights, device=dino.device, dtype=dino.dtype)
                regions.append((weights / weights.sum(1, keepdim=True).clamp_min(1)) @ dino[i])
                ids.extend([int(c)]*(end-begin))
    if not regions:
        return dino.new_empty((0, dino.shape[-1])), labels.new_empty((0,))
    return torch.cat(regions), labels.new_tensor(ids)


def top_margin(scores):
    """Per-class advantage over the strongest alternative; arbitrary K/order."""
    if scores.shape[-1] == 1:
        return torch.zeros_like(scores)
    values, indices = scores.topk(2, dim=-1)
    return scores - torch.where(torch.arange(scores.shape[-1], device=scores.device)==indices[..., :1], values[..., 1:], values[..., :1])


class SeparateCostEvidence(nn.Module):
    """RSKT-Seg-inspired separate spatial cost embedding, then fusion.
    The output is a gate correction, not a fixed-class classifier.
    """
    def __init__(self):
        super().__init__()
        def branch(): return nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.GroupNorm(1,8),nn.GELU())
        self.clip_branch,self.dino_branch=branch(),branch()
        self.fuse=nn.Sequential(nn.Conv2d(16,16,3,padding=1),nn.GroupNorm(1,16),nn.GELU(),nn.Conv2d(16,1,1))
        nn.init.normal_(self.fuse[-1].weight,std=.01)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, cs, ds, grid):
        b,n,k=cs.shape
        c=map_grid(cs,grid).reshape(b*k,1,*grid)
        d=map_grid(ds,grid).reshape(b*k,1,*grid)
        return self.fuse(torch.cat((self.clip_branch(c),self.dino_branch(d)),1)).reshape(b,k,n).transpose(1,2)


class ContextCostFusion(nn.Module):
    """Low-rank text-conditioned branch reliability; no [B,N,K,D] feature copy.
    Separately embedded cost neighborhoods, local visual context, category query
    compatibility and branch rivalry jointly control a shared gate.
    """
    def __init__(self, cd, dd, width=32):
        super().__init__()
        self.spatial=SeparateCostEvidence()
        self.cv = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd,width))
        self.dv = nn.Sequential(nn.LayerNorm(dd), nn.Linear(dd,width))
        self.ct = nn.Linear(cd,width)
        self.dt = nn.Linear(dd,width)
        self.gate = nn.Sequential(nn.Linear(8,32),nn.LayerNorm(32),nn.GELU(),nn.Linear(32,1))
        nn.init.normal_(self.gate[-1].weight,std=.01)
        nn.init.constant_(self.gate[-1].bias,-1.5)

    def forward(self, c, d, text, aligned, cs, ds, grid):
        cv,dv = self.cv(c),self.dv(d)
        def local(x):
            return F.avg_pool2d(map_grid(x,grid),3,1,1,count_include_pad=False).flatten(2).transpose(1,2)
        ct,dt = norm(self.ct(text)),norm(self.dt(aligned))
        cp,dp = norm(cv)@ct.T,norm(dv)@dt.T
        cn,dn = norm(local(cv))@ct.T,norm(local(dv))@dt.T
        features = torch.stack((cs,ds,top_margin(cs),top_margin(ds),cp,dp,cn,dn),-1)
        return (self.gate(features).squeeze(-1)+self.spatial(cs,ds,grid)).sigmoid()


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
    """Chunked exact top-k search, sparse gather propagation. O(N^2) search
    remains; peak pair workspace O(chunk*N), propagation O(N*k*K).
    Frozen/reference scores are detached by caller; affinities/gates learn.
    """
    def __init__(self, mode, neighbors, source="native", control="global", chunk=128):
        super().__init__()
        self.mode,self.neighbors,self.source,self.control,self.chunk=mode,neighbors,source,control,chunk
        if mode == "dual":
            self.mix=nn.Parameter(torch.zeros(()))
        if mode != "visual" and source == "joint":
            self.semantic_mix=nn.Parameter(torch.zeros(()))
        self.strength=nn.Parameter(torch.tensor(-2.0))
        if control == "conditional":
            self.gate=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,1))
            nn.init.normal_(self.gate[-1].weight,std=.01)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, seed, dino, reference, fused=None):
        b,n,k=seed.shape
        if n > 16384:
            raise ValueError("Quadratic neighbor search exceeds 16384 tokens")
        def profile(x): return norm(x-x.mean(-1,keepdim=True))
        v=norm(dino) if self.mode!="semantic" else None
        r=f=None
        if self.mode!="visual":
            if fused is None: fused=reference
            r=profile(reference) if self.source!="fused" else None
            f=profile(fused) if self.source!="native" else None
        batch=torch.arange(b,device=seed.device)[:,None,None]
        outputs=[]
        full=(fused if fused is not None else reference).detach()
        conf=full.softmax(-1).amax(-1)
        for start in range(0,n,self.chunk):
            end=min(start+self.chunk,n)
            # Discrete neighborhood selection has no gradient. Recompute the weighted
            # affinity of selected edges only, so backward does not retain N*N maps.
            with torch.no_grad():
                if v is not None: av=v[:,start:end]@v.transpose(1,2)
                if self.mode!="visual":
                    if r is not None: ar=r[:,start:end]@r.transpose(1,2)
                    if f is not None: af=f[:,start:end]@f.transpose(1,2)
                    ass=ar if self.source=="native" else af if self.source=="fused" else self.semantic_mix.sigmoid()*af+(1-self.semantic_mix.sigmoid())*ar
                a=av if self.mode=="visual" else ass if self.mode=="semantic" else self.mix.sigmoid()*av+(1-self.mix.sigmoid())*ass
                eye=(torch.arange(start,end,device=seed.device)[:,None]==torch.arange(n,device=seed.device)[None,:])[None]
                idx=a.masked_fill(eye,2.).topk(min(n,self.neighbors),dim=-1).indices
                if v is not None: av=av.gather(-1,idx)
                if self.mode!="visual":
                    if r is not None: ar=ar.gather(-1,idx)
                    if f is not None: af=af.gather(-1,idx)
                self_edge=idx==torch.arange(start,end,device=seed.device)[None,:,None]
            if self.mode!="visual":
                ass=ar if self.source=="native" else af if self.source=="fused" else self.semantic_mix.sigmoid()*af+(1-self.semantic_mix.sigmoid())*ar
            a=av if self.mode=="visual" else ass if self.mode=="semantic" else self.mix.sigmoid()*av+(1-self.mix.sigmoid())*ass
            a=torch.where(self_edge,torch.ones_like(a),a)
            weights=(a/.1).softmax(-1)
            context=(seed[batch,idx]*weights[...,None]).sum(-2)
            current=seed[:,start:end]
            strength=self.strength.sigmoid()
            if self.control=="conditional":
                confidence=conf[:,start:end]
                coherence=(a*weights).sum(-1)
                # Full-vocabulary confidence of neighbors, independent of routing.
                change=(conf[batch,idx]*weights).sum(-1)-confidence
                strength=strength*self.gate(torch.stack((confidence,coherence,change),-1)).sigmoid()
            outputs.append(current+strength*(context-current))
        return torch.cat(outputs,dim=1)


class CategoryRouter(nn.Module):
    """Dual coarse evidence before expensive class-dependent decoding. A SED-style
    union control is retained. Dual route uses class-wise peak evidence (small-object
    safeguard) and confident patch winners from both branches; no GT at inference.
    """
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GroupNorm(1,16),nn.GELU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self, c, d, fused, grid):
        b,n,k=c.shape
        x=torch.stack((c,d,fused),-1).permute(0,2,3,1).reshape(b*k,3,*grid)
        return fused+self.head(x).reshape(b,k,n).transpose(1,2)

    @staticmethod
    def select(aux,c,d,topk,mode):
        k=aux.shape[-1]; topk=min(topk,k)
        if mode=="union": return aux.topk(topk,-1).indices.unique(sorted=True)
        candidates=[x.amax(1).topk(topk,-1).indices.flatten() for x in (aux,c,d)]
        # Strong local winners can survive even if global peak rank is low.
        for x in (c,d,aux):
            prob=x.softmax(-1); conf,win=prob.max(-1)
            candidates.append(win[conf>=.8])
        return torch.cat(candidates).unique(sorted=True)


def unknown_consistency(native, dino_score, grid, labels, status, confidence, margin):
    """Conservative detached teacher agreement; never assert unknown is negative.
    Return low-resolution soft targets and selection mask. Only wholly unknown patches
    qualify; confidence/margin in both branches and spatial agreement are required.
    """
    with torch.no_grad():
        p,q=native.softmax(-1),dino_score.softmax(-1)
        a,ci=p.max(-1);b,di=q.max(-1)
        if p.shape[-1]<2:
            return p,torch.zeros_like(a,dtype=torch.bool)
        pa=p.topk(2,-1).values;pb=q.topk(2,-1).values
        good=(ci==di)&(a>=confidence)&(b>=confidence)&(pa[...,0]-pa[...,1]>=margin)&(pb[...,0]-pb[...,1]>=margin)
        unknown=F.adaptive_avg_pool2d((status==0).float()[:,None],grid).flatten(1)>=1-1e-6
        # Reject isolated agreement: winning category must match pooled neighborhood.
        local=F.avg_pool2d(map_grid((p+q)*.5,grid),3,1,1,count_include_pad=False).flatten(2).transpose(1,2).argmax(-1)
        return (p+q)*.5,good&unknown&(ci==local)


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


class SharedReconstruction(nn.Module):
    """Competing decoder: reconstruct visual fields once, then match words.
    No per-category hidden feature copies; no claim of native high-res ViT features.
    """
    def __init__(self,text_dim,cw,dw):
        super().__init__()
        def projection(): return nn.Sequential(nn.Conv2d(cw+dw,64,1),nn.GroupNorm(8,64),nn.GELU())
        self.guide1,self.guide2=projection(),projection()
        self.up1=nn.ConvTranspose2d(64,64,2,stride=2)
        self.up2=nn.ConvTranspose2d(64,64,2,stride=2)
        self.mix=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU())
        self.residual=nn.Conv2d(64,64,1)
        self.text=nn.Linear(text_dim,64)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
    def forward(self,seed,reference,text,guides,grid):
        first,second=(map_grid(torch.cat((c,d),-1),grid) for c,d in guides)
        x=self.up1(self.guide1(first))
        skip=F.interpolate(self.guide2(second),x.shape[-2:],mode="bilinear",align_corners=False)
        x=self.up2(self.mix(torch.cat((x,skip),1)))
        return torch.einsum("bdhw,kd->bkhw",self.residual(x),norm(self.text(text)))


class CombinationCore(nn.Module):
    def __init__(self, cfg, train_classes, clip_dim=512, dino_dim=768, clip_width=768, clip_layers=12, dino_layers=4):
        super().__init__()
        self.cfg = cfg
        self.base = identity_projection(clip_dim + (dino_dim if cfg.concat_seed else 0), clip_dim)
        if cfg.multilayer_residual:
            self.clip_merge = WeightedLayers(clip_width, clip_layers, 128)
            self.dino_merge = WeightedLayers(dino_dim, dino_layers, 128)
            self.merge = nn.Sequential(nn.Linear(256, clip_dim), nn.GELU(), nn.Linear(clip_dim, clip_dim))
            nn.init.zeros_(self.merge[-1].weight)
            nn.init.zeros_(self.merge[-1].bias)
        if cfg.dual_cost:
            self.text_map = nn.Sequential(nn.Linear(clip_dim, dino_dim), nn.Tanh(), nn.Linear(dino_dim, dino_dim))
            self.dino_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))
            if cfg.fusion == "context":
                self.fusion = ContextCostFusion(clip_dim, dino_dim)
            elif cfg.fusion == "separate":
                self.fusion=SeparateCostEvidence()
            elif cfg.fusion == "learned":
                self.fusion = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
                nn.init.zeros_(self.fusion[-1].weight)
                nn.init.constant_(self.fusion[-1].bias, -1.5)
        if cfg.relation:
            self.relation = RelationPropagation(cfg.affinity, cfg.neighbors, cfg.relation_source, cfg.relation_control, cfg.relation_chunk)
        if cfg.use_rs:
            self.memory = PrototypeMemory(train_classes, dino_dim)
        if cfg.has_decoder:
            self.decoder = (GuidedDecoder if cfg.decoder=="guided" else SharedReconstruction)(clip_dim, clip_width, dino_dim)
        if cfg.variant == "C6":
            self.selector=CategoryRouter()

    def forward(self, clip, dino, text, scale, grid, output_size, clip_layers=(), dino_layers=(), guides=(), labels=None, ignore=255, force_keep_all=False, status=None, pseudo_enabled=True):
        cfg = self.cfg
        if clip.shape[1] != grid[0] * grid[1] or text.ndim != 2 or len(text) < 1:
            raise ValueError("Invalid token grid or vocabulary")
        if cfg.variant != "C0" and (dino is None or dino.shape[:2] != clip.shape[:2]):
            raise ValueError("DINO and CLIP grids differ")
        if not self.training and (labels is not None or status is not None):
            raise ValueError("Ground truth cannot enter inference")
        text = norm(text)
        native = norm(clip) @ text.T
        z = self.base(torch.cat((clip, dino), -1) if cfg.concat_seed else clip)
        if cfg.multilayer_residual:
            z = z + self.merge(torch.cat((self.clip_merge(clip_layers), self.dino_merge(dino_layers)), -1))
        cs = norm(z) @ text.T
        full_k = len(text)
        keep = torch.arange(full_k, device=text.device)
        result = {}
        seed = scale * cs
        if cfg.dual_cost:
            aligned = norm(self.text_map(text))
            raw_aligned=aligned
            if cfg.use_rs:
                aligned = norm(self.memory(aligned))
            ds = norm(dino) @ aligned.T
            ds_scale = self.dino_scale.exp().clamp(max=100)
            if cfg.fusion == "mean":
                weight = .5
            elif cfg.fusion == "separate":
                weight=(self.fusion(cs,ds,grid)-1.5).sigmoid()
            elif cfg.fusion == "learned":
                weight = self.fusion(torch.stack((cs, ds, ds - cs), -1)).squeeze(-1).sigmoid()
            else:
                weight = self.fusion(clip,dino,text,aligned,cs,ds,grid)
            seed = seed + weight * (ds_scale * ds - seed)
            if self.training and labels is not None:
                regions, ids = pool_regions(dino.detach(), labels, grid, full_k, ignore,cfg.region_mode)
                result["alignment_loss"] = region_contrastive(regions, ids, aligned, ds_scale)
                result["regions"], result["region_ids"] = regions, ids
                if cfg.use_pixel and cfg.pseudo and pseudo_enabled:
                    status=(labels!=ignore).long() if status is None else status
                    # No memory-conditioned teacher, no novel-class names, no hard negative from ignore.
                    target,mask=unknown_consistency((scale*native).detach(), (ds_scale*(norm(dino)@raw_aligned.T)).detach(),grid,labels,status,cfg.pseudo_conf,cfg.pseudo_margin)
                    result["pseudo_target"],result["pseudo_mask"]=target,mask
        full_seed=seed
        if cfg.variant=="C6" and (self.training or not force_keep_all):
            aux=self.selector((scale*cs).detach(),(ds_scale*ds).detach(),seed.detach(),grid)
            if self.training:
                result["aux_logits"]=F.interpolate(map_grid(aux,grid),output_size,mode="bilinear",align_corners=False)
            # Optional train-time routing with GT union safeguards. Inference never reads GT.
            route=not force_keep_all and (not self.training or (cfg.route_train and labels is not None and bool(torch.rand((),device=seed.device)<.5)))
            if route:
                keep=self.selector.select(aux.detach(),(scale*cs).detach(),(ds_scale*ds).detach(),cfg.topk,cfg.router)
                if self.training:
                    keep=torch.cat((keep,labels[labels!=ignore].unique())).unique(sorted=True)
                    if full_k>1 and len(keep)==1:
                        alt=aux.detach().amax((0,1)).index_fill(0,keep,-torch.inf).argmax()[None]
                        keep=torch.cat((keep,alt)).sort().values
                # Keep the complete vocabulary for consistency and verified-negative training.
                if self.training and cfg.use_pixel and status is not None and (status==2).any():
                    keep=torch.arange(full_k,device=text.device)
                text,seed=text[keep],seed[...,keep]
        if cfg.relation:
            seed = self.relation(seed, dino, (scale*native).detach(), full_seed.detach())
        if "pseudo_mask" in result:
            mask=result.pop("pseudo_mask"); target=result.pop("pseudo_target")
            result["pseudo_count"]=mask.sum().detach()
            result["unknown_count"]=(F.adaptive_avg_pool2d((status==0).float()[:,None],grid).flatten(1)>=1-1e-6).sum().detach()
            # Full coarse prediction means routed training never changes pseudo target vocabulary.
            result["pseudo_loss"]=F.kl_div(full_seed[mask].log_softmax(-1),target[mask],reduction="batchmean") if mask.any() else full_seed.sum()*0
        delta = None
        if cfg.has_decoder:
            if len(guides) != 2:
                raise ValueError("Two guide levels required")
            delta = self.decoder(seed, scale * native[..., keep], text, guides, grid)
        logits = F.interpolate(map_grid(seed, grid), output_size, mode="bilinear", align_corners=False)
        if delta is not None:
            logits = logits + F.interpolate(delta, output_size, mode="bilinear", align_corners=False)
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
    if status.shape != labels.shape or not ((status == 0) | (status == 1) | (status == 2)).all() or not torch.equal(status == 1, valid):
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
