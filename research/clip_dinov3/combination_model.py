"""Integration with the original DualPatchSegmenter without altering its code."""
import torch

from model import DualPatchSegmenter, CLIP_WIDTH, DINO_WIDTH, PATCH_SIZE
from combination_core import CombinationCore


class CSeriesSegmenter(DualPatchSegmenter):
    def __init__(self, clip_path, dino_path, size, config, train_classes):
        if config.uses_dino != (dino_path is not None):
            raise ValueError(f'DINOv3 weight path does not match variant {config.variant}')
        super().__init__(clip_path, dino_path, size)
        if self.dino is not None:
            del self.projection  # original B0 projection is reproduced in the new core
        self.config = config
        self.core = CombinationCore(config, len(train_classes), CLIP_WIDTH, DINO_WIDTH)
        self.train_classes = tuple(train_classes)

    @torch.no_grad()
    def extract(self, images):
        grid = (self.size // PATCH_SIZE,) * 2
        if self.config.variant == 'B0' or self.config.matched_fusion:
            c, d = self.encode_patches(images)
            return c.float(), d.float(), (), (), (), grid
        rgb = self.resize_rgb(images)
        if not self.config.uses_clip_visual:
            return None, self.dino_tokens(rgb), (), (), (), grid
        if not self.config.uses_dino:
            return self.clip_tokens(rgb)[:, 1:].float(), None, (), (), (), grid
        blocks = self.clip.visual.transformer.resblocks
        if len(blocks) != 12 or len(self.dino.blocks) != 12:
            raise ValueError('The audited C-series assumes the two specified ViT-B/16 checkpoints')
        capture = {}
        indices = list(range(12)) if self.config.multilayer_residual else [3, 7]
        def hook_at(i):
            def record(module, inputs, output):
                capture[i] = output.transpose(0, 1)[:, 1:].float()
            return record
        handles = [blocks[i].register_forward_hook(hook_at(i)) for i in indices]
        try:
            c = self.clip_tokens(rgb)[:, 1:].float()
        finally:
            for handle in handles:
                handle.remove()
        dino_input = (rgb - self.dino_mean) / self.dino_std
        ds = tuple(x.float() for x in self.dino.get_intermediate_layers(dino_input, n=[8, 9, 10, 11], norm=True))
        d = ds[-1]
        guides = ((capture[7], ds[2]), (capture[3], ds[0])) if self.config.has_decoder else ()
        return c, d, tuple(capture[i] for i in indices), ds, guides, grid

    def forward(self, images, text, labels=None, ignore=255, force_keep_all=False, status=None, pseudo_enabled=True, alignment_enabled=True):
        c, d, cs, ds, guides, grid = self.extract(images)
        return self.core(c, d, text, self.clip.logit_scale.exp().float(), grid, images.shape[-2:],
                         cs, ds, guides, labels=labels, ignore=ignore, force_keep_all=force_keep_all,
                         status=status, pseudo_enabled=pseudo_enabled, alignment_enabled=alignment_enabled)

    @torch.no_grad()
    def audit_features(self, images):
        """Extra encoder passes only in explicit audit, never in measured inference."""
        c, d, _, _, _, grid = self.extract(images)
        if self.dino is not None and c is not None:
            ref_c, ref_d = self.encode_patches(images)
            torch.testing.assert_close(c, ref_c.float(), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(d, ref_d.float(), atol=1e-5, rtol=1e-5)
        if c is not None:
            assert c.shape[1:] == (grid[0] * grid[1], CLIP_WIDTH)
        if d is not None:
            assert d.shape[1:] == (grid[0] * grid[1], DINO_WIDTH)
        assert not self.clip.training and all(not p.requires_grad for p in self.clip.parameters())
        if self.dino is not None:
            assert not self.dino.training and all(not p.requires_grad for p in self.dino.parameters())
        return {'clip': list(c.shape) if c is not None else None, 'dino': list(d.shape) if d is not None else None,
                'grid': list(grid), 'encoder_equivalence_checked': self.dino is not None and c is not None}
