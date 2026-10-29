"""Explicit full CAT-Seg ViT-B/16 host, not the former cached-feature prototype."""
import torch
from torch import nn
from torch.nn import functional as F


class CATSegHost(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        if model.clip_resolution != (384, 384):
            raise ValueError('Only the repository ViT-B/16 dense path is supported')
        model.requires_grad_(True)
        self.clip.requires_grad_(False)
        self._texts = {}

    @property
    def predictor(self):
        return self.model.sem_seg_head.predictor

    @property
    def clip(self):
        return self.predictor.clip_model

    def text_vectors(self, names):
        return F.normalize(self.text_features(names).mean(1), dim=-1)

    @torch.no_grad()
    def text_features(self, names):
        if not names or len(set(names)) != len(names):
            raise ValueError('Vocabulary must be nonempty and unique')
        key = tuple(names)
        if key not in self._texts:
            # Frozen CLIP permits caching; vocabulary order is part of the key.
            self._texts[key] = self.predictor.get_text_embeds(
                names, self.predictor.prompt_templates, self.clip).detach()
        return self._texts[key]

    def _low_logits(self, rgb, names):
        """Float RGB [0,255], BCHW; return raw K-channel logits at host head size.

        All images in a tensor have the same size, so no padded image pixels are
        introduced. Normalization/384 resize/dense CLIP/guidance match CATSeg.
        Text is explicitly supplied: it never switches based on train()/eval().
        """
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not torch.isfinite(rgb).all():
            raise ValueError('Expected finite BCHW RGB')
        if rgb.min() < 0 or rgb.max() > 255:
            raise ValueError('Expected RGB range [0,255], not normalized features')
        m = self.model
        with torch.no_grad():
            x = (rgb.float()-m.clip_pixel_mean)/m.clip_pixel_std
            x = F.interpolate(x, size=m.clip_resolution, mode='bilinear', align_corners=False)
            m.layers = []
            encoded = self.clip.encode_image(x, dense=True)
            if len(m.layers) != 2 or encoded.shape[1] != 577:
                raise RuntimeError('Unexpected dense CLIP feature/hook layout')
            intermediate = list(m.layers)
            m.layers = []
        b = len(rgb)
        res3 = encoded[:, 1:].transpose(1, 2).reshape(b, -1, 24, 24)
        res4 = m.upsample1(intermediate[0][1:].permute(1, 2, 0).reshape(b, -1, 24, 24))
        res5 = m.upsample2(intermediate[1][1:].permute(1, 2, 0).reshape(b, -1, 24, 24))
        text = self.text_features(names).unsqueeze(0).expand(b, -1, -1, -1)
        logits = self.predictor.transformer(res3, text, [res3, res4, res5])
        return logits

    def forward(self, rgb, names):
        # Native training path: interpolate logits BEFORE the BCE loss.
        return F.interpolate(self._low_logits(rgb, names), size=rgb.shape[-2:],
                             mode='bilinear', align_corners=False)

    @torch.no_grad()
    def predict_logits(self, rgb, names):
        # Native inference path: sigmoid BEFORE interpolation. Logit conversion
        # exposes scores to the selector without changing native argmax/fusion.
        scores = F.interpolate(self._low_logits(rgb, names).sigmoid(), size=rgb.shape[-2:],
                               mode='bilinear', align_corners=False)
        return torch.logit(scores.clamp(1e-7, 1-1e-7))

    @torch.no_grad()
    def check_native(self, image, names):
        """Real-weight preflight: compare to native host, no silent fallback."""
        if image.shape[-2:] != (384, 384):
            raise ValueError('Parity image must be 384 square (no native padding)')
        m = self.model
        old_names, old_slide = self.predictor.test_class_texts, m.sliding_window
        try:
            self.predictor.test_class_texts = list(names)
            m.sliding_window = False
            m.eval()
            expected = m([{'image': image, 'height': 384, 'width': 384}])[0]['sem_seg']
            actual = self.predict_logits(image[None], names)[0].sigmoid()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            return float((actual-expected).abs().max())
        finally:
            self.predictor.test_class_texts = old_names
            m.sliding_window = old_slide


def build_host(config, weights, train_classes, test_classes, device, legacy_weights=False):
    # Imports are lazy: CPU algorithm tests do not require Detectron2 or weights.
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config
    from cat_seg import add_cat_seg_config
    from cat_seg.engine.checkpoint import EvaluationCheckpointer, model_config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config)
    cfg.MODEL.DEVICE = device
    if cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED != 'ViT-B/16' or cfg.INPUT.FORMAT != 'RGB':
        raise ValueError('Supply the matching RGB ViT-B/16 full-host config')
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = str(train_classes)
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = str(test_classes)
    cfg.TEST.SLIDING_WINDOW = False
    cfg.freeze()
    model = build_model(cfg)
    EvaluationCheckpointer(model, expected_config=model_config(cfg),
                           allow_unverified=legacy_weights).load(weights)
    return CATSegHost(model).eval(), cfg.dump()
