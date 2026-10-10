"""Checkpoint metadata for computation settings not encoded in tensor shapes."""
import json

from detectron2.checkpoint import DetectionCheckpointer


def model_config(cfg):
    """Forward settings, excluding vocabulary, runtime paths and initialization.

    Open-vocabulary evaluation may intentionally change classes and image size.
    Learned residual scales come from weights, not their config initializers.
    Only active optional-module settings participate in the contract.
    """
    keys = (
        'MODEL.META_ARCHITECTURE', 'MODEL.SEM_SEG_HEAD.NAME',
        'MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED', 'MODEL.PROMPT_ENSEMBLE_TYPE',
        'MODEL.SEM_SEG_HEAD.PROMPT_DEPTH', 'MODEL.SEM_SEG_HEAD.PROMPT_LENGTH',
        'MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM', 'MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM',
        'MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM', 'MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM',
        'MODEL.SEM_SEG_HEAD.DECODER_DIMS', 'MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS',
        'MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS', 'MODEL.SEM_SEG_HEAD.NUM_LAYERS',
        'MODEL.SEM_SEG_HEAD.NUM_HEADS', 'MODEL.SEM_SEG_HEAD.HIDDEN_DIMS',
        'MODEL.SEM_SEG_HEAD.POOLING_SIZES', 'MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION',
        'MODEL.SEM_SEG_HEAD.WINDOW_SIZES', 'MODEL.SEM_SEG_HEAD.ATTENTION_TYPE',
        'MODEL.CLIP_PIXEL_MEAN', 'MODEL.CLIP_PIXEL_STD', 'MODEL.MASK_FORMER.SIZE_DIVISIBILITY',
        'INPUT.FORMAT',
        'MODEL.OVRS.AGGREGATION', 'MODEL.OVRS.CONTEXT_ADAPTER', 'MODEL.OVRS.CAD_STAGES',
    )
    if cfg.MODEL.OVRS.CONTEXT_ADAPTER == 'lsk':
        keys += ('MODEL.OVRS.LSK_DILATION',)
    if cfg.MODEL.OVRS.AGGREGATION == 'parallel_epl':
        keys += ('MODEL.OVRS.EPL_EXPERTS',)
    if cfg.MODEL.OVRS.CAD_STAGES:
        keys += ('MODEL.OVRS.CAD_POOLING',)
    result = {}
    for key in keys:
        value = cfg
        for part in key.split('.'):
            value = value[part]
        result[key] = value
    return json.loads(json.dumps(result))


class OVRSCheckpointer(DetectionCheckpointer):
    """Permissive weight initialization; verified evaluation/resume when strict."""

    def __init__(self, model, save_dir='', *, expected_config=None, strict=False,
                 allow_unverified=False, provenance=None, **checkpointables):
        super().__init__(model, save_dir, **checkpointables)
        self.expected_config = expected_config
        self.strict = strict
        self.allow_unverified = allow_unverified
        self.provenance = provenance

    def save(self, name, **kwargs):
        if self.expected_config is not None:
            kwargs['ovrs_metadata'] = {
                'version': 1, 'model_config': self.expected_config,
                'provenance': self.provenance,
            }
        super().save(name, **kwargs)

    def _load_model(self, checkpoint):
        if self.strict and self.expected_config is not None:
            metadata = checkpoint.get('ovrs_metadata')
            if metadata is None:
                if not self.allow_unverified:
                    raise ValueError(
                        'Checkpoint has no OVRS model configuration metadata. '
                        'For a verified legacy checkpoint, explicitly pass '
                        '--allow-unverified-checkpoint; only tensor compatibility can then be checked.'
                    )
                self.logger.warning('Unverified legacy checkpoint: model configuration cannot be checked.')
            else:
                if metadata.get('version') != 1 or not isinstance(metadata.get('model_config'), dict):
                    raise ValueError('Invalid or unsupported OVRS checkpoint metadata')
                saved = metadata['model_config']
                changed = {
                    key: {'checkpoint': saved.get(key), 'requested': self.expected_config.get(key)}
                    for key in saved.keys() | self.expected_config.keys()
                    if saved.get(key) != self.expected_config.get(key)
                }
                if changed:
                    raise ValueError('Checkpoint model configuration mismatch: ' + json.dumps(changed, sort_keys=True))
        incompatible = super()._load_model(checkpoint)
        if self.strict and (incompatible.missing_keys or incompatible.unexpected_keys or incompatible.incorrect_shapes):
            raise ValueError(
                'Checkpoint does not match the evaluation architecture or resume model. '
                f'missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}, '
                f'incorrect_shapes={incompatible.incorrect_shapes}'
            )
        return incompatible


class EvaluationCheckpointer(OVRSCheckpointer):
    def __init__(self, model, save_dir='', **kwargs):
        super().__init__(model, save_dir, strict=True, **kwargs)
