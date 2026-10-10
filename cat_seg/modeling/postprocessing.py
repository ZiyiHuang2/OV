"""Restore predictions from the padded CLIP input canvas to each image."""
from torch.nn import functional as F
from detectron2.modeling.postprocessing import sem_seg_postprocess


def postprocess_padded_prediction(prediction, padded_size, image_size, output_size):
    """Crop in input-image coordinates, never directly in the low-res grid.

    Without padding, one resize suffices and preserves the normal evaluation
    path. With padding, first restore the full canvas, then crop and resize.
    """
    if tuple(padded_size) == tuple(image_size):
        return F.interpolate(
            prediction.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False,
        )[0]
    canvas = F.interpolate(
        prediction.unsqueeze(0), size=padded_size, mode='bilinear', align_corners=False,
    )[0]
    return sem_seg_postprocess(canvas, image_size, *output_size)
