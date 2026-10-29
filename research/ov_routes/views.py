"""Route B: learn which real image crop to re-encode, using source labels only."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def windows(height, width):
    """Eight shared candidate actions; coordinates always refer to original RGB."""
    result = []
    for scale in (.5, .75):
        h, w = max(1, round(height*scale)), max(1, round(width*scale))
        for y in (0, height-h):
            for x in (0, width-w):
                box = (y, x, y+h, x+w)
                if box not in result:
                    result.append(box)
    return result


def action_features(logits, text, boxes):
    """No labels or seen/unseen IDs. Features equivariant to vocabulary ordering.

    Pair semantics use |t1-t2|; nine scalar features describe the initial
    competition, spatial disagreement, and action geometry. Entropy is only an
    input feature, never asserted to be calibrated correctness probability.
    """
    if logits.ndim != 3 or len(text) != len(logits) or len(text) < 2:
        raise ValueError('Expected KHW logits and KD text with K >= 2')
    if not torch.isfinite(logits).all() or not torch.isfinite(text).all():
        raise ValueError('Non-finite evidence')
    k, height, width = logits.shape
    p = logits.float().softmax(0)
    text = F.normalize(text.float(), dim=-1)
    rows = []
    for y0, x0, y1, x1 in boxes:
        q = p[:, y0:y1, x0:x1]
        mean = q.mean((1, 2))
        values, ids = mean.topk(2)
        entropy = -(q*q.clamp_min(1e-12).log()).sum(0).mean()/math.log(k)
        disagreement = (q.argmax(0) != ids[0]).float().mean()
        pair = (text[ids[0]]-text[ids[1]]).abs()
        scalars = torch.stack((values[0], values[1], values[0]-values[1], entropy,
                               disagreement, mean.new_tensor((y1-y0)/height),
                               mean.new_tensor((x1-x0)/width),
                               mean.new_tensor((y0+y1)/(2*height)),
                               mean.new_tensor((x0+x1)/(2*width))))
        rows.append(torch.cat((pair, scalars)))
    return torch.stack(rows)


class ViewSelector(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.text_dim = text_dim
        self.net = nn.Sequential(nn.Linear(text_dim+9, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


def choose(features, budget, policy, generator=None, selector=None):
    if not 0 <= budget <= len(features):
        raise ValueError('Crop budget outside candidate count')
    if budget == 0:
        return []
    if policy == 'learned':
        if selector is None:
            raise ValueError('learned policy requires trained selector')
        return selector(features).argsort(descending=True, stable=True)[:budget].tolist()
    if policy == 'random':
        return torch.randperm(len(features), generator=generator)[:budget].tolist()
    if policy == 'fixed':
        # Fixed order alternates candidate scales and corners; never reads GT.
        order = [0, 7, 1, 6, 2, 5, 3, 4] if len(features) == 8 else list(range(len(features)))
        return order[:budget]
    raise ValueError('Unknown view policy')


def fuse(base, crops):
    """Average sigmoid scores, matching host score convention; overlap is symmetric."""
    result = base.sigmoid().clone()
    count = result.new_ones(result.shape[-2:])
    for (y0, x0, y1, x1), logits in crops:
        if logits.shape != (len(base), y1-y0, x1-x0):
            raise ValueError('Crop output not aligned with original coordinates')
        result[:, y0:y1, x0:x1] += logits.sigmoid()
        count[y0:y1, x0:x1] += 1
    return result/count


@torch.no_grad()
def predict_views(model, image, names, policy='fixed', budget=2, selector=None, generator=None):
    """One initial encoding plus EXACTLY budget fresh crop encodings. No label input."""
    if image.ndim != 3:
        raise ValueError('Expected CHW RGB image')
    base = model.predict_logits(image[None], names)[0]
    boxes = windows(*image.shape[-2:])
    features = action_features(base, model.text_vectors(names), boxes)
    selected = choose(features, budget, policy, generator, selector)
    crops = []
    for index in selected:
        y0, x0, y1, x1 = boxes[index]
        crop = image[:, y0:y1, x0:x1].contiguous()
        crops.append((boxes[index], model.predict_logits(crop[None], names)[0]))
    return fuse(base, crops), {'selected': selected, 'boxes': [boxes[i] for i in selected],
                              'encoder_calls': 1+len(selected)}


@torch.no_grad()
def training_actions(model, image, labels, names):
    """Enumerate crop benefit ONLY on source TRAIN labels, not validation targets.

    Target is reduction in whole-image valid-pixel mean BCE if this action alone
    is fused. Positive means improvement. Labels=255 contribute neither positive
    nor negative evidence. Benefits of overlapping actions need not be additive.
    """
    base = model.predict_logits(image[None], names)[0]
    valid = labels != 255
    if not valid.any():
        return None
    if ((labels[valid] < 0) | (labels[valid] >= len(names))).any():
        raise ValueError('Training labels must use SOURCE contiguous IDs')
    boxes = windows(*image.shape[-2:])
    features = action_features(base, model.text_vectors(names), boxes)
    original = base.sigmoid()
    gains = []
    for box in boxes:
        y0, x0, y1, x1 = box
        mask = valid[y0:y1, x0:x1]
        crop = model.predict_logits(image[None, :, y0:y1, x0:x1].contiguous(), names)[0]
        old = original[:, y0:y1, x0:x1].permute(1, 2, 0)[mask]
        new = ((original[:, y0:y1, x0:x1]+crop.sigmoid())/2).permute(1, 2, 0)[mask]
        if not mask.any():
            gains.append(base.new_zeros(()))
            continue
        target = F.one_hot(labels[y0:y1, x0:x1][mask], len(names)).float()
        delta = (F.binary_cross_entropy(old, target, reduction='sum') -
                 F.binary_cross_entropy(new, target, reduction='sum'))/(valid.sum()*len(names))
        gains.append(delta)
    return features.detach(), torch.stack(gains).detach()
