"""First-order held-class adaptation. No target-unseen labels enter this module."""
import torch
from torch.nn import functional as F
from torch.func import functional_call


def masked_bce(logits, labels, keep=None):
    """BCE over valid pixels and the supplied vocabulary, as in the CAT-Seg host."""
    if labels.dtype != torch.long or labels.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ValueError('Expected BCHW logits and long BHW labels')
    valid = labels != 255
    if ((labels[valid] < 0) | (labels[valid] >= logits.shape[1])).any():
        raise ValueError('Label is outside the source vocabulary')
    if keep is not None:
        allowed = torch.zeros(logits.shape[1], device=labels.device, dtype=torch.bool)
        allowed[list(keep)] = True
        valid &= allowed[labels.clamp(0, logits.shape[1]-1)]
    if not valid.any():
        return None
    x = logits.permute(0, 2, 3, 1)[valid]
    y = F.one_hot(labels[valid], logits.shape[1]).to(x.dtype)
    return F.binary_cross_entropy_with_logits(x, y)


def remap_support(labels, support_classes):
    result = torch.full_like(labels, 255)
    for new, old in enumerate(support_classes):
        result[labels == old] = new
    return result


def objective(model, support_image, support_labels, query_image, query_labels,
              names, held, inner_lr=1e-3, weight=1., mode='meta'):
    """Return loss without updating model state.

    Inner: support classes only, including the softmax/BCE candidate axis.
    Outer: different image, held-class pixels, FULL source vocabulary competition.
    The identity derivative through p - lr * stop_gradient(inner_grad) is FOMAML.
    It intentionally omits the Hessian, not the query gradient.
    'joint' uses exactly the same supervision WITHOUT a fast update; 'erm' uses
    all valid source pixels in both images. All modes freeze host dropout/BN.
    """
    if mode not in ('meta', 'joint', 'erm') or inner_lr <= 0 or weight <= 0:
        raise ValueError('Invalid objective settings')
    held = tuple(sorted(set(held)))
    if not held or min(held) < 0 or max(held) >= len(names) or len(held) == len(names):
        raise ValueError('held must be a nonempty proper subset of SOURCE classes')
    support = [i for i in range(len(names)) if i not in held]
    support_labels_local = remap_support(support_labels, support)
    held_present = torch.zeros_like(query_labels, dtype=torch.bool)
    for c in held:
        held_present |= query_labels == c
    # Identical skip rule across meta/joint/erm, including labels lost on resize.
    if not (support_labels_local != 255).any() or not held_present.any():
        return None, {'skip': 'empty_episode_after_resize'}
    if mode == 'erm':
        a = masked_bce(model(support_image, names), support_labels)
        b = masked_bce(model(query_image, names), query_labels)
    else:
        a = masked_bce(model(support_image, [names[i] for i in support]),
                       support_labels_local)
        if a is None:
            return None, {'skip': 'no_support_pixels'}
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if mode == 'meta':
            gradients = torch.autograd.grad(a, tuple(params.values()), retain_graph=True,
                                            allow_unused=True)
            fast = {n: p if g is None else p - inner_lr * g.detach()
                    for (n, p), g in zip(params.items(), gradients)}
            query = functional_call(model, fast, (query_image, names), strict=False)
        else:
            query = model(query_image, names)
        b = masked_bce(query, query_labels, held)
    if a is None or b is None:
        return None, {'skip': 'no_valid_pixels'}
    return a + weight*b, {'support_loss': float(a.detach()), 'query_loss': float(b.detach()),
                          'held_source_ids': list(held), 'mode': mode}
