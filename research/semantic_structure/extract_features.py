"""Explicit local-checkpoint adapter: bundled CLIP ViT + torchvision ResNet-50.
No implicit downloads, random fallback, masks derived from predictions, or test access.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from data import load_cache, sha256


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_clip(path, checkpoint_format, source_root):
    module = import_file('ov_clip_model', source_root / 'cat_seg/third_party/model_vpt.py')
    if checkpoint_format == 'torchscript':
        state = dict(torch.jit.load(str(path), map_location='cpu').state_dict())
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or 'visual.proj' not in state or not any(k.endswith('attn.in_proj_weight') for k in state):
        raise ValueError('Requires an OpenAI-format CLIP ViT checkpoint with unsplit QKV; no auto prefix conversion')
    width = state['visual.conv1.weight'].shape[0]
    patch = state['visual.conv1.weight'].shape[-1]
    grid = int((state['visual.positional_embedding'].shape[0]-1)**0.5)
    if grid*grid+1 != state['visual.positional_embedding'].shape[0]:
        raise ValueError('Invalid square positional grid')
    layers = len([k for k in state if k.startswith('visual.') and k.endswith('attn.in_proj_weight')])
    twidth = state['ln_final.weight'].shape[0]
    tlayers = len([k for k in state if k.startswith('transformer.') and k.endswith('attn.in_proj_weight')])
    model = module.CLIP(state['text_projection'].shape[1], grid*patch, layers, width, patch,
                        state['positional_embedding'].shape[0], state['token_embedding.weight'].shape[0],
                        twidth, twidth//64, tlayers)
    converted = {}
    for key, value in state.items():
        if key in ('input_resolution','context_length','vocab_size'):
            continue
        if key.endswith('attn.in_proj_weight'):
            for name, tensor in zip(('q','k','v'), value.chunk(3, dim=0)):
                converted[key.replace('in_proj_weight', name+'_proj_weight')] = tensor
        else:
            converted[key] = value
    model.load_state_dict(converted, strict=True)
    return model.float().eval().requires_grad_(False)


def load_structure(path):
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.float().eval().requires_grad_(False)


def tokens_for(texts, tokenizer, context_length):
    result = torch.zeros(len(texts), context_length, dtype=torch.long)
    for i, text in enumerate(texts):
        ids = [tokenizer.encoder['<|startoftext|>']] + tokenizer.encode(text) + [tokenizer.encoder['<|endoftext|>']]
        if len(ids) > context_length:
            raise ValueError('Description exceeds CLIP context length; truncation is not allowed')
        result[i, :len(ids)] = torch.tensor(ids)
    return result


def prepare_pair(image_path, mask_path, size, mapping):
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError('Image/mask spatial mismatch')
        # Full-image deterministic resize, no independent crops between encoders.
        rgb = np.asarray(image.convert('RGB').resize((size, size), Image.Resampling.BICUBIC)).copy()
        raw = np.asarray(mask).copy()
    if raw.ndim != 2:
        raise ValueError('Mask must be a single-channel label image')
    output = np.full(raw.shape, 255, dtype=np.int64)
    for label in np.unique(raw):
        if str(int(label)) not in mapping:
            raise ValueError(f'Unmapped raw label {label}')
        output[raw == label] = mapping[str(int(label))]
    return torch.from_numpy(rgb).permute(2,0,1).float().div(255), torch.from_numpy(output)


@torch.no_grad()
def encode_image_pair(clip, structure, image):
    device = next(clip.parameters()).device
    x = image[None].to(device)
    mean = x.new_tensor([0.48145466,0.4578275,0.40821073])[None,:,None,None]
    std = x.new_tensor([0.26862954,0.26130258,0.27577711])[None,:,None,None]
    tokens = clip.encode_image((x-mean)/std, dense=True)
    h, w = x.shape[-2]//clip.visual.patch_size, x.shape[-1]//clip.visual.patch_size
    if tokens.shape[1] != h*w+1:
        raise ValueError('Unexpected CLIP token layout')
    visual = tokens[:,1:].transpose(1,2).reshape(1,-1,h,w)
    mean = x.new_tensor([0.485,0.456,0.406])[None,:,None,None]
    std = x.new_tensor([0.229,0.224,0.225])[None,:,None,None]
    z = structure.maxpool(structure.relu(structure.bn1(structure.conv1((x-mean)/std))))
    z = structure.layer1(z)
    fine = structure.layer2(z)
    coarse = structure.layer3(fine)
    return visual.cpu(), [coarse.cpu(), fine.cpu()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--clip-checkpoint', required=True)
    p.add_argument('--clip-format', choices=['torchscript','state_dict'], required=True)
    p.add_argument('--clip-source', required=True, help='Declared pretraining/model provenance')
    p.add_argument('--structure-checkpoint', required=True)
    p.add_argument('--structure-encoder', choices=['resnet50'], required=True)
    p.add_argument('--structure-source', required=True, help='e.g. torchvision ImageNet1K_V2; never infer remote-sensing provenance')
    p.add_argument('--output', required=True)
    p.add_argument('--size', type=int, default=384)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    manifest_path = Path(a.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('split') not in ('train','validation','research'):
        raise ValueError('Only train/validation/research splits allowed; test is sealed')
    samples, classes = manifest['samples'], manifest['classes']
    if not samples or not classes or len({c['name'] for c in classes}) != len(classes):
        raise ValueError('Empty samples/classes or duplicate classes')
    descriptions = []
    for c in classes:
        unique = {}
        for text in c['descriptions']:
            text = text.strip()
            if text:
                unique.setdefault(text.casefold(), text)
        if not unique:
            raise ValueError('Every class needs a nonempty description')
        descriptions.append(list(unique.values()))
    mapping = manifest['label_map']
    if any(not isinstance(v, int) or (v != 255 and not 0 <= v < len(classes)) for v in mapping.values()):
        raise ValueError('Invalid label map')
    root = Path(__file__).resolve().parents[2]
    clip_path, structure_path = Path(a.clip_checkpoint), Path(a.structure_checkpoint)
    clip = load_clip(clip_path, a.clip_format, root).to(a.device)
    if a.size <= 0 or a.size % clip.visual.patch_size:
        raise ValueError('Square extraction size must be a positive multiple of CLIP patch size')
    structure = load_structure(structure_path).to(a.device)
    tokenizer_module = import_file('ov_tokenizer', root / 'cat_seg/third_party/simple_tokenizer.py')
    tokenizer = tokenizer_module.SimpleTokenizer()
    pmax = max(map(len, descriptions))
    bank = torch.zeros(len(classes), pmax, clip.text_projection.shape[-1])
    valid = torch.zeros(len(classes), pmax, dtype=torch.bool)
    for c, strings in enumerate(descriptions):
        tokens = tokens_for(strings, tokenizer, clip.context_length).to(a.device)
        bank[c, :len(strings)] = F.normalize(clip.encode_text(tokens), dim=-1).cpu()
        valid[c, :len(strings)] = True
    visuals, pyramids, labels, hashes, mask_hashes = [], [[],[]], [], [], []
    for record in samples:
        image_path = manifest_path.parent / record['image']
        mask_path = manifest_path.parent / record['mask']
        image, label = prepare_pair(image_path, mask_path, a.size, mapping)
        if labels and label.shape != labels[0].shape:
            raise ValueError('Pilot cache needs equal native mask sizes; prepare a documented tile manifest first')
        visual, pyramid = encode_image_pair(clip, structure, image)
        visuals.append(visual)
        for target, feature in zip(pyramids, pyramid):
            target.append(feature)
        labels.append(label); hashes.append(sha256(image_path)); mask_hashes.append(sha256(mask_path))
    metadata = {'schema_version':2, 'dataset':manifest['dataset'], 'split':manifest['split'],
        'sample_ids':[r['id'] for r in samples], 'image_sha256':hashes, 'mask_sha256':mask_hashes,
        'class_names':[c['name'] for c in classes], 'descriptions':descriptions,
        'manifest_sha256':sha256(manifest_path), 'label_map':mapping,
        'encoders': {'clip': {'sha256':sha256(clip_path), 'source':a.clip_source,
                            'implementation':'bundled model_vpt dense=True, final projected tokens; not HG',
                            'implementation_sha256':sha256(root/'cat_seg/third_party/model_vpt.py')},
                     'structure':{'sha256':sha256(structure_path),'source':a.structure_source,
                                  'architecture':'torchvision resnet50 layer3,layer2'}},
        'preprocessing':{'rgb':'full image bicubic square resize', 'size':a.size,
                         'clip_normalization':'OpenAI CLIP', 'structure_normalization':'ImageNet',
                         'labels':'native grid, explicit label_map',
                         'extractor_sha256':sha256(Path(__file__))}}
    result = dict(visual=torch.cat(visuals), text=bank, valid=valid,
                  structures=[torch.cat(x) for x in pyramids], labels=torch.stack(labels), metadata=metadata)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix+'.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch.save(result, temporary)
        load_cache(temporary, manifest['split'])
        temporary.rename(out)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({'cache':str(out), 'samples':len(samples), 'sha256':sha256(out)}))

if __name__ == '__main__':
    main()
