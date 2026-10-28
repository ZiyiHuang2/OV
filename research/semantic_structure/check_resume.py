"""Check a saved optimizer and sampler with an exact save/restore continuation."""
import argparse
import io
import json
from pathlib import Path

import torch
from model import SemanticStructureDecoder, segmentation_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = torch.load(Path(args.features) / 'train.pt', weights_only=True)
    results = {}
    for arm in 'BCDE':
        def build(checkpoint):
            model = SemanticStructureDecoder(
                data['visual'].shape[1], [s.shape[1] for s in data['structures']],
                semantic_mode=checkpoint['semantic'], spatial_mode=checkpoint['spatial']).cuda()
            model.load_state_dict(checkpoint['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
            optimizer.load_state_dict(checkpoint['optimizer'])
            generator = torch.Generator()
            generator.set_state(checkpoint['rng_sampler'])
            return model, optimizer, generator

        checkpoint = torch.load(Path(args.checkpoints) / f'{arm}.pt', map_location='cpu', weights_only=True)
        model, optimizer, generator = build(checkpoint)
        # Serialize and deserialize, so optimizer step tensors cannot alias.
        buffer = io.BytesIO()
        torch.save({**checkpoint, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'rng_sampler': generator.get_state()}, buffer)
        buffer.seek(0)
        restored, restored_opt, restored_gen = build(torch.load(buffer, map_location='cpu', weights_only=True))
        indices = torch.randint(len(data['visual']), (2,), generator=generator)
        assert torch.equal(indices, torch.randint(len(data['visual']), (2,), generator=restored_gen))
        for m, opt in [(model, optimizer), (restored, restored_opt)]:
            opt.zero_grad(set_to_none=True)
            target = data['labels'][indices].cuda()
            logits = m(data['visual'][indices].cuda(), data['text'].cuda(), data['valid'].cuda(),
                       [s[indices].cuda() for s in data['structures']], target.shape[-2:])
            loss = segmentation_loss(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        for key, state in optimizer.state_dict()['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, restored_opt.state_dict()['state'][key][name], rtol=0, atol=0)
        results[arm] = {'sampler_indices': indices.tolist(), 'next_update_exact': True,
                        'optimizer_state_exact': True}
        print(arm, results[arm], flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
