"""Apply tiny, fail-closed adapter to *local* OV repo only.

Example (from OV repo root, with the package copied into research/ov_competition_v1):
    python research/ov_competition_v1/patch_repo.py --check
    python research/ov_competition_v1/patch_repo.py --apply

Do not apply on an unknown or modified repository without reviewing diff.
"""
from pathlib import Path
import argparse

ANCHOR = """        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)
"""
REPLACEMENT = """        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0 and getattr(self, 'competition_update', None) is not None:
                if getattr(layer, 'aggregation', 'serial') != 'serial' or getattr(layer, 'cost_context', None) is not None:
                    raise RuntimeError('Competition V1 requires native serial aggregation without cost_context')
                spatial = layer.swin_block(corr_embed, projected_guidance)
                # 'corr' is the aligned raw [B,P,K,H,W] correlation AFTER native
                # category truncation, if any. Prompt reduction matches the
                # current single-prompt protocol; never use postdecoder logits.
                spatial = self.competition_update(spatial, corr.mean(dim=1))
                corr_embed = layer.attention(spatial, projected_text_guidance)
            else:
                corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=Path,
                        default=Path('cat_seg/modeling/transformer/model.py'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    content = args.target.read_text(encoding='utf-8')
    if content.count(ANCHOR) != 1:
        raise SystemExit('FAILED: exact source anchor missing or duplicated; no modification')
    new = content.replace(ANCHOR, REPLACEMENT)
    if args.check:
        print('PASS: expected source anchor found once; patch can be applied')
    else:
        args.target.write_text(new, encoding='utf-8')
        print(f'Patched local {args.target}; inspect git diff before running')

if __name__ == '__main__':
    main()
