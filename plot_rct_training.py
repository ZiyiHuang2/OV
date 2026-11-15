"""Plot the logged training loss and RCT telemetry for completed 2k arms."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root, output = map(Path, sys.argv[1:])
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
series = [('total_loss', 'Training loss'), ('rival_bridge/update_rms', 'RCT update RMS'),
          ('rival_bridge/gate_mean', 'Gate mean'), ('rival_bridge/mix_mean', 'Mix mean')]
for arm in ('A', 'C', 'M', 'V', 'R'):
    rows = [json.loads(line) for line in (root / ('explore_' + arm) / 'metrics.json').read_text().splitlines()]
    for ax, (key, title) in zip(axes.flat, series):
        if arm == 'M' and key == 'rival_bridge/mix_mean':
            continue  # Logged by the hook, but unused by M's forward path.
        selected = [(row['iteration'], row[key]) for row in rows if key in row]
        if selected:
            ax.plot(*zip(*selected), label=arm, linewidth=1.2)
        ax.set_title(title); ax.set_xlabel('Iteration'); ax.grid(alpha=.2)
        if key in ('rival_bridge/gate_mean', 'rival_bridge/mix_mean'):
            ax.set_ylim(0, 1)
        if selected:
            ax.legend()
fig.suptitle('OVRSIS95K, 2k exploration, seed 42')
fig.savefig(output, dpi=180)
