"""Generate fixed-endpoint tables, class contrasts and diagnostic curves."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);root=Path(p.parse_args().root)
    read=lambda name:list(csv.DictReader((root/name).open()))
    rows=read('summary.csv');curves=read('learning_curve.csv');classes=read('per_class_17.csv')
    for x in rows:
        for k in ('seen14','miou17','seen17','unseen17','hiou17','elapsed_s','peak_allocated_mib'):x[k]=float(x[k])
    results={r['arm']:r for r in rows};curve={(r['arm'],int(r['step'])):r for r in curves};byclass={(r['arm'],r['class_name']):r for r in classes}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key,title in zip(axes.flat,['loss_mean_last100','seen17','unseen17','hiou17'],['Training loss (last 100 steps)','Seen mIoU, 17-class competition','Unseen mIoU, 17-class competition','hIoU, 17-class competition']):
        for arm in 'BCDEFGH':
            subset=[x for x in curves if x['arm']==arm];ax.plot([int(x['step'])/1000 for x in subset],[float(x[key]) for x in subset],marker='o',label=arm)
        ax.set(xlabel='Optimizer updates (thousands)',ylabel='Cross entropy' if key.startswith('loss') else 'IoU (fraction)',title=title);ax.grid(alpha=.25)
    axes.flat[1].legend(ncol=4,fontsize=8);fig.savefig(root/'learning_curves.png',dpi=180);plt.close(fig)
    lines=['# 七组 80k 固定终点结果','',
           '单种子 DLRSD 自定义划分，训练 1604 张、验证 200 张；缓存特征训练，不含编码器计算费用。完整 17 类列均基于同一次 17 类竞争；未读取封存 test。','',
           '|组|配置|14类 mIoU|17类总 mIoU|17类 seen|17类 unseen|17类 hIoU|时间 s|峰值 MiB|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        lines.append(f"|{x['arm']}|{x['semantic']} + {x['spatial']} (template {x['template_index'] or 'both'})|{x['seen14']*100:.3f}|{x['miou17']*100:.3f}|{x['seen17']*100:.3f}|{x['unseen17']*100:.3f}|{x['hiou17']*100:.3f}|{x['elapsed_s']:.1f}|{x['peak_allocated_mib']:.1f}|")
    lines+=['','## 预设对比（百分点）','']
    for a,b in [('C','B'),('C','F'),('C','G'),('D','B'),('E','C'),('B','H')]:
        lines.append(f"- {a}−{b}：14 类 {100*(results[a]['seen14']-results[b]['seen14']):+.3f}，17 类 seen {100*(results[a]['seen17']-results[b]['seen17']):+.3f}，unseen {100*(results[a]['unseen17']-results[b]['unseen17']):+.3f}，hIoU {100*(results[a]['hiou17']-results[b]['hiou17']):+.3f}。")
    lines+=['','## 类别对照：17 类竞争','', '|类别|B IoU|C IoU|E IoU|B precision/recall|C precision/recall|','|---|---:|---:|---:|---:|---:|']
    def fmt(x):return 'null' if x=='' else f'{100*float(x):.2f}'
    for name in ('dock','ship','court','field','grass','pavement'):
        b=byclass['B',name];c=byclass['C',name];e=byclass['E',name]
        lines.append(f"|{name}|{fmt(b['iou'])}|{fmt(c['iou'])}|{fmt(e['iou'])}|{fmt(b['precision'])}/{fmt(b['recall'])}|{fmt(c['precision'])}/{fmt(c['recall'])}|")
    lines+=['','## 60k 到 80k 的变化','']
    for arm in 'BCDEFGH':
        a=curve[arm,60000];b=curve[arm,80000]
        lines.append(f"- {arm}：训练 loss {float(a['loss_mean_last100']):.4f} → {float(b['loss_mean_last100']):.4f}；17类 seen {100*(float(b['seen17'])-float(a['seen17'])):+.3f} pp，unseen {100*(float(b['unseen17'])-float(a['unseen17'])):+.3f} pp，hIoU {100*(float(b['hiou17'])-float(a['hiou17'])):+.3f} pp。")
    lines+=['','## 描述性决策提示','']
    c=results['C']; competitors=[results[x] for x in 'BFG']
    if all(c['hiou17']>x['hiou17'] for x in competitors):
        lines.append('- C 的 17 类 hIoU 高于 B/F/G，动态模板选择可作为下一步复核候选；需同时核对 seen 与 unseen 变化。')
    else:
        lines.append('- C 的 17 类 hIoU 未同时超过 B/F/G，此次不支持仅凭动态模板选择扩大路线；需检查 seen 与 unseen 的取舍。')
    if results['D']['hiou17']>results['B']['hiou17'] and results['E']['hiou17']>results['C']['hiou17']:
        lines.append('- 两条配对条件传播比较的 hIoU 点估计均为正，值得复核，尚不足以证明跨种子稳定。')
    else:
        lines.append('- 两条配对条件传播比较的 hIoU 未同时为正，本次优先简化条件传播，不将其判断推广到其他数据或种子。')
    lines.append('- B/H 比较的是局部传播开关；H 仍有直接结构输入，因此不是移除 RSIB 的消融。若所有简单和动态配置的 17 类 hIoU 均不可用或很低，应先检查开放词汇路径，再决定是否结束此原型。')
    lines+=['','所有数值是一个种子的点估计；60k 与 80k 是预设评测点，不能仅凭两点证明平台、过拟合或跨域稳定性。逐类指标、原始混淆矩阵和其余四个评测点保存在各组结果 JSON。','',
            '训练配方和资源身份见 protocol.json；匹配的初始化、采样轨迹、实际学习率序列见 matched_checks.json；七组日志、优化器状态和六个 checkpoint 位于 runs/。']
    (root/'EXECUTION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Created report and curves',flush=True)

if __name__=='__main__':main()
