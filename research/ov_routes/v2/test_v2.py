"""CPU-only algorithm tests; toy host never replaces real-weight CUDA preflight."""
from types import SimpleNamespace
import json

import pytest
import torch
from torch import nn

from .evidence import build_evidence, windows
from .engine import batch_evidence, fuse_probabilities, predict, training_example, first_pass
from .objective import pairwise_ranking, selector_loss
from .selector import SpatialSemanticSelector
from . import run_v2


torch.set_num_threads(1)


class ToyHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls=[]

    @torch.no_grad()
    def text_vectors(self,names):
        base=torch.tensor([[.1,.5,.3,.8,.4,.2],[-.5,.1,.6,.2,.8,.7],[.3,.6,-.2,.4,.9,.1]])
        return base[:len(names)]

    @torch.no_grad()
    def predict_logits(self,image,names):
        self.calls.append(tuple(image.shape))
        x=image[0]/255
        # Cropping genuinely changes the local predictions via image means.
        v=x.mean((1,2))
        return torch.stack((x[0]+v[0]*.3, x[1]*.8+v[1]*.3,
                            x[2]*.9+v[2]*.3),dim=0)[:len(names)].unsqueeze(0)


class FixedSelector(nn.Module):
    def __init__(self,scores):
        super().__init__()
        self.register_buffer('scores',torch.tensor(scores,dtype=torch.float32))

    def forward(self,evidence):
        return self.scores[None].expand(evidence['spatial'].shape[0],-1)


def sample(h=32,w=32):
    torch.manual_seed(19)
    return torch.rand(3,h,w)*255, torch.randint(3,(h,w)), ['airplane','building','road']


def test_eight_windows_are_exact_old_policy_geometry():
    assert windows(384,384)==[
        (0,0,192,192),(0,192,192,384),(192,0,384,192),(192,192,384,384),
        (0,0,288,288),(0,96,288,384),(96,0,384,288),(96,96,384,384)]


def test_evidence_shapes_and_gt_free_class_permutation():
    rgb,_,names=sample()
    m=ToyHost()
    logits=m.predict_logits(rgb[None],names)[0]
    txt=m.text_vectors(names)
    a=build_evidence(logits,txt)
    assert a['spatial'].shape==(8,8,8,8)
    assert a['legacy'].shape==(8,15)
    assert a['class_tokens'].shape==(8,4,8)
    assert a['class_valid'][:,3].sum()==0
    perm=torch.tensor([2,0,1])
    b=build_evidence(logits[perm],txt[perm])
    for key in a:
        torch.testing.assert_close(a[key],b[key],rtol=1e-5,atol=1e-6)
    assert not any(x in build_evidence.__code__.co_varnames for x in ('gt','label','mask'))


def test_spatial_maps_preserve_layout_while_mean_summary_unchanged():
    logits=torch.zeros(2,16,16)
    logits[0,:8]=2; logits[1,8:]=2
    txt=torch.tensor([[1.,0.],[0.,1.]])
    x=build_evidence(logits,txt,boxes=[(0,0,16,16)])
    logits2=logits.flip(-2)
    y=build_evidence(logits2,txt,boxes=[(0,0,16,16)])
    torch.testing.assert_close(x['legacy'],y['legacy'],atol=1e-6,rtol=0)
    assert not torch.allclose(x['spatial'],y['spatial'])


def test_selector_modes_and_gradient_are_finite():
    rgb,_,names=sample()
    m=ToyHost()
    ev=batch_evidence(build_evidence(m.predict_logits(rgb[None],names)[0],m.text_vectors(names)))
    for mode in ('legacy_only','spatial_only','spatial_cross','full_rank'):
        args=run_v2.model_settings(mode,64,6,8,4)
        net=SpatialSemanticSelector(**args)
        gains=net(ev)
        assert gains.shape==(1,8)
        assert torch.isfinite(gains).all()
        (gains.square().mean()).backward()
        assert sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)>0


def test_loss_prefers_correct_window_order():
    target=torch.tensor([[1.,.2,-.4,.1]])
    assert pairwise_ranking(target,target)<pairwise_ranking(-target,target)
    good,_=selector_loss(target,target/100)
    bad,_=selector_loss(-target,target/100)
    assert good<bad


def test_train_targets_match_old_bce_definition():
    rgb,gt,names=sample()
    m=ToyHost()
    out=training_example(m,rgb,gt,names)
    assert out['gains'].shape==(8,)
    assert len(m.calls)==9
    m2=ToyHost()
    base=m2.predict_logits(rgb[None],names)[0].sigmoid()
    box=windows(*rgb.shape[-2:])[0]
    y0,x0,y1,x1=box
    crop=m2.predict_logits(rgb[None,:,y0:y1,x0:x1],names)[0].sigmoid()
    valid=gt!=255
    tgt=torch.nn.functional.one_hot(gt[y0:y1,x0:x1],len(names)).float()
    old=base[:,y0:y1,x0:x1].permute(1,2,0)
    new=((base[:,y0:y1,x0:x1]+crop)/2).permute(1,2,0)
    ref=(torch.nn.functional.binary_cross_entropy(old,tgt,reduction='sum')-
         torch.nn.functional.binary_cross_entropy(new,tgt,reduction='sum'))/(valid.sum()*len(names))
    torch.testing.assert_close(ref,out['gains'][0])


def test_inference_no_op_exact_zero_and_call_budget():
    rgb,gt,names=sample()
    host=ToyHost()
    pred,trace=predict(host,rgb,names,FixedSelector([-2]*8),allow_skip=True)
    assert trace['encoder_calls']==1 and trace['skip'] and not trace['selected']
    torch.testing.assert_close(pred,host.predict_logits(rgb[None],names)[0].sigmoid())
    host.calls.clear()
    pred2,trace2=predict(host,rgb,names,FixedSelector([0,0,0,2,0,0,0,0]))
    assert trace2['encoder_calls']==2 and trace2['selected']==[3]
    assert len(host.calls)==2 and pred2.shape==(3,32,32)
    assert not torch.allclose(pred,pred2)


def test_fusion_matches_unmodified_pixels_and_average_inside():
    base=torch.zeros(3,8,8)
    box=(0,0,4,4)
    crop=torch.full((3,4,4),2.)
    prob=fuse_probabilities(base,[(box,crop)])
    torch.testing.assert_close(prob[:,:4,:4],torch.full((3,4,4),(.5+torch.sigmoid(torch.tensor(2.)).item())/2))
    assert torch.equal(prob[:,4:,4:],torch.full((3,4,4),.5))


def test_checkpoint_cache_train_smoke(tmp_path):
    rgb,gt,names=sample()
    host=ToyHost()
    sample1=training_example(host,rgb,gt,names)
    sample2=training_example(host,rgb.flip(-1),gt.flip(-1),names)
    cache=tmp_path/'cache';cache.mkdir()
    meta={'format':'route-b-v2-cache-v2','split':'train','source_only':True,
          'code':run_v2.source_identity(), 'identity':{'fake':'for_cpu_test'},
          'smoke_only':True,'examples_written':2,'grid_size':8,'top_m':4}
    (cache/'meta.json').write_text(json.dumps(meta))
    torch.save(sample1,cache/'train_00000_a00.pt')
    torch.save(sample2,cache/'train_00001_a00.pt')
    out=tmp_path/'train'
    a=SimpleNamespace(cache_dir=str(cache),output=str(out),steps=3,batch_size=2,
                      lr=1e-3,device='cpu',seed=42,hidden_dim=64,ablation='full_rank',save_every=2,
                      resume=None,allow_smoke=True,internal_val_fraction=.25,split_seed=20261121)
    run_v2.do_train(a)
    state=torch.load(out/'selector_final.pt',map_location='cpu',weights_only=True)
    assert state['step']==3 and state['config']['sample_count']==2
    assert (out/'COMPLETE.json').exists()
    assert (out/'internal_val_diagnostics.json').exists()
    assert (out/'internal_fit_diagnostics.json').exists()
    assert not set(json.loads((out/'split.json').read_text())['fit_groups']) & set(json.loads((out/'split.json').read_text())['internal_val_groups'])
    assert (out/'checkpoint_000002.pt').exists()
    resumed_dir=tmp_path/'train_resumed'
    a.output=str(resumed_dir)
    a.resume=str(out/'checkpoint_000002.pt')
    run_v2.do_train(a)
    resumed=torch.load(resumed_dir/'selector_final.pt',map_location='cpu',weights_only=True)
    for k in state['selector']:
        torch.testing.assert_close(state['selector'][k],resumed['selector'][k],rtol=0,atol=0)


class HookHost(ToyHost):
    def __init__(self):
        super().__init__()
        self.predictor=nn.Module()
        self.predictor.transformer=nn.Module()
        self.predictor.transformer.conv1=nn.Identity()

    def predict_logits(self,image,names):
        # Synthetic CAT-Seg conv1 input, only for hook interface tests.
        raw=torch.nn.functional.interpolate(image[:,0:1],size=(8,8),mode='bilinear',align_corners=False)[0]
        tensor=torch.stack([raw[0]+i for i in range(len(names))]).unsqueeze(1)
        self.predictor.transformer.conv1(tensor)
        return super().predict_logits(image,names)


def test_real_cost_hook_interface_and_optional_ablation():
    rgb,gt,names=sample()
    host=HookHost()
    a=training_example(host,rgb,gt,names,with_raw_cost=True)
    assert a['evidence']['spatial'].shape==(8,12,8,8)
    assert len(host.calls)==9  # hook only on first full-image pass
    net=SpatialSemanticSelector(**run_v2.model_settings('cost_full_rank',64,6,8,4))
    score=net(batch_evidence(a['evidence']))
    assert score.shape==(1,8)
    host.calls.clear()
    out,trace=predict(host,rgb,names,net)
    assert out.shape==(3,32,32) and trace['encoder_calls']==2
    assert len(host.calls)==2
    assert len(host.predictor.transformer.conv1._forward_pre_hooks)==0


def test_unmodified_v1_legacy_feature_definition():
    rgb,_,names=sample()
    host=ToyHost()
    logits=host.predict_logits(rgb[None],names)[0]
    text=torch.nn.functional.normalize(host.text_vectors(names),dim=-1)
    ev=build_evidence(logits,host.text_vectors(names))
    p=logits.softmax(0)
    y0,x0,y1,x1=windows(32,32)[0]
    q=p[:,y0:y1,x0:x1]
    means=q.mean((1,2))
    values,idx=means.topk(2)
    entropy=-(q*q.clamp_min(1e-12).log()).sum(0).mean()/__import__('math').log(len(names))
    disagree=(q.argmax(0)!=idx[0]).float().mean()
    scalars=torch.stack((values[0],values[1],values[0]-values[1],entropy,disagree,
                         values.new_tensor(.5),values.new_tensor(.5),
                         values.new_tensor(.25),values.new_tensor(.25)))
    torch.testing.assert_close(ev['legacy'][0],torch.cat(((text[idx[0]]-text[idx[1]]).abs(),scalars)))


def test_posthoc_summary_never_changes_selection():
    from .summarize_v2 import summarize
    zero={'miou':.29,'seen':.32,'unseen':.26,'hiou':.28,
          'per_image':[{'index':0,'file':'a.png','miou':.4},
                       {'index':1,'file':'b.png','miou':.5}]}
    v2={'miou':.31,'seen':.34,'unseen':.28,'hiou':.30,
        'per_image':[{'index':0,'file':'a.png','miou':.45,'selected':[1],
                      'encoder_calls':2,'skip':False},
                     {'index':1,'file':'b.png','miou':.49,'selected':[],
                      'encoder_calls':1,'skip':True}]}
    result=summarize(zero,v2)
    assert abs(result['deltas_pp']['miou']-2.)<1e-8
    assert result['window_selection_counts']['1']==1
    assert result['mean_host_calls']==1.5 and result['skipped_images']==1
    assert result['imagewise_miou_change']['improved']==1
    assert result['imagewise_miou_change']['harmed']==1
    changed={'miou':.31,'per_image':[dict(v2['per_image'][0],file='wrong.png'),v2['per_image'][1]]}
    with pytest.raises(ValueError):summarize(zero,changed)


def test_exact_v1_selector_architecture_and_matched_cache():
    from .selector import LegacySelectorExact
    m=LegacySelectorExact(512,64)
    assert sum(p.numel() for p in m.parameters())==33473
    ev={'legacy':torch.randn(2,8,521)}
    assert m(ev).shape==(2,8)
    assert set(m.state_dict())=={'net.0.weight','net.0.bias','net.2.weight','net.2.bias'}


# V2 revision validation: decision-readiness before expensive CAT-Seg training.
from .diagnostics import (gt_present_miou, summarize_predictions,
                          ranking_pair_counts, evaluate_cached_selector)
from .selector import HeuristicSelector, IndexSelector, LegacySelectorExact


def test_source_holdout_keeps_original_augmentations_together():
    samples=[{'meta':{'train_index':i, 'source_group':str(i//2)}}
             for i in range(12) for _ in range(3)]
    tr,va,summary=run_v2.internal_split(samples,.25,20261121)
    a=set(samples[i]['meta']['source_group'] for i in tr)
    b=set(samples[i]['meta']['source_group'] for i in va)
    assert a.isdisjoint(b)
    assert len(tr)+len(va)==len(samples)
    assert run_v2.internal_split(samples,.25,20261121)==(tr,va,summary)


def test_internal_split_fails_single_source_group():
    samples=[{'meta':{'source_group':'one'}} for _ in range(5)]
    with pytest.raises(ValueError,match='>=2'):
        run_v2.internal_split(samples,.1,20261121)


def test_heuristics_rank_without_learning_or_gt():
    rgb,_,names=sample()
    host=ToyHost()
    ev=batch_evidence(build_evidence(host.predict_logits(rgb[None],names)[0],host.text_vectors(names)))
    for mode in ('entropy','low_margin','disagreement'):
        selector=HeuristicSelector(mode)
        x=selector(ev)
        assert x.shape==(1,8) and torch.isfinite(x).all()
        assert sum(p.numel() for p in selector.parameters())==0
        host.calls.clear()
        prob,trace=predict(host,rgb,names,selector,scores_are_gain=False)
        assert prob.shape==(3,32,32) and trace['encoder_calls']==2
        assert trace['score_semantics']=='heuristic_rank'
        assert 'estimated_gains_scaled_bce' not in trace
        assert len(host.calls)==2
    with pytest.raises(ValueError,match='Skip requires'):
        predict(host,rgb,names,HeuristicSelector('entropy'),allow_skip=True,scores_are_gain=False)


def test_legacy_wide_capacity_is_matched_for_clip512():
    st=run_v2.model_settings('legacy_wide',64,512,8,4)
    selector=run_v2.create_selector('legacy_wide',st)
    assert st['hidden_dim']==120
    assert isinstance(selector,LegacySelectorExact)
    assert sum(p.numel() for p in selector.parameters())==62761
    base=LegacySelectorExact(512,64)
    assert sum(p.numel() for p in base.parameters())==33473


def test_training_cache_contains_gt_present_iou_diagnostics():
    rgb,gt,names=sample()
    example=training_example(ToyHost(),rgb,gt,names)
    assert example['iou_gains'].shape==(8,)
    assert torch.isfinite(example['iou_gains']).all()
    assert torch.isfinite(example['gains']).all()
    # IoU signal is never present in the inference evidence.
    assert 'iou_gains' not in example['evidence']
    assert gt_present_miou(gt,gt,3)==pytest.approx(1.,abs=1e-6)


def test_diagnostic_can_find_bce_iou_conflict_without_claiming_dataset_miou():
    rows=[{'predicted':[.8,.2], 'gains':[.5,-.3],
           'iou_gains':[-.05,.04]},
          {'predicted':[.2,.9], 'gains':[.1,.4],
           'iou_gains':[.02,.03]}]
    summary=summarize_predictions(rows)
    assert summary['n_images']==2
    assert summary['top1_bce_oracle_hit_rate']==1.
    assert summary['gt_present_iou']['bce_best_hurts_iou_count']==1
    assert summary['gt_present_iou']['bce_best_hurts_iou_rate']==.5
    assert summary['mean_bce_regret']==pytest.approx(0.,abs=1e-10)
    assert summary['ranking_pairs']['possible']==2


def test_rank_pairs_report_actual_cutoff_support():
    g=torch.tensor([[0.001,0.0011,0.002], [0.,0.,0.]])
    x=ranking_pair_counts(g, gain_scale=100.,min_gap=.02)
    assert x=={'eligible':2,'possible':6}


def test_exact_source_content_hash_changes_with_mask_file(tmp_path):
    (tmp_path/'rgb').write_bytes(b'raw image bytes')
    (tmp_path/'gt').write_bytes(b'mask bytes')
    class M:
        rows={'train':[{'file_name':'rgb','sem_seg_file_name':'gt'}]}
        def resolve(self,p):return tmp_path/p
    a=run_v2.dataset_files_sha256(M(),'train')
    (tmp_path/'gt').write_bytes(b'MASK bytes')
    b=run_v2.dataset_files_sha256(M(),'train')
    assert a['unique_files']==2 and a['sha256']!=b['sha256']


def test_hook_does_not_mutate_first_pass_scores():
    rgb,_,names=sample()
    host=HookHost()
    a,_=first_pass(host,rgb,names,with_raw_cost=False)
    b,raw=first_pass(host,rgb,names,with_raw_cost=True)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert raw.shape==(3,8,8)
    assert not host.predictor.transformer.conv1._forward_pre_hooks


def test_policy_evaluation_is_gt_free_for_selection(tmp_path, monkeypatch):
    from PIL import Image
    rgb, gt, names = sample()
    Image.fromarray(rgb.permute(1,2,0).byte().numpy()).save(tmp_path/'sample.png')
    Image.fromarray(gt.byte().numpy()).save(tmp_path/'label.png')

    class FakeData:
        all_names=names
        seen=[0,1]
        rows={'val':[{'file_name':'sample.png','sem_seg_file_name':'label.png'}]}
        def read(self,split,i):
            assert split=='val'
            return rgb, gt
        def resolve(self,relative):
            return tmp_path/relative

    # The standalone review package omits original OV ../data.py; use a
    # fully-qualified import and stub only that existing parent's metrics.
    import importlib, types, sys
    production_entry=importlib.import_module('research.ov_routes.v2.run_v2')
    fake_data_module=types.ModuleType('research.ov_routes.data')
    def fake_metrics(cm,seen):
        k=len(cm); tp=cm.diag().float()
        union=cm.sum(0).float()+cm.sum(1).float()-tp
        iou=tp/union.clamp_min(1)
        s=float(iou[list(seen)].mean())
        u=float(iou[[i for i in range(k) if i not in seen]].mean())
        return {'miou':float(iou.mean()),'seen':s,'unseen':u,
                'hiou':2*s*u/(s+u) if s+u else 0.}
    fake_data_module.metrics=fake_metrics
    monkeypatch.setitem(sys.modules,'research.ov_routes.data',fake_data_module)
    host=ToyHost()
    monkeypatch.setattr(production_entry,'get_host',lambda args:(host,FakeData()))
    for policy,calls in [('zero',1),('fixed',2),('random',2),('entropy',2),
                         ('low_margin',2),('disagreement',2)]:
        host.calls.clear()
        args=SimpleNamespace(policy=policy,allow_skip=False,selector_checkpoint=None,
                            manifest='fake',config='fake',weights='fake',legacy_weights=False,
                            device='cpu',seed=42,output=str(tmp_path/f'out_{policy}'))
        production_entry.do_evaluate(args)
        report=json.loads((tmp_path/f'out_{policy}'/'validation.json').read_text())
        assert report['oracle_or_gt_in_selection'] is False
        assert report['total_host_calls']==calls
        assert len(host.calls)==calls
        assert len(report['per_image'])==1
    args.policy='entropy';args.allow_skip=True;args.output=str(tmp_path/'fail')
    with pytest.raises(ValueError,match='No-op threshold'):
        production_entry.do_evaluate(args)


def test_source_target_diagnostic_before_training(tmp_path):
    from .diagnostics import summarize_target_alignment
    rgb,gt,names=sample()
    s1=training_example(ToyHost(),rgb,gt,names)
    s2=training_example(ToyHost(),rgb.flip(-1),gt.flip(-1),names)
    d=summarize_target_alignment([s1,s2])
    assert d['n_images_augmented']==2 and d['n_actions']==8
    assert 0<=d['top1_bce_iou_same_window_rate']<=1
    assert d['rank_pairs']['possible']==2*8*7//2
    assert d['scope'].startswith('SOURCE TRAIN')
    assert d['no_gt_in_inference'] is True
    cache=tmp_path/'cache';cache.mkdir()
    for i,example in enumerate((s1,s2)):
        example['meta'].update(train_index=i,augmentation=0,source_group=str(i))
        torch.save(example,cache/f'train_{i:05d}_a00.pt')
    (cache/'meta.json').write_text(json.dumps({
        'format':'route-b-v2-cache-v2','split':'train','source_only':True,
        'code':run_v2.source_identity(),'smoke_only':True,
        'examples_written':2}))
    args=SimpleNamespace(cache_dir=str(cache),output=str(tmp_path/'audit'),
                         internal_val_fraction=.1,split_seed=20261121,allow_smoke=True)
    run_v2.do_analyze_cache(args)
    assert (tmp_path/'audit'/'source_fit_target_alignment.json').exists()
    assert (tmp_path/'audit'/'source_internal_val_target_alignment.json').exists()


def test_formal_source_subset_is_deterministic_and_not_prefix():
    ids=run_v2.select_source_indices(100,10,42)
    assert len(ids)==10 and len(set(ids))==10 and ids==sorted(ids)
    assert ids==run_v2.select_source_indices(100,10,42)
    assert ids!=list(range(10))
    assert run_v2.select_source_indices(10,0,42)==list(range(10))
    assert run_v2.select_source_indices(10,20,42)==list(range(10))


def test_external_evaluation_identity_checks_host_not_dataset(monkeypatch):
    from types import SimpleNamespace
    source={'manifest_sha256':'source-different-manifest',
            'config_sha256':'c','weights_sha256':'w',
            'host_and_catseg_source':{'sha256':'h','files':3},
            'train_names':['a','b']}
    cfg={'cache_identity':source,'settings':{'text_dim':6}}
    monkeypatch.setattr(run_v2,'sha',lambda p:{'target_manifest':'target','cfg':'c','weight':'w'}[p])
    monkeypatch.setattr(run_v2,'base_model_source_sha256',lambda:{'sha256':'h','files':3})
    fake_host=SimpleNamespace(text_vectors=lambda names:torch.ones(len(names),6))
    data=SimpleNamespace(all_names=['a','b','c'])
    a=SimpleNamespace(manifest='target_manifest',config='cfg',weights='weight')
    result=run_v2.verify_evaluation_identity(cfg,a,fake_host,data)
    assert result['cross_dataset'] and result['evaluation_class_count']==3
    cfg['cache_identity']['weights_sha256']='altered'
    with pytest.raises(ValueError,match='weights_sha256'):
        run_v2.verify_evaluation_identity(cfg,a,fake_host,data)
