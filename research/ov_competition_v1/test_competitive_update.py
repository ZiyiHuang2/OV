import torch
from .competitive_update import CompetitiveSpatialUpdate


def sample(b=2, k=4, d=8, h=5, w=6):
    torch.manual_seed(9)
    return torch.randn(b,d,k,h,w), (torch.randn(b,k,h,w) * 0.2)


def test_strict_bypass():
    x, s = sample()
    mod = CompetitiveSpatialUpdate(8, 'competitive')
    y = mod(x,s,enabled=False)
    assert y is x and torch.equal(y,x)


def test_gradients_reach_real_new_path():
    x,s = sample()
    x = x.detach().requires_grad_()
    s = s.detach().requires_grad_()
    mod = CompetitiveSpatialUpdate(8, 'competitive', initial_scale=0.01)
    y = mod(x,s)
    (y - 0.7*x).square().mean().backward()
    for part in [mod.alpha,mod.relation[0].weight,mod.relation[2].weight,
                 mod.accept[0].weight,mod.accept[2].weight,mod.value.weight]:
        assert part.grad is not None
        assert part.grad.abs().sum().item()>0, 'missing genuine gradient path'
    assert x.grad is not None and s.grad is not None


def test_modes_identical_trainable_architecture():
    b = CompetitiveSpatialUpdate(8,'self_only')
    c = CompetitiveSpatialUpdate(8,'competitive')
    assert [(n,tuple(p.shape)) for n,p in b.named_parameters()] == [(n,tuple(p.shape)) for n,p in c.named_parameters()]
    c.load_state_dict(b.state_dict(),strict=True)


def test_cross_class_is_only_added_information():
    x,s=sample(b=1)
    # Keep the target class 0 evidence constant; perturb only other class logits.
    s_changed=s.clone()
    s_changed[:,1:]+=0.7
    b=CompetitiveSpatialUpdate(8,'self_only')
    c=CompetitiveSpatialUpdate(8,'competitive')
    b0=b.contrast(s)[:,0]
    b1=b.contrast(s_changed)[:,0]
    c0=c.contrast(s)[:,0]
    c1=c.contrast(s_changed)[:,0]
    torch.testing.assert_close(b0,b1,rtol=0,atol=0)
    assert (c0-c1).abs().max().item()>0.6


def test_permutation_equivariant_without_native_truncation():
    x,s=sample()
    mod=CompetitiveSpatialUpdate(8,'competitive')
    perm=torch.tensor([2,0,3,1]); inv=perm.argsort()
    y=mod(x,s)
    y_perm=mod(x[:,:,perm],s[:,perm])[:,:,inv]
    torch.testing.assert_close(y,y_perm,rtol=1e-5,atol=1e-6)


def test_k_one_competition_falls_back_to_self_only():
    x,s=sample(k=1)
    b=CompetitiveSpatialUpdate(8,'self_only')
    c=CompetitiveSpatialUpdate(8,'competitive')
    c.load_state_dict(b.state_dict(),strict=True)
    torch.testing.assert_close(b.contrast(s),c.contrast(s),rtol=0,atol=0)
    torch.testing.assert_close(b(x,s),c(x,s),rtol=0,atol=0)


def test_class_padding_and_spatial_invalid_leave_original_unchanged():
    x,s=sample(b=1)
    mod=CompetitiveSpatialUpdate(8,'competitive')
    cm=torch.tensor([True,True,False,False])
    pm=torch.ones((1,5,6),dtype=torch.bool)
    pm[:,-1,:]=False
    y=mod(x,s,class_mask=cm,spatial_mask=pm)
    assert torch.equal(y[:,:,2:],x[:,:,2:])
    assert torch.equal(y[:,:,:,-1,:],x[:,:,:,-1,:])
    assert torch.isfinite(y).all()


def test_single_spatial_location_has_no_forced_update():
    x,s=sample(b=1,h=1,w=1)
    mod=CompetitiveSpatialUpdate(8,'competitive')
    y=mod(x,s)
    torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_directionality_has_receiver_neighbor_interaction():
    x,s=sample(b=1,h=3,w=3)
    mod=CompetitiveSpatialUpdate(8,'competitive')
    _,aux=mod(x,s,return_aux=True)
    # Neighbor j=1 for center i=0 is direction right; reverse uses j=3 for i=1.
    a=aux['pair_logits'][0,0,0,5].item()
    b=aux['pair_logits'][0,0,1,3].item()
    assert abs(a-b)>1e-8


def test_rejection_alpha_and_bias_behavior():
    x,s=sample(b=1)
    mod=CompetitiveSpatialUpdate(8,'competitive')
    with torch.no_grad(): mod.alpha.zero_()
    assert torch.equal(mod(x,s),x)
    # The alpha=0 initialization does NOT have immediate relation gradients.
    mod.zero_grad(set_to_none=True)
    mod(x,s).square().mean().backward()
    assert mod.relation[0].weight.grad is None or mod.relation[0].weight.grad.abs().sum().item()==0


def test_gate_can_reject_all_neighbors_exactly():
    x,s=sample(b=1)
    mod=CompetitiveSpatialUpdate(8,'competitive',initial_scale=0.01)
    with torch.no_grad():
        mod.accept[2].weight.zero_()
        mod.accept[2].bias.fill_(-10.0)
    y,aux=mod(x,s,return_aux=True)
    assert aux['gate'].max().item()==0.0
    assert torch.equal(x,y)


def test_mixed_precision_features_cpu():
    x,s=sample(b=1)
    mod=CompetitiveSpatialUpdate(8,'competitive')
    y=mod(x.half(),s.float())
    assert y.dtype==torch.float16 and torch.isfinite(y).all()
