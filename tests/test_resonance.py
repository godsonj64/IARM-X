import torch
from iarmx.model.resonance import ResonanceTransform

def test_resonance_shapes_and_gate_simplex():
    m=ResonanceTransform(dim=64,n_heads=4,n_operators=3,rank=4)
    x=torch.randn(2,7,4,16); c=torch.randn(2,7,64)
    y,g,d=m(x,c)
    assert y.shape==x.shape and d.shape==x.shape
    assert g.shape==(2,7,4,3)
    assert torch.allclose(g.sum(-1),torch.ones_like(g.sum(-1)),atol=1e-5)
