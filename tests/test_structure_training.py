import importlib.util
from pathlib import Path
import torch
import numpy as np


def module():
 spec=importlib.util.spec_from_file_location('structure_training',Path('scripts/structure_training.py'))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_area_downsample_retains_tiny_foreground_lost_by_nearest():
 m=module();x=torch.zeros(1,8,8);x[0,1,1]=1
 assert torch.nn.functional.interpolate(x[None],(2,2),mode='nearest').sum()==0
 reduced=m.coverage_targets(x,(2,2))
 assert torch.allclose(reduced.sum()*16,x.sum()) and reduced.max()>0


def test_structure_loss_prefers_connected_correct_mask_and_has_gradient():
 m=module();gt=torch.zeros(1,1,24,24);gt[:,:,12,3:21]=1
 good=gt*.98+.01;broken=good.clone();broken[:,:,12,10:14]=.01
 assert m.structure_terms(good,gt)[1].item()<m.structure_terms(broken,gt)[1].item()
 logits=torch.zeros_like(gt,requires_grad=True);terms=m.structure_terms(logits.sigmoid(),gt)
 sum(x.mean() for x in terms).backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
 empty=torch.zeros_like(gt);assert all(torch.isfinite(x).all() for x in m.structure_terms(empty,empty))


def test_small_target_sampling_is_bounded_and_geometry_aligned():
 m=module();masks=[]
 for size in [2,4,16]:
  a=np.zeros((32,32),np.uint8);a[3:3+size,4:4+size]=1;masks.append(a)
 weights=m.small_weights(masks);assert max(weights)/min(weights)<=4 and weights[0]>weights[-1]
 image=np.stack([masks[1]*200]*3,axis=-1)
 rotated,targets=m.rotate_pair(image,[masks[1]],1)
 assert np.array_equal(rotated[:,:,0]>0,targets[0]>0)


def test_coverage_handles_empty_image_batch_targets():
 m=module();assert m.coverage_targets(torch.zeros(0,32,32),(8,8)).shape==(0,8,8)


def test_complete_instance_loss_backward_does_not_mutate_sigmoid_or_gt():
 m=module();proto=torch.randn(4,16,16,requires_grad=True);pred=torch.randn(2,4,requires_grad=True)
 gt=torch.zeros(2,16,16);gt[:,4:10,5:11]=1;original=gt.clone()
 boxes=torch.tensor([[3.,3.,12.,12.],[3.,3.,12.,12.]])
 loss=m.StructureLoss.single_mask_loss(gt,pred,proto,boxes,torch.tensor([.3,.3]))
 loss.backward();assert torch.isfinite(pred.grad).all() and torch.isfinite(proto.grad).all()
 assert torch.equal(gt,original)


def test_subprototype_line_keeps_nonzero_complete_loss_gradient():
 import sys
 sys.path.insert(0,str(Path('scripts').resolve()))
 import structure_method_v2 as v2
 import structure_training as m
 original=m.differentiable_crop;m.differentiable_crop=v2.coverage_crop
 try:
  g=m.coverage_targets(torch.tensor([[[0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,1.,1.,1.,1.,1.,0.]]+[[0.]*8]*6]),(2,2))
  proto=torch.zeros(1,2,2,requires_grad=True);coeff=torch.ones(1,1,requires_grad=True)
  box=torch.tensor([[.25,.25,1.75,.5]])
  loss=m.StructureLoss.single_mask_loss(g,coeff,proto,box,torch.tensor([.09375]));loss.backward()
  assert proto.grad.abs().sum()>0 and torch.isfinite(proto.grad).all()
  assert v2.coverage_crop(g,box).sum()>0
 finally:m.differentiable_crop=original
