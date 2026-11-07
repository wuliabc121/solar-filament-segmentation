import numpy as np
import pytest
import torch
from team_pipeline.models import predict_m1,predict_m2

def test_m1_original_coordinate_sigmoid():
 class Fake(torch.nn.Module):
  def forward(self,x):
   assert x.shape==(1,3,512,512)
   return torch.zeros((1,1,512,512),device=x.device)
 p=predict_m1(Fake(),np.zeros((27,41,3),np.uint8),torch.device('cpu'))
 assert p.shape==(27,41) and np.allclose(p,.5)

def test_m2_small_nonsquare_pad_crop_and_tile_validation():
 class Fake(torch.nn.Module):
  def forward(self,pixel_values):
   from types import SimpleNamespace
   return SimpleNamespace(logits=torch.zeros((len(pixel_values),1,pixel_values.shape[2]//4,pixel_values.shape[3]//4),device=pixel_values.device))
 p=predict_m2(Fake(),np.zeros((27,41),np.uint8),torch.device('cpu'),tile=32,overlap=8,batch=2)
 assert p.shape==(27,41) and np.allclose(p,.5)
 for kw in ({'tile':32,'overlap':32},{'tile':32,'overlap':-1},{'batch':0}):
  with pytest.raises(ValueError):predict_m2(Fake(),np.zeros((16,17),np.uint8),torch.device('cpu'),**kw)

def test_yolo_record_requires_original_coordinate_masks():
 from types import SimpleNamespace
 from team_pipeline.models import yolo_record
 from m3_filament.schema import validate_record
 box=SimpleNamespace(conf=torch.tensor([.8]));p=SimpleNamespace(orig_shape=(8,9),masks=SimpleNamespace(data=torch.ones((1,8,9))),boxes=box)
 r=yolo_record(p,'synthetic',{'run_id':'model-smoke'});validate_record(r);assert r['instances'][0]['bbox']==[0,0,9,8]
 p.masks.data=torch.ones((1,8,8))
 with pytest.raises(ValueError):yolo_record(p,'synthetic',{'run_id':'model-smoke'})
