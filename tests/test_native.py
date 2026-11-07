import json
from pathlib import Path
import numpy as np
from PIL import Image
from m3_filament.io import write_json,write_jsonl
from m3_filament.masks import encode
from m3_filament.native import NativeMaskDataset


def test_native_training_targets_preserve_holes_and_disconnected_fragments(tmp_path):
    m=np.zeros((32,32),np.uint8);m[2:12,3:14]=1;m[5:8,6:10]=0;m[25:28,25:29]=1
    image=tmp_path/'sample.jpg';Image.fromarray(np.full((32,32,3),150,np.uint8)).save(image)
    target=tmp_path/'target.json';write_json(target,{'masks':[encode(m)]})
    write_jsonl(tmp_path/'samples.jsonl',[{'sample_id':'pass_a','image_path':str(image),
        'target_path':str(target),'split':'train','instance_count':1}])
    data=NativeMaskDataset(tmp_path,'train',32)
    batch=data[0]
    assert batch['img'].shape==(3,32,32)
    assert np.array_equal(batch['masks'][0].numpy(),m)
    assert batch['cls'].shape==(1,1)
    assert np.allclose(batch['bboxes'][0].numpy(),[16/32,15/32,26/32,26/32])


def test_native_collation_retains_instance_to_image_association(tmp_path):
    m=np.zeros((32,32),np.uint8);m[2:10,3:11]=1
    image=tmp_path/'sample.jpg';Image.fromarray(np.zeros((32,32,3),np.uint8)).save(image)
    target=tmp_path/'target.json';write_json(target,{'masks':[encode(m),encode(m)]})
    write_jsonl(tmp_path/'samples.jsonl',[{'sample_id':'a','image_path':str(image),'target_path':str(target),
         'split':'train','instance_count':2},{'sample_id':'b','image_path':str(image),'target_path':str(target),
         'split':'train','instance_count':2}])
    d=NativeMaskDataset(tmp_path,'train',32);b=d.collate_fn([d[0],d[1]])
    assert b['batch_idx'].tolist()==[0,0,1,1]
    assert b['masks'].shape==(4,32,32)


def test_training_crop_preserves_original_coordinates_and_binary_target(tmp_path):
    m=np.zeros((64,96),np.uint8);m[25:39,40:56]=1;m[29:33,45:49]=0
    rgb=np.zeros((64,96,3),np.uint8);rgb[:,:,0]=np.arange(96,dtype=np.uint8)[None,:]
    image=tmp_path/'im.png';Image.fromarray(rgb).save(image)
    target=tmp_path/'target.json';write_json(target,{'masks':[encode(m)]})
    write_jsonl(tmp_path/'samples.jsonl',[{'sample_id':'a','image_path':str(image),'target_path':str(target),
       'split':'train','instance_count':1}])
    d=NativeMaskDataset(tmp_path,'train',32,crop_size=32,crop_probability=1.)
    b=d[0];x0,y0,x1,y1=b['crop_xyxy']
    assert (x1-x0,y1-y0)==(32,32)
    assert np.array_equal(b['masks'][0].numpy(),m[y0:y1,x0:x1])
    assert np.array_equal(b['img'].numpy().transpose(1,2,0),rgb[y0:y1,x0:x1])
