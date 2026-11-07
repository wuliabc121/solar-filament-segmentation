import importlib.util
from pathlib import Path
import numpy as np
from m3_filament.masks import decode,encode


def module():
    spec=importlib.util.spec_from_file_location('sliced_prototype',Path('scripts/sliced_prototype.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_tiles_cover_image_edges_and_mask_offset_is_exact():
    m=module();tiles=m.tile_windows(2000,2048,1024,.5)
    assert len(tiles)==9 and tiles[-1]==[1024,976,2048,2000]
    small=np.ones((3,4),np.uint8);r=m.place_mask(small,[5,6,9,9],[12,14])
    expected=np.zeros((12,14),np.uint8);expected[6:9,5:9]=1
    assert np.array_equal(decode(r),expected)


def test_mask_ios_stitches_cross_tile_partial_instances_only():
    m=module();a=np.zeros((12,12),np.uint8);b=a.copy();a[2:5,2:7]=1;b[2:5,4:9]=1
    groups=m.mask_groups([encode(a),encode(b)],[0,1],.5)
    assert groups==[[0,1]]
    assert m.mask_groups([encode(a),encode(b)],[0,0],.5)==[[0],[1]]
