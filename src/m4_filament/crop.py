"""Native-resolution M4 crop/paste interface. No model or split is defined here."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CropTransform:
    original_hw: tuple[int, int]
    origin_xy: tuple[int, int]
    crop_hw: tuple[int, int]
    target_bbox_xyxy_exclusive: tuple[int, int, int, int]
    context_px: int
    scale_xy: tuple[float, float] = (1.0, 1.0)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, record):
        return cls(**{key: tuple(value) if isinstance(value, list) else value
                      for key, value in record.items()})


def plan_crop(coarse_mask, context_px=32, buckets=(384, 512, 768, 1152)):
    coarse_mask = np.asarray(coarse_mask)
    if coarse_mask.ndim != 2:
        raise ValueError('Expected one target mask in H×W coordinates.')
    ys, xs = np.nonzero(coarse_mask)
    if not len(xs):
        raise ValueError('empty_target_prompt')
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
    needed = max(x1-x0, y1-y0) + 2*context_px
    side = next((int(v) for v in buckets if v >= needed), None)
    if side is None:
        raise ValueError('crop_exceeds_validated_size_buckets')
    x = int(np.floor((x0+x1-side)/2))
    y = int(np.floor((y0+y1-side)/2))
    return CropTransform(tuple(coarse_mask.shape), (x, y), (side, side), (x0,y0,x1,y1), int(context_px))


def crop_array(array, transform):
    array = np.asarray(array)
    if array.ndim != 2 or tuple(array.shape) != transform.original_hw:
        raise ValueError('Array must match original H×W coordinates.')
    x, y = transform.origin_xy
    ch, cw = transform.crop_hw
    h, w = transform.original_hw
    x0, y0, x1, y1 = max(0,x), max(0,y), min(w,x+cw), min(h,y+ch)
    result = np.zeros((ch,cw), dtype=array.dtype)
    result[y0-y:y1-y, x0-x:x1-x] = array[y0:y1, x0:x1]
    return result


def paste_array(local_array, transform):
    local_array = np.asarray(local_array)
    if local_array.ndim != 2 or tuple(local_array.shape) != transform.crop_hw:
        raise ValueError('Local output shape must match the recorded crop.')
    x, y = transform.origin_xy
    ch, cw = transform.crop_hw
    h, w = transform.original_hw
    x0, y0, x1, y1 = max(0,x), max(0,y), min(w,x+cw), min(h,y+ch)
    result = np.zeros((h,w), dtype=local_array.dtype)
    result[y0:y1,x0:x1] = local_array[y0-y:y1-y,x0-x:x1-x]
    return result


def make_input(gray_uint8, semantic_probability, coarse_mask, transform):
    gray_uint8 = np.asarray(gray_uint8)
    probability = np.asarray(semantic_probability)
    coarse_mask = np.asarray(coarse_mask)
    if gray_uint8.dtype != np.uint8:
        raise ValueError('Expected the original 8-bit grayscale image.')
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError('Semantic probability must contain finite values in [0,1].')
    return np.stack([
        crop_array(gray_uint8, transform).astype(np.float32)/255.0,
        crop_array(probability, transform).astype(np.float32),
        crop_array(coarse_mask != 0, transform).astype(np.float32),
    ])
