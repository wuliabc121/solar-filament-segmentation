"""Submission encoding only. This module never uploads to Kaggle."""
import csv
import io
from .schema import validate_record
from .masks import encode,decode
from .io import write_text


def write_submission(records,path,expected_shape=(2048,2048)):
    stream=io.StringIO(newline='');writer=csv.writer(stream)
    writer.writerow(['filament_id','segmentation_rle']);seen=set()
    for r in records:
        validate_record(r,exclusive=True)
        if r['observation_id'] in seen:raise ValueError('duplicate observation_id')
        seen.add(r['observation_id'])
        if tuple(r['image_size'])!=tuple(expected_shape):raise ValueError(f'submission requires shape {expected_shape}')
        for inst in r['instances']:
            if inst.get('requires_refiner'):raise ValueError('unreviewed semantic residual cannot be submitted as a final physical instance')
            counts=encode(decode(inst['segmentation']))['counts']
            writer.writerow([f"{r['observation_id']}_{inst['instance_id']}",counts])
    write_text(path,stream.getvalue())
