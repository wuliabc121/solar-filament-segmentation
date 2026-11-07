import json
import os
from collections import defaultdict
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils


class COCOParser:
    def __init__(self, ann_file):
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        self.images = {img['id']: img for img in self.data['images']}
        self.anns = defaultdict(list)
        for ann in self.data['annotations']:
            self.anns[ann['image_id']].append(ann)
        self.cats = {cat['id']: cat for cat in self.data['categories']}
    
    def get_image_info(self, image_id):
        return self.images[image_id]
    
    def get_annotations(self, image_id):
        return self.anns[image_id]
    
    def get_all_image_ids(self):
        return list(self.images.keys())
