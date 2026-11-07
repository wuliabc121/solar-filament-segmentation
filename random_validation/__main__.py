import argparse
from .inference import m3_inference,semantic_inference
p=argparse.ArgumentParser();p.add_argument('phase',choices=['M3_C','M3_S','semantics','score']);a=p.parse_args()
if a.phase=='semantics':semantic_inference()
elif a.phase=='score':
 from .scoring import comparison
 comparison()
else:m3_inference(a.phase)
