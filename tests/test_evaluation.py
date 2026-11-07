import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from m3_filament.evaluation import pq_counts, aggregate_counts, pairwise_iou,validate_evaluation_provenance,ensure_eval_images
from m3_filament.masks import encode
from m3_filament.schema import make_record
from m3_filament.io import content_hash,write_jsonl


def original_functions():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'references/official-self-evaluation-v6.py').read_text())
    names={'get_overlap_matrices','fp_count_hit','fn_count_hit','get_pq_score'}
    selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    env={'torch':torch,'pd':SimpleNamespace(DataFrame=object)}
    exec(compile(selected,'official-pq-v6','exec'),env)
    return env


class Rows:
    def __init__(self,matrices):self.matrices=matrices
    def itertuples(self,index=False):
        return iter(SimpleNamespace(iou_matrix=torch.tensor(m),n_gt=m.shape[0],n_pred=m.shape[1]) for m in self.matrices)


def test_pq_strict_half_boundary_and_pooled_counts():
    a=pq_counts(np.array([[.5,.6],[0,0]],np.float32))
    assert a['tp']==1 and a['fp']==1 and a['fn']==1
    assert np.isclose(aggregate_counts([a])['pq'],.3)


def test_official_all_qualified_pairs_are_not_hungarian_matching():
    c=pq_counts(np.array([[.6,.7]],np.float32))
    assert (c['tp'],c['fp'],c['fn'])==(2,0,0)


def test_local_score_matches_published_notebook_boundaries_and_random_matrices():
    rng=np.random.default_rng(42)
    matrices=[np.zeros((0,0),np.float32),np.zeros((0,3),np.float32),np.zeros((2,0),np.float32),
              np.array([[.5,.50001],[.6,0]],np.float32)]
    matrices += [rng.random((int(rng.integers(1,6)),int(rng.integers(1,6))),dtype=np.float32) for _ in range(30)]
    official=original_functions()['get_pq_score'](Rows(matrices))
    assert np.isclose(aggregate_counts([pq_counts(m) for m in matrices])['pq'],official,atol=1e-7)


def test_rle_iou_matches_original_torch_overlap_on_asymmetric_masks():
    rng=np.random.default_rng(13)
    gt=(rng.random((3,5,9))>.65).astype(np.float32)
    pred=(rng.random((4,5,9))>.7).astype(np.float32)
    expected=original_functions()['get_overlap_matrices'](torch.from_numpy(gt),torch.from_numpy(pred))[0].numpy()
    got=pairwise_iou([encode(m) for m in gt],[encode(m) for m in pred])
    assert np.allclose(got,expected,atol=1e-7)


def provenance_fixture():
    cfg={'conf':.003,'imgsz':512,'max_det':100}
    source={'run_id':'p','checkpoint_sha256':'a'*64,'split_hash':'s','training_run_id':'t',
            'unseen_check_passed':True,'inference_config':cfg,'inference_config_sha256':content_hash(cfg),
            'source_manifest_sha256':'c'*64}
    records=[make_record(obs,[3,7],[],dict(source)) for obs in ['obs_a','obs_b']]
    manifest={'run_id':'t','checkpoint_sha256':'a'*64,'training_observation_ids':['train_obs']}
    return records,{'split_hash':'s'},manifest


@pytest.mark.parametrize('key,value',[
    ('checkpoint_sha256','b'*64),('split_hash','wrong'),('unseen_check_passed',False),
    ('inference_config_sha256','bad'),('source_manifest_sha256','d'*64)])
def test_incompatible_or_unverified_prediction_caches_are_rejected(key,value):
    records,info,manifest=provenance_fixture();records[1]['source'][key]=value
    with pytest.raises(ValueError):validate_evaluation_provenance(records,info,manifest,[.01])


def test_calibration_grid_cannot_go_below_cached_inference_floor():
    records,info,manifest=provenance_fixture()
    with pytest.raises(ValueError,match='floor'):validate_evaluation_provenance(records,info,manifest,[.001,.01])
    assert validate_evaluation_provenance(records,info,manifest,[.003,.01])['inference_floor']==.003


def test_training_overlap_is_checked_against_the_actual_manifest():
    records,info,manifest=provenance_fixture();manifest['training_observation_ids']=['obs_b']
    with pytest.raises(ValueError,match='trained'):validate_evaluation_provenance(records,info,manifest,[.01])


def test_eval_images_can_be_built_from_fresh_per_annotator_preparation(tmp_path):
    image=tmp_path/'pass_prefix.jpg';image.write_bytes(b'original jpeg bytes')
    rows=[{'split':'calibration','observation_id':'obs','image_path':str(image),'sample_id':s} for s in ['a','b']]
    write_jsonl(tmp_path/'samples.jsonl',rows)
    directory=ensure_eval_images(tmp_path)
    files=list(directory.iterdir())
    assert len(files)==1 and files[0].stem=='obs' and files[0].read_bytes()==image.read_bytes()
