from m3_filament.joint import derive_metrics, choose_joint_candidate


def row(tp=10,fp=5,fn=10,pq=.3,sq=.6,rq=.5,small_matched=2):
    return dict(tp=tp,fp=fp,fn=fn,n_gt=tp+fn,n_pred_across_annotators=tp+fp,
                pq=pq,sq=sq,rq=rq,small_gt=10,small_matched=small_matched)


def test_joint_rejects_recall_only_and_precision_only_improvements():
    old=row();recall=row(tp=14,fp=12,fn=6,pq=.35,rq=.6,small_matched=4)
    balanced=row(tp=15,fp=4,fn=5,pq=.45,sq=.65,rq=.7,small_matched=5)
    choice=choose_joint_candidate([recall|{'name':'recall'},row(fp=1,fn=14)|{'name':'precision'},balanced|{'name':'joint'}],[old,recall])
    assert choice['joint_gate_passed'] and choice['candidate']['name']=='joint'


def test_no_domination_is_explicit_failure_even_if_pq_improves():
    ref=row()
    choice=choose_joint_candidate([row(fp=20,pq=.5)|{'name':'higher_pq'}],[ref])
    assert not choice['joint_gate_passed']
    assert 'fp' in choice['violations'] and 'precision' in choice['violations']


def test_metrics_use_unique_object_counts_not_all_pairs_tp():
    r=row(tp=12,fp=2,fn=2);r.update(n_gt=10,n_pred_across_annotators=10)
    m=derive_metrics(r)
    assert m['precision']==.8 and m['recall']==.8 and m['small_recall']==.2
