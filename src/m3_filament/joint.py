"""Joint instance-quality selection: no recall-for-precision trade accepted."""
import math

QUALITY=('pq','sq','rq','precision','recall','f1','small_recall')


def derive_metrics(row):
    result=dict(row)
    for key in ('tp','fp','fn','n_gt','n_pred_across_annotators','small_gt','small_matched'):
        if key not in row or not math.isfinite(row[key]) or row[key]<0:
            raise ValueError('invalid count: '+key)
    n_pred=row['n_pred_across_annotators'];n_gt=row['n_gt']
    if row['fp']>n_pred or row['fn']>n_gt or row['small_matched']>row['small_gt']:
        raise ValueError('unmatched count exceeds object count')
    # Organizer matching counts all IoU>.5 pairs. Precision/recall below use
    # unique matched objects, so overlapping GT cannot inflate these ratios.
    p=(n_pred-row['fp'])/n_pred if n_pred else 0.
    r=(n_gt-row['fn'])/n_gt if n_gt else 0.
    result.update(precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.,
                  small_recall=row['small_matched']/row['small_gt'] if row['small_gt'] else None)
    for key in ('pq','sq','rq'):
        if not math.isfinite(result[key]) or not 0<=result[key]<=1:
            raise ValueError('invalid quality metric: '+key)
    return result


def choose_joint_candidate(rows,references):
    if not rows or not references:
        raise ValueError('measured candidates and fixed references are required')
    refs=[derive_metrics(r) for r in references]
    lower={key:max(r[key] for r in refs if r[key] is not None)
           for key in QUALITY if any(r[key] is not None for r in refs)}
    upper={key:min(r[key] for r in refs) for key in ('fp','fn')}
    assessed=[]
    for raw in rows:
        r=derive_metrics(raw);violations=[];margins=[];strict=False
        for key,bound in lower.items():
            value=r[key]
            if value is None:
                violations.append(key);margins.append(float('-inf'));continue
            delta=value-bound
            if delta < -1e-12:violations.append(key)
            strict |= delta>1e-12
            margins.append(delta/max(bound,.05))
        for key,bound in upper.items():
            delta=bound-r[key]
            if delta < 0:violations.append(key)
            strict |= delta>0
            margins.append(delta/max(bound,1))
        if not strict and not violations:violations.append('no_strict_improvement')
        assessed.append({'candidate':r,'violations':violations,'joint_gate_passed':not violations,
                         'worst_normalized_margin':min(margins)})
    feasible=[a for a in assessed if a['joint_gate_passed']]
    winner=max(feasible,key=lambda a:(a['candidate']['pq'],a['candidate']['f1'],-a['candidate']['fp'])) if feasible else max(assessed,key=lambda a:(a['worst_normalized_margin'],a['candidate']['pq']))
    return winner|{'quality_lower_bounds':lower,'count_upper_bounds':upper,
                   'feasible_candidates':len(feasible),'candidates_measured':len(assessed),
                   'selection_policy':'joint dominance first, PQ second; infeasible winner is diagnostic only'}
