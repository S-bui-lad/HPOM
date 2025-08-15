import yaml
import numpy as np
from pathlib import Path


def load_cfg(cfg_path="configs/pipeline.yaml"):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_relations(rel_path="configs/relations.yaml"):
    if not Path(rel_path).exists():
        return {}
    import yaml
    with open(rel_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def geom_prior(box_s, box_o, relation: str):
    # simple priors to boost plausible geometric relations
    x1s,y1s,x2s,y2s = box_s; x1o,y1o,x2o,y2o = box_o
    cxs, cys = (x1s+x2s)/2.0, (y1s+y2s)/2.0
    cxo, cyo = (x1o+x2o)/2.0, (y1o+y2o)/2.0
    if relation in ["on", "on top of", "trên"]:
        return 1.0 if cys < cyo else -0.2
    if relation in ["left of", "bên trái", "trái"]:
        return 1.0 if cxs < cxo else -0.2
    if relation in ["right of", "bên phải", "phải"]:
        return 1.0 if cxs > cxo else -0.2
    if relation in ["holding", "cầm"]:
        # closer better
        dist = np.hypot(cxs-cxo, cys-cyo)
        return 1.0 / (1.0 + dist/100.0)
    return 0.0


def match_triplets(parsed_query, triplets, s_sub, s_obj, yolo_scores, boxes, cfg_path="configs/pipeline.yaml"):
    cfg = load_cfg(cfg_path)
    w = cfg.get('match', {
        'alpha': 0.45, 'beta': 0.20, 'gamma': 0.20, 'delta': 0.10, 'eps': 0.05, 'final_thresh': 0.42
    })
    rel_map = load_relations(cfg.get('relations_path','configs/relations.yaml'))

    # normalized relation expected from parsed_query
    target_rel = parsed_query.get('relation', '').lower()

    best = None
    for t in triplets:
        s_idx = t['sub_idx']; o_idx = t['obj_idx']
        if s_idx >= len(s_sub) or o_idx >= len(s_obj):
            # skip if indices misalign
            continue
        s_reltr = float(t.get('rel_score', 0.0))
        s_clip_sub = float(s_sub[s_idx])
        s_clip_obj = float(s_obj[o_idx])
        s_yolo = float((yolo_scores[s_idx] + yolo_scores[o_idx]) / 2.0)
        s_geom = geom_prior(boxes[s_idx], boxes[o_idx], target_rel)

        score = (w['alpha'] * s_reltr
                 + w['beta'] * s_clip_sub
                 + w['gamma'] * s_clip_obj
                 + w['delta'] * s_yolo
                 + w['eps'] * s_geom)

        # optional: relation label matching using synonyms in relations.yaml
        # If the triplet label exactly equals target_rel or target_rel maps to triplet label synonyms,
        # we can boost it; if not matching and triplet label is not 'unknown', penalize slightly.
        trip_rel_label = str(t.get('rel_label', '')).lower()
        if trip_rel_label != 'unknown' and target_rel and trip_rel_label != target_rel:
            # check synonyms
            mapped = False
            for k,v in rel_map.items():
                if trip_rel_label == k:
                    # v contains synonyms
                    if isinstance(v, list) and target_rel in [x.lower() for x in v]:
                        mapped = True
                        break
            if not mapped:
                score *= 0.9

        # attribute filters placeholder (not implemented): assume pass
        passed_attrs = True

        if passed_attrs:
            if best is None or score > best['score']:
                best = {
                    'score': score,
                    'sub_idx': s_idx,
                    'obj_idx': o_idx,
                    'rel_label': trip_rel_label
                }

    # final threshold check
    if best and best['score'] >= w.get('final_thresh', 0.0):
        return best
    return None


def score_triplet(triplet, s_sub, s_obj, yolo_scores, boxes, target_rel, weights):
    """Score a single triplet based on various factors
    
    Args:
        triplet: Triplet dictionary with sub_idx, obj_idx, rel_score, rel_label
        s_sub: Subject similarity scores tensor
        s_obj: Object similarity scores tensor  
        yolo_scores: YOLO confidence scores
        boxes: Bounding boxes
        target_rel: Target relation string
        weights: Weight dictionary for scoring components
        
    Returns:
        tuple: (score, sub_idx, obj_idx, rel_label)
    """
    s_idx = triplet['sub_idx']
    o_idx = triplet['obj_idx']
    
    if s_idx >= len(s_sub) or o_idx >= len(s_obj):
        return -1.0, s_idx, o_idx, triplet.get('rel_label', 'unknown')
    
    s_reltr = float(triplet.get('rel_score', 0.0))
    s_clip_sub = float(s_sub[s_idx])
    s_clip_obj = float(s_obj[o_idx])
    s_yolo = float((yolo_scores[s_idx] + yolo_scores[o_idx]) / 2.0)
    s_geom = geom_prior(boxes[s_idx], boxes[o_idx], target_rel)
    
    score = (weights['alpha'] * s_reltr
             + weights['beta'] * s_clip_sub
             + weights['gamma'] * s_clip_obj
             + weights['delta'] * s_yolo
             + weights['eps'] * s_geom)
    
    # Relation label matching using synonyms
    trip_rel_label = str(triplet.get('rel_label', '')).lower()
    if trip_rel_label != 'unknown' and target_rel and trip_rel_label != target_rel:
        # Check synonyms
        rel_map = load_relations("configs/relations.yaml")
        mapped = False
        for k, v in rel_map.items():
            if trip_rel_label == k:
                if isinstance(v, list) and target_rel in [x.lower() for x in v]:
                    mapped = True
                    break
        if not mapped:
            score *= 0.9
    
    return score, s_idx, o_idx, trip_rel_label
