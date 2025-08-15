import yaml
import numpy as np

try:
    from models.reltr_wrap import RelTRWrapper
    _has_reltr = True
except Exception:
    _has_reltr = False


def load_cfg(cfg_path="configs/pipeline.yaml"):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def geom_relation_score(box_a, box_b):
    # rough geometric cue: distance & vertical relation
    (x1a, y1a, x2a, y2a) = box_a
    (x1b, y1b, x2b, y2b) = box_b
    cxa, cya = (x1a + x2a) / 2.0, (y1a + y2a) / 2.0
    cxb, cyb = (x1b + x2b) / 2.0, (y1b + y2b) / 2.0
    dist = np.hypot(cxa - cxb, cya - cyb)
    # normalize by diagonal approx (avoid zero)
    diag = max(np.hypot(x2a - x1a, y2a - y1a), 1.0)
    score = 1.0 / (1.0 + (dist / (50.0 + diag)))
    return float(score)


def relate_objects(image, boxes, cfg_path="configs/pipeline.yaml"):
    cfg = load_cfg(cfg_path)
    if _has_reltr and cfg.get('reltr', {}).get('weights'):
        try:
            reltr_cfg = cfg.get('reltr', {})
            wrapper = RelTRWrapper(cfg_path=reltr_cfg.get('cfg'), weights=reltr_cfg.get('weights'))
            # wrapper.predict should accept image + boxes or just image depending on your RelTR implementation
            triplets = wrapper.predict(image, boxes)
            # expected: list of dicts with keys sub_idx,obj_idx,rel_label,rel_score
            return triplets
        except Exception as e:
            print("RelTR failed to run, falling back to heuristic relate. Error:", e)

    # fallback: enumerate pairs and give a weak geometric score
    N = len(boxes)
    triplets = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            triplets.append({
                'sub_idx': i,
                'obj_idx': j,
                'rel_label': 'unknown',
                'rel_score': geom_relation_score(boxes[i], boxes[j])
            })
    # sort descending by rel_score (helpful downstream)
    triplets = sorted(triplets, key=lambda x: -x['rel_score'])
    return triplets