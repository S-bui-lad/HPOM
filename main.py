import yaml, cv2
import logging
from PIL import Image
import torch, numpy as np
from functools import lru_cache
from typing import Tuple, Dict, List

from models.llm_parser import heuristic_parse
from models.yolo_wrap import YOLOWrapper
from models.region_wrap import RegionCLIPWrapper
from models.reltr_wrap import RelTRWrapper
from pipeline.match import match_triplets, score_triplet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_models(cfg_path: str) -> Tuple[YOLOWrapper, RegionCLIPWrapper, RelTRWrapper]:
    """Load and cache models to avoid reloading"""
    cfg = yaml.safe_load(open(cfg_path, "r"))
    yolo = YOLOWrapper(**cfg["yolo"])
    rc = RegionCLIPWrapper(
        model_name=cfg["regionclip"]["model_name"],
        checkpoint_path=cfg["regionclip"]["weights"]
    )
    reltr = RelTRWrapper(cfg["reltr"]["weights"])
    return yolo, rc, reltr, cfg

def crop_boxes(image, boxes):
    """Crop image regions based on bounding boxes"""
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)
    return crops

def draw(img, box, color, text=None):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    if text:
        cv2.putText(img, text, (x1,max(y1-5,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def run(image_path: str, query: str, cfg_path: str = "configs/pipeline.yaml") -> None:
    try:
        # Load models
        yolo, rc, reltr, cfg = load_models(cfg_path)
        
        # Parse query
        q = heuristic_parse(query)
        subj_text = q["subject"]["class"] + (" " + " ".join(q["subject"]["attributes"]) if q["subject"]["attributes"] else "")
        obj_text = q["object"]["class"] + (" " + " ".join(q["object"]["attributes"]) if q["object"]["attributes"] else "")
        logger.info(f"Parsed query - Subject: {subj_text}, Object: {obj_text}, Relation: {q['relation']}")

        # YOLO detection
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img_cv = img_cv[:, :, ::-1]
        
        try:
            boxes, yscores, ylabels, names = yolo.detect(img_cv)
            logger.info(f"YOLO found {len(boxes)} objects")
            if len(boxes) == 0:
                raise ValueError("No objects detected in image")
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            raise

        # RegionCLIP processing
        try:
            img_pil = Image.fromarray(img_cv)
            with torch.cuda.amp.autocast():  # Mixed precision for faster inference
                reg_feats = rc.encode_regions(img_pil, boxes)
                sub_emb = rc.encode_text(subj_text)
                obj_emb = rc.encode_text(obj_text)
                s_sub = rc.similarity(reg_feats, sub_emb).cpu().numpy()
                s_obj = rc.similarity(reg_feats, obj_emb).cpu().numpy()
        except Exception as e:
            logger.error(f"RegionCLIP processing failed: {e}")
            raise

        # Top-K selection
        topk = min(cfg["regionclip"]["topk_regions"], len(boxes))
        sel_idx = np.argsort(-(np.maximum(s_sub, s_obj) + yscores*0.2))[:topk]
        boxes_k = boxes[sel_idx]
        yscores_k = yscores[sel_idx]
        s_sub_k = s_sub[sel_idx]
        s_obj_k = s_obj[sel_idx]
        logger.debug(f"Selected top-{topk} regions")

        # RelTR processing
        try:
            img_t = torch.from_numpy(img_cv.transpose(2,0,1)).float()/255.0
            img_t = img_t.unsqueeze(0).cuda()
            triplets = reltr.predict_relations(img_t, boxes_k)
            logger.info(f"RelTR found {len(triplets)} relations")
        except Exception as e:
            logger.error(f"RelTR processing failed: {e}")
            raise

        # Matching and scoring
        weights = cfg["match"]
        best = None
        for t in triplets:
            sc, s_idx, o_idx, rel_label = score_triplet(
                t, torch.tensor(s_sub_k), torch.tensor(s_obj_k),
                yscores_k, boxes_k, q["relation"], weights
            )
            if best is None or sc > best[0]:
                best = (sc, s_idx, o_idx, rel_label)

        # Visualization
        out = img_cv[:, :, ::-1].copy()
        if best and best[0] > weights["final_thresh"]:
            logger.info(f"Found match with score {best[0]:.3f}")
            draw(out, boxes_k[best[1]], (0,255,0), f"{q['subject']['class']}")
            draw(out, boxes_k[best[2]], (255,0,0), f"{q['object']['class']}")
            mid = (boxes_k[best[1]] + boxes_k[best[2]])/2
            draw(out, mid, (0,255,255), q["relation"])
        else:
            logger.info("No matching relation found")
            cv2.putText(out, "Khong tim thay cap doi thoa man", (20,40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.imwrite("output.jpg", out)
        logger.info("Saved: output.jpg")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--query", required=True, help="Natural language query")
    ap.add_argument("--cfg", default="configs/pipeline.yaml", help="Path to config file")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = ap.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    run(args.image, args.query, args.cfg)
