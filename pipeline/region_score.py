import numpy as np
from models.region_wrap import RegionCLIPWrapper


def score_regions(image_pil, boxes, subject_prompt, object_prompt, device='cuda', batch_size=64):
    rc = RegionCLIPWrapper(device=device)

    # encode text prompts
    sub_emb = rc.encode_text(subject_prompt)  # (1,D) torch.Tensor
    obj_emb = rc.encode_text(object_prompt)   # (1,D)

    # encode regions -> (N,D)
    reg_feats = rc.encode_regions(image_pil, boxes)

    # cosine sims
    s_sub = rc.sim(reg_feats, sub_emb).cpu().numpy()
    s_obj = rc.sim(reg_feats, obj_emb).cpu().numpy()

    return s_sub, s_obj