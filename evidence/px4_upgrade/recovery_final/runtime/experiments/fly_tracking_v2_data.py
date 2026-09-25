"""Predeclared, appearance-separated data for the second tracking experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

DT, SIDE = .02, 391
CELL_TYPES = ["L1", "L2", "L3", "Mi1", "Tm1", "Tm2", "Mi4", "Mi9",
              "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
KINDS = ["stationary", "horizontal", "vertical", "diagonal", "stop_restart", "reversal"]
REAL_FIRST, REAL_LAST = 315, 360
REAL_BOX = [1128., 431., 1307., 944.]


def specs():
    result = []
    for split, count, seed_start in [("train",24,5100),("validation",6,6100),("test",6,8100)]:
        for i in range(count):
            kind = KINDS[i%6]
            repetition=i//6
            speed = [20.,36.,36.,20.][repetition] if split=="train" else [0.,26.,34.,30.,38.,42.][i]
            sign_x=1 if repetition<2 else -1
            sign_y=1 if repetition%2==0 else -1
            if split!="train":
                sign_x=-1 if i%2 else 1
                sign_y=-1 if i in (1,2,5) else 1
            if split=="test":
                kind = ["stationary","diagonal","stop_restart","reversal","fast","dim"][i]
                speed = [0.,32.,34.,38.,54.,26.][i]
            result.append({"name":f"{split}_{i:02d}_{kind}","split":split,"seed":seed_start+i,
                "kind":kind,"speed":speed,"size":[80,96,112][(i+repetition)%3],"duration_s":3.,
                "sign":sign_x,"sign_y":sign_y,"stationary_prefix_s":.6,
                "low_contrast":kind=="dim"})
    return result


def generate(spec):
    rng = np.random.default_rng(spec["seed"])
    times = np.arange(151,dtype=np.float64)*DT
    v = np.zeros((len(times),2),dtype=float)
    moving = times>=.6-1e-10
    speed, direction, kind = spec["speed"], spec["sign"], spec["kind"]
    direction_y=spec["sign_y"]
    if kind=="horizontal":v[moving]=[direction*speed,0]
    elif kind=="vertical":v[moving]=[0,direction_y*speed]
    elif kind in ("diagonal","dim"):v[moving]=[direction*.8*speed,direction_y*.6*speed]
    elif kind=="stop_restart":
        v[moving & (times<1.4-1e-10)]=[direction*speed,0]
        v[times>=2.-1e-10]=[-direction*.6*speed,direction_y*.8*speed]
    elif kind=="reversal":
        v[moving & (times<1.8-1e-10)]=[direction*speed,0]
        v[times>=1.8-1e-10]=[-direction*speed,0]
    elif kind=="fast":v[moving]=[speed,0]
    elif kind!="stationary":raise ValueError(kind)
    offsets=np.vstack([np.zeros(2),np.cumsum(v[:-1]*DT,axis=0)])
    center=np.array([195.,195.])-(offsets.min(0)+offsets.max(0))/2+rng.uniform(-18,18,2)
    size=spec["size"]
    boxes=np.r_[center-size/2,center+size/2]+offsets[:,[0,1,0,1]]
    if np.min(boxes)<20 or np.max(boxes)>371:raise ValueError("Stimulus exceeds fixed geometry")
    level=float(rng.uniform(.25,.65));contrast=.10 if spec["low_contrast"] else float(rng.uniform(.35,.7))
    background=cv2.resize(rng.uniform(level-.10,level+.10,(24,24)).astype(np.float32),(SIDE,SIDE))
    texture=cv2.resize(rng.uniform(level-contrast/2,level+contrast/2,(10+(spec["seed"]%5),)*2).astype(np.float32),(size,size))
    texture=np.clip(texture,0,1)
    patch=np.zeros((SIDE,SIDE),np.float32);mask=patch.copy()
    patch[:size,:size],mask[:size,:size]=texture,1.
    frames=[]
    for box in boxes:
        transform=np.array([[1,0,box[0]],[0,1,box[1]]],np.float32)
        moved=cv2.warpAffine(patch,transform,(SIDE,SIDE))
        alpha=cv2.warpAffine(mask,transform,(SIDE,SIDE))
        frames.append(background*(1-alpha)+moved)
    # Save and infer the same quantized image stream, with an explicit unit range.
    frames=np.round(np.clip(frames,0,1)*255).astype(np.uint8)
    return {"spec":spec,"name":spec["name"],"frames":frames.astype(np.float32)/255.,
            "times":times,"truth_boxes":boxes,"velocities":v,"kind":"synthetic"}


def real_clip(project):
    project=Path(project);source=project/"inputs/sabana_grande/source.webm"
    provenance=json.loads((source.parent/"provenance.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=provenance["sha256"]:raise ValueError("Video hash mismatch")
    pts=json.loads((source.parent/"timestamps.json").read_text())["frames"]
    times=np.array([float(f["best_effort_timestamp_time"]) for f in pts[REAL_FIRST:REAL_LAST+1]])
    cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,REAL_FIRST)
    frames,color=[],[]
    for i in range(REAL_FIRST,REAL_LAST+1):
        ok,f=cap.read()
        if not ok:raise ValueError(f"Cannot decode frame{i}")
        cropped=cv2.resize(f[:,420:1500],(SIDE,SIDE),interpolation=cv2.INTER_AREA)
        color.append(cropped);frames.append(cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.)
    cap.release()
    return {"name":"real_new_target","kind":"real_video","frames":np.asarray(frames),
            "color":np.asarray(color),"times":times,"initial_box":(np.array(REAL_BOX)-[420,0,420,0])*391/1080,
            "source_sequences":np.arange(REAL_FIRST,REAL_LAST+1),"provenance":provenance}
