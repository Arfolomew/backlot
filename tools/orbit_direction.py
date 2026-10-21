#!/usr/bin/env python3
"""Usage: python3 tools/orbit_direction.py clip.mp4 [...]  (needs opencv-python-headless + numpy)
Used Oct 20 2026 to prove which way the Camera Motion LoRA orbits; see moves.json orbit-left/right."""
"""Orbit direction from background drift (background slides the way the camera travels).
Dense optical flow, median dx over the top and bottom bands (away from the subject), summed."""
import sys, cv2, numpy as np
def score(path):
    cap=cv2.VideoCapture(path); prev=None; per=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        g=cv2.cvtColor(cv2.resize(f,(432,240)),cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow=cv2.calcOpticalFlowFarneback(prev,g,None,0.5,3,21,3,5,1.2,0)
            h=g.shape[0]; band=np.concatenate([flow[:int(h*.28),:,0].ravel(), flow[int(h*.72):,:,0].ravel()])
            per.append(float(np.median(band)))
        prev=g
    per=np.array(per); return per.sum(), float(np.median(per)), float((per>0).mean())
for p in sys.argv[1:]:
    s,m,fr=score(p); print(f"{p.split('/')[-1]:22s} sum_dx={s:7.1f}px median={m:6.2f} frames_right={fr:4.0%} -> {'RIGHT' if s>0 else 'LEFT'}")
