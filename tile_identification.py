#!/usr/bin/env python3
"""
kingdomino_scan_tiles.py – dominant-HSV matcher with adjustable weights
────────────────────────────────────────────────────────────────────────
Distance  d = wH·|zH_tile − zH_sample| +
              wS·|zS_tile − zS_sample| +
              wV·|zV_tile − zV_sample|

Change COEF_H / COEF_S / COEF_V to tweak importance.
Everything else (walkthrough, large windows, stats overlay) is as before.
"""

import cv2 as cv, numpy as np, json, sys, re, random, math
from pathlib import Path
import collections

# ───────── adjustable weights ─────────
COEF_H, COEF_S, COEF_V = 80/100, 5/100, 15/100   # should sum to 1.0

# ───────── other config ─────────
TILES_DIR, TOP_K = Path("tiles"), 3
SHOW_BANK, SHOW_DEBUG = True, True

CODE2TERR = {"f":"forest","me":"meadow","mi":"mine",
             "w":"water","wa":"wasteland","wh":"wheat"}
TERR2CODE = {v:k for k,v in CODE2TERR.items()}

# ───────── utility: dominant of 1-D channel ─────────
def dominant_val(channel, bins, rng, weights=None, wrap=False):
    hist, edges = np.histogram(channel, bins=bins, range=rng, weights=weights)
    k = int(hist.argmax())
    idx = [(k-1)%bins if wrap else max(k-1,0),
           k,
           (k+1)%bins if wrap else min(k+1,bins-1)]
    centres = 0.5*(edges[:-1]+edges[1:])
    w = hist[idx]
    return float(np.sum(w*centres[idx]) / (w.sum()+1e-9))

# ───────── SV moment-match ─────────
def match_sv(img, tgt_s, tgt_v):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
    s_m, v_m = hsv[:,:,1].mean(), hsv[:,:,2].mean()
    if s_m>1: hsv[:,:,1] *= tgt_s/s_m
    if v_m>1: hsv[:,:,2] *= tgt_v/v_m
    hsv[:,:,1:] = np.clip(hsv[:,:,1:],0,255)
    return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

# ───────── feature extraction ─────────
def feature_domHSV(img_bgr):
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    H,S,V = cv.split(hsv)
    mask  = S>20
    if not mask.any(): mask = np.ones_like(H,dtype=bool)
    h_sel,s_sel,v_sel = H[mask], S[mask], V[mask]

    domH = dominant_val(h_sel, 18, (0,180), weights=s_sel, wrap=True)   # 10°
    domS = dominant_val(s_sel, 16, (0,256))                            # 16-bins
    domV = dominant_val(v_sel, 16, (0,256))
    stdH = float(h_sel.std())

    # hue histogram (for display only)
    hist,_ = np.histogram(h_sel,bins=18,range=(0,180),weights=s_sel)
    hist = (hist/(hist.sum()+1e-9)).astype(np.float32)

    sat_mean, val_mean = float(S.mean()), float(V.mean())
    return np.hstack([domH, domS, domV, stdH, hist, sat_mean, val_mean])

# ───────── sample bank ─────────
Sample = collections.namedtuple(
    "Sample","terrain crowns img name vec zH zS zV sat val")

def parse_name(stem):
    if "_c" in stem:
        code,c = stem.split("_c"); crowns=int(re.match(r"\d+",c).group(0))
    else:
        code = re.match(r"[a-z]+",stem).group(0); crowns=0
    return code,crowns

def build_bank():
    bank=[]
    for p in sorted(TILES_DIR.glob("*.png")):
        code,crowns = parse_name(p.stem.lower())
        terr,img = CODE2TERR.get(code,"unknown"), cv.imread(str(p))
        hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        bank.append(Sample(terr,crowns,img,p.name,None,0,0,0,
                           float(hsv[:,:,1].mean()),
                           float(hsv[:,:,2].mean())))
    if not bank: raise RuntimeError("No PNGs in ./tiles/")

    tgt_S = np.mean([s.sat for s in bank])
    tgt_V = np.mean([s.val for s in bank])

    # fill feature vec & raw dominants
    domH_list, domS_list, domV_list = [], [], []
    for i,s in enumerate(bank):
        vec = feature_domHSV(s.img)
        domH_list.append(vec[0]); domS_list.append(vec[1]); domV_list.append(vec[2])
        bank[i] = s._replace(vec=vec)

    # z-score constants
    muH, sdH = np.mean(domH_list), np.std(domH_list)+1e-9
    muS, sdS = np.mean(domS_list), np.std(domS_list)+1e-9
    muV, sdV = np.mean(domV_list), np.std(domV_list)+1e-9

    for i,s in enumerate(bank):
        bank[i] = s._replace(
            zH=(s.vec[0]-muH)/sdH,
            zS=(s.vec[1]-muS)/sdS,
            zV=(s.vec[2]-muV)/sdV
        )
    return bank, (muH,sdH,muS,sdS,muV,sdV), (tgt_S,tgt_V)

# ───────── matcher ─────────
def classify(tile, bank, z_stats, tgtSV):
    muH,sdH,muS,sdS,muV,sdV = z_stats
    tile_adj = match_sv(tile,*tgtSV)
    vec = feature_domHSV(tile_adj)
    zH = (vec[0]-muH)/sdH
    zS = (vec[1]-muS)/sdS
    zV = (vec[2]-muV)/sdV

    scored=[]
    for s in bank:
        d = (COEF_H*abs(zH-s.zH) +
             COEF_S*abs(zS-s.zS) +
             COEF_V*abs(zV-s.zV))
        scored.append((s,d, abs(zH-s.zH), abs(zS-s.zS), abs(zV-s.zV), vec))
    scored.sort(key=lambda x:x[1])
    return scored[:TOP_K], vec  # top matches + tile vec

# ───────── small overlay helper ─────────
def block(img, lines):
    h = 18*len(lines)+4
    cv.rectangle(img,(0,0),(img.shape[1],h),(0,0,0),-1)
    y=16
    for ln in lines:
        cv.putText(img,ln,(3,y),cv.FONT_HERSHEY_SIMPLEX,
                   0.45,(255,255,255),1,cv.LINE_AA); y+=18
    return img

# ───────── visualisation windows ─────────
def show_bank(bank):
    for i,s in enumerate(bank,1):
        lines=[f"{i}/{len(bank)}  {s.name}",
               f"{s.terrain}  crowns={s.crowns}",
               f"domH={s.vec[0]:.1f}  domS={s.vec[1]:.0f}  domV={s.vec[2]:.0f}",
               f"sat={s.sat:.1f}  val={s.val:.1f}"]
        view=s.img.copy(); block(view,lines)
        cv.imshow("sample",view)
        k=cv.waitKey(0); cv.destroyAllWindows()
        if k==27: break

def show_tile(tile, comps, vec_tile):
    h = tile.shape[0]*2
    R = lambda im: cv.resize(im,(h,h),cv.INTER_AREA)
    board = block(R(tile.copy()),[
        "BOARD TILE",
        f"domH={vec_tile[0]:.1f}  domS={vec_tile[1]:.0f}  domV={vec_tile[2]:.0f}",
        f"stdH={vec_tile[3]:.2f}  hist={np.linalg.norm(vec_tile[4:22]):.2f}",
        f"satM={vec_tile[22]:.1f} valM={vec_tile[23]:.1f}"
    ])
    thumbs=[]
    for s,d,dH,dS,dV,_ in comps:
        t=block(R(s.img.copy()),[
            s.name,
            f"score={d:.2f}",
            f"ΔH={dH:.2f}  ΔS={dS:.2f}  ΔV={dV:.2f}"
        ])
        thumbs.append(t)
    cv.imshow("tile-debug",cv.hconcat([board]+thumbs))
    k=cv.waitKey(0)
    if k==27: cv.destroyAllWindows(); sys.exit(0)
    cv.destroyWindow("tile-debug")

# ───────── board helpers ─────────
def annotate(board,res):
    out=board.copy(); H,W=out.shape[:2]; s=min(H,W)//7
    font=cv.FONT_HERSHEY_SIMPLEX; scale=s/140; thick=max(1,s//70)
    for r,row in enumerate(res):
        for c,cell in enumerate(row):
            y,x=r*s,c*s
            tag=f"{TERR2CODE.get(cell['terrain'],'?')}{cell['crowns']}"
            cv.putText(out,tag,(x+5,y+s-5),font,scale,
                       (0,0,0),thick+2,cv.LINE_AA)
            cv.putText(out,tag,(x+5,y+s-5),font,scale,
                       (255,255,255),thick,cv.LINE_AA)
            cv.rectangle(out,(x,y),(x+s,y+s),(0,0,0),1)
    return out

def crop_tiles(b):
    H,W=b.shape[:2]; s=min(H,W)//7
    return [[b[r*s:(r+1)*s,c*s:(c+1)*s] for c in range(7)] for r in range(7)]

# ───────── pipeline ─────────
def analyse(board):
    bank,z_stats,tgtSV = build_bank()
    if SHOW_BANK: show_bank(bank)
    res=[]
    for r,row in enumerate(crop_tiles(board)):
        line=[]
        for c,tile in enumerate(row):
            top, vec_tile = classify(tile,bank,z_stats,tgtSV)
            best=top[0][0]
            line.append({"terrain":best.terrain,"crowns":best.crowns})
            if SHOW_DEBUG:
                print(f"# Tile ({r},{c}) {TERR2CODE.get(best.terrain,'?')}{best.crowns}")
                print(json.dumps([{
                    "example":s.name,"terr":s.terrain,"crowns":s.crowns,
                    "score":round(d,3),
                    "d_domH":round(dH,3),"d_domS":round(dS,3),"d_domV":round(dV,3)}
                    for s,d,dH,dS,dV,_ in top],indent=2))
                show_tile(tile,top,vec_tile)
        res.append(line)
    return res

# ───────── CLI ─────────
if __name__=="__main__":
    if len(sys.argv)!=2:
        print("usage: python kingdomino_scan_tiles.py board.jpg"); sys.exit(1)
    board=cv.imread(sys.argv[1])
    if board is None: print("cannot read",sys.argv[1]); sys.exit(1)

    summary=analyse(board)
    print("\n=== 7×7 summary ==="); print(json.dumps(summary,indent=2))
    cv.imshow("Kingdomino – annotated", annotate(board,summary))
    cv.waitKey(0); cv.destroyAllWindows()

    if random.random()<0.5:
        print("\nDad-quip: with hue, sat, and val in harmony, every match is a colourful symphony!")
