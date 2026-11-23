"""
box_editor.py -- a small Streamlit component for editing boxes on an image.

    from box_editor import box_editor
    out = box_editor(image, boxes, ...)

You can click a box to select it, drag it to move it, drag a handle to resize it, drag on empty space to draw a
new box, and press Delete (or the button) to remove the selected box. Arrow keys nudge the selected box by 1 pixel
(5 with Shift). It needs no extra packages: the front end is one HTML file that this module writes next to itself
(folder `_box_editor`) the first time it is used.

Coordinates in and out are in pixels of the ORIGINAL image.
"""
import base64
import hashlib
import io
import os

import streamlit.components.v1 as components
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.join(_HERE, "_box_editor")

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html,body{margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;font-size:13px;background:#fff;color:#111}
  #bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:5px 6px;background:#f3f4f6;border:1px solid #d0d3d9;border-bottom:0}
  #bar button{padding:3px 9px;border:1px solid #9aa0aa;background:#fff;border-radius:4px;cursor:pointer;font-size:13px}
  #bar button:disabled{opacity:.45;cursor:default}
  #info{color:#374151;margin-left:6px}
  #wrap{overflow:auto;position:relative;border:1px solid #d0d3d9;outline:none;background:#e5e7eb}
  canvas{display:block;touch-action:none}
</style></head>
<body>
<div id="bar">
  <button id="del" disabled>Delete box</button>
  <button id="undo" disabled>Undo</button>
  <button id="fit">Fit width</button>
  <span id="info">Click a box to select it. Drag it to move, drag a handle to resize, drag on empty space to draw a new box.</span>
</div>
<div id="wrap" tabindex="0"><canvas id="c"></canvas></div>
<script>
(function(){
  const COLORS = {pending:"#2563eb", flag:"#dc2626", done:"#16a34a", skip:"#9ca3af", new:"#f59e0b"};
  const HANDLE = 5, HIT = 9, MIN = 4;
  const wrap = document.getElementById("wrap"), cv = document.getElementById("c"), ctx = cv.getContext("2d");
  const btnDel = document.getElementById("del"), btnUndo = document.getElementById("undo"), info = document.getElementById("info");
  let st = {token:null, imgToken:null, img:null, natW:1, natH:1, sx:1, boxes:[], statuses:{}, labels:{}, sel:null,
            rev:0, selN:null, nextNew:1, undo:[], drag:null, height:640};

  function post(type, data){ window.parent.postMessage(Object.assign({isStreamlitMessage:true, type:type}, data||{}), "*"); }
  function setHeight(h){ post("streamlit:setFrameHeight", {height:h}); }
  function sendValue(){
    st.rev += 1;
    post("streamlit:setComponentValue", {dataType:"json", value:{
      token:st.token, rev:st.rev, selected:st.sel,
      boxes: st.boxes.map(b => ({id:b.id, x1:Math.round(b.x1*10)/10, y1:Math.round(b.y1*10)/10, x2:Math.round(b.x2*10)/10, y2:Math.round(b.y2*10)/10, isNew:!!b.isNew}))}});
  }
  function clone(a){ return JSON.parse(JSON.stringify(a)); }
  function pushUndo(){ st.undo.push(clone(st.boxes)); if(st.undo.length > 60) st.undo.shift(); btnUndo.disabled = false; }
  function byId(id){ return st.boxes.find(b => b.id === id); }
  function clampBox(b){
    b.x1 = Math.max(0, Math.min(b.x1, st.natW)); b.x2 = Math.max(0, Math.min(b.x2, st.natW));
    b.y1 = Math.max(0, Math.min(b.y1, st.natH)); b.y2 = Math.max(0, Math.min(b.y2, st.natH));
  }
  function normBox(b){
    const x1 = Math.min(b.x1, b.x2), x2 = Math.max(b.x1, b.x2), y1 = Math.min(b.y1, b.y2), y2 = Math.max(b.y1, b.y2);
    b.x1 = x1; b.x2 = x2; b.y1 = y1; b.y2 = y2;
  }
  function colorOf(b){ return b.isNew ? COLORS.new : (COLORS[st.statuses[b.id]] || COLORS.pending); }

  function draw(){
    if(!st.img) return;
    cv.width = st.img.width; cv.height = st.img.height;
    ctx.drawImage(st.img, 0, 0);
    ctx.font = "bold 12px Arial"; ctx.textBaseline = "top";
    for(const b of st.boxes){
      const x = b.x1*st.sx, y = b.y1*st.sx, w = (b.x2-b.x1)*st.sx, h = (b.y2-b.y1)*st.sx;
      const col = colorOf(b), sel = (b.id === st.sel), skip = st.statuses[b.id] === "skip";
      ctx.lineWidth = sel ? 3 : 1.6; ctx.strokeStyle = sel ? "#7c3aed" : col;
      ctx.setLineDash(skip ? [5,3] : []);
      ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
      const tag = (b.num != null ? String(b.num) : "") ;
      if(tag){
        const tw = ctx.measureText(tag).width + 6;
        ctx.fillStyle = sel ? "#7c3aed" : col; ctx.fillRect(x, Math.max(0, y-14), tw, 14);
        ctx.fillStyle = "#fff"; ctx.fillText(tag, x+3, Math.max(0, y-13));
      }
      const lab = st.labels[b.id];
      if(lab && (sel || st.statuses[b.id] === "done")){
        const tw = ctx.measureText(lab).width + 6;
        ctx.fillStyle = "rgba(17,24,39,.85)"; ctx.fillRect(x, y+h+1, tw, 14);
        ctx.fillStyle = "#fff"; ctx.fillText(lab, x+3, y+h+2);
      }
      if(sel){
        ctx.fillStyle = "#fff"; ctx.strokeStyle = "#7c3aed"; ctx.lineWidth = 1.5;
        for(const p of handlePoints(b)){ ctx.fillRect(p.x-HANDLE, p.y-HANDLE, 2*HANDLE, 2*HANDLE); ctx.strokeRect(p.x-HANDLE, p.y-HANDLE, 2*HANDLE, 2*HANDLE); }
      }
    }
    if(st.drag && st.drag.mode === "draw"){
      const d = st.drag, x = Math.min(d.x0,d.x1)*st.sx, y = Math.min(d.y0,d.y1)*st.sx, w = Math.abs(d.x1-d.x0)*st.sx, h = Math.abs(d.y1-d.y0)*st.sx;
      ctx.strokeStyle = COLORS.new; ctx.lineWidth = 2; ctx.setLineDash([6,3]); ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
    }
    const s = byId(st.sel);
    btnDel.disabled = !s;
    info.textContent = s ? ("Selected box " + (s.num != null ? s.num : s.id) + ": " + Math.round(s.x2-s.x1) + " x " + Math.round(s.y2-s.y1) + " px at (" + Math.round(s.x1) + ", " + Math.round(s.y1) + "). Drag to move, handles to resize, Delete to remove.")
                            : "Click a box to select it. Drag it to move, drag a handle to resize, drag on empty space to draw a new box.";
  }
  function handlePoints(b){
    const x1=b.x1*st.sx, y1=b.y1*st.sx, x2=b.x2*st.sx, y2=b.y2*st.sx, mx=(x1+x2)/2, my=(y1+y2)/2;
    return [{k:"nw",x:x1,y:y1},{k:"n",x:mx,y:y1},{k:"ne",x:x2,y:y1},{k:"e",x:x2,y:my},{k:"se",x:x2,y:y2},{k:"s",x:mx,y:y2},{k:"sw",x:x1,y:y2},{k:"w",x:x1,y:my}];
  }
  function hitHandle(b, px, py){
    for(const p of handlePoints(b)) if(Math.abs(px-p.x) <= HIT && Math.abs(py-p.y) <= HIT) return p.k;
    return null;
  }
  function hitBox(px, py){
    // the smallest box under the pointer wins (so a small glyph inside a frame can still be picked)
    let best = null, bestA = 1e18;
    const nx = px/st.sx, ny = py/st.sx;
    for(const b of st.boxes){
      if(nx >= b.x1 && nx <= b.x2 && ny >= b.y1 && ny <= b.y2){
        const a = (b.x2-b.x1)*(b.y2-b.y1); if(a < bestA){ best = b; bestA = a; }
      }
    }
    return best;
  }
  function pos(e){ const r = cv.getBoundingClientRect(); return {x:e.clientX - r.left, y:e.clientY - r.top}; }

  cv.addEventListener("pointerdown", function(e){
    if(!st.img) return;
    wrap.focus(); cv.setPointerCapture(e.pointerId);
    const p = pos(e), s = byId(st.sel);
    if(s){
      const hk = hitHandle(s, p.x, p.y);
      if(hk){ pushUndo(); st.drag = {mode:"resize", k:hk, id:s.id, moved:false}; return; }
    }
    const hb = hitBox(p.x, p.y);
    if(hb){
      st.sel = hb.id; pushUndo();
      st.drag = {mode:"move", id:hb.id, ox:p.x/st.sx - hb.x1, oy:p.y/st.sx - hb.y1, w:hb.x2-hb.x1, h:hb.y2-hb.y1, moved:false};
      draw(); return;
    }
    st.sel = null;
    st.drag = {mode:"draw", x0:p.x/st.sx, y0:p.y/st.sx, x1:p.x/st.sx, y1:p.y/st.sx};
    draw();
  });
  cv.addEventListener("pointermove", function(e){
    const p = pos(e), d = st.drag;
    if(!d){
      const s = byId(st.sel); let cur = "crosshair";
      if(s){ const hk = hitHandle(s, p.x, p.y); if(hk) cur = (hk==="n"||hk==="s") ? "ns-resize" : (hk==="e"||hk==="w") ? "ew-resize" : (hk==="nw"||hk==="se") ? "nwse-resize" : "nesw-resize"; }
      if(cur === "crosshair" && hitBox(p.x, p.y)) cur = "move";
      cv.style.cursor = cur; return;
    }
    const nx = Math.max(0, Math.min(st.natW, p.x/st.sx)), ny = Math.max(0, Math.min(st.natH, p.y/st.sx));
    if(d.mode === "draw"){ d.x1 = nx; d.y1 = ny; }
    else {
      const b = byId(d.id); if(!b) return; d.moved = true;
      if(d.mode === "move"){
        let x1 = nx - d.ox, y1 = ny - d.oy;
        x1 = Math.max(0, Math.min(st.natW - d.w, x1)); y1 = Math.max(0, Math.min(st.natH - d.h, y1));
        b.x1 = x1; b.y1 = y1; b.x2 = x1 + d.w; b.y2 = y1 + d.h;
      } else {
        if(d.k.indexOf("n") >= 0) b.y1 = ny; if(d.k.indexOf("s") >= 0) b.y2 = ny;
        if(d.k.indexOf("w") >= 0) b.x1 = nx; if(d.k.indexOf("e") >= 0) b.x2 = nx;
      }
    }
    draw();
  });
  function finish(e){
    const d = st.drag; if(!d) return; st.drag = null;
    try{ cv.releasePointerCapture(e.pointerId); }catch(_){}
    if(d.mode === "draw"){
      const w = Math.abs(d.x1-d.x0), h = Math.abs(d.y1-d.y0);
      if(w >= MIN && h >= MIN){
        pushUndo();
        const id = "n" + (st.nextNew++);
        st.boxes.push({id:id, x1:Math.min(d.x0,d.x1), y1:Math.min(d.y0,d.y1), x2:Math.max(d.x0,d.x1), y2:Math.max(d.y0,d.y1), isNew:true, num:"+"});
        st.sel = id; sendValue();
      } else { sendValue(); }   // a click on empty space only clears the selection
    } else {
      const b = byId(d.id);
      if(b){
        normBox(b); clampBox(b);
        if(b.x2 - b.x1 < MIN) b.x2 = b.x1 + MIN; if(b.y2 - b.y1 < MIN) b.y2 = b.y1 + MIN;
      }
      if(!d.moved){ st.undo.pop(); btnUndo.disabled = st.undo.length === 0; }
      sendValue();
    }
    draw();
  }
  cv.addEventListener("pointerup", finish);
  cv.addEventListener("pointercancel", finish);

  function deleteSel(){
    const s = byId(st.sel); if(!s) return;
    pushUndo(); st.boxes = st.boxes.filter(b => b.id !== s.id); st.sel = null; sendValue(); draw();
  }
  btnDel.addEventListener("click", deleteSel);
  btnUndo.addEventListener("click", function(){
    if(!st.undo.length) return; st.boxes = st.undo.pop(); btnUndo.disabled = st.undo.length === 0;
    if(!byId(st.sel)) st.sel = null; sendValue(); draw();
  });
  document.getElementById("fit").addEventListener("click", function(){ wrap.scrollLeft = 0; wrap.scrollTop = 0; });
  wrap.addEventListener("keydown", function(e){
    const s = byId(st.sel); if(!s) return;
    if(e.key === "Delete" || e.key === "Backspace"){ e.preventDefault(); deleteSel(); return; }
    const step = e.shiftKey ? 5 : 1; let dx = 0, dy = 0;
    if(e.key === "ArrowLeft") dx = -step; else if(e.key === "ArrowRight") dx = step; else if(e.key === "ArrowUp") dy = -step; else if(e.key === "ArrowDown") dy = step; else return;
    e.preventDefault(); pushUndo();
    if(e.altKey){ s.x2 += dx; s.y2 += dy; } else { s.x1 += dx; s.x2 += dx; s.y1 += dy; s.y2 += dy; }
    normBox(s); clampBox(s); sendValue(); draw();
  });

  function scrollTo(b){
    const cx = (b.x1+b.x2)/2*st.sx, cy = (b.y1+b.y2)/2*st.sx;
    wrap.scrollLeft = Math.max(0, cx - wrap.clientWidth/2); wrap.scrollTop = Math.max(0, cy - wrap.clientHeight/2);
  }

  function onRender(a){
    st.statuses = a.statuses || {}; st.labels = a.labels || {};
    const newToken = (a.token !== st.token);
    if(newToken){
      st.token = a.token; st.rev = 0; st.undo = []; btnUndo.disabled = true; st.drag = null;
      st.boxes = (a.boxes || []).map(b => ({id:b.id, x1:b.x1, y1:b.y1, x2:b.x2, y2:b.y2, num:b.num, isNew:false}));
      st.sel = a.selected || null; st.nextNew = a.next_new || 1;
    } else {
      // keep the geometry the user has edited; refresh numbers for known boxes; add boxes the app created
      const known = new Set(st.boxes.map(b => b.id));
      for(const b of (a.boxes || [])){
        const k = byId(b.id);
        if(k){ k.num = b.num; } else if(!st.removed || !st.removed.has(b.id)){ /* a box that was deleted here stays deleted */ }
      }
    }
    st.removed = null;
    st.height = a.height || 640; wrap.style.height = st.height + "px"; setHeight(st.height + 40);
    st.natW = a.nat_w; st.natH = a.nat_h;
    if(a.sel_n !== st.selN){ st.selN = a.sel_n; if(a.select_id){ const b = byId(a.select_id); if(b){ st.sel = b.id; scrollTo(b); } } }
    if(a.img_token !== st.imgToken){
      st.imgToken = a.img_token; const im = new Image();
      im.onload = function(){ st.img = im; st.sx = im.width / st.natW; draw(); };
      im.src = "data:image/jpeg;base64," + a.img_b64;
    } else { draw(); }
  }
  window.addEventListener("message", function(ev){
    if(ev.data && ev.data.type === "streamlit:render"){ onRender(ev.data.args); }
  });
  post("streamlit:componentReady", {apiVersion:1});
  setHeight(680);
})();
</script></body></html>
"""


def _ensure_frontend():
    os.makedirs(_DIR, exist_ok=True)
    path = os.path.join(_DIR, "index.html")
    want = hashlib.md5(HTML.encode("utf-8")).hexdigest()
    have = None
    if os.path.exists(path):
        with open(path, "rb") as f:
            have = hashlib.md5(f.read()).hexdigest()
    if have != want:
        with open(path, "w", encoding="utf-8") as f:
            f.write(HTML)
    return _DIR


_component = None


def _get_component():
    global _component
    if _component is None:
        _component = components.declare_component("nsibidi_box_editor", path=_ensure_frontend())
    return _component


def _encode(img, width):
    """JPEG, scaled to `width` pixels, as a base64 string."""
    w = max(200, int(width))
    if img.width != w:
        img = img.resize((w, max(1, round(img.height * w / img.width))), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_B64 = {}


def box_editor(image, boxes, statuses=None, labels=None, selected=None, select_request=0, token="t0",
               width=1100, height=640, next_new=1, key=None):
    """Show `image` with editable boxes.

    boxes: list of dicts with id, x1, y1, x2, y2 (original pixels) and optionally num (text shown on the box).
    statuses: {id: 'pending' | 'flag' | 'done' | 'skip'} sets the colour.   labels: {id: text} shown under a box.
    selected: id to select when `select_request` changes (scrolls it into view).
    token: change it to throw away the user's edits and load `boxes` afresh (new plate, new detection run).
    Returns None until the user does something, then a dict:
        {'token', 'rev', 'selected', 'boxes': [{'id','x1','y1','x2','y2','isNew'}, ...]}
    """
    comp = _get_component()
    img_token = f"{token}:{int(width)}"
    if img_token not in _B64:
        if len(_B64) > 4:
            _B64.pop(next(iter(_B64)))
        _B64[img_token] = _encode(image, width)
    return comp(img_b64=_B64[img_token], img_token=img_token, nat_w=image.width, nat_h=image.height,
                boxes=[{"id": b["id"], "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"], "num": b.get("num")}
                       for b in boxes],
                statuses=statuses or {}, labels=labels or {}, selected=None, select_id=selected, sel_n=select_request,
                token=token, height=int(height), next_new=int(next_new), key=key, default=None)
