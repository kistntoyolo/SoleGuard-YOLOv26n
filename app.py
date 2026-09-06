import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import gc
import time
import shutil
import numpy as np
import pandas as pd
import io
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from realtime_detection import render_realtime_tab

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shoe Defect Detection",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="collapsed"   # collapsed = lebih nyaman di HP
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: #080810;
    color: #d4d4e8;
}
.stApp { background: #080810; }

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { color: #6c63ff !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid #1a1a2e;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d1a; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 99px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #12102a 100%);
    border: 1px solid #1a1a2e;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #6c63ff22 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #6c63ff;
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(2.2rem, 6vw, 3.5rem);
    letter-spacing: 2px;
    color: #fff;
    line-height: 1;
    margin: 0 0 0.5rem 0;
}
.hero-title span { color: #6c63ff; }
.hero-desc {
    font-size: 0.82rem;
    color: #555578;
    max-width: 480px;
}

/* ── Model status chip ── */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.chip-green { background: #0b2318; border: 1px solid #1a4030; color: #3dd68c; }
.chip-red   { background: #230b0b; border: 1px solid #401a1a; color: #f87171; }
.chip-dot   { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

/* ── Cards ── */
.card {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 16px;
    padding: 1.4rem;
    height: 100%;
    transition: border-color .2s;
}
.card:hover { border-color: #6c63ff33; }
.card-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #3a3a5a;
    margin-bottom: 0.8rem;
}

/* ── Metric card ── */
.mcard {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.mcard-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: #fff;
    line-height: 1;
}
.mcard-label { font-size: 0.68rem; color: #3a3a5a;
    text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }
.mcard-sub { font-size: 0.72rem; color: #6c63ff; margin-top: 2px; }

/* ── Stat bar ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.9rem;
}
.stat-name { font-size: 0.8rem; display: flex; align-items: center; gap: 7px; }
.stat-count { font-size: 0.78rem; color: #aaa; }
.stat-bar-bg {
    height: 3px; background: #1a1a2e; border-radius: 99px;
    margin-top: 4px; margin-bottom: 0.9rem;
}
.stat-bar-fill { height: 3px; border-radius: 99px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #1a1a2e;
    margin: 1.2rem 0;
}

/* ── Status badge big ── */
.status-ok {
    background: #0b2318; color: #3dd68c;
    border: 1px solid #1a4030;
    padding: 6px 16px; border-radius: 99px;
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 1px;
}
.status-defect {
    background: #230b0b; color: #f87171;
    border: 1px solid #401a1a;
    padding: 6px 16px; border-radius: 99px;
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 1px;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: #6c63ff !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: all .2s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: #7c74ff !important;
    box-shadow: 0 0 24px #6c63ff55 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: #1e1e2e !important;
    color: #3a3a5a !important;
}
[data-testid="stFileUploader"] {
    background: #0d0d1a;
    border: 1px dashed #2a2a4a;
    border-radius: 12px;
}
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stProgress > div > div {
    background: linear-gradient(90deg, #6c63ff, #a78bfa) !important;
    border-radius: 99px !important;
}
[data-testid="stDataFrame"] {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 8px;
    color: #555578;
    font-size: 0.78rem;
    padding: 6px 14px;
}
.stTabs [aria-selected="true"] {
    background: #6c63ff !important;
    border-color: #6c63ff !important;
    color: #fff !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }
div[data-testid="metric-container"] {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    padding: 0.8rem;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero { padding: 1.4rem 1.2rem; }
    .hero-title { font-size: 2rem; }
    .mcard-val { font-size: 1.8rem; }
    .card { padding: 1rem; }
}
</style>
""", unsafe_allow_html=True)



# ── Helpers ──────────────────────────────────────────────────────────────────
PALETTE = ["#6c63ff","#f87171","#34d399","#fbbf24","#60a5fa",
           "#f472b6","#a78bfa","#2dd4bf","#fb923c","#e879f9"]

def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

def safe_remove(path):
    """Remove file safely — handles Windows file-lock."""
    if not path or not os.path.exists(path):
        return
    for _ in range(5):
        try:
            os.remove(path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.4)
    # Last resort: move to temp and leave OS to clean up
    try:
        shutil.move(path, tempfile.gettempdir())
    except Exception:
        pass

def get_output_dir():
    """Use a fixed output folder in working directory — avoids temp locking."""
    p = Path("soleguard_output")
    p.mkdir(exist_ok=True)
    return p


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("last_result", None),
    ("processing", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("best.pt")

try:
    model       = load_model()
    class_names = model.names
    model_ok    = True
except Exception as e:
    model       = None
    class_names = {}
    model_ok    = False


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <p style='font-family:Bebas Neue,sans-serif;font-size:1.4rem;
    letter-spacing:2px;color:#fff;margin:0;'>SISTEM DETEKSI DEFECT SEPATU</p>
    <p style='font-size:0.62rem;color:#3a3a5a;text-transform:uppercase;
    letter-spacing:2.5px;margin:0 0 1.2rem;'>Detection System</p>
    """, unsafe_allow_html=True)

    chip = "chip-green" if model_ok else "chip-red"
    dot_label = "Model Ready" if model_ok else "Model Error"
    st.markdown(f'<span class="chip {chip}"><span class="chip-dot"></span>{dot_label}</span>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**⚙️ Detection Settings**")

    confidence = st.slider("Confidence", 0.10, 1.0, 0.50, 0.05)
    iou_thresh = st.slider("IoU Threshold", 0.10, 1.0, 0.45, 0.05)
    skip_frame = st.slider("Proses setiap N frame", 1, 5, 1,
                           help="1 = semua frame, 5 = lebih cepat tapi kurang akurat")

    st.markdown("<br>**🏷️ Filter Kelas**")
    selected_classes = []
    if class_names:
        for idx, name in class_names.items():
            if st.checkbox(f"{name.replace('_',' ').title()}", value=True,
                           key=f"cls_{idx}"):
                selected_classes.append(idx)
    else:
        st.caption("Kelas muncul setelah model dimuat.")

    st.markdown(f"""
    <hr style='border-color:#1a1a2e;margin:1rem 0;'>
    <p style='font-size:0.68rem;color:#3a3a5a;line-height:1.7;'>
    <b style='color:#555578;'>Model</b> YOLOv26n<br>
    <b style='color:#555578;'>Kelas</b> {len(class_names)} defect type<br>
    <b style='color:#555578;'>Framework</b> Ultralytics<br>
    <b style='color:#555578;'>Interface</b> Streamlit
    </p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <p class="hero-tag">Computer Vision · YOLOv26n · Adidas Quality Control</p>
    <h1 class="hero-title">SISTEM DETEKSI SEPATU</span><br>(SOLEGUARD)</h1>
    <p class="hero-desc">
        Sistem deteksi cacat sepatu berbasis deep learning.
        Upload video, analisis otomatis, hasil instan.
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — ONE PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab_detect, tab_stats, tab_history, tab_realtime = st.tabs([
    "🎯  Deteksi",
    "📊  Statistik",
    "🕒  Riwayat",
    "📡  Real-Time"
])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — DETEKSI                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab_detect:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: Upload ─────────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="card"><p class="card-label">📤 Input Video</p>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload video",
            type=["mp4","avi","mov"],
            label_visibility="collapsed"
        )

        if uploaded:
            st.video(uploaded)
            st.caption(
                f"📁 **{uploaded.name}** &nbsp;·&nbsp; "
                f"{uploaded.size/1024/1024:.1f} MB"
            )
        else:
            st.markdown("""
            <div style='text-align:center;padding:3rem 1rem;color:#2a2a4a;'>
                <div style='font-size:3rem;'>👟</div>
                <p style='font-size:0.8rem;margin:0.5rem 0 0;'>
                MP4 &nbsp;·&nbsp; AVI &nbsp;·&nbsp; MOV</p>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: Result / Action ────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="card"><p class="card-label">🎯 Hasil Deteksi</p>',
                    unsafe_allow_html=True)

        if not uploaded:
            st.markdown("""
            <div style='text-align:center;padding:3rem 1rem;color:#2a2a4a;'>
                <div style='font-size:2.5rem;'>📡</div>
                <p style='font-size:0.8rem;margin:0.5rem 0 0;'>
                Upload video untuk memulai</p>
            </div>""", unsafe_allow_html=True)

        else:
            run = st.button("🚀  Mulai Deteksi", disabled=not model_ok)

            if run:
                if not selected_classes:
                    st.warning("Pilih minimal satu kelas di sidebar.")
                else:
                    # ── Tulis ke folder lokal, bukan temp ──
                    out_dir    = get_output_dir()
                    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
                    input_path = str(out_dir / f"input_{ts}.mp4")
                    output_path= str(out_dir / f"result_{ts}.mp4")

                    with open(input_path, "wb") as f:
                        f.write(uploaded.getvalue())

                    cap      = cv2.VideoCapture(input_path)
                    fps_vid  = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
                    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = round(total_f / fps_vid, 1)

                    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
                    writer     = cv2.VideoWriter(output_path, fourcc, fps_vid, (W, H))

                    # Counters
                    class_counts    = defaultdict(int)
                    frame_defects   = []
                    total_dets      = 0
                    defect_frames   = 0

                    prog      = st.progress(0, text="Menganalisis video...")
                    preview   = st.empty()
                    fidx      = 0
                    t0        = time.time()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if fidx % skip_frame == 0:
                            results  = model(frame, conf=confidence,
                                             iou=iou_thresh,
                                             classes=selected_classes,
                                             verbose=False)
                            annotated = results[0].plot(line_width=2)
                            boxes     = results[0].boxes
                            n         = len(boxes)
                            total_dets  += n
                            if n > 0: defect_frames += 1
                            frame_defects.append(n)
                            for b in boxes:
                                class_counts[int(b.cls[0])] += 1
                        else:
                            annotated = frame
                            frame_defects.append(0)

                        writer.write(annotated)

                        if fidx % 10 == 0:
                            preview.image(
                                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                channels="RGB",  width="stretch"
                            )

                        elapsed = max(time.time()-t0, 0.01)
                        eta     = int((total_f - fidx) / (fidx/elapsed+0.01))
                        prog.progress(
                            min(fidx/max(total_f,1), 1.0),
                            text=f"Frame {fidx}/{total_f}  ·  ETA {eta}s"
                        )
                        fidx += 1

                    # ── Release semua handle SEBELUM hapus file ──
                    cap.release()
                    writer.release()
                    cv2.destroyAllWindows()
                    gc.collect()

                    prog.empty()
                    preview.empty()

                    # Hapus input dengan safe_remove
                    safe_remove(input_path)

                    # Simpan hasil ke session
                    result = {
                        "timestamp"       : datetime.now().strftime("%d %b %Y, %H:%M"),
                        "filename"        : uploaded.name,
                        "duration"        : duration,
                        "total_frames"    : total_f,
                        "defect_frames"   : defect_frames,
                        "total_detections": total_dets,
                        "class_counts"    : dict(class_counts),
                        "frame_defects"   : frame_defects,
                        "output_path"     : output_path,
                    }
                    st.session_state.last_result = result
                    st.session_state.history.append(
                        {k: v for k, v in result.items() if k != "frame_defects"}
                    )
                    st.rerun()

            # ── Tampilkan hasil ──────────────────────────────────────────
            r = st.session_state.last_result
            if r:
                pct_f = round(r["defect_frames"] / max(r["total_frames"],1) * 100, 1)
                status = "DEFECT" if r["total_detections"] > 0 else "OK"
                badge  = "status-defect" if status == "DEFECT" else "status-ok"

                # Status
                st.markdown(
                    f'<span class="{badge}">{status}</span>',
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # 3 metric cards
                mc1, mc2, mc3 = st.columns(3)
                for col, val, label, sub in [
                    (mc1, r["total_detections"], "Total Defect", "deteksi"),
                    (mc2, r["defect_frames"],    "Frame Defect", f"{pct_f}%"),
                    (mc3, r["duration"],         "Durasi (s)",   "video"),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class="mcard">
                            <div class="mcard-val">{val}</div>
                            <div class="mcard-label">{label}</div>
                            <div class="mcard-sub">{sub}</div>
                        </div>""", unsafe_allow_html=True)

                # Per-class breakdown
                if r["class_counts"]:
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    st.markdown(
                        '<p class="card-label" style="margin-top:.5rem;">Breakdown Kelas</p>',
                        unsafe_allow_html=True
                    )
                    total_c = sum(r["class_counts"].values())
                    for i, (cid, cnt) in enumerate(
                        sorted(r["class_counts"].items(),
                               key=lambda x: x[1], reverse=True)
                    ):
                        nm    = class_names.get(int(cid), f"Class {cid}").replace("_"," ").title()
                        pct_c = round(cnt/total_c*100, 1)
                        col   = PALETTE[i % len(PALETTE)]
                        bar_w = int(pct_c)
                        st.markdown(f"""
                        <div class="stat-row">
                            <span class="stat-name">
                                <span class="dot" style="background:{col};"></span>
                                {nm}
                            </span>
                            <span class="stat-count">{cnt}× &nbsp;
                                <span style="color:{col};">{pct_c}%</span>
                            </span>
                        </div>
                        <div class="stat-bar-bg">
                            <div class="stat-bar-fill"
                                style="background:{col};width:{bar_w}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)

                # Download
                if os.path.exists(r["output_path"]):
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    with open(r["output_path"], "rb") as f:
                        st.download_button(
                            "⬇️  Download Video Hasil",
                            data=f,
                            file_name=f"soleguard_{r['filename']}",
                            mime="video/mp4",
                            use_container_width=True
                        )

        st.markdown('</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — STATISTIK                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab_stats:
    r = st.session_state.last_result

    if not r:
        st.markdown("""
        <div style='text-align:center;padding:4rem;color:#2a2a4a;'>
            <div style='font-size:3rem;'>📊</div>
            <p style='font-size:0.85rem;margin:0.5rem 0 0;'>
            Jalankan deteksi dulu untuk melihat statistik.</p>
        </div>""", unsafe_allow_html=True)
    else:
        # ── Row summary metrics ───────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        peak = max(r["frame_defects"]) if r.get("frame_defects") else 0
        avg  = round(sum(r["frame_defects"]) / max(len(r["frame_defects"]),1), 2) \
               if r.get("frame_defects") else 0

        for col, val, label, sub in [
            (s1, r["total_detections"], "Total Deteksi", "bounding box"),
            (s2, r["defect_frames"],    "Frame Defect",  "frame"),
            (s3, peak,                  "Peak / Frame",  "maks deteksi"),
            (s4, avg,                   "Rata-rata",     "deteksi/frame"),
        ]:
            with col:
                st.markdown(f"""
                <div class="mcard">
                    <div class="mcard-val">{val}</div>
                    <div class="mcard-label">{label}</div>
                    <div class="mcard-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        gc1, gc2 = st.columns(2, gap="large")

        # ── Bar chart ─────────────────────────────────────────────────────
        with gc1:
            st.markdown('<div class="card"><p class="card-label">Distribusi Defect per Kelas</p>',
                        unsafe_allow_html=True)
            if r["class_counts"]:
                labels = [
                    class_names.get(int(k),"?").replace("_"," ").title()
                    for k in r["class_counts"]
                ]
                values = list(r["class_counts"].values())
                df_bar = pd.DataFrame({"Kelas": labels, "Deteksi": values}
                                      ).set_index("Kelas")
                st.bar_chart(df_bar, color="#6c63ff", height=260)

                # Proporsi
                total_c = sum(values)
                st.markdown('<p class="card-label" style="margin-top:1rem;">Proporsi</p>',
                            unsafe_allow_html=True)
                for i, (label, val) in enumerate(zip(labels, values)):
                    pct = round(val/total_c*100,1)
                    col = PALETTE[i % len(PALETTE)]
                    st.markdown(f"""
                    <div class="stat-row">
                        <span class="stat-name">
                            <span class="dot" style="background:{col};"></span>{label}
                        </span>
                        <span class="stat-count">{val} ({pct}%)</span>
                    </div>
                    <div class="stat-bar-bg">
                        <div class="stat-bar-fill"
                            style="background:{col};width:{int(pct)}%;"></div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Tidak ada defect terdeteksi.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Line chart timeline ───────────────────────────────────────────
        with gc2:
            st.markdown('<div class="card"><p class="card-label">Timeline Defect per Frame</p>',
                        unsafe_allow_html=True)
            if r.get("frame_defects"):
                fd   = r["frame_defects"]
                step = max(len(fd)//200, 1)
                df_line = pd.DataFrame({
                    "Frame"  : list(range(0, len(fd), step)),
                    "Defect" : fd[::step]
                }).set_index("Frame")
                st.line_chart(df_line, color="#6c63ff", height=220)

                # Insight otomatis
                peak_frame = fd.index(max(fd)) if fd else 0
                peak_sec   = round(peak_frame / max(r["total_frames"]/r["duration"],1), 1)
                st.markdown(f"""
                <div style='background:#0a0a18;border:1px solid #1a1a2e;
                border-radius:10px;padding:.9rem 1rem;margin-top:.8rem;
                font-size:0.78rem;color:#aaa;line-height:1.7;'>
                    🔍 <b style='color:#d4d4e8;'>Insight Otomatis</b><br>
                    Peak deteksi di frame <b style='color:#6c63ff;'>{peak_frame}</b>
                    (~{peak_sec}s)<br>
                    {round(r['defect_frames']/max(r['total_frames'],1)*100,1)}%
                    frame mengandung defect<br>
                    Rata-rata <b style='color:#6c63ff;'>{avg}</b> deteksi per frame
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
            st.markdown('<hr class="divider">', unsafe_allow_html=True)

        df_summary = pd.DataFrame({
            "Metrik": ["Total Deteksi", "Frame Defect", "Peak per Frame", "Rata-rata per Frame"],
            "Nilai": [r["total_detections"], r["defect_frames"], peak, avg],
        })
        if r["class_counts"]:
            df_kelas = pd.DataFrame({
                "Kelas": [class_names.get(int(k), "?").replace("_", " ").title() for k in r["class_counts"]],
                "Jumlah Deteksi": list(r["class_counts"].values()),
                "Persentase (%)": [round(v / sum(r["class_counts"].values()) * 100, 1) for v in r["class_counts"].values()],
            })
        else:
            df_kelas = pd.DataFrame(columns=["Kelas", "Jumlah Deteksi", "Persentase (%)"])

        buf_stats = io.StringIO()
        buf_stats.write("sep=,\n")
        df_summary.to_csv(buf_stats, index=False)
        buf_stats.write("\n")
        df_kelas.to_csv(buf_stats, index=False)

        st.download_button(
            "⬇️  Export Statistik ke CSV",
            data=buf_stats.getvalue().encode("utf-8-sig"),
            file_name=f"soleguard_statistik_{r['filename']}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — RIWAYAT                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab_history:
    if not st.session_state.history:
        st.markdown("""
        <div style='text-align:center;padding:4rem;color:#2a2a4a;'>
            <div style='font-size:3rem;'>🕒</div>
            <p style='font-size:0.85rem;margin:0.5rem 0 0;'>
            Riwayat akan muncul setelah deteksi pertama.</p>
        </div>""", unsafe_allow_html=True)
    else:
        rows = []
        for h in reversed(st.session_state.history):
            top = "-"
            if h.get("class_counts"):
                best_id = max(h["class_counts"], key=h["class_counts"].get)
                top = class_names.get(int(best_id),"?").replace("_"," ").title()
            rows.append({
                "Waktu"           : h["timestamp"],
                "File"            : h["filename"],
                "Durasi (s)"      : h["duration"],
                "Total Deteksi"   : h["total_detections"],
                "Frame Defect"    : h["defect_frames"],
                "Defect Dominan"  : top,
                "Status"          : "DEFECT" if h["total_detections"] > 0 else "OK",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb = st.columns([1,1])

        with ca:
            buf = io.StringIO()
            buf.write("sep=,\n")            # memaksa Excel pakai koma, apa pun locale-nya
            df.to_csv(buf, index=False)
            st.download_button(
                "⬇️  Export CSV",
                data=buf.getvalue().encode("utf-8-sig"),   # BOM agar karakter UTF-8 terbaca benar di Excel
                file_name="soleguard_riwayat.csv",
                mime="text/csv",
                use_container_width=True
            )

        with cb:
            if st.button("🗑️  Hapus Riwayat", use_container_width=True):
                st.session_state.history = []
                st.session_state.last_result = None
                st.rerun()
                
with tab_realtime:
    render_realtime_tab(model, class_names, confidence, iou_thresh,
                         selected_classes, skip_frame, model_ok)