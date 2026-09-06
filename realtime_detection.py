"""
realtime_detection.py
Modul deteksi real-time untuk SoleGuard menggunakan streamlit-webrtc.
Disesuaikan dengan struktur app.py yang sudah ada (model, class_names,
confidence, iou_thresh, skip_frame, selected_classes dari sidebar).

Cara pakai di app.py (tidak mengubah apa pun yang sudah ada):

    from realtime_detection import render_realtime_tab

    tab_detect, tab_stats, tab_history, tab_realtime = st.tabs([
        "🎯  Deteksi", "📊  Statistik", "🕒  Riwayat", "📡  Real-Time"
    ])

    with tab_realtime:
        render_realtime_tab(model, class_names, confidence, iou_thresh,
                             selected_classes, skip_frame, model_ok)
"""

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# -----------------------------------------------------------------------------
# STUN + TURN. TURN dari OpenRelay dipakai sebagai fallback saat koneksi P2P
# gagal (umum di jaringan seluler/WiFi publik saat pakai kamera HP).
# Untuk sidang, sebaiknya ganti dengan akun TURN gratis pribadi dari
# https://www.metered.ca/tools/openrelay/ agar tidak bergantung kredensial publik.
# -----------------------------------------------------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)

# Lebar frame yang dikirim ke model (lebih kecil = lebih cepat di CPU tanpa GPU)
INFERENCE_WIDTH = 480


class YOLOVideoProcessor(VideoProcessorBase):
    """Memproses tiap frame webcam/HP dengan model YOLOv26n yang sama dengan tab Deteksi."""

    def __init__(self, model, confidence, iou_thresh, selected_classes, skip_frame):
        self.model = model
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.selected_classes = selected_classes
        self.skip_frame = max(skip_frame, 1)
        self._frame_counter = 0
        self._last_annotated = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_counter += 1

        # Skip frame — pakai slider "Proses setiap N frame" yang sama dengan tab Deteksi
        if self._frame_counter % self.skip_frame != 0 and self._last_annotated is not None:
            return av.VideoFrame.from_ndarray(self._last_annotated, format="bgr24")

        # Resize menjaga aspect ratio, biar tidak gepeng
        h, w = img.shape[:2]
        new_w = INFERENCE_WIDTH
        new_h = int(h * (new_w / w))
        resized = cv2.resize(img, (new_w, new_h))

        if not self.selected_classes:
            # Tidak ada kelas dipilih -> tampilkan frame polos saja
            self._last_annotated = resized
            return av.VideoFrame.from_ndarray(resized, format="bgr24")

        results = self.model(
            resized,
            conf=self.confidence,
            iou=self.iou_thresh,
            classes=self.selected_classes,
            verbose=False,
        )
        annotated = results[0].plot(line_width=2)  # sama seperti di tab Deteksi

        self._last_annotated = annotated
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def render_realtime_tab(model, class_names, confidence, iou_thresh,
                         selected_classes, skip_frame, model_ok):
    """Dipanggil dari dalam tab_realtime di app.py."""

    st.markdown('<div class="card"><p class="card-label">📡 Deteksi Real-Time</p>',
                unsafe_allow_html=True)

    if not model_ok:
        st.markdown("""
        <div style='text-align:center;padding:3rem 1rem;color:#2a2a4a;'>
            <div style='font-size:2.5rem;'>⚠️</div>
            <p style='font-size:0.8rem;margin:0.5rem 0 0;'>Model belum siap.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.caption(
        "Confidence, IoU, dan filter kelas mengikuti pengaturan di sidebar — "
        "sama seperti tab Deteksi. Kamera belakang otomatis dipakai di HP."
    )

    camera_source = st.radio(
        "Sumber kamera", ["Laptop / Webcam", "HP (kamera belakang)"],
        horizontal=True,
    )
    facing_mode = "environment" if camera_source.startswith("HP") else "user"

    webrtc_streamer(
        key="soleguard-realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"facingMode": facing_mode}, "audio": False},
        video_processor_factory=lambda: YOLOVideoProcessor(
            model=model,
            confidence=confidence,
            iou_thresh=iou_thresh,
            selected_classes=selected_classes,
            skip_frame=skip_frame,
        ),
        async_processing=True,
    )

    st.markdown("""
    <div style='background:#0a0a18;border:1px solid #1a1a2e;border-radius:10px;
    padding:.9rem 1rem;margin-top:.8rem;font-size:0.78rem;color:#aaa;line-height:1.7;'>
        ℹ️ Akses kamera hanya diizinkan browser lewat <b style='color:#d4d4e8;'>HTTPS</b>
        (Streamlit Community Cloud) atau <b style='color:#d4d4e8;'>localhost</b>.
        Akses lewat IP lokal biasa (http://) akan ditolak browser, terutama di HP.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
