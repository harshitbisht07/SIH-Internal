"""
SAMVAAD — Immediate speech on detection (no click), overlay text persistent

Usage:
- Install dependencies: streamlit, streamlit-webrtc, mediapipe, opencv-python, numpy, sklearn, gTTS (optional), pyttsx3 (optional)
- Run: streamlit run this_file.py
"""

import os
import time
import pickle
import tempfile
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# TTS options
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTS_AVAILABLE = True
except Exception:
    PYTTS_AVAILABLE = False

from sklearn.neighbors import KNeighborsClassifier

# ----------------- CONFIG -----------------
st.set_page_config(page_title="SAMVAAD — Instant Speech", layout="wide")

DATA_DIR = "sign_dataset"
MODEL_PATH = "sign_model.pkl"

import string
LETTERS = list(string.ascii_uppercase) + ["SPACE"]
LABEL_TO_CHAR = lambda lbl: " " if lbl == "SPACE" else lbl
CHAR_TO_LABEL = lambda ch: "SPACE" if ch == " " else ch

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ----------------- session-state init -----------------
if "GLOBAL_CONFIG" not in st.session_state:
    st.session_state.GLOBAL_CONFIG = {"collect_letter": None, "is_collecting": False, "model_loaded": False, "mode": "live"}
GLOBAL_CONFIG = st.session_state.GLOBAL_CONFIG

if "predict_text" not in st.session_state:
    st.session_state.predict_text = ""
if "latest_spoken_char" not in st.session_state:
    st.session_state.latest_spoken_char = None
if "text_updated" not in st.session_state:
    st.session_state.text_updated = False
if "clear_requested" not in st.session_state:
    st.session_state.clear_requested = False
if "_force_model_reload" not in st.session_state:
    st.session_state["_force_model_reload"] = False
if "auto_speak_mode" not in st.session_state:
    st.session_state.auto_speak_mode = "Speak each letter"
# store latest audio bytes produced by processor (gTTS)
if "_latest_audio_bytes" not in st.session_state:
    st.session_state["_latest_audio_bytes"] = None

# ----------------- helpers -----------------
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def extract_features_from_landmarks(hand_landmarks) -> np.ndarray:
    xs, ys = [], []
    for lm in hand_landmarks.landmark:
        xs.append(lm.x)
        ys.append(lm.y)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    # normalize
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-6)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-6)
    feat = np.empty(42, dtype=np.float32)
    feat[0::2] = xs
    feat[1::2] = ys
    return feat

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def train_knn_and_save(k_neighbors=7):
    ensure_dirs()
    X, y = [], []
    for label in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if not f.endswith(".npy"): continue
            arr = np.load(os.path.join(folder, f))
            if arr.shape == (42,):
                X.append(arr); y.append(label)
    if len(X) == 0:
        return False, "No data found - collect samples first."
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights="distance", metric="euclidean")
    knn.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(knn, f)
    return True, f"Trained on {len(X)} samples, saved {MODEL_PATH}"

def synthesize_gtts_bytes(text: str):
    """Return mp3 bytes using gTTS or None if not available/fails."""
    if not GTTS_AVAILABLE or not text.strip():
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            gTTS(text).save(tmp.name)
            tmp.flush()
            tmp.seek(0)
            data = open(tmp.name, "rb").read()
        try:
            os.remove(tmp.name)
        except Exception:
            pass
        return data
    except Exception as e:
        # swallow and return None
        return None

def server_speak_now(text: str):
    """Speak immediately on server speaker (pyttsx3). Returns True/False."""
    if not PYTTS_AVAILABLE or not text.strip():
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False

# ----------------- Video processors -----------------
class CollectorProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.saved = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if res.multi_hand_landmarks:
            hls = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hls, mp_hands.HAND_CONNECTIONS)
            if GLOBAL_CONFIG.get("is_collecting") and GLOBAL_CONFIG.get("collect_letter"):
                feat = extract_features_from_landmarks(hls)
                folder = os.path.join(DATA_DIR, GLOBAL_CONFIG["collect_letter"])
                os.makedirs(folder, exist_ok=True)
                fname = os.path.join(folder, f"{int(time.time()*1000)}.npy")
                np.save(fname, feat)
                self.saved += 1
        cv2.rectangle(img, (0,0), (img.shape[1], 70), (0,0,0), -1)
        text = f"Mode: DATA COLLECTION | Letter: {GLOBAL_CONFIG.get('collect_letter')} | Collecting: {GLOBAL_CONFIG.get('is_collecting')} | Saved: {self.saved}"
        cv2.putText(img, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

class PredictorProcessor(VideoProcessorBase):
    """
    Immediate speech behavior:
    - short smoothing window (pred_history)
    - on stable detection: set st.session_state.predict_text (persistent overlay)
    - create gTTS bytes and store in st.session_state["_latest_audio_bytes"]
    - optionally call pyttsx3 to speak immediately on server speakers
    """
    def __init__(self):
        self.model = load_model()
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        # smaller window for more responsive detection
        self.pred_history = deque(maxlen=6)
        GLOBAL_CONFIG["model_loaded"] = self.model is not None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # hot reload model request
        if st.session_state.get("_force_model_reload"):
            self.model = load_model()
            GLOBAL_CONFIG["model_loaded"] = self.model is not None
            st.session_state["_force_model_reload"] = False

        # clear request handling
        if st.session_state.get("clear_requested", False):
            self.pred_history.clear()
            st.session_state.predict_text = ""
            st.session_state.clear_requested = False

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # overlay persistent text
        overlay_text = st.session_state.get("predict_text", "")
        h, w = img.shape[:2]
        # dark rectangle at bottom
        cv2.rectangle(img, (0, h-120), (w, h), (0,0,0), -1)
        # wrap text
        max_chars = max(20, w // 20)
        lines = [overlay_text[i:i+max_chars] for i in range(0, len(overlay_text), max_chars)] if overlay_text else []
        for i, line in enumerate(lines[-3:]):  # last 3 lines
            cv2.putText(img, line, (10, h-80 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if self.model is None:
            cv2.putText(img, "No model loaded - train & press Reload model", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        res = self.hands.process(rgb)
        if res.multi_hand_landmarks:
            hls = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hls, mp_hands.HAND_CONNECTIONS)
            feat = extract_features_from_landmarks(hls).reshape(1,-1)
            try:
                pred_label = self.model.predict(feat)[0]
                self.pred_history.append(pred_label)
            except Exception:
                pred_label = None

            # stability check
            if len(self.pred_history) >= 4:
                counts = Counter(self.pred_history)
                stable_label, freq = counts.most_common(1)[0]
                # require majority frequency (>= half) and at least 3
                if freq >= max(3, len(self.pred_history)//2 + 1):
                    stable_char = LABEL_TO_CHAR(stable_label)
                    current = st.session_state.get("predict_text", "")
                    if not current or current[-1] != stable_char:
                        # append to persistent overlay text
                        st.session_state.predict_text = current + stable_char

                        # Immediately attempt server playback (pyttsx3) for instant real-world audio
                        if PYTTS_AVAILABLE:
                            try:
                                # speak short char/word on server immediately (non-blocking-ish)
                                # Note: pyttsx3.runAndWait() blocks the thread; keep short text.
                                # To avoid blocking streaming for long, speak very short strings.
                                server_speak_now(stable_char)
                            except Exception:
                                pass

                        # Prepare gTTS bytes for browser playback and store in session_state
                        if GTTS_AVAILABLE:
                            mp3_bytes = synthesize_gtts_bytes(stable_char)
                            if mp3_bytes:
                                st.session_state["_latest_audio_bytes"] = mp3_bytes

                        # Signal main thread to play audio (and optionally re-run UI)
                        st.session_state.latest_spoken_char = stable_char
                        st.session_state.text_updated = True

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------- UI -----------------
st.title("SAMVAAD — Instant speech on detection")
st.markdown("As soon as a stable letter is detected, it is appended to the overlay and spoken (no clicks).")

# Sidebar mode
mode = st.sidebar.radio("Mode", ["Data Collection", "Train Model", "Live Recognition", "Delete Model"])
GLOBAL_CONFIG["mode"] = mode

# dataset summary
with st.expander("Dataset summary"):
    ensure_dirs()
    counts = {}
    for lb in sorted(os.listdir(DATA_DIR)) if os.path.exists(DATA_DIR) else []:
        p = os.path.join(DATA_DIR, lb)
        if os.path.isdir(p):
            counts[lb] = len([f for f in os.listdir(p) if f.endswith(".npy")])
    st.write(counts or "No data yet")

# Data Collection
if mode == "Data Collection":
    st.subheader("Data Collection")
    col1, col2 = st.columns([2,1])
    with col1:
        letter = st.selectbox("Choose letter to collect (A–Z or SPACE)", LETTERS)
        samples_goal = st.number_input("Target samples (per session)", min_value=10, max_value=2000, value=200, step=10)
        st.write("Position hand, press Start Collection, then Stop when done.")
    with col2:
        if st.button("Start Collection"):
            st.session_state.GLOBAL_CONFIG["collect_letter"] = letter
            st.session_state.GLOBAL_CONFIG["is_collecting"] = True
        if st.button("Stop Collection"):
            st.session_state.GLOBAL_CONFIG["is_collecting"] = False
        st.write("Collecting:", "✅" if GLOBAL_CONFIG["is_collecting"] else "❌")
        st.write("Letter:", GLOBAL_CONFIG["collect_letter"] or "None")

    st.write("---")
    webrtc_streamer(
        key="collect",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=CollectorProcessor,
        async_processing=True,
    )

# Train Model
elif mode == "Train Model":
    st.subheader("Train Model (KNN)")
    k = st.slider("k (neighbors)", 1, 21, 7, step=2)
    if st.button("Train KNN"):
        with st.spinner("Training..."):
            ok, msg = train_knn_and_save(k_neighbors=k)
            if ok: st.success(msg)
            else: st.error(msg)
    st.write("After training, go to Live Recognition and press 'Reload model' if needed.")

# Live Recognition
elif mode == "Live Recognition":
    st.subheader("Live Recognition — Immediate speech")
    if st.button("🔄 Reload model"):
        st.session_state["_force_model_reload"] = True

    col1, col2 = st.columns([3,1])
    with col2:
        if st.button("Clear overlay text"):
            st.session_state.predict_text = ""
            st.session_state.latest_spoken_char = None
            st.session_state.clear_requested = True
            st.session_state.text_updated = False
        st.write("---")
        st.write("Auto-speak mode (processor speaks char server-side if available; browser plays gTTS bytes):")
        st.session_state.auto_speak_mode = st.radio("Mode:", ["Speak each letter", "Speak entire recognized string"], index=0)
        st.write("gTTS available (browser playback):", GTTS_AVAILABLE)
        st.write("pyttsx3 available (server speakers):", PYTTS_AVAILABLE)

    with col1:
        webrtc_streamer(
            key="predict",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=PredictorProcessor,
            async_processing=True,
        )

    # MAIN THREAD: play audio immediately when processor signals text_updated.
    # This block runs during Streamlit reruns; processor sets text_updated True to trigger.
    if st.session_state.text_updated:
        # consume flag
        st.session_state.text_updated = False

        # Decide what to speak based on auto_speak_mode
        if st.session_state.auto_speak_mode == "Speak each letter":
            to_speak = st.session_state.latest_spoken_char or ""
        else:  # whole string
            to_speak = st.session_state.predict_text or ""

        # Play browser audio if gTTS bytes produced
        audio_bytes = st.session_state.get("_latest_audio_bytes")
        if audio_bytes and to_speak.strip():
            try:
                st.audio(audio_bytes, format="audio/mp3")
            finally:
                # clear after playing
                st.session_state["_latest_audio_bytes"] = None
        else:
            # If no gTTS, fallback to pyttsx3 server playback (if available)
            if PYTTS_AVAILABLE and to_speak.strip():
                server_speak_now(to_speak)

        # Re-run to immediately reflect overlay/update UI (fast)
        st.experimental_rerun()

# Delete Model
elif mode == "Delete Model":
    st.subheader("Delete model")
    if st.button("Delete sign_model.pkl"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            st.success("Deleted model.")
        else:
            st.warning("No model found.")

st.markdown("---")
st.caption("Instant: stable detection -> spoken immediately. Use gTTS + browser audio for client playback and pyttsx3 for server playback.")
