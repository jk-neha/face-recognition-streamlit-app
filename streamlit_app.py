## python -m streamlit run streamlit_app.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# ─────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FaceID Attendance",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
#  GLOBAL CSS  – dark-tech / cyber aesthetic
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

/* ── root palette ── */
:root {
    --bg:        #060b14;
    --bg2:       #0c1526;
    --bg3:       #111e35;
    --accent:    #00d4ff;
    --accent2:   #7b5ea7;
    --green:     #00ff9d;
    --red:       #ff4466;
    --text:      #c8dff0;
    --muted:     #4a6a8a;
    --border:    rgba(0,212,255,0.18);
    --font-mono: 'Share Tech Mono', monospace;
    --font-ui:   'Exo 2', sans-serif;
}

/* ── global resets ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--bg2) !important; }
section[data-testid="stMain"] > div { padding-top: 0 !important; }

/* ── animated grid background ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
    animation: gridMove 20s linear infinite;
}
@keyframes gridMove {
    0%   { background-position: 0 0; }
    100% { background-position: 48px 48px; }
}

/* ── hero banner ── */
.hero-banner {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #060b14 0%, #0a1929 50%, #060b14 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.5rem 3rem 2rem;
    margin-bottom: 2rem;
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::before {
    content: '';
    position: absolute;
    bottom: -80px; left: 10%;
    width: 250px; height: 250px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(123,94,167,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: var(--font-mono);
    font-size: 2.4rem;
    color: var(--accent);
    letter-spacing: 0.06em;
    margin: 0 0 0.25rem;
    text-shadow: 0 0 30px rgba(0,212,255,0.35);
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,255,157,0.08);
    border: 1px solid rgba(0,255,157,0.25);
    color: var(--green);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 2px;
    letter-spacing: 0.15em;
    margin-top: 1rem;
    animation: pulse 2.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.55; }
}

/* ── scan animation overlay (decorative) ── */
.scan-frame {
    position: absolute;
    right: 3rem; top: 50%;
    transform: translateY(-50%);
    width: 120px; height: 120px;
}
.scan-frame .corner {
    position: absolute;
    width: 22px; height: 22px;
    border-color: var(--accent);
    border-style: solid;
    opacity: 0.7;
}
.scan-frame .tl { top:0; left:0;  border-width: 2px 0 0 2px; }
.scan-frame .tr { top:0; right:0; border-width: 2px 2px 0 0; }
.scan-frame .bl { bottom:0; left:0;  border-width: 0 0 2px 2px; }
.scan-frame .br { bottom:0; right:0; border-width: 0 2px 2px 0; }
.scan-line {
    position: absolute;
    left: 4px; right: 4px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    box-shadow: 0 0 8px var(--accent);
    animation: scanDown 2.2s ease-in-out infinite;
}
@keyframes scanDown {
    0%   { top: 4px; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 112px; opacity: 0; }
}

/* ── stat cards ── */
.stat-row {
    display: flex;
    gap: 16px;
    margin-bottom: 2rem;
}
.stat-card {
    flex: 1;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 8px 0 0 8px;
}
.stat-card.blue::before  { background: var(--accent); }
.stat-card.green::before { background: var(--green); }
.stat-card.purple::before{ background: var(--accent2); }
.stat-card:hover { border-color: rgba(0,212,255,0.4); }
.stat-label {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.stat-value {
    font-family: var(--font-mono);
    font-size: 1.9rem;
    color: var(--accent);
    line-height: 1;
}
.stat-card.green .stat-value  { color: var(--green); }
.stat-card.purple .stat-value { color: #a87fd4; }

/* ── section headers ── */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
}
.section-hdr .line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}
.section-hdr .label {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
}

/* ── panels ── */
.panel {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.panel-title {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
}

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 0.9rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.12) !important;
}
.stTextInput label {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-family: var(--font-mono) !important;
}

/* buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.8rem !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.08) !important;
    box-shadow: 0 0 18px rgba(0,212,255,0.2) !important;
}

/* file uploader */
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: var(--font-mono) !important;
}

/* alerts */
.stAlert {
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    border: none !important;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* tabs */
[data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    padding: 0.8rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabContent"] {
    padding-top: 1.5rem !important;
}

/* divider */
hr { border-color: var(--border) !important; }

/* column gaps */
[data-testid="stHorizontalBlock"] { gap: 1.5rem !important; }

/* spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────
os.makedirs("Attendance", exist_ok=True)
os.makedirs("static/faces", exist_ok=True)

attendance_file = "Attendance/attendance.csv"
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time", "Date", "Status"]).to_csv(attendance_file, index=False)

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.3, 5)


def train_model():
    faces, labels = [], []
    users = os.listdir("static/faces")
    for user in users:
        user_path = f"static/faces/{user}"
        for imgname in os.listdir(user_path):
            img = cv2.imread(f"{user_path}/{imgname}")
            if img is None:
                continue
            faces.append(cv2.resize(img, (50, 50)).ravel())
            labels.append(user)
    if not faces:
        return False
    knn = KNeighborsClassifier(n_neighbors=min(3, len(faces)))
    knn.fit(faces, labels)
    joblib.dump(knn, "static/face_recognition_model.pkl")
    return True


def identify_face(face_array):
    model = joblib.load("static/face_recognition_model.pkl")
    return model.predict(face_array)[0]


def mark_attendance(name):
    df = pd.read_csv(attendance_file)
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    # prevent duplicate per day
    already = df[(df["Name"] == name) & (df.get("Date", pd.Series(dtype=str)) == today)]
    if len(already) == 0:
        new_row = pd.DataFrame([[name, now.strftime("%H:%M:%S"), today, "✅ Present"]],
                               columns=["Name", "Time", "Date", "Status"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        return True
    return False  # already marked today


def get_stats():
    users = len(os.listdir("static/faces")) if os.path.exists("static/faces") else 0
    df = pd.read_csv(attendance_file)
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = len(df[df.get("Date", pd.Series(dtype=str)) == today]) if "Date" in df.columns else 0
    model_exists = os.path.exists("static/face_recognition_model.pkl")
    return users, len(df), today_count, model_exists


# ─────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">⬡ FACEID ATTENDANCE</div>
    <div class="hero-sub">Neural Recognition &nbsp;|&nbsp; Real-Time Tracking &nbsp;|&nbsp; Automated Logging</div>
    <div class="hero-badge">● SYSTEM ONLINE</div>
    <div class="scan-frame">
        <div class="corner tl"></div>
        <div class="corner tr"></div>
        <div class="corner bl"></div>
        <div class="corner br"></div>
        <div class="scan-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  STATS ROW
# ─────────────────────────────────────────
users_count, total_records, today_count, model_ready = get_stats()

st.markdown(f"""
<div class="stat-row">
    <div class="stat-card blue">
        <div class="stat-label">Registered Users</div>
        <div class="stat-value">{users_count:02d}</div>
    </div>
    <div class="stat-card green">
        <div class="stat-label">Today's Attendance</div>
        <div class="stat-value">{today_count:02d}</div>
    </div>
    <div class="stat-card purple">
        <div class="stat-label">Total Records</div>
        <div class="stat-value">{total_records:04d}</div>
    </div>
    <div class="stat-card {'green' if model_ready else 'blue'}">
        <div class="stat-label">Model Status</div>
        <div class="stat-value" style="font-size:1rem; margin-top:6px; color:{'var(--green)' if model_ready else 'var(--red)'}">
            {'◉ READY' if model_ready else '○ UNTRAINED'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  REGISTER USER  ", "  FACE RECOGNITION  ", "  ATTENDANCE LOG  "])


# ══════════════════════════════════════════
#  TAB 1 — REGISTER
# ══════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-hdr">
        <span class="label">New User Registration</span>
        <span class="line"></span>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_info = st.columns([1.4, 1])

    with col_form:
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Identity Setup</div>', unsafe_allow_html=True)

        username = st.text_input("Full Name", placeholder="e.g. Neha Vardhini")

        uploaded_images = st.file_uploader(
            "Upload Face Images (multiple angles recommended)",
            accept_multiple_files=True,
            type=["jpg", "png", "jpeg"],
            help="Upload 5–10 clear face photos for better accuracy"
        )

        if uploaded_images:
            st.caption(f"📁 {len(uploaded_images)} image(s) selected")

        col_btn, _ = st.columns([1, 2])
        with col_btn:
            register_clicked = st.button("⬡  REGISTER", use_container_width=True)

        if register_clicked:
            if not username:
                st.error("⚠ Name is required.")
            elif not uploaded_images:
                st.error("⚠ Please upload at least one face image.")
            else:
                user_folder = f"static/faces/{username}"
                os.makedirs(user_folder, exist_ok=True)
                with st.spinner("Processing images & training model…"):
                    for i, img_file in enumerate(uploaded_images):
                        img = np.array(Image.open(img_file))
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(f"{user_folder}/{i}.jpg", img_bgr)
                    success = train_model()
                if success:
                    st.success(f"✅ **{username}** registered & model retrained.")
                    st.rerun()
                else:
                    st.error("Training failed — no valid face data found.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Registered Identities</div>', unsafe_allow_html=True)

        if os.path.exists("static/faces"):
            users_list = os.listdir("static/faces")
            if users_list:
                for u in users_list:
                    img_count = len(os.listdir(f"static/faces/{u}"))
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;justify-content:space-between;
                                padding:8px 0;border-bottom:1px solid rgba(0,212,255,0.08);">
                        <span style="font-family:var(--font-mono);font-size:0.82rem;color:#c8dff0;">◈ {u}</span>
                        <span style="font-size:0.68rem;color:var(--muted);letter-spacing:0.1em;">{img_count} imgs</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:var(--muted);font-size:0.82rem;">No users registered yet.</p>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="panel" style="margin-top:0;">
            <div class="panel-title"><span class="dot"></span>Tips for Best Accuracy</div>
            <ul style="color:var(--muted);font-size:0.78rem;font-family:var(--font-mono);line-height:1.9;padding-left:1.2rem;">
                <li>Upload 5–10 photos</li>
                <li>Vary angles & lighting</li>
                <li>Avoid heavy shadows</li>
                <li>Use clear, hi-res images</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════
#  TAB 2 — RECOGNITION
# ══════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-hdr">
        <span class="label">Live Face Recognition</span>
        <span class="line"></span>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists("static/face_recognition_model.pkl"):
        st.markdown("""
        <div style="background:rgba(255,68,102,0.06);border:1px solid rgba(255,68,102,0.25);
                    border-radius:8px;padding:1rem 1.4rem;font-family:var(--font-mono);
                    font-size:0.82rem;color:#ff7a94;">
            ⚠  No trained model found. Please register at least one user first.
        </div>
        """, unsafe_allow_html=True)
    else:
        col_upload, col_result = st.columns([1, 1])

        with col_upload:
            st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Upload Test Image</div>', unsafe_allow_html=True)
            test_image = st.file_uploader("Select image to identify", type=["jpg", "png", "jpeg"], key="test_img")

            if test_image:
                preview = Image.open(test_image)
                st.image(preview, caption="Input Image", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Recognition Output</div>', unsafe_allow_html=True)

            if test_image:
                with st.spinner("Scanning…"):
                    image = Image.open(test_image)
                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    faces = extract_faces(image_bgr)

                if len(faces) == 0:
                    st.markdown("""
                    <div style="text-align:center;padding:2rem 0;color:var(--red);
                                font-family:var(--font-mono);font-size:0.85rem;">
                        ✕ NO FACE DETECTED<br>
                        <span style="color:var(--muted);font-size:0.72rem;">Ensure clear lighting & frontal pose</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    prediction = None
                    for (x, y, w, h) in faces:
                        face = image_bgr[y:y+h, x:x+w]
                        resized = cv2.resize(face, (50, 50)).ravel()
                        prediction = identify_face([resized])

                        # draw bounding box
                        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 212, 255), 2)
                        # corners accent
                        sz = 14
                        for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
                            cv2.line(image_bgr,(px,py),(px+dx*sz,py),(0,212,255),3)
                            cv2.line(image_bgr,(px,py),(px,py+dy*sz),(0,212,255),3)
                        # label pill
                        label = prediction
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                        cv2.rectangle(image_bgr,(x, y-30),(x+tw+16, y),(0,212,255),-1)
                        cv2.putText(image_bgr, label,(x+8, y-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,(6,11,20),2)

                    final = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    st.image(final, use_container_width=True)

                    if prediction:
                        marked = mark_attendance(prediction)
                        st.markdown(f"""
                        <div style="margin-top:1rem;background:rgba(0,255,157,0.06);
                                    border:1px solid rgba(0,255,157,0.2);border-radius:6px;
                                    padding:0.9rem 1.2rem;font-family:var(--font-mono);">
                            <div style="color:var(--green);font-size:0.95rem;margin-bottom:4px;">
                                ◉ IDENTIFIED: {prediction}
                            </div>
                            <div style="color:var(--muted);font-size:0.72rem;letter-spacing:0.1em;">
                                {'ATTENDANCE MARKED · ' + datetime.now().strftime('%H:%M:%S') if marked else 'ALREADY MARKED TODAY'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style="text-align:center;padding:3rem 0;color:var(--muted);
                            font-family:var(--font-mono);font-size:0.82rem;">
                    ○ AWAITING INPUT<br>
                    <span style="font-size:0.7rem;letter-spacing:0.1em;">Upload an image to begin scanning</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════
#  TAB 3 — ATTENDANCE LOG
# ══════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-hdr">
        <span class="label">Attendance Records</span>
        <span class="line"></span>
    </div>
    """, unsafe_allow_html=True)

    df = pd.read_csv(attendance_file)

    if df.empty:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:var(--muted);font-family:var(--font-mono);font-size:0.82rem;">
            ○ NO RECORDS YET<br>
            <span style="font-size:0.7rem;">Attendance will appear here after recognition.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Filter controls
        col_f1, col_f2, col_dl = st.columns([1.5, 1.5, 1])
        with col_f1:
            search = st.text_input("Search by name", placeholder="Type to filter…")
        with col_f2:
            if "Date" in df.columns:
                dates = ["All"] + sorted(df["Date"].dropna().unique().tolist(), reverse=True)
                date_filter = st.selectbox("Filter by date", dates)
            else:
                date_filter = "All"
        with col_dl:
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ EXPORT CSV", csv_data, "attendance.csv", "text/csv", use_container_width=True)

        filtered = df.copy()
        if search:
            filtered = filtered[filtered["Name"].str.contains(search, case=False, na=False)]
        if date_filter != "All" and "Date" in filtered.columns:
            filtered = filtered[filtered["Date"] == date_filter]

        st.dataframe(
            filtered.sort_values("Date" if "Date" in filtered.columns else "Name", ascending=False)
                    .reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

        st.markdown(f"""
        <div style="text-align:right;font-family:var(--font-mono);font-size:0.7rem;
                    color:var(--muted);margin-top:0.5rem;letter-spacing:0.1em;">
            SHOWING {len(filtered)} OF {len(df)} RECORDS
        </div>
        """, unsafe_allow_html=True)
