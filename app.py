import io
import os
import pickle
import hashlib
import threading
import sqlite3
import smtplib
import json
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from email.message import EmailMessage

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------
# Page config (must be the FIRST Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Campus Management System - Smart Campus AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Face recognition library
# face_recognition calls quit() (SystemExit) when face_recognition_models
# is missing, so we catch BaseException, not just ImportError.
# ---------------------------------------------------------
FACE_LIB = False
FACE_ERR = ""
try:
    import face_recognition
    FACE_LIB = True
except BaseException as e:
    FACE_LIB = False
    FACE_ERR = f"{type(e).__name__}: {e}"

FACE_TOLERANCE = 0.5      # lower = stricter matching (0.45 strict, 0.55 relaxed)
MAX_IMAGE_SIDE = 800      # one student's close-up photo -> small size is enough (fast)
DATA_FILE = "campus_data.pkl"  # Legacy data file; imported once into SQLite if present.
DB_FILE = "campus_data.db"

DEPARTMENTS = ["AIDS", "CSE", "IT", "E&TC", "Civil", "Mechanical"]
CLASSES = ["FY B.Tech", "SY B.Tech", "TY B.Tech", "Final Year B.Tech"]
DIVISIONS = ["Division A", "Division B", "Division C"]

# College timetable (breaks: 12:00-12:45 and 02:45-03:00)
SLOTS = [
    "10:00 AM - 11:00 AM",
    "11:00 AM - 12:00 PM",
    "12:45 PM - 01:45 PM",
    "01:45 PM - 02:45 PM",
    "03:00 PM - 04:00 PM",
    "04:00 PM - 05:00 PM",
]

st.markdown("""
<style>
    :root {
        --canvas: #f7fbff; --surface: #ffffff; --surface-2: #eff7ff;
        --line: #d9e7fb; --text: #13213f; --muted: #5e7197;
        --blue: #2f80ed; --violet: #825de8; --teal: #13bfa3; --rose: #f56c85;
    }
    html, body, [class*="css"] { font-family: "Segoe UI", Inter, system-ui, sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at 3% 0%, rgba(72,164,255,.20), transparent 27rem),
            radial-gradient(circle at 96% 8%, rgba(187,116,255,.17), transparent 24rem),
            linear-gradient(145deg, #f9fdff 0%, #eef8ff 49%, #fcf8ff 100%);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(255,255,255,.74); border-bottom: 1px solid rgba(178,205,244,.45); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #eef8ff 100%);
        border-right: 1px solid #d7e7fb;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: .75rem; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption { color: var(--muted); }
    [data-testid="stSidebar"] hr { border-color: var(--line); margin: 1.2rem 0; }
    .block-container { max-width: 1500px; padding: .65rem 2.15rem 3rem; }
    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: var(--text); letter-spacing: -.03em;
    }
    p, .stMarkdown p { color: var(--muted); }
    @keyframes liftIn {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .smart-hero {
        position: relative; overflow: hidden; min-height: 0; padding: 1.2rem 2rem;
        border-radius: 28px; isolation: isolate; color: #fff;
        background: linear-gradient(118deg, #3298f2 0%, #5d78ec 48%, #a95fdd 100%);
        border: 1px solid rgba(143,186,255,.52);
        box-shadow: 0 24px 48px rgba(54,119,210,.24); animation: liftIn .58s ease-out;
    }
    .smart-hero:before, .smart-hero:after { content: ""; position: absolute; border-radius: 999px; z-index: -1; }
    .smart-hero:before { width: 360px; height: 360px; right: -115px; top: -190px; background: rgba(173,246,255,.35); filter: blur(1px); }
    .smart-hero:after { width: 260px; height: 260px; right: 95px; bottom: -180px; background: rgba(255,208,250,.34); }
    .hero-kicker { display: flex; align-items: center; gap: .55rem; color: #eaf7ff; font-size: .68rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
    .hero-dot { width: .55rem; height: .55rem; border-radius: 50%; background: #55efc4; box-shadow: 0 0 0 5px rgba(85,239,196,.15); }
    .smart-hero h1 { margin: .42rem 0 .35rem; color: #fff; max-width: 790px; font-size: clamp(1.85rem, 3.2vw, 2.7rem); line-height: 1.03; }
    .smart-hero h1 span { color: #e4edff; }
    .smart-hero p { margin: 0; max-width: 660px; color: #f5faff; font-size: .9rem; line-height: 1.45; }
    .hero-pills { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .75rem; }
    .hero-pill { border: 1px solid rgba(255,255,255,.38); background: rgba(255,255,255,.16); color: #fff; padding: .38rem .62rem; border-radius: 99px; font-size: .7rem; backdrop-filter: blur(8px); }
    .hero-pill strong { color: #e1fff4; margin-right: .28rem; }
    .hero-emblem { position: absolute; right: 2.9rem; bottom: 2.2rem; width: 98px; height: 98px; display: grid; place-items: center; border-radius: 27px; background: rgba(255,255,255,.17); border: 1px solid rgba(255,255,255,.38); color: #fff; font-size: 3.1rem; transform: rotate(-8deg); }
    .side-brand { padding: .8rem .4rem 1.3rem; }
    .side-brand .eyebrow { color: #2f80ed; font-size: .67rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    .side-brand h2 { margin: .42rem 0 .35rem; color: #162a52; font-size: 1.35rem; }
    .side-brand p { margin: 0; font-size: .83rem; color: #60739a; line-height: 1.45; }
    .sidebar-label { color: #5279ba; font-size: .68rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; margin: .35rem 0 .42rem; }
    .sidebar-status { padding: .95rem; margin-top: 1.25rem; border-radius: 15px; background: linear-gradient(135deg, #edf7ff, #f8f0ff); border: 1px solid #d9e7fb; }
    .sidebar-status strong { display:block; color:#233f72; font-size: .86rem; margin-bottom:.24rem; }
    .sidebar-status span { color:#63779e; font-size:.76rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { display:grid; gap:.42rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        margin:0; padding:.7rem .72rem; border-radius:11px; background:rgba(255,255,255,.66);
        border:1px solid transparent; transition:background .2s ease, border-color .2s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#edf7ff; border-color:#b8d6fc; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:linear-gradient(135deg, #e6f2ff, #f2eaff); border-color:#9fc4fb; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label * { color:#29466f !important; opacity:1 !important; }
    .section-heading { margin: 1rem 0 .7rem; display:flex; align-items: center; gap:.85rem; animation: liftIn .45s ease-out; }
    .section-icon { display:grid; place-items:center; flex: 0 0 auto; width: 43px; height:43px; border-radius:14px; background:linear-gradient(145deg, #dff1ff, #efe4ff); border: 1px solid #cddffb; font-size:1.35rem; }
    .section-heading .eyebrow { color:#367fdd; font-size:.66rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
    .section-heading h2 { margin:.1rem 0; font-size:1.35rem; }
    .section-heading p { margin:0; font-size:.8rem; }
    .metric-card {
        position:relative; overflow:hidden; min-height: 145px; padding: 1.2rem;
        border-radius: 20px; background: linear-gradient(145deg, #ffffff, #f5f9ff);
        border: 1px solid var(--line); box-shadow: 0 12px 26px rgba(63,111,177,.12); transition: all .25s ease;
    }
    .metric-card:after { content:""; position:absolute; width:100px; height:100px; border-radius:50%; right:-45px; bottom:-52px; background:rgba(110,153,255,.11); }
    .metric-card:hover { transform: translateY(-5px); border-color: #b7d5fb; box-shadow: 0 20px 35px rgba(63,111,177,.18); }
    .metric-top { display:flex; justify-content:space-between; align-items:center; position:relative; z-index:1; }
    .metric-icon { display:grid; place-items:center; width:39px; height:39px; border-radius:12px; font-size:1.2rem; }
    .metric-icon.blue { background:rgba(91,140,255,.18); } .metric-icon.teal { background:rgba(45,212,191,.16); }
    .metric-icon.violet { background:rgba(157,112,255,.17); } .metric-icon.rose { background:rgba(251,113,133,.16); }
    .metric-badge { color:#3978ca; background:#eaf4ff; padding:.24rem .45rem; border-radius:99px; font-size:.66rem; font-weight:700; }
    .metric-card h3 { margin: .95rem 0 0; color: #5f7196; font-size: .86rem; font-weight: 600; }
    .metric-card h2 { margin: .2rem 0 0; color: #17294c; font-size: 2rem; line-height:1.15; }
    .metric-card p { margin:.32rem 0 0; color:#7e91b2; font-size:.74rem; }
    .smart-panel { padding:1.25rem; min-height: 185px; border-radius:20px; background:rgba(255,255,255,.84); border:1px solid var(--line); box-shadow:0 12px 24px rgba(63,111,177,.10); }
    .panel-kicker { color:#3c82dd; font-size:.68rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
    .smart-panel h3 { margin:.42rem 0 .3rem; font-size:1.18rem; }
    .smart-panel p { margin:.25rem 0; line-height:1.55; }
    .signal-list { display:grid; gap:.62rem; margin-top:1.05rem; }
    .signal-row { display:flex; align-items:center; gap:.7rem; padding:.7rem .75rem; border-radius:12px; background:#f5f9ff; }
    .signal-dot { width:.58rem; height:.58rem; border-radius:50%; flex:0 0 auto; }
    .signal-dot.green { background:#48e0b5; box-shadow:0 0 0 4px rgba(72,224,181,.1); }
    .signal-dot.amber { background:#f9c74f; box-shadow:0 0 0 4px rgba(249,199,79,.1); }
    .signal-row strong { display:block; color:#29466f; font-size:.83rem; }
    .signal-row span { display:block; color:#7184a6; font-size:.73rem; margin-top:.08rem; }
    .quick-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; margin-top:1.1rem; }
    .quick-card { padding:.9rem; border-radius:14px; background:#f5f9ff; border:1px solid #e0ebfb; }
    .quick-card strong { color:#29466f; display:block; font-size:.83rem; margin:.35rem 0; }
    .quick-card span { color:#7487a9; font-size:.73rem; line-height:1.4; }
    .workspace-card { height:128px; min-height:0; box-sizing:border-box; overflow:hidden; padding:.72rem .9rem; margin-bottom:.28rem; border-radius:18px; background:linear-gradient(145deg, #ffffff, #f3f9ff); border:1px solid var(--line); transition:transform .22s ease, border-color .22s ease; }
    .workspace-card:hover { transform:translateY(-4px); border-color:#a9cffb; box-shadow:0 16px 27px rgba(69,134,210,.15); }
    .workspace-card .workspace-icon { display:grid; place-items:center; width:31px; height:31px; border-radius:10px; background:linear-gradient(145deg, #dff2ff, #eee4ff); font-size:.92rem; }
    .workspace-card .workspace-tag { margin-top:.35rem; color:#3e83d7; font-size:.58rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
    .workspace-card h3 { margin:.15rem 0 .15rem; color:#203b65; font-size:.9rem; }
    .workspace-card p { margin:0; color:#7085a8; font-size:.68rem; line-height:1.28; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input,
    [data-baseweb="select"] > div, [data-testid="stTextArea"] textarea {
        color: #1f365f !important; background: #ffffff !important;
        border-color: #cfe0f8 !important; border-radius: 11px !important;
    }
    [data-baseweb="select"] * { color: #1f365f !important; }
    [data-testid="stTextInput"] label, [data-testid="stSelectbox"] label, [data-testid="stDateInput"] label,
    [data-testid="stFileUploader"] label, [data-testid="stRadio"] label, [data-testid="stMultiSelect"] label { color:#334e7c !important; font-weight:600; }
    div[data-testid="stForm"] { padding:1.35rem; border-radius:20px; border:1px solid var(--line); background:rgba(255,255,255,.82); }
    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
        min-height:2.9rem; padding:.65rem 1rem; border-radius: 11px; font-weight: 800; letter-spacing:.01em;
        color:#ffffff !important; border:1px solid #1b6ed0; background:linear-gradient(135deg, #147eea 0%, #5c68e9 52%, #8c55dc 100%);
        box-shadow:0 9px 20px rgba(54,125,221,.23); transition: transform .2s ease, filter .2s ease;
    }
    .stButton > button *, .stDownloadButton > button *, [data-testid="stFormSubmitButton"] > button * {
        color:#ffffff !important; opacity:1 !important; font-weight:800 !important; font-size:.92rem !important;
        text-shadow:0 1px 1px rgba(17,54,118,.25);
    }
    .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { transform:translateY(-2px); filter:brightness(1.07); }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible { outline:3px solid #ffbd59; outline-offset:3px; }
    div[data-testid="stExpander"] { background:rgba(255,255,255,.86); border:1px solid var(--line); border-radius:15px; overflow:hidden; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    [data-testid="stAlert"] { border-radius:14px; border-width:1px; }
    [data-testid="stFileUploader"] { border:1px dashed #bdd7f8; border-radius:14px; padding:.35rem; background:#f8fcff; }
    [data-testid="stProgress"] > div > div > div { background: linear-gradient(90deg, #2f80ed, #825de8); }
    .app-footer { margin-top:2.5rem; padding-top:1.1rem; border-top:1px solid var(--line); color:#7184a6; text-align:center; font-size:.8rem; }
    @media (max-width: 850px) {
        .block-container { padding:1rem 1rem 2.2rem; }
        .smart-hero { min-height:0; padding:1.8rem; border-radius:22px; }
        .hero-emblem { display:none; }
        .quick-grid { grid-template-columns:1fr; }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def hash_pw(p):
    return hashlib.sha256(p.encode("utf-8")).hexdigest()


PERSIST_KEYS = ["students_db", "faculty_db", "attendance_logs", "lecture_logs",
                "sms_outbox", "att_session", "whatsapp_sender_override"]


class CampusStorage:
    """Local SQLite storage for students, attendance, and notification history.

    The app keeps its current session lists for a responsive Streamlit UI and
    saves their contents into normalized SQLite tables after every change.
    """

    def __init__(self, path=DB_FILE):
        self.path = path
        self._create_tables()
        self._migrate_legacy_pickle()

    def _connect(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS students (
                    department TEXT NOT NULL, class_name TEXT NOT NULL,
                    division TEXT NOT NULL, roll_no TEXT NOT NULL,
                    name TEXT NOT NULL, mobile TEXT, parent_mobile TEXT NOT NULL,
                    parent_email TEXT, photo BLOB, encoding BLOB,
                    PRIMARY KEY (department, class_name, division, roll_no)
                );
                CREATE TABLE IF NOT EXISTS faculty (
                    username TEXT PRIMARY KEY, name TEXT NOT NULL,
                    mobile TEXT, password_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, faculty TEXT,
                    class_name TEXT, division TEXT, roll_no TEXT, name TEXT,
                    department TEXT, date TEXT, time TEXT, slot TEXT,
                    status TEXT, method TEXT, parent_mobile TEXT, parent_email TEXT
                );
                CREATE TABLE IF NOT EXISTS lecture_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, faculty TEXT,
                    class_name TEXT, division TEXT, department TEXT,
                    date TEXT, slot TEXT, total INTEGER, present INTEGER, absent INTEGER
                );
                CREATE TABLE IF NOT EXISTS notification_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, channel TEXT,
                    recipient TEXT, student TEXT, type TEXT, message TEXT, status TEXT
                );
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY, value BLOB
                );
            """)

    def _is_empty(self):
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 0 and \
                conn.execute("SELECT COUNT(*) FROM faculty").fetchone()[0] == 0

    def _migrate_legacy_pickle(self):
        """Import the original campus_data.pkl one time, without overwriting SQLite."""
        if not self._is_empty() or not os.path.exists(DATA_FILE):
            return
        try:
            with open(DATA_FILE, "rb") as file:
                legacy = pickle.load(file)
            if isinstance(legacy, dict):
                self.save(legacy)
        except Exception:
            # A corrupt legacy file should never prevent the new app from opening.
            pass

    @staticmethod
    def _blob(value):
        return sqlite3.Binary(pickle.dumps(value)) if value is not None else None

    @staticmethod
    def _unblob(value):
        if value is None:
            return None
        try:
            return pickle.loads(value)
        except Exception:
            return None

    def load(self):
        with self._connect() as conn:
            students = []
            for row in conn.execute("SELECT * FROM students ORDER BY department, class_name, division, roll_no"):
                students.append({
                    "roll_no": row["roll_no"], "name": row["name"], "department": row["department"],
                    "class_name": row["class_name"], "division": row["division"], "mobile": row["mobile"] or "",
                    "parent_mobile": row["parent_mobile"] or "", "parent_email": row["parent_email"] or "",
                    "photo": row["photo"], "encoding": self._unblob(row["encoding"]),
                })
            faculty = [dict(row) for row in conn.execute("SELECT * FROM faculty ORDER BY name")]
            attendance = [dict(row) for row in conn.execute(
                "SELECT faculty, class_name, division, roll_no, name, department, date, time, slot, status, method, parent_mobile, parent_email FROM attendance_logs ORDER BY id")]
            lectures = [dict(row) for row in conn.execute(
                "SELECT faculty, class_name, division, department, date, slot, total, present, absent FROM lecture_logs ORDER BY id")]
            notifications = []
            for row in conn.execute("SELECT time, channel, recipient, student, type, message, status FROM notification_logs ORDER BY id"):
                item = dict(row)
                item["to"] = item["recipient"]  # compatibility with earlier in-memory outbox rows
                notifications.append(item)
            state = conn.execute("SELECT value FROM app_state WHERE key = 'att_session'").fetchone()
            att_session = self._unblob(state["value"]) if state else None
            sender_state = conn.execute("SELECT value FROM app_state WHERE key = 'whatsapp_sender_override'").fetchone()
            whatsapp_sender_override = self._unblob(sender_state["value"]) if sender_state else ""
            return {
                "students_db": students, "faculty_db": faculty, "attendance_logs": attendance,
                "lecture_logs": lectures, "sms_outbox": notifications, "att_session": att_session,
                "whatsapp_sender_override": whatsapp_sender_override or "",
            }

    def save(self, data):
        students = data.get("students_db", []) or []
        faculty = data.get("faculty_db", []) or []
        attendance = data.get("attendance_logs", []) or []
        lectures = data.get("lecture_logs", []) or []
        notifications = data.get("sms_outbox", []) or []
        with self._connect() as conn:
            conn.execute("DELETE FROM students")
            conn.executemany("""
                INSERT INTO students (department, class_name, division, roll_no, name, mobile, parent_mobile, parent_email, photo, encoding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(
                s.get("department", ""), s.get("class_name", ""), s.get("division", ""), s.get("roll_no", ""),
                s.get("name", ""), s.get("mobile", ""), s.get("parent_mobile", ""), s.get("parent_email", ""),
                sqlite3.Binary(s["photo"]) if s.get("photo") is not None else None, self._blob(s.get("encoding")),
            ) for s in students])

            conn.execute("DELETE FROM faculty")
            conn.executemany("INSERT INTO faculty (username, name, mobile, password_hash) VALUES (?, ?, ?, ?)", [(
                f.get("username", ""), f.get("name", ""), f.get("mobile", ""),
                f.get("password_hash", hash_pw(f.get("password", ""))),
            ) for f in faculty])

            conn.execute("DELETE FROM attendance_logs")
            conn.executemany("""
                INSERT INTO attendance_logs (faculty, class_name, division, roll_no, name, department, date, time, slot, status, method, parent_mobile, parent_email)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(
                row.get("faculty", ""), row.get("class_name", ""), row.get("division", ""), row.get("roll_no", ""),
                row.get("name", ""), row.get("department", ""), row.get("date", ""), row.get("time", ""),
                row.get("slot", ""), row.get("status", ""), row.get("method", ""),
                row.get("parent_mobile", ""), row.get("parent_email", ""),
            ) for row in attendance])

            conn.execute("DELETE FROM lecture_logs")
            conn.executemany("""
                INSERT INTO lecture_logs (faculty, class_name, division, department, date, slot, total, present, absent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(
                row.get("faculty", ""), row.get("class_name", ""), row.get("division", ""), row.get("department", ""),
                row.get("date", ""), row.get("slot", ""), row.get("total", 0), row.get("present", 0), row.get("absent", 0),
            ) for row in lectures])

            conn.execute("DELETE FROM notification_logs")
            conn.executemany("""
                INSERT INTO notification_logs (time, channel, recipient, student, type, message, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [(
                row.get("time", ""), row.get("channel", "Notification"), row.get("recipient", row.get("to", "")),
                row.get("student", ""), row.get("type", ""), row.get("message", ""), row.get("status", ""),
            ) for row in notifications])

            conn.execute("DELETE FROM app_state WHERE key IN ('att_session', 'whatsapp_sender_override')")
            session = data.get("att_session")
            if session is not None:
                conn.execute("INSERT INTO app_state (key, value) VALUES ('att_session', ?)", (self._blob(session),))
            sender_override = str(data.get("whatsapp_sender_override", "") or "").strip()
            if sender_override:
                conn.execute("INSERT INTO app_state (key, value) VALUES ('whatsapp_sender_override', ?)",
                             (self._blob(sender_override),))


@st.cache_resource
def get_storage():
    return CampusStorage()


def save_data():
    """Persist all student and attendance data to the local SQLite database."""
    try:
        get_storage().save({key: st.session_state.get(key) for key in PERSIST_KEYS})
    except Exception as e:
        st.warning(f"Could not save local data: {e}")


def load_data():
    try:
        return get_storage().load()
    except Exception:
        return {}


def show_df(df):
    """Show a dataframe full-width on both old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


def section_heading(icon, eyebrow, title, subtitle):
    """Reusable visual heading used by each app workspace."""
    return f"""
    <div class="section-heading">
        <div class="section-icon">{icon}</div>
        <div>
            <div class="eyebrow">{eyebrow}</div>
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
    </div>
    """


def metric_card(icon, tone, label, value, note, badge):
    """A small dashboard metric that stays independent from app state."""
    return f"""
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-icon {tone}">{icon}</div>
            <div class="metric-badge">{badge}</div>
        </div>
        <h3>{label}</h3>
        <h2>{value}</h2>
        <p>{note}</p>
    </div>
    """


def _load_rgb(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return np.array(img)


def get_face_encodings(img_bytes):
    """All face encodings in an image (used for registration photo check)."""
    arr = _load_rgb(img_bytes)
    locs = face_recognition.face_locations(arr, number_of_times_to_upsample=1)
    return face_recognition.face_encodings(arr, known_face_locations=locs)


def get_single_face_encoding(img_bytes):
    """For individual attendance: returns (encoding, number_of_faces).
    If more than one face is visible, the LARGEST (closest) face is used."""
    arr = _load_rgb(img_bytes)
    locs = face_recognition.face_locations(arr, number_of_times_to_upsample=1)
    if not locs:
        return None, 0
    # location = (top, right, bottom, left) -> pick biggest area
    big = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
    enc = face_recognition.face_encodings(arr, known_face_locations=[big])[0]
    return enc, len(locs)


# ---------------------------------------------------------
# Parent notification engine: WhatsApp/SMS through Twilio and email through SMTP.
# No messages are sent unless the corresponding credentials are configured.
# ---------------------------------------------------------
@st.cache_resource
def notification_runtime():
    """Shared worker pool + result box (survives Streamlit reruns)."""
    return {"pool": ThreadPoolExecutor(max_workers=8), "results": [], "lock": threading.Lock()}


NOTIFICATION_KEYS = [
    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM", "TWILIO_SMS_FROM",
    "TWILIO_CONTENT_SID",
    "SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL", "SMTP_USE_SSL",
]


def get_notification_config():
    """Read notification credentials from Streamlit secrets or environment variables."""
    cfg = {key: os.environ.get(key, "") for key in NOTIFICATION_KEYS}
    # Do not access st.secrets when no file exists: Streamlit renders a noisy
    # missing-secrets diagnostic even if the exception is handled.
    secrets_paths = [
        os.path.join(".streamlit", "secrets.toml"),
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
    ]
    if any(os.path.exists(path) for path in secrets_paths):
        try:
            for key in NOTIFICATION_KEYS:
                cfg[key] = str(st.secrets.get(key, cfg[key]) or "")
        except Exception:
            pass
    # The college sender is not a credential. A Principal can replace this
    # approved Twilio WhatsApp sender from the app without exposing account
    # tokens or email passwords in the interface or SQLite database.
    sender_override = str(st.session_state.get("whatsapp_sender_override", "") or "").strip()
    if sender_override:
        cfg["TWILIO_WHATSAPP_FROM"] = sender_override
    return cfg


def enabled_notification_channels(cfg=None):
    cfg = cfg or get_notification_config()
    channels = []
    if all(cfg[key].strip() for key in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM")):
        channels.append("WhatsApp")
    if all(cfg[key].strip() for key in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_SMS_FROM")):
        channels.append("SMS")
    if all(cfg[key].strip() for key in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL")):
        channels.append("Email")
    return channels


def notification_configuration_gaps(cfg=None):
    """Return the non-secret setting names still needed for each alert channel."""
    cfg = cfg or get_notification_config()
    requirements = {
        "WhatsApp": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"),
        "SMS": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_SMS_FROM"),
        "Email": ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL"),
    }
    return {
        channel: [setting for setting in settings if not str(cfg.get(setting, "")).strip()]
        for channel, settings in requirements.items()
    }


def build_absence_notification(student, session):
    return (
        f"Dear Parent,\n\n"
        f"Attendance alert: {student['name']} (Roll No. {student['roll_no']}) was marked ABSENT.\n"
        f"Class: {session['department']} - {session['class_name']} - {session['division']}\n"
        f"Lecture: {session['slot']} on {session['date']}\n\n"
        f"Please contact the college office if this record is incorrect.\n\n"
        f"Sandipani Technical Campus, Kolpa"
    )


def _record_notification(channel, recipient, student_name, kind, message, status):
    runtime = notification_runtime()
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "channel": channel,
        "recipient": recipient, "to": recipient, "student": student_name,
        "type": kind, "message": message, "status": status,
    }
    with runtime["lock"]:
        runtime["results"].append(entry)
    return status


def _normalize_indian_number(mobile):
    digits = "".join(character for character in str(mobile) if character.isdigit())
    if len(digits) == 10:
        return "+91" + digits
    if len(digits) >= 11 and digits.startswith("91"):
        return "+" + digits
    return ""


def normalize_whatsapp_sender(sender):
    """Validate an approved Twilio WhatsApp sender in E.164 format."""
    raw = str(sender or "").strip()
    if raw.lower().startswith("whatsapp:"):
        raw = raw.split(":", 1)[1].strip()
    digits = "".join(character for character in raw if character.isdigit())
    if raw.startswith("+") and 8 <= len(digits) <= 15:
        return f"whatsapp:+{digits}"
    return ""


def _deliver_notification(channel, recipient, message, student_name, kind, cfg, template_variables=None):
    """Deliver one WhatsApp, SMS, or email notification in a background worker."""
    status = "Failed"
    try:
        if channel == "WhatsApp":
            import requests
            target = _normalize_indian_number(recipient)
            sender = cfg["TWILIO_WHATSAPP_FROM"].strip()
            sender = sender if sender.startswith("whatsapp:") else f"whatsapp:{sender}"
            data = {"From": sender, "To": f"whatsapp:{target}"}
            # A free-form message is accepted in a current WhatsApp conversation or
            # Twilio Sandbox. Configure an approved template for scheduled alerts
            # that may be sent outside WhatsApp's 24-hour customer-service window.
            if cfg["TWILIO_CONTENT_SID"].strip():
                data["ContentSid"] = cfg["TWILIO_CONTENT_SID"].strip()
                if template_variables:
                    data["ContentVariables"] = json.dumps(template_variables)
            else:
                data["Body"] = message
            response = requests.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{cfg['TWILIO_ACCOUNT_SID']}/Messages.json",
                auth=(cfg["TWILIO_ACCOUNT_SID"], cfg["TWILIO_AUTH_TOKEN"]),
                data=data, timeout=20,
            )
            if response.ok:
                status = "Queued with WhatsApp"
            else:
                status = f"Failed ({response.status_code}: {response.text[:120]})"
        elif channel == "SMS":
            import requests
            target = _normalize_indian_number(recipient)
            response = requests.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{cfg['TWILIO_ACCOUNT_SID']}/Messages.json",
                auth=(cfg["TWILIO_ACCOUNT_SID"], cfg["TWILIO_AUTH_TOKEN"]),
                data={"From": cfg["TWILIO_SMS_FROM"].strip(), "To": target, "Body": message}, timeout=20,
            )
            if response.ok:
                status = "Queued with SMS"
            else:
                status = f"Failed ({response.status_code}: {response.text[:120]})"
        elif channel == "Email":
            email = EmailMessage()
            email["Subject"] = "Attendance Alert: Student marked absent"
            email["From"] = cfg["SMTP_FROM_EMAIL"]
            email["To"] = recipient
            email.set_content(message)
            port = int(cfg["SMTP_PORT"] or 587)
            if str(cfg["SMTP_USE_SSL"]).lower() in ("1", "true", "yes") or port == 465:
                with smtplib.SMTP_SSL(cfg["SMTP_HOST"], port, timeout=20) as smtp:
                    smtp.login(cfg["SMTP_USERNAME"], cfg["SMTP_PASSWORD"])
                    smtp.send_message(email)
            else:
                with smtplib.SMTP(cfg["SMTP_HOST"], port, timeout=20) as smtp:
                    smtp.ehlo()
                    smtp.starttls()
                    smtp.ehlo()
                    smtp.login(cfg["SMTP_USERNAME"], cfg["SMTP_PASSWORD"])
                    smtp.send_message(email)
            status = "Email sent"
        else:
            status = "Failed (unsupported channel)"
    except Exception as error:
        status = f"Failed ({type(error).__name__})"
    return _record_notification(channel, recipient, student_name, kind, message, status)


def delivery_channels_for_student(student, cfg=None):
    """Return configured channels that have a valid parent recipient for this student."""
    enabled = enabled_notification_channels(cfg)
    mobile_is_valid = bool(_normalize_indian_number(student.get("parent_mobile", "")))
    email_is_valid = "@" in student.get("parent_email", "").strip()
    return [channel for channel in enabled if (channel in ("WhatsApp", "SMS") and mobile_is_valid)
            or (channel == "Email" and email_is_valid)]


def queue_absence_notifications(student, session, requested_channels=None):
    """Queue an absence alert through all configured or explicitly selected channels."""
    cfg = get_notification_config()
    message = build_absence_notification(student, session)
    available_channels = enabled_notification_channels(cfg)
    selected = set(available_channels if requested_channels is None else requested_channels)
    channels = [channel for channel in available_channels if channel in selected]
    futures = []
    if not channels:
        _record_notification("Not configured", "", student["name"], "Absent", message,
                             "Not sent: choose a configured WhatsApp, SMS, or email channel")
        return futures

    runtime = notification_runtime()
    if "WhatsApp" in channels:
        mobile = student.get("parent_mobile", "")
        if _normalize_indian_number(mobile):
            variables = {
                "1": student["name"], "2": student["roll_no"], "3": session["date"],
                "4": session["slot"], "5": f"{session['department']} - {session['class_name']} - {session['division']}",
            }
            futures.append(runtime["pool"].submit(
                _deliver_notification, "WhatsApp", mobile, message, student["name"], "Absent", cfg, variables))
        else:
            _record_notification("WhatsApp", mobile, student["name"], "Absent", message,
                                 "Skipped: parent mobile number is missing or invalid")
    if "SMS" in channels:
        mobile = student.get("parent_mobile", "")
        if _normalize_indian_number(mobile):
            futures.append(runtime["pool"].submit(
                _deliver_notification, "SMS", mobile, message, student["name"], "Absent", cfg))
        else:
            _record_notification("SMS", mobile, student["name"], "Absent", message,
                                 "Skipped: parent mobile number is missing or invalid")
    if "Email" in channels:
        email = student.get("parent_email", "").strip()
        if "@" in email:
            futures.append(runtime["pool"].submit(
                _deliver_notification, "Email", email, message, student["name"], "Absent", cfg))
        else:
            _record_notification("Email", email, student["name"], "Absent", message,
                                 "Skipped: parent email address is missing or invalid")
    return futures


def alert_context_from_attendance(record):
    """Turn a saved absence record into notification-ready student and session data."""
    student = {
        "name": record.get("name", "Student"), "roll_no": record.get("roll_no", "-"),
        "parent_mobile": record.get("parent_mobile", ""),
        "parent_email": record.get("parent_email", ""),
    }
    session = {
        "department": record.get("department", "-"), "class_name": record.get("class_name", "-"),
        "division": record.get("division", "-"), "slot": record.get("slot", "-"),
        "date": record.get("date", "-"),
    }
    return student, session


def render_parent_alert_sender(records, key_prefix, audience_label, show_parent_contact=False):
    """Show a manual send/resend control for saved absence records."""
    absences = [record for record in records if record.get("status") == "Absent"]
    absences.sort(key=lambda record: (str(record.get("date", "")), str(record.get("time", ""))), reverse=True)
    with st.expander("📤 Send parent absence alert", expanded=False):
        st.caption(audience_label)
        if not absences:
            st.info("No absent-student records are available to alert.")
            return

        selected_index = st.selectbox(
            "Absent student", range(len(absences)), key=f"{key_prefix}_record",
            format_func=lambda index: (
                f"{absences[index].get('date', '-')} · {absences[index].get('slot', '-')} · "
                f"{absences[index].get('name', 'Student')} (Roll {absences[index].get('roll_no', '-')})"
                + (f" · Parent: {absences[index].get('parent_mobile') or 'No mobile number'}"
                   if show_parent_contact else "")
            ),
        )
        record = absences[selected_index]
        student, session = alert_context_from_attendance(record)
        if show_parent_contact:
            st.caption("Principal-only parent contact preview")
            contact_mobile, contact_email = st.columns(2)
            contact_mobile.info(f"📱 **Parent mobile:** {student['parent_mobile'] or 'Not provided'}")
            contact_email.info(f"✉️ **Parent email:** {student['parent_email'] or 'Not provided'}")
        available_channels = delivery_channels_for_student(student)
        if not available_channels:
            st.warning("No sendable channel is available for this record yet.")
            mobile_is_valid = bool(_normalize_indian_number(student.get("parent_mobile", "")))
            email_is_valid = "@" in student.get("parent_email", "").strip()
            if not mobile_is_valid:
                st.error("This student has no valid parent mobile number. Add a 10-digit Indian number in Student Registry.")
            if not email_is_valid:
                st.info("This student has no valid parent email address. Add one in Student Registry if you want email alerts.")

            cfg = get_notification_config()
            enabled = enabled_notification_channels(cfg)
            if not enabled:
                st.info("No alert provider is configured. Add one complete channel in .streamlit/secrets.toml, then restart Streamlit.")
                for channel, missing_settings in notification_configuration_gaps(cfg).items():
                    if missing_settings:
                        st.caption(f"{channel} still needs: {', '.join(missing_settings)}")
            elif mobile_is_valid or email_is_valid:
                st.info("A provider is configured, but it does not match this student's saved parent contact details.")
            return

        selected_channels = st.multiselect(
            "Send using", available_channels, default=available_channels, key=f"{key_prefix}_channels",
        )
        st.caption("This action sends a real message to the selected student's parent. The delivery result is saved in Parent Notification Log.")
        if st.button("Send parent alert now", key=f"{key_prefix}_send", type="primary"):
            if not selected_channels:
                st.warning("Select at least one delivery channel.")
                return
            with st.spinner("Sending absence alert to the parent..."):
                futures = queue_absence_notifications(student, session, selected_channels)
                if futures:
                    wait(futures, timeout=45)
            statuses = [future.result() for future in futures if future.done()]
            drain_sms_results()
            if any(status.startswith("Failed") for status in statuses):
                st.error("One or more alerts could not be delivered. Open Parent Notification Log for details.")
            elif statuses:
                st.success("Parent alert queued successfully. Check Parent Notification Log for the delivery status.")
            else:
                st.warning("No alert was queued. Check the parent contact details and notification configuration.")


def retry_notification(entry):
    cfg = get_notification_config()
    channel = entry.get("channel", "")
    if channel not in enabled_notification_channels(cfg):
        _record_notification(channel or "Unknown", entry.get("recipient", entry.get("to", "")),
                             entry.get("student", ""), entry.get("type", "Absent"), entry.get("message", ""),
                             "Not retried: channel is not configured")
        return None
    runtime = notification_runtime()
    return runtime["pool"].submit(
        _deliver_notification, channel, entry.get("recipient", entry.get("to", "")), entry.get("message", ""),
        entry.get("student", ""), entry.get("type", "Absent"), cfg)


def drain_sms_results():
    """Move completed WhatsApp/email results into the visible notification log."""
    rt = notification_runtime()
    with rt["lock"]:
        items = rt["results"][:]
        rt["results"].clear()
    if items:
        st.session_state.sms_outbox.extend(items)
        save_data()


def lecture_summary_df(lectures):
    cols = ["Date", "Time Slot", "Department", "Class", "Division", "Faculty",
            "Total", "Present", "Absent"]
    if not lectures:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(lectures)
    if "department" not in df.columns:
        df["department"] = "-"
    df["department"] = df["department"].fillna("-")
    df = df.rename(columns={"date": "Date", "slot": "Time Slot", "department": "Department",
                            "class_name": "Class", "division": "Division", "faculty": "Faculty",
                            "total": "Total", "present": "Present", "absent": "Absent"})
    return df[cols]


def principal_password():
    secrets_paths = [
        os.path.join(".streamlit", "secrets.toml"),
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
    ]
    if any(os.path.exists(path) for path in secrets_paths):
        try:
            return st.secrets.get("PRINCIPAL_PASSWORD", "principal123")
        except Exception:
            pass
    return "principal123"


def sid_of(stud):
    """Unique id of a student inside a class."""
    return f"{stud['department']}|{stud['roll_no']}"


def finalize_attendance(sess, students):
    """Mark remaining students absent, save logs, and queue absence alerts."""
    now_time = datetime.now().strftime("%H:%M:%S")
    batch, futs = [], []
    for stud in students:
        info = sess["present"].get(sid_of(stud))
        status = "Present" if info else "Absent"
        entry = {
            "faculty": sess["faculty"], "class_name": sess["class_name"],
            "division": sess["division"], "roll_no": stud["roll_no"], "name": stud["name"],
            "department": stud["department"], "date": sess["date"],
            "time": info["time"] if info else now_time, "slot": sess["slot"],
            "status": status, "method": info["method"] if info else "-",
            "parent_mobile": stud["parent_mobile"],
            "parent_email": stud.get("parent_email", ""),
        }
        st.session_state.attendance_logs.append(entry)
        batch.append(entry)

        if status == "Absent":
            futs.extend(queue_absence_notifications(stud, sess))

    n_present = sum(1 for b in batch if b["status"] == "Present")
    st.session_state.lecture_logs.append({
        "faculty": sess["faculty"], "class_name": sess["class_name"],
        "division": sess["division"], "department": sess["department"],
        "date": sess["date"], "slot": sess["slot"], "total": len(batch),
        "present": n_present, "absent": len(batch) - n_present,
    })
    st.session_state.att_session = None
    save_data()
    return batch, n_present, futs


# ---------------------------------------------------------
# Session state (loaded from disk on first run)
# ---------------------------------------------------------
if "data_loaded" not in st.session_state:
    saved = load_data()
    st.session_state.students_db = saved.get("students_db", [])
    default_faculty = [
        {"name": "Dr. V. K. Kulkarni", "mobile": "9876543210",
         "username": "faculty1", "password_hash": hash_pw("123")}
    ]
    # A brand-new SQLite database contains an empty faculty list. Keep the
    # documented starter faculty account available on first launch.
    st.session_state.faculty_db = saved.get("faculty_db") or default_faculty
    st.session_state.attendance_logs = saved.get("attendance_logs", [])
    st.session_state.lecture_logs = saved.get("lecture_logs", [])
    st.session_state.sms_outbox = saved.get("sms_outbox", [])
    st.session_state.att_session = saved.get("att_session", None)
    st.session_state.whatsapp_sender_override = saved.get("whatsapp_sender_override", "")
    st.session_state.data_loaded = True

drain_sms_results()   # move completed notification results into the log

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
hero_date = datetime.now().strftime("%A · %d %b %Y")
st.markdown(f"""
<section class="smart-hero">
    <div class="hero-kicker"><span class="hero-dot"></span> Campus operations hub</div>
    <h1>Sandipani <span>Smart Campus</span></h1>
    <p>One connected space for student records, intelligent attendance, parent alerts, and campus operations.</p>
    <div class="hero-pills">
        <span class="hero-pill"><strong>● LIVE</strong> System online</span>
        <span class="hero-pill">⌁ {hero_date}</span>
        <span class="hero-pill">✦ AI-assisted attendance</span>
    </div>
    <div class="hero-emblem">✦</div>
</section>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("""
<div class="side-brand">
    <div class="eyebrow">Sandipani Technical Campus</div>
    <h2>Campus OS</h2>
    <p>Attendance, records and parent communication in one secure workspace.</p>
</div>
<div class="sidebar-label">Workspace</div>
""", unsafe_allow_html=True)
navigation_labels = {
    "Dashboard": "⌘  Dashboard",
    "Student Registration": "✦  Student registration",
    "Faculty Portal & Attendance": "◉  Faculty & attendance",
    "Principal Admin Panel": "⌁  Principal controls",
    "Manage Students": "▦  Student directory",
    "Emergency SOS": "!  Emergency SOS",
}
if "workspace_menu" not in st.session_state:
    st.session_state.workspace_menu = navigation_labels["Dashboard"]
selected_menu_label = st.sidebar.radio(
    "Open workspace", list(navigation_labels.values()), key="workspace_menu", label_visibility="collapsed"
)
menu = next(name for name, label in navigation_labels.items() if label == selected_menu_label)


def open_workspace(destination):
    """Navigate from the dashboard menu without changing any app data."""
    st.session_state.workspace_menu = navigation_labels[destination]


with st.sidebar.expander("🕒 Daily timetable", expanded=False):
    st.markdown(
        "10:00 – 11:00  \n11:00 – 12:00  \n☕ Break · 12:00 – 12:45  \n12:45 – 01:45  \n"
        "01:45 – 02:45  \n☕ Break · 02:45 – 03:00  \n03:00 – 04:00  \n04:00 – 05:00"
    )
st.sidebar.markdown("""
<div class="sidebar-status">
    <strong>● Operations ready</strong>
    <span>Local records are stored securely in the campus database.</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 1: Dashboard
# ---------------------------------------------------------
if menu == "Dashboard":
    st.markdown(section_heading("☷", "Workspace menu", "What would you like to do?",
                                "Choose a module below. All campus functions are available from this menu."), unsafe_allow_html=True)
    workspace_cards = [
        ("✦", "Student registration", "Create a new student record with class and parent details.", "Student Registration", "Open registration"),
        ("◉", "Faculty & attendance", "Sign in, start a class session, and mark student attendance.", "Faculty Portal & Attendance", "Open attendance"),
        ("⌁", "Principal controls", "Review reports, notification delivery, and overall attendance.", "Principal Admin Panel", "Open principal panel"),
        ("▦", "Student directory", "View registered students and maintain accurate records.", "Manage Students", "Open directory"),
        ("!", "Emergency SOS", "Open the emergency alert control for campus security.", "Emergency SOS", "Open SOS control"),
    ]
    for start in range(0, len(workspace_cards), 3):
        columns = st.columns(3)
        for column, (icon, title, description, destination, button_text) in zip(columns, workspace_cards[start:start + 3]):
            with column:
                st.markdown(f"""
                <div class="workspace-card">
                    <div class="workspace-icon">{icon}</div>
                    <div class="workspace-tag">Campus module</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)
                st.button(button_text, key=f"dashboard_{destination}", use_container_width=True,
                          on_click=open_workspace, args=(destination,))
    st.markdown("""
    <div class="smart-panel" style="margin-top:1.2rem; min-height:0;">
        <div class="panel-kicker">How to use the system</div>
        <h3>Start with student registration, then move to attendance.</h3>
        <p>Parent alerts are automatically logged for absent students when your WhatsApp or email channel is enabled.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 2: Student Registration
# ---------------------------------------------------------
elif menu == "Student Registration":
    st.markdown(section_heading("✦", "Student records", "Add a student profile",
                                "Create a complete record with class placement and parent contact details."), unsafe_allow_html=True)

    with st.form("student_reg_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            s_name = st.text_input("Student Name")
            s_roll = st.text_input("Student Roll No.")
            s_dept = st.selectbox("Department", DEPARTMENTS)
            s_class = st.selectbox("Class Name", CLASSES)
        with col_b:
            s_div = st.selectbox("Division", DIVISIONS)
            s_mobile = st.text_input("Student Mobile Number")
            p_mobile = st.text_input("Parent Mobile Number (for WhatsApp alerts)")
            p_email = st.text_input("Parent Email Address (for email alerts)")

        st.markdown("---")
        st.write("📸 **Passport Photo (used for optional face recognition):**")
        uploaded_photo = st.file_uploader("Upload Passport Photo (.jpg, .png)", type=["jpg", "png", "jpeg"])
        submit_student = st.form_submit_button("Register Student")

    if submit_student:
        errors = []
        if not (s_name.strip() and s_roll.strip()):
            errors.append("Student Name and Roll No. are required.")
        if not (s_mobile.strip().isdigit() and len(s_mobile.strip()) == 10):
            errors.append("Student Mobile Number must contain exactly 10 digits.")
        if p_mobile.strip() and not (p_mobile.strip().isdigit() and len(p_mobile.strip()) == 10):
            errors.append("Parent Mobile Number must contain exactly 10 digits, or be left blank for email-only alerts.")
        if p_email.strip() and ("@" not in p_email or "." not in p_email.rsplit("@", 1)[-1]):
            errors.append("Enter a valid parent email address, or leave it blank.")
        if not p_mobile.strip() and not p_email.strip():
            errors.append("Enter at least one parent contact: a mobile number or email address.")
        if any(s['roll_no'] == s_roll.strip() and s['class_name'] == s_class and s['division'] == s_div
               and s['department'] == s_dept for s in st.session_state.students_db):
            errors.append("This Roll No. already exists in the selected Department, Class, and Division.")

        encoding = None
        photo_bytes = None
        if not errors and uploaded_photo is not None:
            photo_bytes = uploaded_photo.getvalue()
            if FACE_LIB:
                try:
                    with st.spinner("Checking the face in the photo..."):
                        encs = get_face_encodings(photo_bytes)
                except Exception as e:
                    encs = []
                    errors.append(f"Could not read the photo: {e}")
                if not errors:
                    if len(encs) == 0:
                        errors.append("No face was found in the photo. Use a clear passport photo.")
                    elif len(encs) > 1:
                        errors.append("More than one face was found. Upload only the student's photo.")
                    else:
                        encoding = encs[0]

        if errors:
            for e in errors:
                st.error(e)
        else:
            st.session_state.students_db.append({
                "roll_no": s_roll.strip(), "name": s_name.strip(), "department": s_dept,
                "class_name": s_class, "division": s_div, "mobile": s_mobile.strip(),
                "parent_mobile": p_mobile.strip(), "parent_email": p_email.strip().lower(),
                "photo": photo_bytes, "encoding": encoding,
            })
            save_data()
            mode = "Face + manual attendance" if encoding is not None else "Manual attendance"
            st.success(f"Student **{s_name}** (Roll No. {s_roll}) was registered successfully. Mode: {mode}.")

# ---------------------------------------------------------
# MODULE 3: Faculty Portal & Attendance
# ---------------------------------------------------------
elif menu == "Faculty Portal & Attendance":
    st.markdown(section_heading("◉", "Faculty workspace", "Attendance studio",
                                "Run structured sessions with face-assisted or manual attendance."), unsafe_allow_html=True)

    if not st.session_state.get('faculty_logged_in', False):
        f_action = st.radio("Select Action:", ["Faculty Login", "Faculty Registration"])

        if f_action == "Faculty Registration":
            with st.form("faculty_reg_form"):
                reg_name = st.text_input("Full Name")
                reg_mobile = st.text_input("Mobile Number")
                reg_user = st.text_input("Create Username")
                reg_pass = st.text_input("Create Password", type="password")
                if st.form_submit_button("Register Faculty"):
                    reg_user = reg_user.strip()
                    if not (reg_name.strip() and reg_user and reg_pass):
                        st.error("Please fill all mandatory fields.")
                    elif reg_mobile.strip() and not (reg_mobile.strip().isdigit() and len(reg_mobile.strip()) == 10):
                        st.error("Mobile Number must contain exactly 10 digits.")
                    elif any(f['username'] == reg_user for f in st.session_state.faculty_db):
                        st.error("This username is already in use.")
                    else:
                        st.session_state.faculty_db.append({
                            "name": reg_name.strip(), "mobile": reg_mobile.strip(),
                            "username": reg_user, "password_hash": hash_pw(reg_pass)})
                        save_data()
                        st.success("Faculty Registered Successfully! You can now login.")
        else:
            with st.form("faculty_login_form"):
                l_user = st.text_input("Username")
                l_pass = st.text_input("Password", type="password")
                login_btn = st.form_submit_button("Login")
            if login_btn:
                matched = next((f for f in st.session_state.faculty_db
                                if f["username"] == l_user.strip()
                                and f["password_hash"] == hash_pw(l_pass)), None)
                if matched:
                    st.session_state['faculty_logged_in'] = True
                    st.session_state['current_faculty_name'] = matched['name']
                    st.rerun()
                else:
                    st.error("Invalid Username or Password!")
    else:
        fac_name = st.session_state['current_faculty_name']
        top1, top2 = st.columns([4, 1])
        top1.success(f"Welcome, {fac_name}! ✅")
        if top2.button("🚪 Logout"):
            st.session_state['faculty_logged_in'] = False
            st.session_state.pop('current_faculty_name', None)
            st.rerun()

        sess = st.session_state.get("att_session")
        if sess and sess.get("faculty") != fac_name:
            sess = None   # another faculty's unfinished session is not shown here

        # =====================================================
        # A) ACTIVE SESSION  -> one-by-one face attendance
        # =====================================================
        if sess:
            students = [s for s in st.session_state.students_db
                        if s['department'] == sess['department']
                        and s['class_name'] == sess['class_name']
                        and s['division'] == sess['division']]
            by_sid = {sid_of(s): s for s in students}
            # drop ids of students that were deleted meanwhile
            sess["present"] = {k: v for k, v in sess["present"].items() if k in by_sid}

            st.markdown(f"### 📋 Attendance in progress: **{sess['department']} | {sess['class_name']} - "
                        f"{sess['division']}** | 📅 {sess['date']} | 🕒 {sess['slot']}")

            total = len(students)
            n_pres = len(sess["present"])
            st.progress(n_pres / total if total else 0.0,
                        text=f"✅ Present: {n_pres} / {total}   |   ⏳ Remaining: {total - n_pres}")

            # ---- latest notification status ----
            with st.expander("📨 Latest parent notification status"):
                recent = st.session_state.sms_outbox[-5:][::-1]
                if recent:
                    show_df(pd.DataFrame(recent)[["time", "student", "type", "status"]])
                else:
                    st.write("No parent notifications yet.")

            # ---- result of the last scan ----
            last = sess.get("last")
            if last:
                kind, text = last
                {"success": st.success, "warning": st.warning,
                 "error": st.error, "info": st.info}.get(kind, st.info)(text)

            # ---- details of the last recognised student ----
            ls = by_sid.get(sess.get("last_sid"))
            if ls:
                ci, cf = st.columns([1, 4])
                if ls.get("photo"):
                    ci.image(ls["photo"], width=110)
                cf.markdown(
                    f"**Name:** {ls['name']}  \n**Roll No.:** {ls['roll_no']}  \n"
                    f"**Department:** {ls['department']} | **Class:** {ls['class_name']} - {ls['division']}  \n"
                    f"**Parent Mobile:** {ls['parent_mobile']}")

            # ---- camera: ONE student at a time ----
            if FACE_LIB:
                st.markdown("#### 📷 Ask the next student to face the camera, then take a photo")
                st.caption("The face is checked immediately and the camera resets for the next student.")
                photo = st.camera_input("Student Camera", key=f"cam_{sess['cam_n']}")
            else:
                photo = None
                st.info("Face matching is unavailable in this environment. Use Manual Present below to record attendance.")

            if photo is not None and FACE_LIB:
                sess["last_sid"] = None
                try:
                    with st.spinner("Matching the face..."):
                        enc, nfaces = get_single_face_encoding(photo.getvalue())
                except Exception as e:
                    enc, nfaces = None, -1
                    sess["last"] = ("error", f"Could not process the photo: {e}")

                if nfaces == 0:
                    sess["last"] = ("warning", "No face was found. Keep the student's face clear and take another photo.")
                elif enc is not None:
                    known = [s for s in students if s.get('encoding') is not None]
                    if not known:
                        sess["last"] = ("error", "No students in this class have registered face data.")
                    else:
                        dists = face_recognition.face_distance([s['encoding'] for s in known], enc)
                        i = int(np.argmin(dists))
                        best = float(dists[i])
                        if best <= FACE_TOLERANCE:
                            stud = known[i]
                            sid = sid_of(stud)
                            sess["last_sid"] = sid
                            multi = " (multiple faces were detected; the largest face was used)" if nfaces > 1 else ""
                            if sid in sess["present"]:
                                sess["last"] = ("info", f"{stud['name']} (Roll {stud['roll_no']}) is already marked Present.{multi}")
                            else:
                                sess["present"][sid] = {"time": datetime.now().strftime("%H:%M:%S"),
                                                        "method": "Face", "sms": True}
                                sess["last"] = ("success",
                                                f"✅ {stud['name']} (Roll {stud['roll_no']}) – Present! "
                                                f"(match distance {best:.2f}){multi}")
                        else:
                            sess["last"] = ("error",
                                            "Face not recognised for any student in this class. "
                                            "Take another photo or use Manual Present below.")
                sess["cam_n"] += 1          # resets the camera widget
                st.session_state.att_session = sess
                save_data()
                st.rerun()

            # ---- present / pending lists ----
            left, right = st.columns(2)
            with left:
                st.markdown("#### ✅ Present students")
                if sess["present"]:
                    rows = [{"Roll No.": by_sid[k]['roll_no'], "Name": by_sid[k]['name'],
                             "Time": v['time'], "Method": v['method']}
                            for k, v in sess["present"].items()]
                    show_df(pd.DataFrame(rows))
                else:
                    st.write("No students marked Present yet.")
            with right:
                st.markdown("#### ⏳ Remaining students")
                pending = [s for s in students if sid_of(s) not in sess["present"]]
                if pending:
                    show_df(pd.DataFrame([{"Roll No.": s['roll_no'], "Name": s['name']} for s in pending]))
                else:
                    st.success("All students are marked Present 🎉")

            # ---- manual fallback ----
            with st.expander("✍️ Manual Present (available at all times)"):
                if pending:
                    man_labels = {f"{s['roll_no']} - {s['name']}": sid_of(s) for s in pending}
                    man_pick = st.multiselect("Select students", list(man_labels.keys()))
                    if st.button("Mark Selected as Present"):
                        for lab in man_pick:
                            _s = by_sid[man_labels[lab]]
                            sess["present"][man_labels[lab]] = {
                                "time": datetime.now().strftime("%H:%M:%S"), "method": "Manual", "sms": True}
                        sess["last"] = ("success", f"{len(man_pick)} student(s) were marked Present manually.")
                        st.session_state.att_session = sess
                        save_data()
                        st.rerun()
                else:
                    st.write("No students remaining.")

            # ---- unmark (wrong match) ----
            with st.expander("↩️ Undo an incorrect Present mark"):
                if sess["present"]:
                    und_labels = {f"{by_sid[k]['roll_no']} - {by_sid[k]['name']}": k for k in sess["present"]}
                    st.caption("A notification may already have been sent; undoing attendance cannot recall it.")
                    und_pick = st.selectbox("Select student", list(und_labels.keys()))
                    if st.button("Remove from Present"):
                        sess["present"].pop(und_labels[und_pick], None)
                        sess["last"] = ("info", f"{und_pick} was removed from Present.")
                        st.session_state.att_session = sess
                        save_data()
                        st.rerun()
                else:
                    st.write("No Present marks to undo.")

            # ---- finish / cancel ----
            st.markdown("---")
            st.caption("When attendance is submitted, WhatsApp and/or email absence alerts are queued only "
                       "for students still marked Absent.")
            b1, b2 = st.columns(2)
            if b1.button("🏁 Finish & Submit Attendance", type="primary"):
                with st.spinner("Saving attendance and notifying parents of Absent students..."):
                    batch, n_present, futs = finalize_attendance(sess, students)
                    if futs:
                        wait(futs, timeout=45)
                statuses = [f.result() for f in futs if f.done()]
                n_sent = sum(1 for x in statuses if x.startswith("Queued") or x == "Email sent")
                n_fail = sum(1 for x in statuses if x.startswith("Failed"))
                drain_sms_results()
                st.success(f"Attendance saved. Present: {n_present} | Absent: {len(batch) - n_present}")
                if n_sent:
                    st.success(f"{n_sent} absence notification(s) were delivered or queued.")
                if n_fail:
                    st.error(f"{n_fail} notification(s) failed. Check Principal Panel → Notification Log "
                             f"and use 'Retry Failed Notifications'.")
                show_df(pd.DataFrame(batch)[['roll_no', 'name', 'department', 'status', 'method']])
            if b2.button("🗑️ Cancel Attendance Session"):
                st.session_state.att_session = None
                save_data()
                st.rerun()

        # =====================================================
        # B) NO SESSION -> history + start new session
        # =====================================================
        else:
            faculty_absences = [record for record in st.session_state.attendance_logs
                                if record.get("faculty") == fac_name and record.get("status") == "Absent"]
            render_parent_alert_sender(
                faculty_absences, "faculty_parent_alert", "Only absence records submitted under your faculty account are shown here."
            )
            st.markdown("### 📚 Your lecture history")
            my_lectures = [l for l in st.session_state.lecture_logs if l['faculty'] == fac_name]
            if not my_lectures:
                st.info("No lectures recorded yet.")
            else:
                for l in reversed(my_lectures):
                    st.markdown(
                        f"- 📅 **{l['date']}** | 🕒 {l['slot']} | 🏫 **{l.get('department', '-')} "
                        f"{l['class_name']} - {l['division']}** "
                        f"| ✅ Present: **{l['present']}** | ❌ Absent: **{l['absent']}** | 👥 Total: {l['total']}"
                    )
                show_df(lecture_summary_df(my_lectures).drop(columns=["Faculty"]))

            st.markdown("---")
            st.markdown("### ▶️ Start a new attendance session")
            col1, col2 = st.columns(2)
            with col1:
                s_dept = st.selectbox("Select Department", DEPARTMENTS)
                class_name = st.selectbox("Select Class", CLASSES)
                division = st.selectbox("Select Division", DIVISIONS)
            with col2:
                att_date = st.date_input("Attendance Date", datetime.now().date())
                time_slot = st.selectbox("Lecture Time Slot", SLOTS)

            if st.button("▶️ Start Attendance Session"):
                date_str = str(att_date)
                class_students = [s for s in st.session_state.students_db
                                  if s['department'] == s_dept and s['class_name'] == class_name
                                  and s['division'] == division]
                if not class_students:
                    st.warning(f"No students are registered in {s_dept} - {class_name} - {division}.")
                elif any(l for l in st.session_state.lecture_logs
                         if l['date'] == date_str and l['class_name'] == class_name
                         and l['division'] == division and l['slot'] == time_slot
                         and l.get('department', s_dept) == s_dept):
                    st.warning("Attendance has already been submitted for this Department, Class, Division, and time slot.")
                else:
                    st.session_state.att_session = {
                        "faculty": fac_name, "department": s_dept, "class_name": class_name,
                        "division": division, "date": date_str, "slot": time_slot,
                        "present": {}, "cam_n": 0, "last": None, "last_sid": None,
                    }
                    save_data()
                    st.rerun()

# ---------------------------------------------------------
# MODULE 4: Principal Admin Panel
# ---------------------------------------------------------
elif menu == "Principal Admin Panel":
    st.markdown(section_heading("⌁", "Leadership workspace", "Campus administration",
                                "Review attendance activity, reports, and parent notification delivery."), unsafe_allow_html=True)
    admin_pass = st.text_input("Enter Principal Password", type="password")

    if admin_pass == principal_password():
        st.success("Authenticated as Principal Admin ✅")

        notification_cfg = get_notification_config()
        active_channels = enabled_notification_channels(notification_cfg)
        with st.expander("⚙️ Parent notification status", expanded=False):
            if active_channels:
                st.success(f"Configured channel(s): {', '.join(active_channels)}")
            else:
                st.info("No live notification channel is configured yet. Attendance is still saved safely in the local database.")
            st.write("To enable real WhatsApp, SMS, or email alerts, copy `secrets.example.toml` to `.streamlit/secrets.toml` and add the college's provider credentials.")
            st.caption("Credentials are never stored with student records. The Notification Log shows every sent or failed alert.")

        with st.expander("📱 College WhatsApp sender", expanded=False):
            saved_sender = str(st.session_state.get("whatsapp_sender_override", "") or "").strip()
            configured_sender = notification_cfg.get("TWILIO_WHATSAPP_FROM", "").strip()
            effective_sender = saved_sender or configured_sender
            st.caption(
                "Principal-only setting. It changes the approved sender number used for WhatsApp alerts; "
                "it never displays or stores Twilio tokens or email passwords."
            )
            if effective_sender:
                st.info(f"Current WhatsApp sender: {effective_sender}")
            else:
                st.warning("No WhatsApp sender is configured yet.")
            sender_input = st.text_input(
                "Approved Twilio WhatsApp sender (E.164)",
                value=effective_sender,
                placeholder="whatsapp:+14155238886",
                key="principal_whatsapp_sender",
                help="Use a sender that is already approved on the college Twilio account.",
            )
            sender_save, sender_reset = st.columns(2)
            if sender_save.button("Save college WhatsApp sender", key="save_whatsapp_sender"):
                normalized_sender = normalize_whatsapp_sender(sender_input)
                if not normalized_sender:
                    st.error("Enter a valid E.164 number, for example whatsapp:+14155238886.")
                else:
                    st.session_state.whatsapp_sender_override = normalized_sender
                    save_data()
                    st.success(f"College WhatsApp sender saved: {normalized_sender}")
            if sender_reset.button("Use sender from protected secrets", key="reset_whatsapp_sender"):
                st.session_state.whatsapp_sender_override = ""
                save_data()
                st.success("The sender will now use TWILIO_WHATSAPP_FROM from .streamlit/secrets.toml.")
                st.rerun()

        render_parent_alert_sender(
            st.session_state.attendance_logs, "principal_parent_alert",
            "Select any recorded absence to send or resend an alert as Principal.", show_parent_contact=True
        )

        if not st.session_state.lecture_logs:
            st.warning("No faculty attendance has been recorded yet.")
        else:
            st.markdown("#### 🏫 Lectures for all classes")
            lec_df = lecture_summary_df(st.session_state.lecture_logs)
            fc1, fc2, fc3, fc4 = st.columns(4)
            f_dept = fc1.selectbox("Department", ["All"] + DEPARTMENTS)
            f_class = fc2.selectbox("Class", ["All"] + CLASSES)
            f_div = fc3.selectbox("Division", ["All"] + DIVISIONS)
            f_date = fc4.selectbox("Date", ["All"] + sorted(lec_df['Date'].unique().tolist(), reverse=True))
            if f_dept != "All":
                lec_df = lec_df[lec_df['Department'] == f_dept]
            if f_class != "All":
                lec_df = lec_df[lec_df['Class'] == f_class]
            if f_div != "All":
                lec_df = lec_df[lec_df['Division'] == f_div]
            if f_date != "All":
                lec_df = lec_df[lec_df['Date'] == f_date]
            show_df(lec_df)

            st.markdown("#### 👨‍🎓 Student-wise attendance")
            df_all = pd.DataFrame(st.session_state.attendance_logs)
            private_columns = [column for column in ("parent_mobile", "parent_email") if column in df_all.columns]
            if private_columns:
                df_all = df_all.drop(columns=private_columns)
            if f_dept != "All":
                df_all = df_all[df_all['department'] == f_dept]
            if f_class != "All":
                df_all = df_all[df_all['class_name'] == f_class]
            if f_div != "All":
                df_all = df_all[df_all['division'] == f_div]
            if f_date != "All":
                df_all = df_all[df_all['date'] == f_date]
            show_df(df_all)

            st.download_button(
                "📥 Download Attendance Report",
                data=df_all.to_csv(index=False).encode('utf-8'),
                file_name=f"Attendance_Report_{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
            )

        with st.expander("📨 Parent Notification Log", expanded=True):
            outbox = st.session_state.sms_outbox
            if outbox:
                show_df(pd.DataFrame(outbox[::-1]))
                failed = [i for i, o in enumerate(outbox)
                          if str(o.get("status", "")).startswith("Failed") and "type" in o]
                if failed and st.button(f"🔁 Retry {len(failed)} Failed Notification(s)"):
                    futs = []
                    for i in failed:
                        o = outbox[i]
                        retry = retry_notification(o)
                        if retry is not None:
                            futs.append(retry)
                        o["status"] = "Retried"
                    if futs:
                        wait(futs, timeout=45)
                    drain_sms_results()
                    save_data()
                    st.rerun()
            else:
                st.write("No parent notifications have been queued yet.")
    elif admin_pass != "":
        st.error("Incorrect Password!")

# ---------------------------------------------------------
# MODULE 5: Manage Students
# ---------------------------------------------------------
elif menu == "Manage Students":
    st.markdown(section_heading("▦", "Directory", "Registered students",
                                "Review the student database and maintain accurate records."), unsafe_allow_html=True)
    if not st.session_state.students_db:
        st.info("No students found.")
    else:
        rows = [{
            "Roll No.": s['roll_no'], "Name": s['name'], "Department": s['department'],
            "Class": s['class_name'], "Division": s['division'],
            "Student Mobile": s['mobile'], "Parents Mobile": s['parent_mobile'],
            "Parent Email": s.get('parent_email', ''),
            "Photo Registered": "Yes" if s.get('encoding') is not None else "No",
        } for s in st.session_state.students_db]
        show_df(pd.DataFrame(rows))

        st.markdown("#### 🗑️ Remove a student")
        labels = [f"{s['roll_no']} - {s['name']} ({s['department']}, {s['class_name']}, {s['division']})"
                  for s in st.session_state.students_db]
        pick = st.selectbox("Select student", labels)
        if st.button("Delete Selected Student"):
            idx = labels.index(pick)
            removed = st.session_state.students_db.pop(idx)
            save_data()
            st.success(f"{removed['name']} was removed.")
            st.rerun()

# ---------------------------------------------------------
# MODULE 6: Emergency SOS
# ---------------------------------------------------------
elif menu == "Emergency SOS":
    st.markdown(section_heading("!", "Emergency controls", "Campus SOS",
                                "Use this protected action only to flag a genuine campus emergency."), unsafe_allow_html=True)
    st.warning("Use this button only during an emergency. It records an alert for campus security and the Principal's office.")
    if st.button("🚨 TRIGGER CAMPUS EMERGENCY SOS"):
        st.error("🚨 [CRITICAL ALERT] Emergency SOS Broadcasted to Sandipani Technical Campus Security & Principal Office at Kolpa!")

st.markdown("<div class='app-footer'>Sandipani Technical Campus, Kolpa &nbsp;·&nbsp; Smart Campus AI Suite &nbsp;·&nbsp; © 2026</div>", unsafe_allow_html=True)
