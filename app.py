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
    :root { --navy: #0f2b5b; --blue: #2563eb; --sky: #eaf2ff; --ink: #172033; }
    .stApp {
        background: radial-gradient(circle at 5% 0%, #eaf2ff 0, transparent 28rem),
                    linear-gradient(135deg, #f8fbff 0%, #edf4ff 48%, #f8fafc 100%);
        color: var(--ink);
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0c2550 0%, #123a78 100%); }
    [data-testid="stSidebar"] * { color: #f8fbff; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #172033; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
    [data-testid="stSidebar"] .stMarkdown p { color: rgba(248,251,255,.82); }
    .block-container { max-width: 1360px; padding-top: 2.4rem; padding-bottom: 2rem; }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animated-header {
        animation: fadeInDown 0.8s ease-out;
        background: linear-gradient(120deg, #0c2450 0%, #1d4f9c 56%, #3b82f6 100%);
        padding: 30px; border-radius: 22px; color: white; text-align: center;
        box-shadow: 0 16px 38px rgba(24, 65, 134, 0.24);
        border: 1px solid rgba(255,255,255,.18);
    }
    .animated-header h1 { margin: 0; letter-spacing: -.8px; font-size: clamp(1.75rem, 3vw, 2.5rem); }
    .animated-header h3 { margin: .6rem 0 0; opacity: .88; font-weight: 500; }
    .metric-card {
        background: rgba(255, 255, 255, 0.94); backdrop-filter: blur(10px);
        padding: 22px; min-height: 128px; border-radius: 16px;
        box-shadow: 0 8px 24px rgba(20, 52, 100, 0.09);
        border: 1px solid #dbeafe; border-left: 6px solid #2563eb; transition: all 0.3s ease;
    }
    .metric-card h3 { margin: 0; color: #52647e; font-size: .96rem; font-weight: 600; }
    .metric-card h2 { margin: .55rem 0 0; color: #102b58; font-size: 2rem; }
    .metric-card:hover { transform: translateY(-6px); box-shadow: 0 16px 30px rgba(20,52,100,.15); }
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px; font-weight: 650; border: 0;
        box-shadow: 0 4px 10px rgba(37,99,235,.16);
    }
    div[data-testid="stExpander"] { background: rgba(255,255,255,.76); border: 1px solid #dbeafe; border-radius: 12px; }
    div[data-testid="stDataFrame"] { border: 1px solid #dbeafe; border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def hash_pw(p):
    return hashlib.sha256(p.encode("utf-8")).hexdigest()


PERSIST_KEYS = ["students_db", "faculty_db", "attendance_logs", "lecture_logs",
                "sms_outbox", "att_session"]


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
            return {
                "students_db": students, "faculty_db": faculty, "attendance_logs": attendance,
                "lecture_logs": lectures, "sms_outbox": notifications, "att_session": att_session,
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

            conn.execute("DELETE FROM app_state WHERE key = 'att_session'")
            session = data.get("att_session")
            if session is not None:
                conn.execute("INSERT INTO app_state (key, value) VALUES ('att_session', ?)", (self._blob(session),))


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
# Parent notification engine: WhatsApp through Twilio and email through SMTP.
# No messages are sent unless the corresponding credentials are configured.
# ---------------------------------------------------------
@st.cache_resource
def notification_runtime():
    """Shared worker pool + result box (survives Streamlit reruns)."""
    return {"pool": ThreadPoolExecutor(max_workers=8), "results": [], "lock": threading.Lock()}


NOTIFICATION_KEYS = [
    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM",
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
    return cfg


def enabled_notification_channels(cfg=None):
    cfg = cfg or get_notification_config()
    channels = []
    if all(cfg[key].strip() for key in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM")):
        channels.append("WhatsApp")
    if all(cfg[key].strip() for key in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL")):
        channels.append("Email")
    return channels


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


def _deliver_notification(channel, recipient, message, student_name, kind, cfg, template_variables=None):
    """Deliver one WhatsApp or email notification in a background worker."""
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


def queue_absence_notifications(student, session):
    """Queue absence alerts only. Returns background worker futures."""
    cfg = get_notification_config()
    message = build_absence_notification(student, session)
    channels = enabled_notification_channels(cfg)
    futures = []
    if not channels:
        _record_notification("Not configured", "", student["name"], "Absent", message,
                             "Not sent: configure WhatsApp or email credentials")
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
    if "Email" in channels:
        email = student.get("parent_email", "").strip()
        if "@" in email:
            futures.append(runtime["pool"].submit(
                _deliver_notification, "Email", email, message, student["name"], "Absent", cfg))
        else:
            _record_notification("Email", email, student["name"], "Absent", message,
                                 "Skipped: parent email address is missing or invalid")
    return futures


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
    st.session_state.faculty_db = saved.get("faculty_db", [
        {"name": "Dr. V. K. Kulkarni", "mobile": "9876543210",
         "username": "faculty1", "password_hash": hash_pw("123")}
    ])
    st.session_state.attendance_logs = saved.get("attendance_logs", [])
    st.session_state.lecture_logs = saved.get("lecture_logs", [])
    st.session_state.sms_outbox = saved.get("sms_outbox", [])
    st.session_state.att_session = saved.get("att_session", None)
    st.session_state.data_loaded = True

drain_sms_results()   # move completed notification results into the log

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="animated-header">
    <h1>🎓 Sandipani Technical Campus, Kolpa</h1>
    <h3>Smart Campus AI Attendance & Parent Notification System</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("🧭 Navigation Panel")
menu = st.sidebar.selectbox("Choose Module:", [
    "Dashboard",
    "Student Registration",
    "Faculty Portal & Attendance",
    "Principal Admin Panel",
    "Manage Students",
    "Emergency SOS",
])
st.sidebar.markdown("**🕒 College Timetable**")
st.sidebar.markdown(
    "- 10:00 – 11:00\n- 11:00 – 12:00\n- ☕ Break 12:00 – 12:45\n- 12:45 – 01:45\n"
    "- 01:45 – 02:45\n- ☕ Break 02:45 – 03:00\n- 03:00 – 04:00\n- 04:00 – 05:00"
)

# ---------------------------------------------------------
# MODULE 1: Dashboard
# ---------------------------------------------------------
if menu == "Dashboard":
    st.subheader("📊 Campus Real-Time Analytics Overview")
    today = str(datetime.now().date())
    today_logs = [l for l in st.session_state.attendance_logs if l['date'] == today]
    if today_logs:
        pct = 100 * sum(1 for l in today_logs if l['status'] == "Present") / len(today_logs)
        today_txt = f"{pct:.1f}%"
    else:
        today_txt = "N/A"

    lectures_today = sum(1 for l in st.session_state.lecture_logs if l["date"] == today)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h3>Total Students</h3><h2>{len(st.session_state.students_db)}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h3>Registered Faculty</h3><h2>{len(st.session_state.faculty_db)}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h3>Today Attendance</h3><h2>{today_txt}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h3>Lectures Today</h3><h2>{lectures_today}</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Attendance is taken one student at a time. The system compares each camera photo "
            "with the registered photo, marks a matching student Present, and marks remaining "
            "students Absent when the session is submitted.")

    _channels = enabled_notification_channels()
    if not _channels:
        st.warning("Parent notifications are in demo mode. Add WhatsApp (Twilio) or email (SMTP) credentials "
                   "to Streamlit secrets to send real absence alerts.")
    else:
        st.success(f"Active parent notification channel(s): **{', '.join(_channels)}**")

    if FACE_LIB:
        st.success("Face recognition is ready.")
    else:
        st.info("Face recognition is not installed. The app remains fully usable with Manual Attendance; "
                "photo matching is temporarily unavailable.")

# ---------------------------------------------------------
# MODULE 2: Student Registration
# ---------------------------------------------------------
elif menu == "Student Registration":
    st.subheader("📝 New Student Registration")

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
    st.subheader("👨‍🏫 Faculty Portal & AI Smart Attendance")

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
    st.subheader("🏛️ Principal Admin Panel (All Classes Attendance)")
    admin_pass = st.text_input("Enter Principal Password", type="password")

    if admin_pass == principal_password():
        st.success("Authenticated as Principal Admin ✅")

        notification_cfg = get_notification_config()
        active_channels = enabled_notification_channels(notification_cfg)
        with st.expander("⚙️ Parent notification setup", expanded=not active_channels):
            if active_channels:
                st.success(f"Configured channel(s): {', '.join(active_channels)}")
            else:
                st.warning("No live notification channel is configured. Attendance is still saved, but absence alerts remain in the Notification Log.")
            st.markdown("**WhatsApp:** Twilio WhatsApp API. **Email:** any SMTP provider, such as Gmail with an App Password.")
            st.caption("A Twilio Content Template is required for proactive WhatsApp alerts outside the 24-hour WhatsApp service window.")
            st.markdown("For local use, copy `secrets.example.toml` to `.streamlit/secrets.toml`. "
                        "For Streamlit Community Cloud, paste the same values into **App settings → Secrets**. "
                        "Credentials are never stored in the student database.")
            st.code(
                'TWILIO_ACCOUNT_SID = "your-twilio-account-sid"\n'
                'TWILIO_AUTH_TOKEN = "your-twilio-auth-token"\n'
                'TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"\n\n'
                'TWILIO_CONTENT_SID = "your-approved-absence-template-id"\n\n'
                'SMTP_HOST = "smtp.gmail.com"\n'
                'SMTP_PORT = "587"\n'
                'SMTP_USERNAME = "college@example.com"\n'
                'SMTP_PASSWORD = "your-email-app-password"\n'
                'SMTP_FROM_EMAIL = "college@example.com"', language="toml")

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
    st.subheader("👨‍🎓 Registered Students")
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
    st.subheader("🚨 Campus Emergency SOS Alert System")
    st.warning("Use this button only during an emergency. It records an alert for campus security and the Principal's office.")
    if st.button("🚨 TRIGGER CAMPUS EMERGENCY SOS"):
        st.error("🚨 [CRITICAL ALERT] Emergency SOS Broadcasted to Sandipani Technical Campus Security & Principal Office at Kolpa!")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #555;'>Sandipani Technical Campus, Kolpa | Smart Campus AI Suite © 2026</p>", unsafe_allow_html=True)
