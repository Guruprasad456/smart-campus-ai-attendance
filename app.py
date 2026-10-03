import io
import os
import pickle
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime

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
DATA_FILE = "campus_data.pkl"

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
                "sms_outbox", "att_session", "sms_config"]


def save_data():
    """Save everything to disk so data is NOT lost when the page refreshes."""
    try:
        with open(DATA_FILE, "wb") as f:
            pickle.dump({k: st.session_state.get(k) for k in PERSIST_KEYS}, f)
    except Exception as e:
        st.warning(f"Could not save local data: {e}")


def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
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
# SMS engine: fast (parallel, background threads) + auto retry
# Message text is fixed as requested by the Principal.
# ---------------------------------------------------------
SIGNATURE = "Principal\nSTC Latur"


def build_sms(status):
    word = "absent" if status == "Absent" else "present"
    return f"Dear Parent,\nYour child is {word} for Today's Lecture.\n\n{SIGNATURE}"


@st.cache_resource
def sms_runtime():
    """Shared worker pool + result box (survives Streamlit reruns)."""
    return {"pool": ThreadPoolExecutor(max_workers=8), "results": [], "lock": threading.Lock()}


SMS_KEYS = ["SMS_PROVIDER", "FAST2SMS_API_KEY", "FAST2SMS_ROUTE", "FAST2SMS_SENDER_ID",
            "FAST2SMS_TEMPLATE_ABSENT", "FAST2SMS_TEMPLATE_PRESENT", "GATEWAY_URL", "GATEWAY_TOKEN"]


def get_sms_config():
    cfg = {k: "" for k in SMS_KEYS}
    # Do not access st.secrets when no file exists: Streamlit renders a noisy
    # missing-secrets diagnostic even if the exception is handled.
    secrets_paths = [
        os.path.join(".streamlit", "secrets.toml"),
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
    ]
    if any(os.path.exists(path) for path in secrets_paths):
        try:
            for k in SMS_KEYS:
                cfg[k] = str(st.secrets.get(k, "") or "")
        except Exception:
            pass
    # A principal can configure a local gateway from the Admin Panel. Secrets
    # take precedence so deployments can keep credentials outside this file.
    for key, value in st.session_state.get("sms_config", {}).items():
        if key in cfg and not cfg[key]:
            cfg[key] = str(value or "")
    if not cfg["SMS_PROVIDER"]:
        cfg["SMS_PROVIDER"] = "fast2sms" if cfg["FAST2SMS_API_KEY"] else "none"
    cfg["SMS_PROVIDER"] = cfg["SMS_PROVIDER"].lower()
    return cfg


def _api_err(r):
    """Short readable error text from the SMS provider's response."""
    try:
        m = r.json().get("message", "")
        return " ".join(m) if isinstance(m, list) else str(m)[:100]
    except Exception:
        return r.text[:100]


def _deliver(rt, cfg, mobile, message, status_word, student_name):
    """Runs in a background thread (no st.* calls here!)."""
    provider = cfg["SMS_PROVIDER"]
    if provider == "none":
        status = "Logged (SMS gateway not configured)"
    else:
        import requests
        status = "Failed"
        for _ in range(2):                       # one automatic retry
            try:
                if provider == "fast2sms":
                    route = (cfg["FAST2SMS_ROUTE"] or "dlt").lower()
                    data = {"route": route, "numbers": mobile, "flash": "0"}
                    if route == "dlt":           # DLT registered template (sender header like STCLTR)
                        data["sender_id"] = cfg["FAST2SMS_SENDER_ID"]
                        data["message"] = cfg["FAST2SMS_TEMPLATE_ABSENT"] if status_word == "Absent" \
                            else cfg["FAST2SMS_TEMPLATE_PRESENT"]
                    else:                        # "q" quick route: free text
                        data["message"] = message
                    r = requests.post("https://www.fast2sms.com/dev/bulkV2",
                                      headers={"authorization": cfg["FAST2SMS_API_KEY"]},
                                      data=data, timeout=15)
                    ok = r.ok and bool(r.json().get("return", False))
                elif provider == "android":      # SMS-gateway app on the Principal's own phone
                    to = mobile if mobile.startswith("+") else "+91" + mobile
                    r = requests.post(cfg["GATEWAY_URL"], json={"to": to, "message": message},
                                      headers={"Authorization": cfg["GATEWAY_TOKEN"]}, timeout=15)
                    ok = r.ok
                else:
                    status = "Failed (unknown SMS_PROVIDER)"
                    break
                if ok:
                    status = "Sent"
                    break
                status = f"Failed ({r.status_code}: {_api_err(r)})"
            except Exception as e:
                status = f"Failed ({type(e).__name__})"
    entry = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "to": mobile,
             "student": student_name, "type": status_word, "message": message, "status": status}
    with rt["lock"]:
        rt["results"].append(entry)
    return status


def queue_sms(mobile, status_word, student_name=""):
    """Send immediately in the background (does not block the screen). Returns a Future."""
    rt = sms_runtime()
    return rt["pool"].submit(_deliver, rt, get_sms_config(), mobile,
                             build_sms(status_word), status_word, student_name)


def _deliver_daily_summary(rt, cfg, mobile, message, student_name):
    """Deliver a custom daily summary where the configured gateway supports free text."""
    provider = cfg["SMS_PROVIDER"]
    if provider == "none":
        status = "Logged (SMS gateway not configured)"
    else:
        try:
            import requests
            if provider == "fast2sms":
                route = (cfg["FAST2SMS_ROUTE"] or "dlt").lower()
                if route == "dlt":
                    status = "Skipped (a DLT daily-summary template is required)"
                else:
                    r = requests.post("https://www.fast2sms.com/dev/bulkV2",
                                      headers={"authorization": cfg["FAST2SMS_API_KEY"]},
                                      data={"route": route, "numbers": mobile, "message": message, "flash": "0"},
                                      timeout=15)
                    status = "Sent" if r.ok and bool(r.json().get("return", False)) else \
                        f"Failed ({r.status_code}: {_api_err(r)})"
            elif provider == "android":
                to = mobile if mobile.startswith("+") else "+91" + mobile
                r = requests.post(cfg["GATEWAY_URL"], json={"to": to, "message": message},
                                  headers={"Authorization": cfg["GATEWAY_TOKEN"]}, timeout=15)
                status = "Sent" if r.ok else f"Failed ({r.status_code}: {_api_err(r)})"
            else:
                status = "Failed (unknown SMS_PROVIDER)"
        except Exception as e:
            status = f"Failed ({type(e).__name__})"
    entry = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "to": mobile,
             "student": student_name, "type": "Daily Summary", "message": message, "status": status}
    with rt["lock"]:
        rt["results"].append(entry)
    return status


def queue_daily_summary(mobile, message, student_name=""):
    """Queue an end-of-day attendance summary without blocking the interface."""
    rt = sms_runtime()
    return rt["pool"].submit(_deliver_daily_summary, rt, get_sms_config(), mobile, message, student_name)


def drain_sms_results():
    """Move finished SMS results into the visible outbox (main thread only)."""
    rt = sms_runtime()
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
    """Mark remaining students ABSENT, save logs, queue SMS, close session.
    Present students already got their SMS at scan time (only missing ones are sent here)."""
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
        }
        st.session_state.attendance_logs.append(entry)
        batch.append(entry)

        if status == "Absent" or not info.get("sms"):
            futs.append(queue_sms(stud["parent_mobile"], status, stud["name"]))

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
    st.session_state.sms_config = saved.get("sms_config", {})
    st.session_state.data_loaded = True

drain_sms_results()   # move finished SMS results into the outbox

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

    _prov = get_sms_config()["SMS_PROVIDER"]
    if _prov == "none":
        st.warning("SMS is in Outbox demo mode. Configure Fast2SMS or an Android SMS gateway in "
                   "Principal Admin Panel to send real messages.")
    else:
        st.success(f"SMS gateway is enabled: **{_prov}**")

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
            p_mobile = st.text_input("Parents Mobile Number")

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
        if not (p_mobile.strip().isdigit() and len(p_mobile.strip()) == 10):
            errors.append("Parent Mobile Number must contain exactly 10 digits.")
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
                "parent_mobile": p_mobile.strip(), "photo": photo_bytes, "encoding": encoding,
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

            # ---- latest SMS status ----
            with st.expander("📨 Latest SMS status"):
                recent = st.session_state.sms_outbox[-5:][::-1]
                if recent:
                    show_df(pd.DataFrame(recent)[["time", "student", "type", "status"]])
                else:
                    st.write("No SMS messages yet.")

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
                                queue_sms(stud["parent_mobile"], "Present", stud["name"])   # instant SMS
                                sess["last"] = ("success",
                                                f"✅ {stud['name']} (Roll {stud['roll_no']}) – Present! "
                                                f"Parent notification queued. (match distance {best:.2f}){multi}")
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
                            queue_sms(_s["parent_mobile"], "Present", _s["name"])   # instant SMS
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
            st.caption("Parent SMS notifications are queued when a student is marked Present. "
                       "Submitting attendance queues notifications for the remaining Absent students.")
            b1, b2 = st.columns(2)
            if b1.button("🏁 Finish & Submit Attendance", type="primary"):
                with st.spinner("Saving attendance and notifying parents of Absent students..."):
                    batch, n_present, futs = finalize_attendance(sess, students)
                    if futs:
                        wait(futs, timeout=45)
                statuses = [f.result() for f in futs if f.done()]
                n_sent = sum(1 for x in statuses if x == "Sent")
                n_logged = sum(1 for x in statuses if x.startswith("Logged"))
                n_fail = len(futs) - n_sent - n_logged
                drain_sms_results()
                st.success(f"Attendance saved. Present: {n_present} | Absent: {len(batch) - n_present}")
                if n_logged:
                    st.warning(f"{n_logged} message(s) were saved to the Outbox because a live SMS gateway is not configured.")
                if n_sent:
                    st.success(f"{n_sent} parent SMS message(s) were sent.")
                if n_fail:
                    st.error(f"{n_fail} SMS message(s) failed or are still in progress. Check Principal Panel → SMS Outbox "
                             f"and use 'Retry Failed SMS'.")
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

        active_sms = get_sms_config()
        sms_options = {
            "Outbox demo (no live SMS)": "none",
            "Fast2SMS": "fast2sms",
            "Android SMS gateway": "android",
        }
        selected_label = next((label for label, value in sms_options.items()
                               if value == active_sms["SMS_PROVIDER"]), "Outbox demo (no live SMS)")
        with st.expander("⚙️ SMS Gateway Setup", expanded=active_sms["SMS_PROVIDER"] == "none"):
            st.caption("Choose a provider, enter your own gateway credentials, then send a test message. "
                       "Without valid credentials, messages are safely saved to the local Outbox only.")
            with st.form("sms_gateway_form"):
                sms_label = st.selectbox("SMS provider", list(sms_options),
                                         index=list(sms_options).index(selected_label))
                sms_provider = sms_options[sms_label]
                configured = {"SMS_PROVIDER": sms_provider}
                test_mobile = st.text_input("Test mobile number (optional)", key="sms_test_mobile")
                if sms_provider == "fast2sms":
                    configured["FAST2SMS_API_KEY"] = st.text_input(
                        "Fast2SMS API key", value=active_sms["FAST2SMS_API_KEY"], type="password")
                    configured["FAST2SMS_ROUTE"] = st.selectbox(
                        "Fast2SMS route", ["q", "dlt"],
                        index=0 if active_sms["FAST2SMS_ROUTE"].lower() != "dlt" else 1)
                    configured["FAST2SMS_SENDER_ID"] = st.text_input(
                        "DLT sender ID (required only for DLT)", value=active_sms["FAST2SMS_SENDER_ID"])
                    configured["FAST2SMS_TEMPLATE_PRESENT"] = st.text_input(
                        "DLT Present template ID (required only for DLT)",
                        value=active_sms["FAST2SMS_TEMPLATE_PRESENT"])
                    configured["FAST2SMS_TEMPLATE_ABSENT"] = st.text_input(
                        "DLT Absent template ID (required only for DLT)",
                        value=active_sms["FAST2SMS_TEMPLATE_ABSENT"])
                elif sms_provider == "android":
                    configured["GATEWAY_URL"] = st.text_input(
                        "Android SMS gateway URL", value=active_sms["GATEWAY_URL"])
                    configured["GATEWAY_TOKEN"] = st.text_input(
                        "Android gateway token", value=active_sms["GATEWAY_TOKEN"], type="password")
                save_sms = st.form_submit_button("Save SMS Configuration")
                test_sms = st.form_submit_button("Save and Send Test SMS")

            if save_sms or test_sms:
                config_errors = []
                if sms_provider == "fast2sms" and not configured.get("FAST2SMS_API_KEY", "").strip():
                    config_errors.append("A Fast2SMS API key is required.")
                if sms_provider == "android":
                    if not configured.get("GATEWAY_URL", "").strip():
                        config_errors.append("An Android SMS gateway URL is required.")
                    if not configured.get("GATEWAY_TOKEN", "").strip():
                        config_errors.append("An Android gateway token is required.")
                if test_sms and not (test_mobile.strip().isdigit() and len(test_mobile.strip()) == 10):
                    config_errors.append("Enter a valid 10-digit test mobile number.")

                if config_errors:
                    for error in config_errors:
                        st.error(error)
                else:
                    st.session_state.sms_config = configured
                    save_data()
                    st.success("SMS configuration saved.")
                    if test_sms:
                        future = queue_sms(test_mobile.strip(), "Present", "SMS gateway test")
                        wait([future], timeout=30)
                        drain_sms_results()
                        st.success("Test SMS was queued. Check the SMS Outbox for the delivery result.")

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
            if "parent_mobile" in df_all.columns:
                df_all = df_all.drop(columns=["parent_mobile"])
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

            st.markdown("#### 📲 Daily attendance summary for parents")
            summary_date = st.date_input("Summary date", datetime.now().date(), key="daily_summary_date")
            if st.button("Send Daily Summary SMS to Parents", type="primary"):
                date_str = str(summary_date)
                per_student = {}
                for log in st.session_state.attendance_logs:
                    if log["date"] != date_str:
                        continue
                    key = (log["department"], log["class_name"], log["division"], log["roll_no"])
                    per_student.setdefault(key, {"name": log["name"], "mobile": log["parent_mobile"], "rows": []})
                    per_student[key]["rows"].append(f"{log['slot']}: {log['status']}")

                if not per_student:
                    st.warning("The selected date has no attendance records.")
                else:
                    futures = []
                    for (_, class_name, division, roll_no), details in per_student.items():
                        message = (f"Sandipani Technical Campus - Daily Report {date_str}: "
                                   f"{details['name']} (Roll No. {roll_no}, {class_name} {division}). "
                                   + " | ".join(details["rows"]))
                        futures.append(queue_daily_summary(details["mobile"], message, details["name"]))
                    wait(futures, timeout=45)
                    drain_sms_results()
                    save_data()
                    st.success(f"Daily summaries have been queued for {len(per_student)} parent(s).")

        with st.expander("📨 SMS Outbox", expanded=True):
            outbox = st.session_state.sms_outbox
            if outbox:
                show_df(pd.DataFrame(outbox[::-1]))
                failed = [i for i, o in enumerate(outbox)
                          if str(o.get("status", "")).startswith("Failed") and "type" in o]
                if failed and st.button(f"🔁 Retry {len(failed)} Failed SMS Message(s)"):
                    futs = []
                    for i in failed:
                        o = outbox[i]
                        futs.append(queue_sms(o["to"], o["type"], o.get("student", "")))
                        o["status"] = "Retried"
                    wait(futs, timeout=45)
                    drain_sms_results()
                    save_data()
                    st.rerun()
            else:
                st.write("No SMS messages have been queued yet.")
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
