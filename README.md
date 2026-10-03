# Smart Campus AI Attendance System

A Streamlit application for student registration, faculty attendance, local SQLite storage, absence notifications, reports, and emergency alerts.

## Run locally

```powershell
& "C:\Users\LAPPY HUB\anaconda3\python.exe" -m streamlit run app.py
```

Open `http://127.0.0.1:8501`.

Default credentials (only when `PRINCIPAL_PASSWORD` is not configured in secrets):

- Faculty: `faculty1` / `123`
- Principal: `principal123`

## Student storage

The local application saves students, faculty, attendance, and notification history in `campus_data.db`. The original `campus_data.pkl`, if present, is imported once automatically.

## Parent absence notifications

When an attendance session is submitted, every student not marked Present is recorded as Absent. The app then queues alerts only for absent students:

- **WhatsApp:** through Twilio's WhatsApp Business API.
- **Email:** through your college SMTP account.

Open **Principal Admin Panel → Parent notification setup** to see whether the channels are configured. Copy `secrets.example.toml` into either a local `.streamlit/secrets.toml` file or Streamlit Community Cloud **App settings → Secrets**, then replace every placeholder with credentials owned by the college. Do not commit live API keys or email passwords.

Twilio Sandbox/free-form messages work for opted-in recipients and active WhatsApp sessions. Production absence alerts that may be sent outside WhatsApp's 24-hour customer-service window require an approved Twilio/Meta template; set its ID as `TWILIO_CONTENT_SID`. The app supplies variables 1-5: student name, roll number, date, lecture slot, and class.

## Free deployment

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Choose **Create app**, select the GitHub repository, and set the entrypoint to `app.py`.

The hosted version supports Manual Attendance immediately. Face recognition is intentionally optional so a missing native face-recognition library never prevents the application from loading. Streamlit Community Cloud local files may be reset during redeploys, so production cloud storage should use a managed database configured by the college; the included SQLite database is persistent for local/self-hosted use.
