# Smart Campus AI Attendance System

A Streamlit application for student registration, faculty attendance, local SQLite storage, absence notifications, reports, and emergency alerts.

## Run locally

```powershell
& "C:\Users\LAPPY HUB\anaconda3\python.exe" -m streamlit run app.py
```

Open `http://127.0.0.1:8501`.

## Run in VS Code

1. Open this folder in VS Code: `C:\Users\LAPPY HUB\Downloads\My project\My project`.
2. Install the Microsoft **Python** extension if VS Code asks.
3. Press **F5** and choose **Run Smart Campus**. The included VS Code settings use your Anaconda Python automatically.

Do not run `./.venv/Scripts/python.exe` for this project: that old environment is incomplete. Use the Anaconda command below or **Terminal → Run Task → Run Smart Campus**.

You can also open the VS Code terminal and run:

```powershell
& "C:\Users\LAPPY HUB\anaconda3\python.exe" -m streamlit run app.py
```

Default credentials (only when `PRINCIPAL_PASSWORD` is not configured in secrets):

- Faculty: `faculty1` / `123`
- Principal: `principal123`

## Student storage

The local application saves students, faculty, attendance, and notification history in `campus_data.db`. The original `campus_data.pkl`, if present, is imported once automatically.

## Parent absence notifications

When an attendance session is submitted, every student not marked Present is recorded as Absent. The app then queues alerts only for absent students:

- **WhatsApp:** through Twilio's WhatsApp Business API.
- **SMS text message:** through an SMS-capable Twilio number.
- **Email:** through your college SMTP account.

Faculty and Principal can also send or resend one selected absence alert from **Send parent absence alert**. Faculty sees only their own submitted attendance; Principal can select from all saved absences. The delivery status appears in **Parent Notification Log**.

Open **Principal Admin Panel → Parent notification status** to see whether the channels are configured. Copy `secrets.example.toml` into either a local `.streamlit/secrets.toml` file or Streamlit Community Cloud **App settings → Secrets**, then replace every placeholder with credentials owned by the college. Set `TWILIO_SMS_FROM` only when the college has an SMS-capable Twilio number. Do not commit live API keys or email passwords.

Twilio Sandbox/free-form messages work for opted-in recipients and active WhatsApp sessions. Production absence alerts that may be sent outside WhatsApp's 24-hour customer-service window require an approved Twilio/Meta template; set its ID as `TWILIO_CONTENT_SID`. The app supplies variables 1-5: student name, roll number, date, lecture slot, and class.

### Send an absence alert manually

1. Add the Twilio WhatsApp sender, SMS sender, or email settings in `.streamlit/secrets.toml`; use `secrets.example.toml` as the template.
2. Register the student with a valid parent mobile number for WhatsApp/SMS and/or a parent email address for email.
3. Submit attendance. Students who were not marked Present are saved as Absent.
4. A Faculty member opens **Faculty & attendance → Send parent absence alert** to send an alert for one of their own absence records. A Principal opens **Principal controls → Send parent absence alert** to send or resend an alert for any saved absence.
5. Choose the student, choose one or more available channels, then click **Send parent alert now**. Open **Parent Notification Log** to review the result.

The app sends to the stored parent contact only after an authorized Faculty or Principal clicks the send button. Do not add provider credentials until the college is authorized to notify parents.

## Free deployment

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Choose **Create app**, select the GitHub repository, and set the entrypoint to `app.py`.

The hosted version supports Manual Attendance immediately. Face recognition is intentionally optional so a missing native face-recognition library never prevents the application from loading. Streamlit Community Cloud local files may be reset during redeploys, so production cloud storage should use a managed database configured by the college; the included SQLite database is persistent for local/self-hosted use.
