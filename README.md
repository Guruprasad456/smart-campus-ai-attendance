# Smart Campus AI Attendance System

A Streamlit application for student registration, faculty attendance, parent notifications, reports, and emergency alerts.

## Run locally

```powershell
& "C:\Users\LAPPY HUB\anaconda3\python.exe" -m streamlit run app.py
```

Open `http://127.0.0.1:8501`.

Default credentials:

- Faculty: `faculty1` / `123`
- Principal: `principal123`

## SMS setup

Open **Principal Admin Panel** and expand **SMS Gateway Setup**. Choose Fast2SMS or an Android SMS gateway, enter credentials owned by the campus, save, and send a test SMS.

For Streamlit Community Cloud, add the same values in the app's **Secrets** settings. Copy `secrets.example.toml` to the Secrets editor and replace its placeholder values. Do not commit live API keys.

## Free deployment

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Choose **Create app**, select the GitHub repository, and set the entrypoint to `app.py`.

The hosted version supports Manual Attendance immediately. Face recognition is intentionally optional so a missing native face-recognition library never prevents the application from loading.
