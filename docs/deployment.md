# Deployment & Live Demo Hosting Guide

This guide walks you through deploying the **Secure Face Recognition System (SFRS)** to free cloud hosting platforms to showcase an interactive live demo in your GitHub repository and portfolio.

---

## Free Cloud Hosting Options Comparison

| Platform | Free Resources | Deployment Method | Recommended For |
| :--- | :--- | :--- | :--- |
| **Hugging Face Spaces** | **16 GB RAM, 2 vCPUs, 50 GB disk** | Streamlit or Docker | ⭐ **Best Choice** (Full ML & Deep Learning performance) |
| **Streamlit Community Cloud** | 1 GB RAM, 1 vCPU | Git Push to GitHub | Lightweight demo & quick setup |
| **Local Docker** | Host Machine hardware | `docker compose up` | Zero-install local testing & demos |

---

## 1. Deploying to Hugging Face Spaces (Recommended)

Hugging Face Spaces provides **16 GB RAM and 2 vCPUs completely free**, which is ideal for computer vision and deep learning inference.

### Step-by-Step Instructions:

1. **Sign Up / Log In**:
   - Go to [Hugging Face](https://huggingface.co/) and create a free account.

2. **Create a New Space**:
   - Click your profile icon (top right) $\rightarrow$ **New Space**.
   - **Space Name**: `secure-face-recognition-system`
   - **License**: `mit` (or choose your preferred license).
   - **Select the Space SDK**: Choose **Streamlit** (or **Docker**).
   - **Space Hardware**: Select **Free: 2 vCPU · 16GB RAM**.
   - Click **Create Space**.

3. **Deploy the Code**:
   - **Option A (GitHub Sync / Direct Push)**:
     Follow Hugging Face's Git instructions to push your repository:
     ```bash
     git remote add space https://huggingface.co/spaces/<your-username>/secure-face-recognition-system
     git push --force space main
     ```
   - **Option B (Configure `app.py` path)**:
     In the Space Settings:
     - Set the Streamlit entrypoint to: `web/app.py`

4. **Add Secret Environment Variables (Optional)**:
   - Go to **Settings** $\rightarrow$ **Variables and secrets**.
   - Add Secret: `DB_KEY` with a random 32-character secret string.

5. **Done!**
   Your live interactive demo will be available at:
   `https://huggingface.co/spaces/<your-username>/secure-face-recognition-system`

---

## 2. Deploying to Streamlit Community Cloud

Streamlit Community Cloud is 100% free and connects directly to your public GitHub repository.

### Step-by-Step Instructions:

1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **Create app** $\rightarrow$ **Deploy a public app from GitHub**.
3. Fill in the details:
   - **Repository**: `<your-github-username>/secure-face-recognition-system`
   - **Branch**: `main`
   - **Main file path**: `web/app.py`
   - **App URL**: Choose a custom subdomain (e.g. `secure-face-recognition.streamlit.app`).
4. Click **Advanced settings** $\rightarrow$ **Secrets**:
   ```toml
   DB_KEY = "your-custom-secure-master-key-here"
   SFRS_MODE = "development"
   ```
5. Click **Deploy!**

---

## 3. Running Locally

### Option A: Desktop Application (CustomTkinter GUI)
Runs the native desktop interface with real-time OpenCV camera widgets:
```bash
python main.py
```

### Option B: Local Web Interface (Streamlit)
Runs the web dashboard locally in your browser:
```bash
python main.py --web
# or
streamlit run web/app.py
```
Open [http://localhost:7860](http://localhost:7860) or [http://localhost:8501](http://localhost:8501).

### Option C: Docker / Docker Compose
Run the entire containerized stack without installing any local Python packages:
```bash
# Build and start container
docker compose up --build

# Open in browser
# http://localhost:7860
```

---

## 4. Adding the Live Demo Link to GitHub

Once deployed, add the live link badge at the top of your `README.md`:

```markdown
[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/<your-username>/secure-face-recognition-system)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-app>.streamlit.app)
```
