# Deployment Guide for PredictWell

This guide covers how to deploy the full stack application:
1. **Backend**: Python Flask app on Render
2. **Frontend**: Next.js app on Vercel

## Prerequisites
- A GitHub account.
- The project code pushed to a GitHub repository.

---

## Part 1: Backend Deployment (Render)

We will use [Render.com](https://render.com) for the backend because it supports Python natively and is free/cheap.

1. **Sign Up/Login** to Render.
2. Click **New +** -> **Web Service**.
3. **Connect your GitHub repository**.
4. Configure the service:
   - **Name**: `predictwell-backend` (or similar)
   - **Region**: Closest to you (e.g., Singapore, Oregon)
   - **Branch**: `main` (or your working branch)
   - **Root Directory**: `backend` (IMPORTANT: set this to `backend`)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run.py`
5. **Environment Variables**:
   Under the "Environment" tab or section, add:
   - `PYTHON_VERSION`: `3.9.0` (Recommended)
   - `PORT`: `10000`
6. Click **Create Web Service**.

Wait for the deployment to finish. Render will provide a URL like `https://predictwell-backend.onrender.com`.
**Copy this URL**. You will need it for the frontend.

---

## Part 2: Frontend Deployment (Vercel)

We will use [Vercel](https://vercel.com) for the Next.js frontend.

1. **Sign Up/Login** to Vercel.
2. Click **Add New...** -> **Project**.
3. **Import** your GitHub repository.
4. Configure the project:
   - **Framework Preset**: `Next.js` (should be auto-detected)
   - **Root Directory**: `./` (default)
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `.next` (default)
5. **Environment Variables**:
   Expand the "Environment Variables" section and add:
   - **Key**: `NEXT_PUBLIC_API_URL`
   - **Value**: The Render URL you copied earlier (e.g., `https://predictwell-backend.onrender.com`)
   - **IMPORTANT**: Do NOT include a trailing slash `/` at the end of the URL.
6. Click **Deploy**.

Vercel will build and deploy your site. Once done, you will get a URL like `https://predictwell.vercel.app`.

---

## Part 3: Final Verification

1. Open your Vercel URL.
2. Open the browser console (F12) to see logs.
3. The app should log "🔗 Backend API URL: https://predictwell-backend.onrender.com" (or whatever you set).
4. Try a prediction (e.g., Heart Disease) to verify the backend connection works.
   - Note: The free tier of Render "spins down" after inactivity. The first request might take 50+ seconds. Please be patient or check the Render dashboard logs.

## Troubleshooting

- **CORS Errors**: If you see CORS errors in the browser console, ensure your Backend `app.py` allows requests from your Vercel domain. The current code allows `*` (all origins), so it should work fine.
- **Backend 500 Error**: Check Render logs.
- **Frontend 404/500**: Check Vercel logs.
