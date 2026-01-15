# Face Verification System

## Overview
Desktop application for face verification using machine learning with anti-spoofing protection. Works like a Ring camera security system - identifies whether a person is registered and detects photo/screen spoofing attempts.

## Current Status
Application is running and fully functional.

## Features
- Face detection and recognition using ML
- Anti-Spoofing / Liveness Detection - Detects if face is real or photo
- SQL Server database support (optional, with separate setup tool)
- SQLite fallback for local storage
- Enhanced camera controls (zoom, brightness, contrast)
- Keyboard shortcuts for quick actions
- Stats sidebar with verification history
- Auto-verify mode for continuous monitoring
- Dark themed professional UI

## Project Files
- `face_verification_app.py` - Main application
- `db_setup.py` - SQL Server database configuration tool (run separately)
- `db_config.json` - Database configuration (created by db_setup.py)
- `faces.db` - SQLite database (auto-created if no SQL Server)

## How to Connect to SQL Server
1. Run the database setup script: `python db_setup.py`
2. Enter your SQL Server credentials (server, database, username, password)
3. Click "Test Connection" to verify
4. Click "Save Configuration" to save settings
5. Restart the main Face Verification App
6. The app will automatically use SQL Server

## Keyboard Shortcuts
- Enter - Verify current face
- R - Register face
- C - Capture/freeze frame
- L - Load image from file
- X - Clear current image
- Esc - Stop camera

## Anti-Spoofing Methods (8 checks)
1. Texture Analysis (LBP variance)
2. Color Distribution (HSV analysis)
3. Frequency Analysis (FFT patterns)
4. Reflection Detection
5. Sharpness Analysis (Laplacian)
6. Skin Detection (YCrCb)
7. Noise Analysis
8. Gradient Analysis

## Technology Stack
- Python 3.12
- PyQt5 - Desktop GUI (VNC mode)
- OpenCV - Face detection (Haar Cascade)
- NumPy/SciPy - Face encoding and similarity
- SQLite - Local database (default)
- pyodbc - SQL Server connection (optional)
