# Face Verification System

## Overview
Desktop application for face verification using machine learning with anti-spoofing protection. Works like a Ring camera security system - identifies whether a person is registered and detects photo/screen spoofing attempts.

## Current Status
Application is running and fully functional with liveness detection.

## Features
- Face detection and recognition using ML
- **Anti-Spoofing / Liveness Detection** - Detects if face is real or photo
- Load images from files or use camera
- Register new faces
- Verify identity against registered faces
- Auto-verify mode for continuous monitoring
- Dark themed professional UI

## Anti-Spoofing Methods (8 checks)
1. **Texture Analysis** - LBP variance (real faces have more texture)
2. **Color Distribution** - HSV analysis for natural skin colors
3. **Frequency Analysis** - FFT to detect print patterns
4. **Reflection Detection** - Unnatural bright spots from photos
5. **Sharpness Analysis** - Laplacian variance
6. **Skin Detection** - YCrCb color space for skin ratio
7. **Noise Analysis** - Natural vs artificial noise patterns
8. **Gradient Analysis** - Edge consistency

## Technology Stack
- Python 3.12
- PyQt5 - Desktop GUI (VNC mode)
- OpenCV - Face detection (Haar Cascade)
- NumPy/SciPy - Face encoding and similarity
- SQLite - Local database storage

## Project Files
- `face_verification_app.py` - Main application
- `faces.db` - SQLite database (auto-created)

## How to Use
1. View the app in the Output/VNC panel
2. Click "Load Image from File" to load a photo with a face
3. The app shows LIVE (green) or PHOTO (orange) on detected faces
4. Enter a name and click "Register Current Face"
5. Load another image and click "Verify Current Face"
6. Results: VERIFIED (green), UNKNOWN (red), or PHOTO DETECTED (orange)

## Liveness Score
- Above 55% = LIVE (real person)
- Below 55% = PHOTO (spoofing attempt)
