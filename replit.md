# Face Verification System

## Overview
Desktop application for face verification using machine learning. Works like a Ring camera security system - identifies whether a person is registered in the database or is unknown.

## Current Status
Application is running and fully functional with local SQLite database.

## Features
- Face detection and recognition using ML (histogram, LBP, ORB features)
- Load images from files or use camera
- Register new faces to database
- Verify identity against registered faces
- Auto-verify mode for continuous monitoring
- Manage registered persons (list, delete)
- Dark themed professional UI

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
3. Enter a name and click "Register Current Face"
4. Load another image and click "Verify Current Face"
5. The app shows "VERIFIED: [name]" or "UNKNOWN PERSON"

## Face Recognition Algorithm
Multi-feature encoding approach:
- Histogram (64-bin grayscale)
- ORB descriptors
- Sobel edge features
- Regional statistics (16 regions)
- LBP texture patterns

Threshold: 70% similarity for positive verification
