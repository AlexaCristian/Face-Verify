# Face Verification Desktop App

## Overview
A Windows desktop application for face verification using machine learning. Works like a Ring camera security system - identifies whether a person is registered in the database or is unknown. Designed to run locally on Windows and connect to your own SQL Server (SSMS) database.

## Features
- Live camera feed with real-time face detection
- Face registration from camera capture or imported images
- Face verification using custom ML-based encoding (histogram, LBP, ORB features)
- SQL Server (SSMS) database integration
- Auto-verify mode for continuous monitoring
- Verification activity logging to database
- Dark themed professional UI with PyQt5

## Technology Stack
- Python 3.11+
- PyQt5 - Desktop GUI framework
- OpenCV - Camera handling and face detection (Haar Cascade)
- NumPy/SciPy - Face encoding and similarity computation
- pyodbc - SQL Server database connectivity

## Project Structure
```
/
├── face_verification_app.py    # Main application
├── db_config.txt               # Saved database configuration (auto-generated)
└── replit.md                   # Project documentation
```

## Windows Installation

### Prerequisites
1. Python 3.11 or higher
2. SQL Server with ODBC Driver 17 installed
3. A webcam

### Install Dependencies
```bash
pip install opencv-python numpy pyodbc Pillow scipy PyQt5
```

### Run the Application
```bash
python face_verification_app.py
```

## Database Setup
The app connects to SQL Server and automatically creates two tables:

### RegisteredFaces Table
- `id` - Auto-increment primary key
- `person_name` - Name of the registered person
- `face_encoding` - ML-generated face features (stored as text)
- `face_image` - Original image (stored as binary)
- `registered_date` - Timestamp

### VerificationLog Table
- `id` - Auto-increment primary key
- `person_name` - Matched person (or "Unknown")
- `verification_result` - "VERIFIED" or "UNKNOWN"
- `confidence` - Similarity score (0-1)
- `timestamp` - When verification occurred

## SQL Server Connection
Supports both authentication methods:
- **Windows Authentication**: Leave username blank
- **SQL Server Authentication**: Provide username and password

Example server names:
- `localhost`
- `.\\SQLEXPRESS`
- `MYSERVER\\SQLINSTANCE`

## Usage Guide
1. Launch the application
2. Enter SQL Server connection details
3. Click "Connect to Database"
4. Click "Start Camera" to begin video feed
5. To register: Enter name → Click "Capture & Register Face"
6. To verify: Click "Verify Current Face"
7. Enable "Auto-Verify" for continuous 3-second checks

## Face Recognition Algorithm
Uses a custom multi-feature encoding approach:
1. **Histogram Features** - 64-bin grayscale intensity distribution
2. **ORB Features** - Oriented FAST and Rotated BRIEF descriptors
3. **Edge Features** - Sobel gradient-based edge detection
4. **Regional Statistics** - Mean and std dev from 16 facial regions
5. **LBP Features** - Local Binary Patterns for texture analysis

Similarity is computed using cosine distance on normalized feature vectors.
Default threshold: 70% similarity for positive verification.

## Troubleshooting

### Camera not opening
- Ensure webcam is connected and not used by another application
- Try running as administrator

### Database connection failed
- Verify SQL Server is running
- Check if ODBC Driver 17 is installed
- Ensure database exists (create it manually in SSMS first)

### Face not detected
- Ensure good lighting
- Face the camera directly
- Keep a reasonable distance from camera
