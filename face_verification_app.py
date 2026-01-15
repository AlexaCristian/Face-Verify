"""
Face Verification System - Ring Camera Style
=============================================
A Windows desktop application for face verification using machine learning.
Connect to your SQL Server (SSMS) database for storing and verifying faces.

REQUIREMENTS (install on Windows):
    pip install opencv-python numpy pyodbc Pillow scipy PyQt5

USAGE:
    python face_verification_app.py

Make sure your SQL Server has:
1. ODBC Driver 17 for SQL Server installed
2. A database created (app will create tables automatically)
"""

import sys
import os
import cv2
import numpy as np
import pyodbc
import threading
from datetime import datetime
from PIL import Image
from scipy.spatial.distance import cosine

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                                 QTextEdit, QGroupBox, QMessageBox, QFileDialog,
                                 QFrame, QSlider)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 not available. Install with: pip install PyQt5")


class FaceEncoder:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def detect_face(self, image):
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        return gray[y:y+h, x:x+w]
    
    def get_encoding(self, image):
        face = self.detect_face(image)
        if face is None:
            return None
            
        face_resized = cv2.resize(face, (128, 128))
        encoding = []
        
        hist = cv2.calcHist([face_resized], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        encoding.extend(hist)
        
        orb = cv2.ORB_create(nfeatures=100)
        keypoints, descriptors = orb.detectAndCompute(face_resized, None)
        
        if descriptors is not None and len(descriptors) > 0:
            desc_mean = np.mean(descriptors, axis=0)
            encoding.extend(desc_mean)
        else:
            encoding.extend(np.zeros(32))
            
        sobelx = cv2.Sobel(face_resized, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(face_resized, cv2.CV_64F, 0, 1, ksize=3)
        
        sx_flat = sobelx.flatten()
        sy_flat = sobely.flatten()
        
        step_x = max(1, len(sx_flat) // 16)
        step_y = max(1, len(sy_flat) // 16)
        
        edge_x = [np.mean(sx_flat[i*step_x:(i+1)*step_x]) for i in range(16)]
        edge_y = [np.mean(sy_flat[i*step_y:(i+1)*step_y]) for i in range(16)]
        encoding.extend(edge_x)
        encoding.extend(edge_y)
        
        regions = []
        for i in range(4):
            for j in range(4):
                region = face_resized[i*32:(i+1)*32, j*32:(j+1)*32]
                regions.append(np.mean(region))
                regions.append(np.std(region))
        encoding.extend(regions)
        
        lbp_features = self._compute_lbp(face_resized)
        encoding.extend(lbp_features)
        
        return np.array(encoding)
    
    def _compute_lbp(self, image):
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
                
        hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-10)
        return hist.tolist()
    
    def compare_faces(self, encoding1, encoding2):
        if encoding1 is None or encoding2 is None:
            return 0.0
            
        e1 = np.array(encoding1)
        e2 = np.array(encoding2)
        
        min_len = min(len(e1), len(e2))
        e1 = e1[:min_len]
        e2 = e2[:min_len]
        
        e1 = e1 / (np.linalg.norm(e1) + 1e-10)
        e2 = e2 / (np.linalg.norm(e2) + 1e-10)
        
        similarity = 1 - cosine(e1, e2)
        
        return max(0, min(1, similarity))


class SignalBridge(QObject):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str, str)
    verification_signal = pyqtSignal(str, str)
    message_signal = pyqtSignal(str, str, str)


class FaceVerificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Verification System - Ring Camera Style")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QLabel { color: white; font-family: 'Segoe UI'; }
            QPushButton { 
                background-color: #0f3460; 
                color: white; 
                border: none; 
                padding: 10px; 
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #16537e; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QLineEdit { 
                background-color: #0f0f23; 
                color: white; 
                border: 1px solid #333; 
                padding: 8px;
                border-radius: 3px;
            }
            QTextEdit { 
                background-color: #0f0f23; 
                color: #00ff88; 
                border: 1px solid #333;
                font-family: 'Consolas', monospace;
            }
            QGroupBox { 
                color: white; 
                font-weight: bold; 
                border: 2px solid #333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        
        self.cap = None
        self.is_running = False
        self.known_faces = {}
        self.db_connection = None
        self.verification_threshold = 0.70
        self.current_frame = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.face_encoder = FaceEncoder()
        
        self.signals = SignalBridge()
        self.signals.log_signal.connect(self.append_log)
        self.signals.status_signal.connect(self.update_status)
        self.signals.verification_signal.connect(self.update_verification)
        self.signals.message_signal.connect(self.show_message)
        
        self.setup_ui()
        self.load_config()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #16213e; border-radius: 10px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        header = QLabel("Live Camera Feed")
        header.setFont(QFont("Segoe UI", 18, QFont.Bold))
        header.setStyleSheet("color: #00d4ff; background: transparent;")
        header.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(header)
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #0f0f23; border-radius: 5px; border: 2px solid #333;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Camera Off\n\nClick 'Start Camera' to begin")
        left_layout.addWidget(self.camera_label)
        
        self.status_label = QLabel("Status: Camera Off")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: #ff6b6b; background: transparent;")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)
        
        self.verification_label = QLabel("")
        self.verification_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.verification_label.setStyleSheet("background: transparent;")
        self.verification_label.setAlignment(Qt.AlignCenter)
        self.verification_label.setMinimumHeight(40)
        left_layout.addWidget(self.verification_label)
        
        main_layout.addWidget(left_panel, stretch=2)
        
        right_panel = QFrame()
        right_panel.setFixedWidth(380)
        right_panel.setStyleSheet("background-color: #16213e; border-radius: 10px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        
        db_group = QGroupBox("SQL Server Connection")
        db_layout = QVBoxLayout(db_group)
        db_layout.setSpacing(5)
        
        db_layout.addWidget(QLabel("Server (e.g., localhost or .\\SQLEXPRESS):"))
        self.server_entry = QLineEdit()
        self.server_entry.setText("localhost")
        self.server_entry.setPlaceholderText("Enter SQL Server name...")
        db_layout.addWidget(self.server_entry)
        
        db_layout.addWidget(QLabel("Database Name:"))
        self.database_entry = QLineEdit()
        self.database_entry.setText("FaceVerificationDB")
        self.database_entry.setPlaceholderText("Enter database name...")
        db_layout.addWidget(self.database_entry)
        
        db_layout.addWidget(QLabel("Username (leave empty for Windows Auth):"))
        self.username_entry = QLineEdit()
        self.username_entry.setPlaceholderText("SQL username or blank...")
        db_layout.addWidget(self.username_entry)
        
        db_layout.addWidget(QLabel("Password:"))
        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.Password)
        self.password_entry.setPlaceholderText("Password...")
        db_layout.addWidget(self.password_entry)
        
        self.connect_btn = QPushButton("Connect to Database")
        self.connect_btn.setStyleSheet("background-color: #1e88e5;")
        self.connect_btn.clicked.connect(self.connect_database)
        db_layout.addWidget(self.connect_btn)
        
        self.db_status_label = QLabel("Not Connected")
        self.db_status_label.setStyleSheet("color: #ff6b6b; background: transparent;")
        self.db_status_label.setAlignment(Qt.AlignCenter)
        db_layout.addWidget(self.db_status_label)
        
        right_layout.addWidget(db_group)
        
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout(camera_group)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.setStyleSheet("background-color: #2e7d32;")
        self.start_btn.clicked.connect(self.start_camera)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.setStyleSheet("background-color: #c62828;")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        camera_layout.addLayout(btn_layout)
        
        right_layout.addWidget(camera_group)
        
        person_group = QGroupBox("Face Registration & Verification")
        person_layout = QVBoxLayout(person_group)
        
        person_layout.addWidget(QLabel("Person Name:"))
        self.person_name_entry = QLineEdit()
        self.person_name_entry.setPlaceholderText("Enter person's name...")
        person_layout.addWidget(self.person_name_entry)
        
        self.capture_btn = QPushButton("Capture & Register Face")
        self.capture_btn.setStyleSheet("background-color: #6a1b9a;")
        self.capture_btn.clicked.connect(self.capture_and_register)
        person_layout.addWidget(self.capture_btn)
        
        self.import_btn = QPushButton("Import Face from Image File")
        self.import_btn.clicked.connect(self.import_face_from_image)
        person_layout.addWidget(self.import_btn)
        
        self.verify_btn = QPushButton("Verify Current Face")
        self.verify_btn.setStyleSheet("background-color: #00897b; font-weight: bold;")
        self.verify_btn.clicked.connect(self.verify_face)
        person_layout.addWidget(self.verify_btn)
        
        self.auto_verify_btn = QPushButton("Toggle Auto-Verify (OFF)")
        self.auto_verify_btn.clicked.connect(self.toggle_auto_verify)
        self.auto_verify = False
        self.auto_verify_timer = QTimer()
        self.auto_verify_timer.timeout.connect(self.verify_face)
        person_layout.addWidget(self.auto_verify_btn)
        
        self.list_btn = QPushButton("List All Registered Persons")
        self.list_btn.clicked.connect(self.list_persons)
        person_layout.addWidget(self.list_btn)
        
        right_layout.addWidget(person_group)
        
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        right_layout.addWidget(log_group)
        right_layout.addStretch()
        
        main_layout.addWidget(right_panel)
        
    def toggle_auto_verify(self):
        self.auto_verify = not self.auto_verify
        if self.auto_verify:
            self.auto_verify_btn.setText("Toggle Auto-Verify (ON)")
            self.auto_verify_btn.setStyleSheet("background-color: #2e7d32;")
            self.auto_verify_timer.start(3000)
            self.log("Auto-verify enabled (every 3 seconds)")
        else:
            self.auto_verify_btn.setText("Toggle Auto-Verify (OFF)")
            self.auto_verify_btn.setStyleSheet("")
            self.auto_verify_timer.stop()
            self.log("Auto-verify disabled")
        
    def append_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def update_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; background: transparent;")
        
    def update_verification(self, text, color):
        self.verification_label.setText(text)
        self.verification_label.setStyleSheet(f"color: {color}; background: transparent;")
        
    def show_message(self, msg_type, title, message):
        if msg_type == "info":
            QMessageBox.information(self, title, message)
        elif msg_type == "warning":
            QMessageBox.warning(self, title, message)
        elif msg_type == "error":
            QMessageBox.critical(self, title, message)
            
    def log(self, message):
        self.signals.log_signal.emit(message)
        
    def load_config(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_config.txt")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        self.server_entry.setText(lines[0].strip())
                        self.database_entry.setText(lines[1].strip())
            except:
                pass
                
    def save_config(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_config.txt")
        with open(config_path, "w") as f:
            f.write(f"{self.server_entry.text()}\n")
            f.write(f"{self.database_entry.text()}\n")
            
    def connect_database(self):
        server = self.server_entry.text().strip()
        database = self.database_entry.text().strip()
        username = self.username_entry.text().strip()
        password = self.password_entry.text()
        
        if not server or not database:
            QMessageBox.warning(self, "Warning", "Please enter server and database name")
            return
        
        try:
            if username:
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
            else:
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes"
            
            self.log(f"Connecting to {server}...")
            self.db_connection = pyodbc.connect(conn_str, timeout=10)
            self.create_tables()
            self.load_known_faces()
            self.db_status_label.setText("Connected")
            self.db_status_label.setStyleSheet("color: #00ff88; background: transparent;")
            self.log(f"Connected to database: {database}")
            self.save_config()
        except pyodbc.Error as e:
            self.db_status_label.setText("Connection Failed")
            self.db_status_label.setStyleSheet("color: #ff6b6b; background: transparent;")
            error_msg = str(e)
            self.log(f"Database error: {error_msg[:100]}")
            QMessageBox.critical(self, "Database Error", 
                f"Failed to connect to SQL Server.\n\n"
                f"Server: {server}\n"
                f"Database: {database}\n\n"
                f"Error: {error_msg}\n\n"
                f"Tips:\n"
                f"1. Make sure SQL Server is running\n"
                f"2. Check if the database exists\n"
                f"3. Verify ODBC Driver 17 is installed\n"
                f"4. For local instances, try: .\\SQLEXPRESS")
            
    def create_tables(self):
        cursor = self.db_connection.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='RegisteredFaces' AND xtype='U')
            CREATE TABLE RegisteredFaces (
                id INT IDENTITY(1,1) PRIMARY KEY,
                person_name NVARCHAR(255) NOT NULL,
                face_encoding TEXT NOT NULL,
                face_image VARBINARY(MAX),
                registered_date DATETIME DEFAULT GETDATE()
            )
        """)
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='VerificationLog' AND xtype='U')
            CREATE TABLE VerificationLog (
                id INT IDENTITY(1,1) PRIMARY KEY,
                person_name NVARCHAR(255),
                verification_result NVARCHAR(50),
                confidence FLOAT,
                timestamp DATETIME DEFAULT GETDATE()
            )
        """)
        self.db_connection.commit()
        self.log("Database tables ready")
        
    def load_known_faces(self):
        self.known_faces = {}
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT id, person_name, face_encoding FROM RegisteredFaces")
        rows = cursor.fetchall()
        for row in rows:
            person_id, name, encoding_str = row
            try:
                encoding = np.array([float(x) for x in encoding_str.split(',')])
                if name not in self.known_faces:
                    self.known_faces[name] = []
                self.known_faces[name].append({'id': person_id, 'encoding': encoding})
            except Exception as e:
                self.log(f"Error loading face: {e}")
        total = sum(len(v) for v in self.known_faces.values())
        self.log(f"Loaded {total} face(s) for {len(self.known_faces)} person(s)")
        
    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.log("Opening camera...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        if self.cap.isOpened():
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Status: Camera Running")
            self.status_label.setStyleSheet("color: #00ff88; background: transparent;")
            self.log("Camera started successfully")
            self.timer.start(33)
        else:
            QMessageBox.critical(self, "Error", "Could not open camera.\n\nMake sure a webcam is connected.")
            self.log("Failed to open camera")
            
    def stop_camera(self):
        self.is_running = False
        self.timer.stop()
        if self.auto_verify:
            self.toggle_auto_verify()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Camera Off")
        self.status_label.setStyleSheet("color: #ff6b6b; background: transparent;")
        self.camera_label.setText("Camera Off\n\nClick 'Start Camera' to begin")
        self.camera_label.setPixmap(QPixmap())
        self.log("Camera stopped")
        
    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_encoder.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_rgb, "Face Detected", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.camera_label.setPixmap(scaled_pixmap)
                
    def capture_and_register(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "No frame captured.\n\nStart the camera first.")
            return
            
        name = self.person_name_entry.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a person name")
            return
            
        if not self.db_connection:
            QMessageBox.warning(self, "Warning", "Please connect to database first")
            return
            
        self.log(f"Registering face for: {name}")
        
        def register_thread():
            encoding = self.face_encoder.get_encoding(self.current_frame)
            
            if encoding is not None:
                encoding_str = ','.join([str(x) for x in encoding])
                
                _, buffer = cv2.imencode('.jpg', self.current_frame)
                image_bytes = buffer.tobytes()
                
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute(
                        "INSERT INTO RegisteredFaces (person_name, face_encoding, face_image) VALUES (?, ?, ?)",
                        (name, encoding_str, image_bytes)
                    )
                    self.db_connection.commit()
                    
                    if name not in self.known_faces:
                        self.known_faces[name] = []
                    self.known_faces[name].append({'encoding': encoding})
                    
                    self.signals.log_signal.emit(f"Successfully registered: {name}")
                    self.signals.message_signal.emit("info", "Success", f"Face registered for {name}")
                except Exception as e:
                    self.signals.log_signal.emit(f"Database error: {str(e)}")
                    self.signals.message_signal.emit("error", "Error", f"Failed to save: {str(e)}")
            else:
                self.signals.log_signal.emit(f"No face detected for: {name}")
                self.signals.message_signal.emit("error", "Error", "No face detected in the frame.\n\nMake sure your face is visible to the camera.")
                
        threading.Thread(target=register_thread, daemon=True).start()
        
    def import_face_from_image(self):
        name = self.person_name_entry.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a person name first")
            return
            
        if not self.db_connection:
            QMessageBox.warning(self, "Warning", "Please connect to database first")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Face Image",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        
        if not file_path:
            return
            
        self.log(f"Importing face for: {name}")
        
        def import_thread():
            image = cv2.imread(file_path)
            if image is None:
                self.signals.message_signal.emit("error", "Error", "Could not read the image file")
                return
                
            encoding = self.face_encoder.get_encoding(image)
            
            if encoding is not None:
                encoding_str = ','.join([str(x) for x in encoding])
                
                with open(file_path, 'rb') as f:
                    image_bytes = f.read()
                
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute(
                        "INSERT INTO RegisteredFaces (person_name, face_encoding, face_image) VALUES (?, ?, ?)",
                        (name, encoding_str, image_bytes)
                    )
                    self.db_connection.commit()
                    
                    if name not in self.known_faces:
                        self.known_faces[name] = []
                    self.known_faces[name].append({'encoding': encoding})
                    
                    self.signals.log_signal.emit(f"Successfully imported: {name}")
                    self.signals.message_signal.emit("info", "Success", f"Face imported for {name}")
                except Exception as e:
                    self.signals.message_signal.emit("error", "Error", f"Database error: {str(e)}")
            else:
                self.signals.log_signal.emit(f"No face detected in image for: {name}")
                self.signals.message_signal.emit("error", "Error", "No face detected in the selected image")
                
        threading.Thread(target=import_thread, daemon=True).start()
        
    def verify_face(self):
        if self.current_frame is None:
            if not self.auto_verify:
                QMessageBox.warning(self, "Warning", "No frame captured.\n\nStart the camera first.")
            return
            
        if not self.known_faces:
            if not self.auto_verify:
                QMessageBox.warning(self, "Warning", "No registered faces.\n\nRegister someone first.")
            return
            
        self.verification_label.setText("Verifying...")
        self.verification_label.setStyleSheet("color: #ffff00; background: transparent;")
        
        def verify_thread():
            current_encoding = self.face_encoder.get_encoding(self.current_frame)
            
            if current_encoding is None:
                self.signals.verification_signal.emit("NO FACE DETECTED", "#ff6b6b")
                return
                
            best_match = None
            best_similarity = 0
            
            for name, face_list in self.known_faces.items():
                for face_data in face_list:
                    known_encoding = face_data['encoding']
                    similarity = self.face_encoder.compare_faces(current_encoding, known_encoding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
                        
            if best_similarity > self.verification_threshold:
                result_text = f"VERIFIED: {best_match}"
                result_color = "#00ff88"
                result_status = "VERIFIED"
                self.signals.log_signal.emit(f"Verified: {best_match} ({best_similarity:.1%})")
            else:
                result_text = "UNKNOWN PERSON"
                result_color = "#ff6b6b"
                result_status = "UNKNOWN"
                if best_match:
                    self.signals.log_signal.emit(f"Unknown (closest: {best_match} at {best_similarity:.1%})")
                else:
                    self.signals.log_signal.emit("Unknown person detected")
                
            if self.db_connection:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute(
                        "INSERT INTO VerificationLog (person_name, verification_result, confidence) VALUES (?, ?, ?)",
                        (best_match or "Unknown", result_status, best_similarity)
                    )
                    self.db_connection.commit()
                except:
                    pass
                
            self.signals.verification_signal.emit(result_text, result_color)
            
        threading.Thread(target=verify_thread, daemon=True).start()
        
    def list_persons(self):
        if not self.db_connection:
            QMessageBox.warning(self, "Warning", "Please connect to database first")
            return
            
        cursor = self.db_connection.cursor()
        cursor.execute("""
            SELECT person_name, COUNT(*) as face_count, MAX(registered_date) as last_registered 
            FROM RegisteredFaces 
            GROUP BY person_name 
            ORDER BY last_registered DESC
        """)
        rows = cursor.fetchall()
        
        if not rows:
            QMessageBox.information(self, "Registered Persons", "No persons registered yet.\n\nUse 'Capture & Register Face' to add someone.")
            return
            
        persons_list = "\n".join([f"- {row[0]} ({row[1]} face(s), last: {row[2]})" for row in rows])
        QMessageBox.information(self, "Registered Persons", f"Registered faces:\n\n{persons_list}")
        
    def closeEvent(self, event):
        self.stop_camera()
        if self.db_connection:
            self.db_connection.close()
        event.accept()


def main():
    if not PYQT_AVAILABLE:
        print("ERROR: PyQt5 is required. Install with: pip install PyQt5")
        sys.exit(1)
        
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(26, 26, 46))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(15, 15, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(26, 26, 46))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(15, 52, 96))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(0, 212, 255))
    dark_palette.setColor(QPalette.Highlight, QColor(0, 212, 255))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    window = FaceVerificationApp()
    window.show()
    
    print("\n" + "="*60)
    print("Face Verification System Started!")
    print("="*60)
    print("1. Connect to your SQL Server database")
    print("2. Start the camera")
    print("3. Register faces or verify identity")
    print("="*60 + "\n")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
