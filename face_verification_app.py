"""
Face Verification System - Ring Camera Style
=============================================
A desktop application for face verification using machine learning.
Uses local SQLite database - works immediately without external setup.
"""

import sys
import os
import sqlite3
import threading
from datetime import datetime

try:
    import cv2
    import numpy as np
    from scipy.spatial.distance import cosine
    from PIL import Image
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                                 QTextEdit, QGroupBox, QMessageBox, QFileDialog,
                                 QFrame, QCheckBox)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QMutex, QMutexLocker
    from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nPlease install required packages with:")
    print("    pip install opencv-python numpy Pillow scipy PyQt5")
    sys.exit(1)


class FaceEncoder:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def detect_face(self, image):
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return gray[y:y+h, x:x+w]
    
    def get_encoding(self, image):
        if image is None:
            return None
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


class LocalDatabase:
    def __init__(self, db_path="faces.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
        
    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS RegisteredFaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    face_encoding TEXT NOT NULL,
                    face_image BLOB,
                    registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS VerificationLog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT,
                    verification_result TEXT,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
            
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def add_face(self, name, encoding, image_bytes):
        encoding_str = ','.join([str(x) for x in encoding])
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO RegisteredFaces (person_name, face_encoding, face_image) VALUES (?, ?, ?)",
                (name, encoding_str, image_bytes)
            )
            conn.commit()
            conn.close()
            
    def get_all_faces(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, person_name, face_encoding FROM RegisteredFaces")
        rows = cursor.fetchall()
        conn.close()
        
        faces = {}
        for row in rows:
            person_id, name, encoding_str = row
            try:
                encoding = np.array([float(x) for x in encoding_str.split(',')])
                if name not in faces:
                    faces[name] = []
                faces[name].append({'id': person_id, 'encoding': encoding})
            except:
                pass
        return faces
    
    def get_persons_list(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT person_name, COUNT(*) as face_count, MAX(registered_date) as last_registered 
            FROM RegisteredFaces 
            GROUP BY person_name 
            ORDER BY last_registered DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def log_verification(self, person_name, result, confidence):
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO VerificationLog (person_name, verification_result, confidence) VALUES (?, ?, ?)",
                (person_name, result, confidence)
            )
            conn.commit()
            conn.close()
            
    def delete_person(self, name):
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM RegisteredFaces WHERE person_name = ?", (name,))
            conn.commit()
            conn.close()


class SignalBridge(QObject):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str, str)
    verification_signal = pyqtSignal(str, str)
    message_signal = pyqtSignal(str, str, str)
    update_faces_signal = pyqtSignal()


class FaceVerificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Verification System")
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
            QCheckBox { color: white; }
        """)
        
        self.db = LocalDatabase()
        self.cap = None
        self.is_running = False
        self.known_faces = {}
        self.verification_threshold = 0.70
        self.current_frame = None
        self.current_image = None
        self.frame_mutex = QMutex()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.face_encoder = FaceEncoder()
        
        self.signals = SignalBridge()
        self.signals.log_signal.connect(self.append_log)
        self.signals.status_signal.connect(self.update_status)
        self.signals.verification_signal.connect(self.update_verification)
        self.signals.message_signal.connect(self.show_message)
        self.signals.update_faces_signal.connect(self.reload_faces)
        
        self.setup_ui()
        self.reload_faces()
        self.log("Application started - Local SQLite database ready")
        self.log(f"Loaded {sum(len(v) for v in self.known_faces.values())} registered face(s)")
        
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
        
        header = QLabel("Face Verification System")
        header.setFont(QFont("Segoe UI", 18, QFont.Bold))
        header.setStyleSheet("color: #00d4ff; background: transparent;")
        header.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(header)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #0f0f23; border-radius: 5px; border: 2px solid #333;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No Image\n\nLoad an image or start camera")
        left_layout.addWidget(self.image_label)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: #00ff88; background: transparent;")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)
        
        self.verification_label = QLabel("")
        self.verification_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.verification_label.setStyleSheet("background: transparent;")
        self.verification_label.setAlignment(Qt.AlignCenter)
        self.verification_label.setMinimumHeight(50)
        left_layout.addWidget(self.verification_label)
        
        main_layout.addWidget(left_panel, stretch=2)
        
        right_panel = QFrame()
        right_panel.setFixedWidth(380)
        right_panel.setStyleSheet("background-color: #16213e; border-radius: 10px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        
        source_group = QGroupBox("Image Source")
        source_layout = QVBoxLayout(source_group)
        
        self.load_image_btn = QPushButton("Load Image from File")
        self.load_image_btn.setStyleSheet("background-color: #1e88e5;")
        self.load_image_btn.clicked.connect(self.load_image)
        source_layout.addWidget(self.load_image_btn)
        
        cam_layout = QHBoxLayout()
        self.start_cam_btn = QPushButton("Start Camera")
        self.start_cam_btn.setStyleSheet("background-color: #2e7d32;")
        self.start_cam_btn.clicked.connect(self.start_camera)
        cam_layout.addWidget(self.start_cam_btn)
        
        self.stop_cam_btn = QPushButton("Stop Camera")
        self.stop_cam_btn.setStyleSheet("background-color: #c62828;")
        self.stop_cam_btn.clicked.connect(self.stop_camera)
        self.stop_cam_btn.setEnabled(False)
        cam_layout.addWidget(self.stop_cam_btn)
        source_layout.addLayout(cam_layout)
        
        right_layout.addWidget(source_group)
        
        register_group = QGroupBox("Register New Face")
        register_layout = QVBoxLayout(register_group)
        
        register_layout.addWidget(QLabel("Person Name:"))
        self.person_name_entry = QLineEdit()
        self.person_name_entry.setPlaceholderText("Enter person's name...")
        register_layout.addWidget(self.person_name_entry)
        
        self.register_btn = QPushButton("Register Current Face")
        self.register_btn.setStyleSheet("background-color: #6a1b9a;")
        self.register_btn.clicked.connect(self.register_face)
        register_layout.addWidget(self.register_btn)
        
        right_layout.addWidget(register_group)
        
        verify_group = QGroupBox("Verify Identity")
        verify_layout = QVBoxLayout(verify_group)
        
        self.verify_btn = QPushButton("Verify Current Face")
        self.verify_btn.setStyleSheet("background-color: #00897b; font-weight: bold; font-size: 14px; padding: 15px;")
        self.verify_btn.clicked.connect(self.verify_face)
        verify_layout.addWidget(self.verify_btn)
        
        self.auto_verify_cb = QCheckBox("Auto-verify every 2 seconds")
        self.auto_verify_cb.stateChanged.connect(self.toggle_auto_verify)
        verify_layout.addWidget(self.auto_verify_cb)
        
        self.auto_verify_timer = QTimer()
        self.auto_verify_timer.timeout.connect(self.verify_face)
        
        right_layout.addWidget(verify_group)
        
        manage_group = QGroupBox("Manage Faces")
        manage_layout = QVBoxLayout(manage_group)
        
        self.list_btn = QPushButton("List Registered Persons")
        self.list_btn.clicked.connect(self.list_persons)
        manage_layout.addWidget(self.list_btn)
        
        delete_layout = QHBoxLayout()
        self.delete_name_entry = QLineEdit()
        self.delete_name_entry.setPlaceholderText("Name to delete...")
        delete_layout.addWidget(self.delete_name_entry)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setStyleSheet("background-color: #c62828;")
        self.delete_btn.clicked.connect(self.delete_person)
        delete_layout.addWidget(self.delete_btn)
        manage_layout.addLayout(delete_layout)
        
        right_layout.addWidget(manage_group)
        
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
        
    def reload_faces(self):
        self.known_faces = self.db.get_all_faces()
        
    def toggle_auto_verify(self, state):
        if state == Qt.Checked:
            self.auto_verify_timer.start(2000)
            self.log("Auto-verify enabled")
        else:
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
        self.verification_label.setStyleSheet(f"color: {color}; background: transparent; font-size: 18px;")
        
    def show_message(self, msg_type, title, message):
        if msg_type == "info":
            QMessageBox.information(self, title, message)
        elif msg_type == "warning":
            QMessageBox.warning(self, title, message)
        elif msg_type == "error":
            QMessageBox.critical(self, title, message)
            
    def log(self, message):
        self.signals.log_signal.emit(message)
    
    def get_current_image(self):
        locker = QMutexLocker(self.frame_mutex)
        if self.current_image is not None:
            return self.current_image.copy()
        return None
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        
        if not file_path:
            return
            
        image = cv2.imread(file_path)
        if image is None:
            self.log("Failed to load image")
            return
            
        locker = QMutexLocker(self.frame_mutex)
        self.current_image = image.copy()
        locker.unlock()
        
        self.display_image(image)
        self.log(f"Loaded image: {os.path.basename(file_path)}")
        self.update_status("Status: Image loaded", "#00ff88")
        
    def display_image(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_encoder.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame_rgb, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        
    def start_camera(self):
        self.log("Starting camera...")
        self.cap = cv2.VideoCapture(0)
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            self.start_cam_btn.setEnabled(False)
            self.stop_cam_btn.setEnabled(True)
            self.update_status("Status: Camera Running", "#00ff88")
            self.log("Camera started")
            self.timer.start(33)
        else:
            self.log("No camera found - use 'Load Image' instead")
            self.update_status("Status: No camera available", "#ff6b6b")
            QMessageBox.warning(self, "Camera", "No camera detected.\n\nUse 'Load Image from File' to test with photos.")
            
    def stop_camera(self):
        self.is_running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_cam_btn.setEnabled(True)
        self.stop_cam_btn.setEnabled(False)
        self.update_status("Status: Camera stopped", "#ffaa00")
        self.log("Camera stopped")
        
    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                locker = QMutexLocker(self.frame_mutex)
                self.current_image = frame.copy()
                locker.unlock()
                self.display_image(frame)
                
    def register_face(self):
        image = self.get_current_image()
        if image is None:
            QMessageBox.warning(self, "Warning", "No image loaded.\n\nLoad an image or start the camera first.")
            return
            
        name = self.person_name_entry.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a person name")
            return
            
        self.log(f"Registering face for: {name}")
        
        def register_thread(img, person_name):
            encoding = self.face_encoder.get_encoding(img)
            
            if encoding is not None:
                _, buffer = cv2.imencode('.jpg', img)
                image_bytes = buffer.tobytes()
                
                try:
                    self.db.add_face(person_name, encoding, image_bytes)
                    self.signals.update_faces_signal.emit()
                    self.signals.log_signal.emit(f"Registered: {person_name}")
                    self.signals.message_signal.emit("info", "Success", f"Face registered for {person_name}")
                except Exception as e:
                    self.signals.log_signal.emit(f"Error: {str(e)}")
                    self.signals.message_signal.emit("error", "Error", f"Failed: {str(e)}")
            else:
                self.signals.log_signal.emit(f"No face detected for: {person_name}")
                self.signals.message_signal.emit("error", "Error", "No face detected in the image.\n\nMake sure a face is clearly visible.")
                
        threading.Thread(target=register_thread, args=(image, name), daemon=True).start()
        
    def verify_face(self):
        image = self.get_current_image()
        if image is None:
            if not self.auto_verify_cb.isChecked():
                QMessageBox.warning(self, "Warning", "No image loaded.\n\nLoad an image or start the camera first.")
            return
            
        if not self.known_faces:
            if not self.auto_verify_cb.isChecked():
                QMessageBox.warning(self, "Warning", "No registered faces.\n\nRegister someone first.")
            return
            
        self.verification_label.setText("Verifying...")
        self.verification_label.setStyleSheet("color: #ffff00; background: transparent;")
        
        locker = QMutexLocker(self.frame_mutex)
        faces_snapshot = {k: [{'encoding': f['encoding'].copy()} for f in v] for k, v in self.known_faces.items()}
        locker.unlock()
        
        def verify_thread(img, faces):
            current_encoding = self.face_encoder.get_encoding(img)
            
            if current_encoding is None:
                self.signals.verification_signal.emit("NO FACE DETECTED", "#ff6b6b")
                self.signals.log_signal.emit("No face detected")
                return
                
            best_match = None
            best_similarity = 0
            
            for name, face_list in faces.items():
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
                    self.signals.log_signal.emit("Unknown person")
                    
            self.db.log_verification(best_match or "Unknown", result_status, best_similarity)
            self.signals.verification_signal.emit(result_text, result_color)
            
        threading.Thread(target=verify_thread, args=(image, faces_snapshot), daemon=True).start()
        
    def list_persons(self):
        rows = self.db.get_persons_list()
        
        if not rows:
            QMessageBox.information(self, "Registered Persons", "No persons registered yet.\n\nUse 'Register Current Face' to add someone.")
            return
            
        persons_list = "\n".join([f"- {row[0]} ({row[1]} face(s))" for row in rows])
        QMessageBox.information(self, "Registered Persons", f"Registered faces:\n\n{persons_list}")
        
    def delete_person(self):
        name = self.delete_name_entry.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Enter a name to delete")
            return
            
        reply = QMessageBox.question(self, "Confirm Delete", 
            f"Delete all faces for '{name}'?",
            QMessageBox.Yes | QMessageBox.No)
            
        if reply == QMessageBox.Yes:
            self.db.delete_person(name)
            self.reload_faces()
            self.log(f"Deleted: {name}")
            self.delete_name_entry.clear()
        
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


def main():
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
    
    print("\n" + "="*50)
    print("Face Verification System - Ready!")
    print("="*50)
    print("Using local SQLite database (faces.db)")
    print("="*50 + "\n")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
