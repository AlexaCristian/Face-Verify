"""
Face Verification System - Ring Camera Style
Beautiful modern desktop application for face verification.
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
                                 QFrame, QCheckBox, QSplitter, QScrollArea,
                                 QGraphicsDropShadowEffect)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QMutex, QMutexLocker
    from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QLinearGradient, QPainter
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)


class LivenessDetector:
    """Anti-spoofing: Detect if face is real or a photo/screen"""
    
    def __init__(self):
        self.min_score = 0.55
        
    def check_liveness(self, image, face_rect):
        """Returns (is_live, confidence, details)"""
        if image is None or face_rect is None:
            return False, 0.0, "No face"
            
        x, y, w, h = face_rect
        pad = int(w * 0.1)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return False, 0.0, "Invalid region"
        
        scores = {}
        
        scores['texture'] = self._analyze_texture(face_region)
        scores['color'] = self._analyze_color(face_region)
        scores['frequency'] = self._analyze_frequency(face_region)
        scores['reflection'] = self._detect_reflection(face_region)
        scores['sharpness'] = self._analyze_sharpness(face_region)
        scores['skin'] = self._analyze_skin(face_region)
        scores['noise'] = self._analyze_noise(face_region)
        scores['gradient'] = self._analyze_gradient(face_region)
        
        weights = {
            'texture': 0.20,
            'color': 0.15,
            'frequency': 0.15,
            'reflection': 0.10,
            'sharpness': 0.10,
            'skin': 0.15,
            'noise': 0.08,
            'gradient': 0.07
        }
        
        final_score = sum(scores[k] * weights[k] for k in scores)
        is_live = final_score >= self.min_score
        
        details = ", ".join([f"{k}:{v:.2f}" for k, v in scores.items()])
        return is_live, final_score, details
    
    def _analyze_texture(self, face):
        """LBP texture variance - real faces have more texture variation"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        gray = cv2.resize(gray, (64, 64))
        
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        variance = np.var(lbp)
        score = min(1.0, variance / 3000)
        return score
    
    def _analyze_color(self, face):
        """Color distribution analysis - photos have different color patterns"""
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        
        h_std = np.std(hsv[:,:,0])
        s_std = np.std(hsv[:,:,1])
        v_std = np.std(hsv[:,:,2])
        
        s_mean = np.mean(hsv[:,:,1])
        
        color_var = (h_std + s_std + v_std) / 3
        score = min(1.0, color_var / 50)
        
        if s_mean < 30:
            score *= 0.7
        
        return score
    
    def _analyze_frequency(self, face):
        """FFT analysis - printed photos have specific frequency patterns"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        gray = cv2.resize(gray, (128, 128))
        
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        low_freq = magnitude[center_y-10:center_y+10, center_x-10:center_x+10].mean()
        high_freq = magnitude.mean()
        
        ratio = high_freq / (low_freq + 1e-10)
        score = min(1.0, ratio * 5)
        
        return score
    
    def _detect_reflection(self, face):
        """Detect unnatural reflections from printed/screen photos"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        
        bright_pixels = np.sum(gray > 240)
        total_pixels = gray.size
        bright_ratio = bright_pixels / total_pixels
        
        if bright_ratio > 0.15:
            return 0.3
        elif bright_ratio > 0.08:
            return 0.6
        else:
            return 1.0
    
    def _analyze_sharpness(self, face):
        """Laplacian variance for sharpness - photos of photos are less sharp"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        gray = cv2.resize(gray, (100, 100))
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        score = min(1.0, variance / 500)
        return score
    
    def _analyze_skin(self, face):
        """Skin color detection in YCrCb space"""
        ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        
        if 0.2 < skin_ratio < 0.8:
            return 1.0
        elif 0.1 < skin_ratio < 0.9:
            return 0.7
        else:
            return 0.3
    
    def _analyze_noise(self, face):
        """Noise pattern analysis - screens/prints have different noise"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        gray = cv2.resize(gray, (64, 64)).astype(np.float32)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.abs(gray - blurred)
        noise_std = np.std(noise)
        
        if 2 < noise_std < 15:
            return 1.0
        elif 1 < noise_std < 20:
            return 0.6
        else:
            return 0.3
    
    def _analyze_gradient(self, face):
        """Gradient consistency - real faces have natural gradient patterns"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        gray = cv2.resize(gray, (64, 64))
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        grad_mean = np.mean(magnitude)
        grad_std = np.std(magnitude)
        
        ratio = grad_std / (grad_mean + 1e-10)
        score = min(1.0, ratio / 1.5)
        
        return score


class FaceEncoder:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.liveness_detector = LivenessDetector()
    
    def detect_face_rect(self, image):
        """Returns (face_gray, (x, y, w, h)) or (None, None)"""
        if image is None:
            return None, None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) == 0:
            return None, None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return gray[y:y+h, x:x+w], (x, y, w, h)
        
    def detect_face(self, image):
        face, _ = self.detect_face_rect(image)
        return face
    
    def check_liveness(self, image):
        """Check if face is real or photo. Returns (is_live, score, details)"""
        _, face_rect = self.detect_face_rect(image)
        if face_rect is None:
            return False, 0.0, "No face detected"
        return self.liveness_detector.check_liveness(image, face_rect)
    
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


class GlowButton(QPushButton):
    def __init__(self, text, color="#667eea", parent=None):
        super().__init__(text, parent)
        self.base_color = color
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {color}, stop:1 {self._adjust_color(color, -30)});
                color: white;
                border: none;
                padding: 12px 20px;
                font-size: 13px;
                font-weight: 600;
                border-radius: 8px;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {self._adjust_color(color, 20)}, stop:1 {color});
            }}
            QPushButton:pressed {{
                background: {self._adjust_color(color, -40)};
            }}
            QPushButton:disabled {{
                background: #3a3a5c;
                color: #666;
            }}
        """)
        
    def _adjust_color(self, color, amount):
        c = QColor(color)
        h, s, l, a = c.getHsl()
        l = max(0, min(255, l + amount))
        c.setHsl(h, s, l, a)
        return c.name()


class FaceVerificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Verification System")
        self.setGeometry(50, 50, 1300, 850)
        self.setMinimumSize(1100, 750)
        
        self.db = LocalDatabase()
        self.cap = None
        self.is_running = False
        self.known_faces = {}
        self.verification_threshold = 0.70
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
        self.log("System ready")
        face_count = sum(len(v) for v in self.known_faces.values())
        if face_count > 0:
            self.log(f"{face_count} registered face(s)")
        
    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f0c29, stop:0.5 #302b63, stop:1 #24243e);
            }
            QLabel {
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLineEdit {
                background: rgba(255, 255, 255, 0.08);
                border: 2px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 10px 15px;
                color: white;
                font-size: 13px;
                selection-background-color: #667eea;
            }
            QLineEdit:focus {
                border: 2px solid #667eea;
                background: rgba(255, 255, 255, 0.12);
            }
            QLineEdit::placeholder {
                color: rgba(255, 255, 255, 0.4);
            }
            QTextEdit {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                color: #00ff88;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 8px;
            }
            QCheckBox {
                color: #e0e0e0;
                font-size: 12px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                background: transparent;
            }
            QCheckBox::indicator:checked {
                background: #667eea;
                border-color: #667eea;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(25)
        
        left_panel = QFrame()
        left_panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(25, 25, 25, 25)
        left_layout.setSpacing(15)
        
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)
        
        header = QLabel("Face Verification")
        header.setFont(QFont("Segoe UI", 28, QFont.Bold))
        header.setStyleSheet("""
            color: transparent;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
            background-clip: text;
            -webkit-background-clip: text;
            color: #667eea;
        """)
        header.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(header)
        
        subtitle = QLabel("AI-Powered Security System")
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.5);")
        subtitle.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(subtitle)
        
        left_layout.addWidget(title_widget)
        
        image_container = QFrame()
        image_container.setStyleSheet("""
            QFrame {
                background: rgba(0, 0, 0, 0.4);
                border-radius: 15px;
                border: 2px solid rgba(255, 255, 255, 0.1);
            }
        """)
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(3, 3, 3, 3)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(600, 450)
        self.image_label.setStyleSheet("""
            background: transparent;
            border-radius: 12px;
        """)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No Image Loaded")
        self.image_label.setFont(QFont("Segoe UI", 14))
        image_layout.addWidget(self.image_label)
        
        left_layout.addWidget(image_container)
        
        status_widget = QFrame()
        status_widget.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 10px;
            }
        """)
        status_layout = QVBoxLayout(status_widget)
        status_layout.setContentsMargins(15, 10, 15, 10)
        status_layout.setSpacing(8)
        
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Segoe UI", 12))
        self.status_label.setStyleSheet("color: #00ff88;")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        self.verification_label = QLabel("")
        self.verification_label.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.verification_label.setAlignment(Qt.AlignCenter)
        self.verification_label.setMinimumHeight(50)
        status_layout.addWidget(self.verification_label)
        
        left_layout.addWidget(status_widget)
        main_layout.addWidget(left_panel, stretch=2)
        
        right_panel = QFrame()
        right_panel.setFixedWidth(380)
        right_panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(15)
        
        source_card = self._create_card("Image Source")
        source_layout = source_card.layout()
        
        self.load_image_btn = GlowButton("Load Image from File", "#3498db")
        self.load_image_btn.clicked.connect(self.load_image)
        source_layout.addWidget(self.load_image_btn)
        
        cam_layout = QHBoxLayout()
        cam_layout.setSpacing(10)
        self.start_cam_btn = GlowButton("Start Camera", "#27ae60")
        self.start_cam_btn.clicked.connect(self.start_camera)
        cam_layout.addWidget(self.start_cam_btn)
        
        self.stop_cam_btn = GlowButton("Stop", "#e74c3c")
        self.stop_cam_btn.clicked.connect(self.stop_camera)
        self.stop_cam_btn.setEnabled(False)
        self.stop_cam_btn.setFixedWidth(80)
        cam_layout.addWidget(self.stop_cam_btn)
        source_layout.addLayout(cam_layout)
        
        right_layout.addWidget(source_card)
        
        register_card = self._create_card("Register Face")
        register_layout = register_card.layout()
        
        name_label = QLabel("Person Name")
        name_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px; margin-bottom: 2px;")
        register_layout.addWidget(name_label)
        
        self.person_name_entry = QLineEdit()
        self.person_name_entry.setPlaceholderText("Enter name...")
        register_layout.addWidget(self.person_name_entry)
        
        self.register_btn = GlowButton("Register Current Face", "#9b59b6")
        self.register_btn.clicked.connect(self.register_face)
        register_layout.addWidget(self.register_btn)
        
        right_layout.addWidget(register_card)
        
        verify_card = self._create_card("Verify Identity")
        verify_layout = verify_card.layout()
        
        self.verify_btn = GlowButton("VERIFY FACE", "#667eea")
        self.verify_btn.setStyleSheet(self.verify_btn.styleSheet() + """
            QPushButton {
                font-size: 16px;
                padding: 18px;
                font-weight: 700;
                letter-spacing: 2px;
            }
        """)
        self.verify_btn.clicked.connect(self.verify_face)
        verify_layout.addWidget(self.verify_btn)
        
        self.auto_verify_cb = QCheckBox("Auto-verify every 2 seconds")
        self.auto_verify_cb.stateChanged.connect(self.toggle_auto_verify)
        verify_layout.addWidget(self.auto_verify_cb)
        
        self.auto_verify_timer = QTimer()
        self.auto_verify_timer.timeout.connect(self.verify_face)
        
        right_layout.addWidget(verify_card)
        
        log_card = self._create_card("Activity Log")
        log_layout = log_card.layout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        
        right_layout.addWidget(log_card)
        right_layout.addStretch()
        
        main_layout.addWidget(right_panel)
        
    def _create_card(self, title):
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 15, 18, 18)
        layout.setSpacing(12)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title_label.setStyleSheet("color: white; border: none; background: transparent;")
        layout.addWidget(title_label)
        
        return card
        
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
        self.status_label.setStyleSheet(f"color: {color};")
        
    def update_verification(self, text, color):
        self.verification_label.setText(text)
        self.verification_label.setStyleSheet(f"color: {color}; font-size: 24px;")
        
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
        self.log(f"Loaded: {os.path.basename(file_path)}")
        self.update_status("Image loaded", "#00ff88")
        
    def display_image(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_encoder.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            is_live, score, _ = self.face_encoder.liveness_detector.check_liveness(image, (x, y, w, h))
            
            if is_live:
                color = (46, 204, 113)
                label = f"LIVE {score:.0%}"
            else:
                color = (231, 126, 34)
                label = f"PHOTO {score:.0%}"
            
            cv2.rectangle(frame_rgb, (x-2, y-2), (x+w+2, y+h+2), color, 3)
            overlay = frame_rgb.copy()
            label_width = max(120, len(label) * 12)
            cv2.rectangle(overlay, (x, y-30), (x+label_width, y), color, -1)
            cv2.addWeighted(overlay, 0.8, frame_rgb, 0.2, 0, frame_rgb)
            cv2.putText(frame_rgb, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
            self.update_status("Camera active", "#00ff88")
            self.log("Camera started")
            self.timer.start(33)
        else:
            self.log("No camera found")
            self.update_status("No camera", "#ff6b6b")
            QMessageBox.warning(self, "Camera", "No camera detected.\n\nUse 'Load Image' to test with photos.")
            
    def stop_camera(self):
        self.is_running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_cam_btn.setEnabled(True)
        self.stop_cam_btn.setEnabled(False)
        self.update_status("Camera stopped", "#ffaa00")
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
            
        self.log(f"Registering: {name}")
        
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
                    self.signals.message_signal.emit("error", "Error", f"Failed: {str(e)}")
            else:
                self.signals.log_signal.emit("No face detected")
                self.signals.message_signal.emit("error", "Error", "No face detected in the image.")
                
        threading.Thread(target=register_thread, args=(image, name), daemon=True).start()
        
    def verify_face(self):
        image = self.get_current_image()
        if image is None:
            if not self.auto_verify_cb.isChecked():
                QMessageBox.warning(self, "Warning", "No image loaded.")
            return
            
        if not self.known_faces:
            if not self.auto_verify_cb.isChecked():
                QMessageBox.warning(self, "Warning", "No registered faces.\n\nRegister someone first.")
            return
            
        self.verification_label.setText("Analyzing...")
        self.verification_label.setStyleSheet("color: #f1c40f; font-size: 24px;")
        
        locker = QMutexLocker(self.frame_mutex)
        faces_snapshot = {k: [{'encoding': f['encoding'].copy()} for f in v] for k, v in self.known_faces.items()}
        locker.unlock()
        
        def verify_thread(img, faces):
            is_live, liveness_score, liveness_details = self.face_encoder.check_liveness(img)
            
            if not is_live:
                self.signals.verification_signal.emit("PHOTO DETECTED", "#e67e22")
                self.signals.log_signal.emit(f"Spoof detected (score: {liveness_score:.2f})")
                self.db.log_verification("SPOOF", "PHOTO_DETECTED", liveness_score)
                return
            
            current_encoding = self.face_encoder.get_encoding(img)
            
            if current_encoding is None:
                self.signals.verification_signal.emit("NO FACE DETECTED", "#e74c3c")
                self.signals.log_signal.emit("No face detected")
                return
            
            self.signals.log_signal.emit(f"Liveness OK ({liveness_score:.0%})")
                
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
                result_color = "#2ecc71"
                result_status = "VERIFIED"
                self.signals.log_signal.emit(f"Verified: {best_match} ({best_similarity:.1%})")
            else:
                result_text = "UNKNOWN PERSON"
                result_color = "#e74c3c"
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
            QMessageBox.information(self, "Registered Persons", "No persons registered yet.")
            return
            
        persons_list = "\n".join([f"  {row[0]} - {row[1]} face(s)" for row in rows])
        QMessageBox.information(self, "Registered Persons", f"Registered:\n\n{persons_list}")
        
    def delete_person(self):
        name = self.delete_name_entry.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Enter a name to delete")
            return
            
        reply = QMessageBox.question(self, "Confirm", 
            f"Delete '{name}'?",
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
    dark_palette.setColor(QPalette.Window, QColor(15, 12, 41))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(20, 20, 40))
    dark_palette.setColor(QPalette.AlternateBase, QColor(30, 30, 50))
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(102, 126, 234))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.Highlight, QColor(102, 126, 234))
    dark_palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(dark_palette)
    
    window = FaceVerificationApp()
    window.show()
    
    print("\n" + "="*50)
    print("Face Verification System")
    print("="*50 + "\n")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
