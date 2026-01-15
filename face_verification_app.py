"""
Face Verification System - Ring Camera Style
Beautiful modern desktop application for face verification.
"""

import sys
import os
import sqlite3
import threading
import json
import base64
from datetime import datetime, date

try:
    import cv2
    import numpy as np
    from scipy.spatial.distance import cosine
    from PIL import Image
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QPushButton, QLineEdit,
                                 QTextEdit, QGroupBox, QMessageBox, QFileDialog,
                                 QFrame, QCheckBox, QSplitter, QScrollArea,
                                 QGraphicsDropShadowEffect, QComboBox, QSlider,
                                 QProgressBar, QToolTip, QShortcut, QListWidget,
                                 QListWidgetItem)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QMutex, QMutexLocker, QPropertyAnimation, QEasingCurve, QRect
    from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QLinearGradient, QPainter, QKeySequence
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

CONFIG_FILE = "db_config.json"


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


class ActionLivenessDetector:
    """Action-based liveness detection - analyzes the PROCESS of smiling/mouth opening"""

    def __init__(self):
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.state_history = []
        self.max_history = 20
        self.challenge_type = None
        self.challenge_start_time = None
        self.challenge_timeout = 8.0
        self.baseline_state = None
        self.action_detected = False
        self.transition_frames = []
        self.min_transition_frames = 5
        self.transition_threshold = 0.25
        self.peak_score_threshold = 0.55

    def reset(self):
        """Reset detector state for new challenge"""
        self.state_history = []
        self.baseline_state = None
        self.action_detected = False
        self.transition_frames = []
        self.challenge_start_time = datetime.now()

    def start_challenge(self, challenge_type="smile"):
        """Start a new challenge: 'smile' or 'open_mouth'"""
        self.challenge_type = challenge_type
        self.reset()
        return challenge_type

    def get_random_challenge(self):
        """Get a random challenge type"""
        import random
        challenges = ["smile", "open_mouth"]
        return random.choice(challenges)

    def _extract_mouth_region(self, image, face_rect):
        """Extract the mouth region from face"""
        if image is None or face_rect is None:
            return None
        x, y, w, h = face_rect
        mouth_y = y + int(h * 0.6)
        mouth_h = int(h * 0.35)
        mouth_x = x + int(w * 0.2)
        mouth_w = int(w * 0.6)

        mouth_y = max(0, mouth_y)
        mouth_x = max(0, mouth_x)
        mouth_h = min(mouth_h, image.shape[0] - mouth_y)
        mouth_w = min(mouth_w, image.shape[1] - mouth_x)

        if mouth_h <= 0 or mouth_w <= 0:
            return None

        return image[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]

    def _analyze_mouth_state(self, mouth_region):
        """Analyze mouth state and return metrics"""
        if mouth_region is None or mouth_region.size == 0:
            return None

        gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY) if len(mouth_region.shape) == 3 else mouth_region
        gray = cv2.resize(gray, (80, 40))

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        upper_half = edges[:20, :]
        lower_half = edges[20:, :]

        upper_edges = np.sum(upper_half > 0)
        lower_edges = np.sum(lower_half > 0)
        total_edges = upper_edges + lower_edges + 1

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        horizontal_gradient = np.mean(np.abs(sobelx))
        vertical_gradient = np.mean(np.abs(sobely))

        center_region = gray[12:28, 20:60]
        center_brightness = np.mean(center_region)
        center_std = np.std(center_region)

        dark_pixels = np.sum(center_region < 60)
        dark_ratio = dark_pixels / center_region.size

        left_corner = gray[15:25, 5:20]
        right_corner = gray[15:25, 60:75]
        corner_brightness = (np.mean(left_corner) + np.mean(right_corner)) / 2

        left_edges = np.sum(edges[:, :20] > 0)
        right_edges = np.sum(edges[:, 60:] > 0)
        corner_edge_count = left_edges + right_edges

        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mouth_area = 0
        mouth_width = 0
        mouth_height = 0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mouth_area = cv2.contourArea(largest)
            x, y, w, h = cv2.boundingRect(largest)
            mouth_width = w
            mouth_height = h

        width_height_ratio = mouth_width / (mouth_height + 1)

        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        contrast = np.std(hist)

        mean_brightness = np.mean(gray)

        return {
            'upper_edges': upper_edges,
            'lower_edges': lower_edges,
            'total_edges': total_edges,
            'horizontal_gradient': horizontal_gradient,
            'vertical_gradient': vertical_gradient,
            'center_brightness': center_brightness,
            'center_std': center_std,
            'contrast': contrast,
            'mean_brightness': mean_brightness,
            'edge_ratio': lower_edges / (upper_edges + 1),
            'dark_ratio': dark_ratio,
            'corner_brightness': corner_brightness,
            'corner_edge_count': corner_edge_count,
            'mouth_area': mouth_area,
            'mouth_width': mouth_width,
            'mouth_height': mouth_height,
            'width_height_ratio': width_height_ratio
        }

    def _calculate_smile_score(self, state, mouth_region=None):
        """Calculate smile likelihood from state using multiple methods"""
        if state is None:
            return 0.0

        score = 0.0

        if mouth_region is not None and self.smile_cascade is not None:
            try:
                gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY) if len(mouth_region.shape) == 3 else mouth_region
                smiles = self.smile_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=12, minSize=(20, 10))
                if len(smiles) > 0:
                    score += 0.5
            except:
                pass

        if state['width_height_ratio'] > 3.0:
            score += 0.3
        elif state['width_height_ratio'] > 2.5:
            score += 0.2
        elif state['width_height_ratio'] > 2.0:
            score += 0.1

        if state['corner_edge_count'] > 40:
            score += 0.15
        elif state['corner_edge_count'] > 25:
            score += 0.1

        if state['horizontal_gradient'] > 8:
            score += 0.1

        if state['mouth_width'] > 40:
            score += 0.1

        return min(1.0, score)

    def _calculate_mouth_open_score(self, state):
        """Calculate mouth openness from state"""
        if state is None:
            return 0.0

        score = 0.0

        if state['dark_ratio'] > 0.3:
            score += 0.35
        elif state['dark_ratio'] > 0.15:
            score += 0.2

        if state['mouth_height'] > 15:
            score += 0.25
        elif state['mouth_height'] > 10:
            score += 0.15

        if state['width_height_ratio'] < 2.5:
            score += 0.2

        if state['center_brightness'] < 100:
            score += 0.15

        if state['mouth_area'] > 400:
            score += 0.1

        return min(1.0, score)

    def _detect_transition(self, current_score):
        """Detect if a valid transition (process) occurred - requires clear change from baseline"""
        if len(self.state_history) < 4:
            return False, 0.0

        baseline_scores = [s['score'] for s in self.state_history[:4]]
        baseline_avg = np.mean(baseline_scores)
        baseline_max = max(baseline_scores)

        if self.baseline_state is None and len(self.state_history) >= 4:
            self.baseline_state = baseline_avg

        if self.baseline_state is None:
            return False, 0.0

        score_change = current_score - self.baseline_state

        if score_change > self.transition_threshold and current_score > baseline_max + 0.15:
            self.transition_frames.append({
                'score': current_score,
                'change': score_change,
                'time': datetime.now()
            })

        if len(self.transition_frames) >= self.min_transition_frames:
            scores = [f['score'] for f in self.transition_frames[-10:]]
            changes = [f['change'] for f in self.transition_frames[-10:]]
            if len(scores) >= 3:
                peak_score = max(scores)
                avg_change = np.mean(changes)

                if peak_score >= self.peak_score_threshold and avg_change >= 0.20:
                    transition_quality = min(1.0, len(self.transition_frames) / 8.0) * peak_score
                    return True, max(0.6, transition_quality)

        return False, 0.0

    def process_frame(self, image, face_rect):
        """Process a frame and check for action transition"""
        if self.challenge_type is None:
            return {
                'status': 'no_challenge',
                'message': 'No challenge active',
                'progress': 0.0,
                'passed': False
            }

        elapsed = (datetime.now() - self.challenge_start_time).total_seconds()
        if elapsed > self.challenge_timeout:
            return {
                'status': 'timeout',
                'message': 'Challenge timeout',
                'progress': 0.0,
                'passed': False
            }

        mouth_region = self._extract_mouth_region(image, face_rect)
        if mouth_region is None:
            return {
                'status': 'no_face',
                'message': 'Position your face',
                'progress': 0.0,
                'passed': False
            }

        state = self._analyze_mouth_state(mouth_region)
        if state is None:
            return {
                'status': 'analysis_failed',
                'message': 'Cannot analyze mouth',
                'progress': 0.0,
                'passed': False
            }

        if self.challenge_type == "smile":
            current_score = self._calculate_smile_score(state, mouth_region)
            action_name = "SMILE"
        else:
            current_score = self._calculate_mouth_open_score(state)
            action_name = "OPEN MOUTH"

        state['score'] = current_score
        self.state_history.append(state)

        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]

        if len(self.state_history) < 4:
            return {
                'status': 'calibrating',
                'message': f'Get ready to {action_name}...',
                'progress': len(self.state_history) / 4.0 * 0.2,
                'current_score': current_score,
                'passed': False
            }

        transition_detected, transition_quality = self._detect_transition(current_score)

        if transition_detected and transition_quality > 0.5:
            self.action_detected = True
            return {
                'status': 'passed',
                'message': f'{action_name} detected!',
                'progress': 1.0,
                'transition_quality': transition_quality,
                'current_score': current_score,
                'passed': True
            }

        progress = min(0.9, 0.2 + (len(self.transition_frames) / self.min_transition_frames) * 0.7)

        if len(self.transition_frames) > 0:
            message = f'Good! Keep {action_name.lower()}ing!'
        else:
            message = f'>>> {action_name} NOW! <<<'

        return {
            'status': 'waiting',
            'message': message,
            'progress': progress,
            'current_score': current_score,
            'passed': False
        }

    def get_challenge_instruction(self):
        """Get instruction text for current challenge"""
        if self.challenge_type == "smile":
            return ">>> SMILE NOW! <<<"
        elif self.challenge_type == "open_mouth":
            return ">>> OPEN MOUTH! <<<"
        return "Ready"


class FaceEncoder:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.liveness_detector = LivenessDetector()
        cat_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml'
        self.cat_cascade = cv2.CascadeClassifier(cat_cascade_path)

    def detect_cat(self, image, exclude_rects=None):
        """Detect cats in the image - returns list of cat rectangles, excluding areas with human faces"""
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        cats = self.cat_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(80, 80))

        if len(cats) == 0:
            return []

        if exclude_rects is not None and len(exclude_rects) > 0:
            filtered_cats = []
            for (cx, cy, cw, ch) in cats:
                cat_center_x = cx + cw // 2
                cat_center_y = cy + ch // 2
                is_overlapping = False
                for (fx, fy, fw, fh) in exclude_rects:
                    if (fx < cat_center_x < fx + fw) and (fy < cat_center_y < fy + fh):
                        is_overlapping = True
                        break
                    overlap_x = max(0, min(cx + cw, fx + fw) - max(cx, fx))
                    overlap_y = max(0, min(cy + ch, fy + fh) - max(cy, fy))
                    overlap_area = overlap_x * overlap_y
                    cat_area = cw * ch
                    if overlap_area > cat_area * 0.3:
                        is_overlapping = True
                        break
                if not is_overlapping:
                    filtered_cats.append((cx, cy, cw, ch))
            return filtered_cats

        return list(cats)

    def detect_all_faces(self, image):
        """Returns list of all face rectangles"""
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        return list(faces) if len(faces) > 0 else []

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

        orb = cv2.ORB_create(nfeatures=150)
        keypoints, descriptors = orb.detectAndCompute(face_resized, None)

        if descriptors is not None and len(descriptors) > 0:
            desc_mean = np.mean(descriptors, axis=0)
            desc_std = np.std(descriptors, axis=0)
            encoding.extend(desc_mean)
            encoding.extend(desc_std)
        else:
            encoding.extend(np.zeros(64))

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

        eye_region = face_resized[20:55, 15:113]
        eye_hist = cv2.calcHist([eye_region], [0], None, [32], [0, 256])
        eye_hist = cv2.normalize(eye_hist, eye_hist).flatten()
        encoding.extend(eye_hist)

        nose_region = face_resized[40:85, 40:88]
        nose_hist = cv2.calcHist([nose_region], [0], None, [32], [0, 256])
        nose_hist = cv2.normalize(nose_hist, nose_hist).flatten()
        encoding.extend(nose_hist)

        mouth_region = face_resized[80:120, 30:98]
        mouth_hist = cv2.calcHist([mouth_region], [0], None, [32], [0, 256])
        mouth_hist = cv2.normalize(mouth_hist, mouth_hist).flatten()
        encoding.extend(mouth_hist)

        dct = cv2.dct(np.float32(face_resized))
        dct_features = dct[:8, :8].flatten()
        encoding.extend(dct_features)

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

    def get_total_registered(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT person_name) FROM RegisteredFaces")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_verified_today(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        today = date.today().isoformat()
        cursor.execute("""
            SELECT COUNT(*) FROM VerificationLog
            WHERE DATE(timestamp) = ? AND verification_result = 'VERIFIED'
        """, (today,))
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_recent_verifications(self, limit=5):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT person_name, verification_result, confidence, timestamp
            FROM VerificationLog
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
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


class SQLServerDatabase:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.connection_string = self._build_connection_string()
        self._init_db()

    def _decode_password(self, encoded):
        try:
            return base64.b64decode(encoded.encode('utf-8')).decode('utf-8')
        except:
            return ""

    def _build_connection_string(self):
        server = self.config.get("server", "")
        database = self.config.get("database", "")
        auth_type = self.config.get("auth_type", "sql")

        if auth_type == "windows":
            return (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Trusted_Connection=yes;"
            )
        else:
            username = self.config.get("username", "")
            password = self._decode_password(self.config.get("password", ""))
            return (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
            )

    def _init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='RegisteredFaces' AND xtype='U')
            CREATE TABLE RegisteredFaces (
                id INT IDENTITY(1,1) PRIMARY KEY,
                person_name NVARCHAR(255) NOT NULL,
                face_encoding NVARCHAR(MAX) NOT NULL,
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

        conn.commit()
        conn.close()

    def get_connection(self):
        return pyodbc.connect(self.connection_string, timeout=10)

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

    def get_total_registered(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT person_name) FROM RegisteredFaces")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_verified_today(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM VerificationLog
            WHERE CAST(timestamp AS DATE) = CAST(GETDATE() AS DATE) AND verification_result = 'VERIFIED'
        """)
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_recent_verifications(self, limit=5):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT TOP {limit} person_name, verification_result, confidence, timestamp
            FROM VerificationLog
            ORDER BY timestamp DESC
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


def create_database():
    if not os.path.exists(CONFIG_FILE):
        print("[Database] No db_config.json found, using SQLite (LocalDatabase)")
        return LocalDatabase(), "sqlite"

    if not PYODBC_AVAILABLE:
        print("[Database] pyodbc not installed, using SQLite (LocalDatabase)")
        return LocalDatabase(), "sqlite"

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

        if not config.get("server") or not config.get("database"):
            print("[Database] Invalid config file, using SQLite (LocalDatabase)")
            return LocalDatabase(), "sqlite"

        db = SQLServerDatabase(config)
        conn = db.get_connection()
        conn.close()

        server = config.get("server", "")
        database = config.get("database", "")
        print(f"[Database] Connected to SQL Server: {server}/{database}")
        return db, "sqlserver"

    except Exception as e:
        print(f"[Database] SQL Server connection failed: {e}")
        print("[Database] Falling back to SQLite (LocalDatabase)")
        return LocalDatabase(), "sqlite"


class SignalBridge(QObject):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str, str)
    verification_signal = pyqtSignal(str, str)
    message_signal = pyqtSignal(str, str, str)
    update_faces_signal = pyqtSignal()
    update_stats_signal = pyqtSignal()
    update_history_signal = pyqtSignal()
    liveness_score_signal = pyqtSignal(float)
    action_challenge_signal = pyqtSignal(str, float, str)
    action_passed_signal = pyqtSignal()
    action_failed_signal = pyqtSignal(str)
    start_action_challenge_signal = pyqtSignal()


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


class PulsingButton(GlowButton):
    def __init__(self, text, color="#667eea", parent=None):
        super().__init__(text, color, parent)
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self._pulse)
        self.pulse_state = 0
        self.is_pulsing = False

    def start_pulsing(self):
        if not self.is_pulsing:
            self.is_pulsing = True
            self.pulse_timer.start(500)

    def stop_pulsing(self):
        self.is_pulsing = False
        self.pulse_timer.stop()
        self.setStyleSheet(self.styleSheet().replace("border: 3px solid #00ff88;", "border: none;"))

    def _pulse(self):
        self.pulse_state = (self.pulse_state + 1) % 2
        if self.pulse_state == 0:
            border = "border: 3px solid #00ff88;"
        else:
            border = "border: 3px solid #667eea;"

        base_style = f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {self.base_color}, stop:1 {self._adjust_color(self.base_color, -30)});
                color: white;
                {border}
                padding: 12px 20px;
                font-size: 16px;
                font-weight: 700;
                border-radius: 8px;
                letter-spacing: 2px;
            }}
        """
        self.setStyleSheet(base_style)


class FaceVerificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Verification System")
        self.setGeometry(50, 50, 1400, 900)
        self.setMinimumSize(1200, 800)

        self.db, self.db_type = create_database()
        self.cap = None
        self.is_running = False
        self.known_faces = {}
        self.verification_threshold = 0.95
        self.current_image = None
        self.frozen_frame = None
        self.is_frozen = False
        self.frame_mutex = QMutex()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.face_encoder = FaceEncoder()

        self.zoom_level = 1.0
        self.brightness = 0
        self.contrast = 1.0
        self.sound_enabled = False
        self.verification_history = []
        self.current_liveness_score = 0.0
        self.current_face_count = 0
        self.scanning_frame = 0
        self.is_scanning = False

        self.signals = SignalBridge()
        self.signals.log_signal.connect(self.append_log)
        self.signals.status_signal.connect(self.update_status)
        self.signals.verification_signal.connect(self.update_verification)
        self.signals.message_signal.connect(self.show_message)
        self.signals.update_faces_signal.connect(self.reload_faces)
        self.signals.update_stats_signal.connect(self.update_stats_display)
        self.signals.update_history_signal.connect(self.update_history_display)
        self.signals.liveness_score_signal.connect(self.update_liveness_meter)
        self.signals.action_challenge_signal.connect(self.update_action_challenge_ui)
        self.signals.action_passed_signal.connect(self.on_action_passed)
        self.signals.action_failed_signal.connect(self.on_action_failed)
        self.signals.start_action_challenge_signal.connect(self.start_action_challenge)

        self.action_liveness = ActionLivenessDetector()
        self.action_challenge_active = False
        self.action_challenge_timer = QTimer()
        self.action_challenge_timer.timeout.connect(self._process_action_frame)
        self.action_liveness_enabled = True

        self.scanning_timer = QTimer()
        self.scanning_timer.timeout.connect(self._update_scanning_animation)

        self.setup_ui()
        self.setup_keyboard_shortcuts()
        self.reload_faces()
        self.update_stats_display()
        self.update_history_display()
        db_label = "SQL Server" if self.db_type == "sqlserver" else "SQLite"
        self.log(f"System ready (Database: {db_label})")
        face_count = sum(len(v) for v in self.known_faces.values())
        if face_count > 0:
            self.log(f"{face_count} registered face(s)")

    def setup_keyboard_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Return), self, self.verify_face)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.verify_face)
        QShortcut(QKeySequence("R"), self, self.register_face)
        QShortcut(QKeySequence("C"), self, self.toggle_freeze_frame)
        QShortcut(QKeySequence("L"), self, self.load_image)
        QShortcut(QKeySequence("X"), self, self.clear_current_image)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.stop_camera)

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
            QComboBox {
                background: rgba(255, 255, 255, 0.08);
                border: 2px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 8px 12px;
                color: white;
                font-size: 12px;
            }
            QComboBox:hover {
                border: 2px solid #667eea;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background: #302b63;
                color: white;
                selection-background-color: #667eea;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #667eea;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #667eea;
                border-radius: 3px;
            }
            QProgressBar {
                background: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 4px;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e74c3c, stop:0.5 #f1c40f, stop:1 #2ecc71);
                border-radius: 4px;
            }
            QListWidget {
                background: rgba(0, 0, 0, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                color: #e0e0e0;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }
            QListWidget::item:selected {
                background: rgba(102, 126, 234, 0.3);
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        left_sidebar = self._create_left_sidebar()
        main_layout.addWidget(left_sidebar)

        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, stretch=2)

        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel)

    def _create_left_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        """)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        stats_title = QLabel("Quick Stats")
        stats_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        stats_title.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(stats_title)

        self.registered_count_label = QLabel("0")
        self.registered_count_label.setFont(QFont("Segoe UI", 32, QFont.Bold))
        self.registered_count_label.setStyleSheet("color: #667eea; background: transparent;")
        self.registered_count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.registered_count_label)

        registered_desc = QLabel("Registered Faces")
        registered_desc.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-size: 11px; background: transparent;")
        registered_desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(registered_desc)

        layout.addSpacing(10)

        self.verified_today_label = QLabel("0")
        self.verified_today_label.setFont(QFont("Segoe UI", 32, QFont.Bold))
        self.verified_today_label.setStyleSheet("color: #2ecc71; background: transparent;")
        self.verified_today_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.verified_today_label)

        verified_desc = QLabel("Verified Today")
        verified_desc.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-size: 11px; background: transparent;")
        verified_desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(verified_desc)

        layout.addSpacing(20)

        history_title = QLabel("Recent Activity")
        history_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        history_title.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(history_title)

        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        layout.addWidget(self.history_list)

        layout.addStretch()

        shortcuts_label = QLabel("Shortcuts:")
        shortcuts_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 10px; background: transparent;")
        layout.addWidget(shortcuts_label)

        shortcuts_text = QLabel("Enter: Verify\nR: Register\nC: Capture\nL: Load\nX: Clear")
        shortcuts_text.setStyleSheet("color: rgba(255, 255, 255, 0.5); font-size: 9px; background: transparent;")
        layout.addWidget(shortcuts_text)

        return sidebar

    def _create_center_panel(self):
        center_panel = QFrame()
        center_panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        layout = QVBoxLayout(center_panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Face Verification")
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setStyleSheet("color: #667eea; background: transparent;")
        title_layout.addWidget(header)

        title_layout.addStretch()

        self.face_count_label = QLabel("No faces detected")
        self.face_count_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 12px; background: transparent;")
        title_layout.addWidget(self.face_count_label)

        layout.addWidget(title_widget)

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
        self.image_label.setMinimumSize(580, 420)
        self.image_label.setStyleSheet("background: transparent; border-radius: 12px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No Image Loaded")
        self.image_label.setFont(QFont("Segoe UI", 14))
        image_layout.addWidget(self.image_label)

        layout.addWidget(image_container)

        feedback_widget = QFrame()
        feedback_widget.setStyleSheet("background: transparent; border: none;")
        feedback_layout = QHBoxLayout(feedback_widget)
        feedback_layout.setContentsMargins(0, 0, 0, 0)
        feedback_layout.setSpacing(20)

        liveness_container = QWidget()
        liveness_layout = QVBoxLayout(liveness_container)
        liveness_layout.setContentsMargins(0, 0, 0, 0)
        liveness_layout.setSpacing(4)

        liveness_label = QLabel("Liveness Score")
        liveness_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px;")
        liveness_layout.addWidget(liveness_label)

        self.liveness_bar = QProgressBar()
        self.liveness_bar.setMaximum(100)
        self.liveness_bar.setValue(0)
        self.liveness_bar.setTextVisible(False)
        self.liveness_bar.setFixedHeight(12)
        liveness_layout.addWidget(self.liveness_bar)

        self.liveness_value_label = QLabel("0%")
        self.liveness_value_label.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
        liveness_layout.addWidget(self.liveness_value_label)

        feedback_layout.addWidget(liveness_container, stretch=1)

        action_container = QWidget()
        action_layout = QVBoxLayout(action_container)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)

        action_header = QLabel("Action Check")
        action_header.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px;")
        action_layout.addWidget(action_header)

        self.action_challenge_label = QLabel("Ready")
        self.action_challenge_label.setStyleSheet("""
            color: #3498db;
            font-size: 16px;
            font-weight: bold;
            padding: 8px 12px;
            background: rgba(52, 152, 219, 0.2);
            border-radius: 8px;
        """)
        self.action_challenge_label.setAlignment(Qt.AlignCenter)
        self.action_challenge_label.setMinimumHeight(40)
        action_layout.addWidget(self.action_challenge_label)

        self.action_progress_bar = QProgressBar()
        self.action_progress_bar.setMaximum(100)
        self.action_progress_bar.setValue(0)
        self.action_progress_bar.setTextVisible(False)
        self.action_progress_bar.setFixedHeight(10)
        self.action_progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 5px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #9b59b6, stop:1 #3498db);
                border-radius: 4px;
            }
        """)
        action_layout.addWidget(self.action_progress_bar)

        feedback_layout.addWidget(action_container, stretch=1)

        self.scanning_label = QLabel("")
        self.scanning_label.setStyleSheet("color: #f1c40f; font-size: 14px; font-weight: bold;")
        self.scanning_label.setAlignment(Qt.AlignCenter)
        self.scanning_label.setMinimumWidth(150)
        feedback_layout.addWidget(self.scanning_label)

        layout.addWidget(feedback_widget)

        status_widget = QFrame()
        status_widget.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 8px;
            }
        """)
        status_layout = QVBoxLayout(status_widget)
        status_layout.setContentsMargins(15, 8, 15, 8)
        status_layout.setSpacing(5)

        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Segoe UI", 12))
        self.status_label.setStyleSheet("color: #00ff88; background: transparent;")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)

        self.verification_label = QLabel("")
        self.verification_label.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.verification_label.setAlignment(Qt.AlignCenter)
        self.verification_label.setMinimumHeight(45)
        self.verification_label.setStyleSheet("background: transparent;")
        status_layout.addWidget(self.verification_label)

        layout.addWidget(status_widget)

        return center_panel

    def _create_right_panel(self):
        right_panel = QFrame()
        right_panel.setFixedWidth(350)
        right_panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(18, 18, 18, 18)
        right_layout.setSpacing(12)

        source_card = self._create_card("Camera Controls")
        source_layout = source_card.layout()

        cam_select_layout = QHBoxLayout()
        cam_label = QLabel("Camera:")
        cam_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px; background: transparent;")
        cam_select_layout.addWidget(cam_label)

        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip("Select camera device")
        self._populate_cameras()
        cam_select_layout.addWidget(self.camera_combo, stretch=1)
        source_layout.addLayout(cam_select_layout)

        cam_btn_layout = QHBoxLayout()
        cam_btn_layout.setSpacing(8)

        self.start_cam_btn = GlowButton("Start", "#27ae60")
        self.start_cam_btn.setToolTip("Start the camera (press Enter to verify)")
        self.start_cam_btn.clicked.connect(self.start_camera)
        cam_btn_layout.addWidget(self.start_cam_btn)

        self.stop_cam_btn = GlowButton("Stop", "#e74c3c")
        self.stop_cam_btn.setToolTip("Stop the camera (Esc)")
        self.stop_cam_btn.clicked.connect(self.stop_camera)
        self.stop_cam_btn.setEnabled(False)
        cam_btn_layout.addWidget(self.stop_cam_btn)

        self.capture_btn = GlowButton("Capture", "#f39c12")
        self.capture_btn.setToolTip("Freeze current frame (C)")
        self.capture_btn.clicked.connect(self.toggle_freeze_frame)
        self.capture_btn.setEnabled(False)
        cam_btn_layout.addWidget(self.capture_btn)
        source_layout.addLayout(cam_btn_layout)

        self.load_image_btn = GlowButton("Load Image from File", "#3498db")
        self.load_image_btn.setToolTip("Load an image file (L)")
        self.load_image_btn.clicked.connect(self.load_image)
        source_layout.addWidget(self.load_image_btn)

        self.clear_btn = GlowButton("Clear Image", "#95a5a6")
        self.clear_btn.setToolTip("Clear current image (X)")
        self.clear_btn.clicked.connect(self.clear_current_image)
        source_layout.addWidget(self.clear_btn)

        right_layout.addWidget(source_card)

        adjust_card = self._create_card("Image Adjustments")
        adjust_layout = adjust_card.layout()

        zoom_layout = QHBoxLayout()
        zoom_label = QLabel("Zoom:")
        zoom_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px; min-width: 60px; background: transparent;")
        zoom_layout.addWidget(zoom_label)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(100, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setToolTip("Digital zoom (100% - 200%)")
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)

        self.zoom_value = QLabel("100%")
        self.zoom_value.setStyleSheet("color: white; font-size: 11px; min-width: 40px; background: transparent;")
        zoom_layout.addWidget(self.zoom_value)
        adjust_layout.addLayout(zoom_layout)

        brightness_layout = QHBoxLayout()
        bright_label = QLabel("Brightness:")
        bright_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px; min-width: 60px; background: transparent;")
        brightness_layout.addWidget(bright_label)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-50, 50)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setToolTip("Adjust image brightness")
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        brightness_layout.addWidget(self.brightness_slider)

        self.brightness_value = QLabel("0")
        self.brightness_value.setStyleSheet("color: white; font-size: 11px; min-width: 40px; background: transparent;")
        brightness_layout.addWidget(self.brightness_value)
        adjust_layout.addLayout(brightness_layout)

        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("Contrast:")
        contrast_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px; min-width: 60px; background: transparent;")
        contrast_layout.addWidget(contrast_label)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 150)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setToolTip("Adjust image contrast")
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        contrast_layout.addWidget(self.contrast_slider)

        self.contrast_value = QLabel("1.0")
        self.contrast_value.setStyleSheet("color: white; font-size: 11px; min-width: 40px; background: transparent;")
        contrast_layout.addWidget(self.contrast_value)
        adjust_layout.addLayout(contrast_layout)

        right_layout.addWidget(adjust_card)

        register_card = self._create_card("Register Face")
        register_layout = register_card.layout()

        name_label = QLabel("Person Name")
        name_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px; margin-bottom: 2px; background: transparent;")
        register_layout.addWidget(name_label)

        self.person_name_entry = QLineEdit()
        self.person_name_entry.setPlaceholderText("Enter name...")
        self.person_name_entry.setToolTip("Name for the person to register")
        register_layout.addWidget(self.person_name_entry)

        self.register_btn = GlowButton("Register Current Face", "#9b59b6")
        self.register_btn.setToolTip("Register the current face (R)")
        self.register_btn.clicked.connect(self.register_face)
        register_layout.addWidget(self.register_btn)

        right_layout.addWidget(register_card)

        verify_card = self._create_card("Verify Identity")
        verify_layout = verify_card.layout()

        self.verify_btn = PulsingButton("VERIFY FACE", "#667eea")
        self.verify_btn.setStyleSheet(self.verify_btn.styleSheet() + """
            QPushButton {
                font-size: 16px;
                padding: 18px;
                font-weight: 700;
                letter-spacing: 2px;
            }
        """)
        self.verify_btn.setToolTip("Verify the current face (Enter)")
        self.verify_btn.clicked.connect(self.verify_face)
        verify_layout.addWidget(self.verify_btn)

        options_layout = QHBoxLayout()

        self.auto_verify_cb = QCheckBox("Auto-verify")
        self.auto_verify_cb.setToolTip("Automatically verify every 2 seconds")
        self.auto_verify_cb.stateChanged.connect(self.toggle_auto_verify)
        options_layout.addWidget(self.auto_verify_cb)

        self.sound_cb = QCheckBox("Sound")
        self.sound_cb.setToolTip("Play sound on verification result")
        self.sound_cb.stateChanged.connect(self.toggle_sound)
        options_layout.addWidget(self.sound_cb)

        verify_layout.addLayout(options_layout)

        self.auto_verify_timer = QTimer()
        self.auto_verify_timer.timeout.connect(self.verify_face)

        right_layout.addWidget(verify_card)

        log_card = self._create_card("Activity Log")
        log_layout = log_card.layout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)

        right_layout.addWidget(log_card)
        right_layout.addStretch()

        return right_panel

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
        layout.setContentsMargins(15, 12, 15, 15)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title_label.setStyleSheet("color: white; border: none; background: transparent;")
        layout.addWidget(title_label)

        return card

    def _populate_cameras(self):
        self.camera_combo.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}", i)
                cap.release()
        if self.camera_combo.count() == 0:
            self.camera_combo.addItem("No cameras found", -1)

    def on_zoom_changed(self, value):
        self.zoom_level = value / 100.0
        self.zoom_value.setText(f"{value}%")
        if self.current_image is not None and not self.is_running:
            self.display_image(self.current_image)

    def on_brightness_changed(self, value):
        self.brightness = value
        self.brightness_value.setText(str(value))
        if self.current_image is not None and not self.is_running:
            self.display_image(self.current_image)

    def on_contrast_changed(self, value):
        self.contrast = value / 100.0
        self.contrast_value.setText(f"{self.contrast:.1f}")
        if self.current_image is not None and not self.is_running:
            self.display_image(self.current_image)

    def toggle_sound(self, state):
        self.sound_enabled = state == Qt.Checked
        self.log(f"Sound {'enabled' if self.sound_enabled else 'disabled'}")

    def toggle_freeze_frame(self):
        if not self.is_running:
            return
        self.is_frozen = not self.is_frozen
        if self.is_frozen:
            self.frozen_frame = self.current_image.copy() if self.current_image is not None else None
            self.capture_btn.setText("Resume")
            self.capture_btn.setStyleSheet(self.capture_btn.styleSheet().replace("#f39c12", "#27ae60"))
            self.log("Frame captured")
            self.update_status("Frame frozen", "#f39c12")
        else:
            self.frozen_frame = None
            self.capture_btn.setText("Capture")
            self.capture_btn.setStyleSheet(self.capture_btn.styleSheet().replace("#27ae60", "#f39c12"))
            self.log("Live feed resumed")
            self.update_status("Camera active", "#00ff88")

    def clear_current_image(self):
        locker = QMutexLocker(self.frame_mutex)
        self.current_image = None
        self.frozen_frame = None
        self.is_frozen = False
        locker.unlock()

        self.image_label.clear()
        self.image_label.setText("No Image Loaded")
        self.face_count_label.setText("No faces detected")
        self.liveness_bar.setValue(0)
        self.liveness_value_label.setText("0%")
        self.verification_label.setText("")
        self.update_status("Ready", "#00ff88")
        self.verify_btn.stop_pulsing()
        self.log("Image cleared")

    def update_stats_display(self):
        registered = self.db.get_total_registered()
        verified = self.db.get_verified_today()
        self.registered_count_label.setText(str(registered))
        self.verified_today_label.setText(str(verified))

    def update_history_display(self):
        self.history_list.clear()
        history = self.db.get_recent_verifications(5)
        for row in history:
            name, result, confidence, timestamp = row
            if result == "VERIFIED":
                icon = "✓"
                color = "#2ecc71"
            elif result == "PHOTO_DETECTED":
                icon = "⚠"
                color = "#e67e22"
            else:
                icon = "✗"
                color = "#e74c3c"

            if isinstance(timestamp, datetime):
                time_str = timestamp.strftime("%H:%M")
            else:
                time_str = str(timestamp).split(" ")[1][:5] if " " in str(timestamp) else str(timestamp)[:5]
            item = QListWidgetItem(f"{icon} {name[:12]} ({confidence:.0%}) {time_str}")
            item.setForeground(QColor(color))
            self.history_list.addItem(item)

    def update_liveness_meter(self, score):
        self.current_liveness_score = score
        percent = int(score * 100)
        self.liveness_bar.setValue(percent)
        self.liveness_value_label.setText(f"{percent}%")

        if score >= 0.7:
            color = "#2ecc71"
        elif score >= 0.55:
            color = "#f1c40f"
        else:
            color = "#e74c3c"
        self.liveness_value_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")

    def update_action_challenge_ui(self, message, progress, status):
        self.action_challenge_label.setText(message)
        self.action_progress_bar.setValue(int(progress * 100))

        if status == "passed":
            style = "color: #2ecc71; font-size: 18px; font-weight: bold; padding: 8px 12px; background: rgba(46, 204, 113, 0.3); border-radius: 8px; border: 2px solid #2ecc71;"
        elif status == "waiting" or status == "calibrating":
            style = "color: #f39c12; font-size: 18px; font-weight: bold; padding: 8px 12px; background: rgba(243, 156, 18, 0.25); border-radius: 8px; border: 2px solid #f39c12;"
        elif status == "timeout" or status == "failed":
            style = "color: #e74c3c; font-size: 18px; font-weight: bold; padding: 8px 12px; background: rgba(231, 76, 60, 0.25); border-radius: 8px; border: 2px solid #e74c3c;"
        else:
            style = "color: #3498db; font-size: 16px; font-weight: bold; padding: 8px 12px; background: rgba(52, 152, 219, 0.2); border-radius: 8px;"
        self.action_challenge_label.setStyleSheet(style)

    def start_action_challenge(self):
        if not self.is_running and self.current_image is None:
            self.log("No camera/image for action check")
            return False

        challenge = self.action_liveness.get_random_challenge()
        self.action_liveness.start_challenge(challenge)
        self.action_challenge_active = True

        instruction = self.action_liveness.get_challenge_instruction()
        self.log(f"Action challenge: {instruction}")
        self.update_action_challenge_ui(instruction, 0.0, "waiting")

        self.action_challenge_timer.start(100)
        return True

    def stop_action_challenge(self):
        self.action_challenge_active = False
        self.action_challenge_timer.stop()
        self.action_liveness.challenge_type = None

    def _process_action_frame(self):
        if not self.action_challenge_active:
            self.stop_action_challenge()
            return

        image = self.get_current_image()
        if image is None:
            return

        _, face_rect = self.face_encoder.detect_face_rect(image)
        if face_rect is None:
            self.signals.action_challenge_signal.emit("Position your face", 0.0, "no_face")
            return

        result = self.action_liveness.process_frame(image, face_rect)

        self.signals.action_challenge_signal.emit(
            result['message'],
            result['progress'],
            result['status']
        )

        if result['passed']:
            self.stop_action_challenge()
            self.signals.action_passed_signal.emit()
        elif result['status'] == 'timeout':
            self.stop_action_challenge()
            self.signals.action_failed_signal.emit("Action timeout - try again")

    def on_action_passed(self):
        self.log("Action check PASSED - face is live")
        self.update_action_challenge_ui("PASSED!", 1.0, "passed")
        self._proceed_with_verification()

    def on_action_failed(self, reason):
        self.log(f"Action check failed: {reason}")
        self.update_action_challenge_ui("FAILED", 0.0, "failed")
        self.verification_label.setText("ACTION CHECK FAILED")
        self.verification_label.setStyleSheet("color: #e74c3c; font-size: 24px; background: transparent;")
        self.stop_scanning_animation()

    def _update_scanning_animation(self):
        self.scanning_frame = (self.scanning_frame + 1) % 4
        dots = "." * (self.scanning_frame + 1)
        self.scanning_label.setText(f"Scanning{dots}")

    def start_scanning_animation(self):
        self.is_scanning = True
        self.scanning_frame = 0
        self.scanning_timer.start(300)

    def stop_scanning_animation(self):
        self.is_scanning = False
        self.scanning_timer.stop()
        self.scanning_label.setText("")

    def reload_faces(self):
        self.known_faces = self.db.get_all_faces()
        self.signals.update_stats_signal.emit()

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
        self.verification_label.setStyleSheet(f"color: {color}; font-size: 24px; background: transparent;")
        self.stop_scanning_animation()
        self.signals.update_stats_signal.emit()
        self.signals.update_history_signal.emit()

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
        if self.is_frozen and self.frozen_frame is not None:
            return self.frozen_frame.copy()
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

        faces = self.face_encoder.detect_all_faces(image)
        if len(faces) > 0:
            self.verify_btn.start_pulsing()
        else:
            self.verify_btn.stop_pulsing()

    def apply_adjustments(self, image):
        adjusted = image.copy().astype(np.float32)
        adjusted = adjusted * self.contrast + self.brightness
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        if self.zoom_level > 1.0:
            h, w = adjusted.shape[:2]
            center_x, center_y = w // 2, h // 2
            new_w, new_h = int(w / self.zoom_level), int(h / self.zoom_level)
            x1 = max(0, center_x - new_w // 2)
            y1 = max(0, center_y - new_h // 2)
            x2 = min(w, x1 + new_w)
            y2 = min(h, y1 + new_h)
            cropped = adjusted[y1:y2, x1:x2]
            adjusted = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return adjusted

    def display_image(self, image):
        adjusted = self.apply_adjustments(image)
        frame_rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        faces = self.face_encoder.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

        self.current_face_count = len(faces)
        if len(faces) == 0:
            self.face_count_label.setText("No faces detected")
            self.face_count_label.setStyleSheet("color: rgba(255, 255, 255, 0.5); font-size: 12px; background: transparent;")
            self.verify_btn.stop_pulsing()
        elif len(faces) == 1:
            self.face_count_label.setText("1 face detected")
            self.face_count_label.setStyleSheet("color: #2ecc71; font-size: 12px; background: transparent;")
            self.verify_btn.start_pulsing()
        else:
            self.face_count_label.setText(f"{len(faces)} faces detected")
            self.face_count_label.setStyleSheet("color: #f1c40f; font-size: 12px; background: transparent;")
            self.verify_btn.start_pulsing()

        for (x, y, w, h) in faces:
            is_live, score, _ = self.face_encoder.liveness_detector.check_liveness(adjusted, (x, y, w, h))
            self.signals.liveness_score_signal.emit(score)

            if is_live:
                color = (46, 204, 113)
                label = f"LIVE {score:.0%}"
            else:
                color = (231, 126, 34)
                label = f"PHOTO {score:.0%}"

            corner_len = min(w, h) // 4
            thickness = 3

            cv2.line(frame_rgb, (x, y), (x + corner_len, y), color, thickness)
            cv2.line(frame_rgb, (x, y), (x, y + corner_len), color, thickness)

            cv2.line(frame_rgb, (x + w, y), (x + w - corner_len, y), color, thickness)
            cv2.line(frame_rgb, (x + w, y), (x + w, y + corner_len), color, thickness)

            cv2.line(frame_rgb, (x, y + h), (x + corner_len, y + h), color, thickness)
            cv2.line(frame_rgb, (x, y + h), (x, y + h - corner_len), color, thickness)

            cv2.line(frame_rgb, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
            cv2.line(frame_rgb, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), color, 1)

            overlay = frame_rgb.copy()
            label_width = max(120, len(label) * 12)
            cv2.rectangle(overlay, (x, y-32), (x+label_width, y-2), color, -1)
            cv2.addWeighted(overlay, 0.85, frame_rgb, 0.15, 0, frame_rgb)
            cv2.putText(frame_rgb, label, (x+8, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cats = self.face_encoder.detect_cat(adjusted, exclude_rects=faces)
        for (x, y, w, h) in cats:
            cat_color = (255, 105, 180)
            corner_len = min(w, h) // 4
            thickness = 3

            cv2.line(frame_rgb, (x, y), (x + corner_len, y), cat_color, thickness)
            cv2.line(frame_rgb, (x, y), (x, y + corner_len), cat_color, thickness)
            cv2.line(frame_rgb, (x + w, y), (x + w - corner_len, y), cat_color, thickness)
            cv2.line(frame_rgb, (x + w, y), (x + w, y + corner_len), cat_color, thickness)
            cv2.line(frame_rgb, (x, y + h), (x + corner_len, y + h), cat_color, thickness)
            cv2.line(frame_rgb, (x, y + h), (x, y + h - corner_len), cat_color, thickness)
            cv2.line(frame_rgb, (x + w, y + h), (x + w - corner_len, y + h), cat_color, thickness)
            cv2.line(frame_rgb, (x + w, y + h), (x + w, y + h - corner_len), cat_color, thickness)
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), cat_color, 1)

            cat_messages = ["MEOW! :3", "CAT DETECTED!", "PISICA! :3", "PURRR~", "KITTY! <3"]
            import random
            cat_label = random.choice(cat_messages)
            overlay = frame_rgb.copy()
            label_width = max(150, len(cat_label) * 14)
            cv2.rectangle(overlay, (x, y-35), (x+label_width, y-2), cat_color, -1)
            cv2.addWeighted(overlay, 0.85, frame_rgb, 0.15, 0, frame_rgb)
            cv2.putText(frame_rgb, cat_label, (x+8, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if len(cats) > 0 and len(faces) == 0:
            self.face_count_label.setText(f"{len(cats)} cat(s) detected! :3")
            self.face_count_label.setStyleSheet("color: #ff69b4; font-size: 12px; background: transparent;")

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def start_camera(self):
        camera_index = self.camera_combo.currentData()
        if camera_index is None or camera_index == -1:
            QMessageBox.warning(self, "Camera", "No camera selected.")
            return

        self.log(f"Starting camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            self.is_frozen = False
            self.start_cam_btn.setEnabled(False)
            self.stop_cam_btn.setEnabled(True)
            self.capture_btn.setEnabled(True)
            self.camera_combo.setEnabled(False)
            self.update_status("Camera active", "#00ff88")
            self.log("Camera started")
            self.timer.start(33)
        else:
            self.log("Camera not available")
            self.update_status("No camera", "#ff6b6b")
            QMessageBox.warning(self, "Camera", "Camera not available.\n\nUse 'Load Image' to test with photos.")

    def stop_camera(self):
        self.is_running = False
        self.is_frozen = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_cam_btn.setEnabled(True)
        self.stop_cam_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setText("Capture")
        self.camera_combo.setEnabled(True)
        self.update_status("Camera stopped", "#ffaa00")
        self.verify_btn.stop_pulsing()
        self.log("Camera stopped")

    def update_frame(self):
        if self.is_running and self.cap and not self.is_frozen:
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

        self.verification_label.setText("Checking liveness...")
        self.verification_label.setStyleSheet("color: #f1c40f; font-size: 24px; background: transparent;")
        self.start_scanning_animation()

        def static_liveness_thread(img):
            is_live, liveness_score, liveness_details = self.face_encoder.check_liveness(img)
            self.signals.liveness_score_signal.emit(liveness_score)

            if not is_live:
                self.signals.verification_signal.emit("PHOTO DETECTED", "#e67e22")
                self.signals.log_signal.emit(f"Spoof detected (score: {liveness_score:.2f})")
                self.db.log_verification("SPOOF", "PHOTO_DETECTED", liveness_score)
                return

            self.signals.log_signal.emit(f"Static liveness OK ({liveness_score:.0%})")

            if self.action_liveness_enabled and (self.is_running or self.current_image is not None):
                self.signals.status_signal.emit("Action check - follow instructions!", "#9b59b6")
                self.signals.start_action_challenge_signal.emit()
            else:
                self._proceed_with_verification()

        threading.Thread(target=static_liveness_thread, args=(image,), daemon=True).start()

    def _proceed_with_verification(self):
        image = self.get_current_image()
        if image is None:
            self.signals.verification_signal.emit("NO IMAGE", "#e74c3c")
            return

        locker = QMutexLocker(self.frame_mutex)
        faces_snapshot = {k: [{'encoding': f['encoding'].copy()} for f in v] for k, v in self.known_faces.items()}
        locker.unlock()

        def verify_thread(img, faces):
            current_encoding = self.face_encoder.get_encoding(img)

            if current_encoding is None:
                self.signals.verification_signal.emit("NO FACE DETECTED", "#e74c3c")
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
                result_text = f"✓ VERIFIED: {best_match} ({best_similarity:.0%}) ✓"
                result_color = "#2ecc71"
                result_status = "VERIFIED"
                self.signals.log_signal.emit(f"Verified: {best_match} ({best_similarity:.1%}) [threshold: {self.verification_threshold:.0%}]")
                QTimer.singleShot(100, self._stop_after_verification)
            else:
                result_text = f"✗ UNKNOWN ({best_similarity:.0%}) ✗"
                result_color = "#e74c3c"
                result_status = "UNKNOWN"
                if best_match:
                    self.signals.log_signal.emit(f"REJECTED: {best_match} at {best_similarity:.1%} < threshold {self.verification_threshold:.0%}")
                else:
                    self.signals.log_signal.emit("Unknown person - no faces in database")
                QTimer.singleShot(100, self._stop_after_verification)

            log_name = best_match if result_status == "VERIFIED" else "UNKNOWN"
            self.db.log_verification(log_name, result_status, best_similarity)
            self.signals.verification_signal.emit(result_text, result_color)

        threading.Thread(target=verify_thread, args=(image, faces_snapshot), daemon=True).start()

    def _stop_after_verification(self):
        """Stop camera and freeze display after verification"""
        if self.is_running:
            self.is_frozen = True
            self.capture_btn.setText("Resume")
            self.log("Verification complete - camera paused")

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
        self.verify_btn.stop_pulsing()
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
