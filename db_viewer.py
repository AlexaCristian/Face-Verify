"""
Database Viewer - Face Verification System
View and manage registered faces in the database.
"""

import sys
import os
import sqlite3
import json
import base64
from datetime import datetime

try:
    import cv2
    import numpy as np
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QPushButton, QTableWidget,
                                 QTableWidgetItem, QMessageBox, QFrame, QTabWidget,
                                 QHeaderView, QAbstractItemView, QSplitter)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QImage
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)

try:
    import pyodbc

    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

CONFIG_FILE = "db_config.json"


def decode_password(encoded):
    """Decode base64 encoded password from db_config.json"""
    try:
        return base64.b64decode(encoded.encode('utf-8')).decode('utf-8')
    except:
        return ""


class DatabaseConnection:
    def __init__(self):
        self.db_type = "sqlite"
        self.config = None
        self._init_database()

    def _init_database(self):
        if os.path.exists(CONFIG_FILE) and PYODBC_AVAILABLE:
            try:
                with open(CONFIG_FILE, 'r') as f:
                    self.config = json.load(f)
                if self.config.get("server") and self.config.get("database"):
                    conn = self._get_sqlserver_connection()
                    conn.close()
                    self.db_type = "sqlserver"
                    return
            except Exception as e:
                print(f"SQL Server connection failed: {e}")

        self.db_type = "sqlite"
        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS RegisteredFaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            face_encoding TEXT NOT NULL,
            face_image BLOB,
            registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS VerificationLog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT,
            verification_result TEXT,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        conn.close()

    def _get_sqlserver_connection(self):
        auth_type = self.config.get('auth_type', 'sql')

        if auth_type == 'windows':
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.config['server']};"
                f"DATABASE={self.config['database']};"
                f"Trusted_Connection=yes;"
            )
        else:
            password = decode_password(self.config.get('password', ''))
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.config['server']};"
                f"DATABASE={self.config['database']};"
                f"UID={self.config['username']};"
                f"PWD={password};"
            )
        return pyodbc.connect(conn_str)

    def get_connection(self):
        if self.db_type == "sqlserver":
            return self._get_sqlserver_connection()
        return sqlite3.connect("faces.db")

    def get_all_faces(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        if self.db_type == "sqlserver":
            cursor.execute(
                "SELECT id, person_name, face_image, registered_date FROM RegisteredFaces ORDER BY registered_date DESC")
        else:
            cursor.execute(
                "SELECT id, person_name, face_image, registered_date FROM RegisteredFaces ORDER BY registered_date DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_verification_log(self, limit=100):
        conn = self.get_connection()
        cursor = conn.cursor()
        if self.db_type == "sqlserver":
            cursor.execute(
                f"SELECT TOP {limit} id, person_name, verification_result, confidence, timestamp FROM VerificationLog ORDER BY timestamp DESC")
        else:
            cursor.execute(
                f"SELECT id, person_name, verification_result, confidence, timestamp FROM VerificationLog ORDER BY timestamp DESC LIMIT {limit}")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def delete_face(self, face_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM RegisteredFaces WHERE id = ?", (face_id,))
        conn.commit()
        conn.close()

    def delete_person(self, name):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM RegisteredFaces WHERE person_name = ?", (name,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    def clear_verification_log(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM VerificationLog")
        conn.commit()
        conn.close()

    def get_stats(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(DISTINCT person_name) FROM RegisteredFaces")
        total_persons = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM RegisteredFaces")
        total_faces = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM VerificationLog")
        total_verifications = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM VerificationLog WHERE verification_result = 'VERIFIED'")
        verified_count = cursor.fetchone()[0]

        conn.close()
        return {
            'total_persons': total_persons,
            'total_faces': total_faces,
            'total_verifications': total_verifications,
            'verified_count': verified_count
        }


class DatabaseViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Database Viewer - Face Verification")
        self.setGeometry(100, 100, 1200, 700)
        self.setMinimumSize(900, 600)

        self.db = DatabaseConnection()

        self.setup_ui()
        self.refresh_all()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_all)
        self.refresh_timer.start(5000)

    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QLabel {
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTableWidget {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                color: white;
                gridline-color: rgba(255, 255, 255, 0.1);
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background: rgba(102, 126, 234, 0.5);
            }
            QHeaderView::section {
                background: rgba(102, 126, 234, 0.3);
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #5a67d8);
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: 600;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #7c8ff0, stop:1 #667eea);
            }
            QPushButton:pressed {
                background: #5a67d8;
            }
            QPushButton#deleteBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #e74c3c, stop:1 #c0392b);
            }
            QPushButton#deleteBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ec7063, stop:1 #e74c3c);
            }
            QTabWidget::pane {
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.03);
            }
            QTabBar::tab {
                background: rgba(255, 255, 255, 0.05);
                color: #aaa;
                padding: 12px 25px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: rgba(102, 126, 234, 0.3);
                color: white;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        header = QLabel("Database Viewer")
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setStyleSheet("color: #667eea;")
        main_layout.addWidget(header)

        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        stats_layout = QHBoxLayout(stats_frame)

        self.stats_labels = {}
        for stat_name, stat_title in [("total_persons", "Persoane"),
                                      ("total_faces", "Fețe"),
                                      ("total_verifications", "Verificări"),
                                      ("verified_count", "Verificate OK")]:
            stat_widget = QWidget()
            stat_widget_layout = QVBoxLayout(stat_widget)
            stat_widget_layout.setSpacing(2)

            value_label = QLabel("0")
            value_label.setFont(QFont("Segoe UI", 28, QFont.Bold))
            value_label.setStyleSheet("color: #2ecc71;")
            value_label.setAlignment(Qt.AlignCenter)

            title_label = QLabel(stat_title)
            title_label.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-size: 12px;")
            title_label.setAlignment(Qt.AlignCenter)

            stat_widget_layout.addWidget(value_label)
            stat_widget_layout.addWidget(title_label)
            stats_layout.addWidget(stat_widget)

            self.stats_labels[stat_name] = value_label

        main_layout.addWidget(stats_frame)

        self.db_type_label = QLabel(f"Baza de date: {self.db.db_type.upper()}")
        self.db_type_label.setStyleSheet("color: #f1c40f; font-size: 12px;")
        main_layout.addWidget(self.db_type_label)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        faces_tab = QWidget()
        faces_layout = QVBoxLayout(faces_tab)

        faces_btn_layout = QHBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_all)
        faces_btn_layout.addWidget(refresh_btn)

        faces_btn_layout.addStretch()

        delete_selected_btn = QPushButton("Șterge Selectat")
        delete_selected_btn.setObjectName("deleteBtn")
        delete_selected_btn.clicked.connect(self.delete_selected_face)
        faces_btn_layout.addWidget(delete_selected_btn)

        faces_layout.addLayout(faces_btn_layout)

        splitter = QSplitter(Qt.Horizontal)

        self.faces_table = QTableWidget()
        self.faces_table.setColumnCount(4)
        self.faces_table.setHorizontalHeaderLabels(["ID", "Nume", "Data Înregistrare", "Acțiuni"])
        self.faces_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.faces_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.faces_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.faces_table.itemSelectionChanged.connect(self.on_face_selected)
        splitter.addWidget(self.faces_table)

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        preview_title = QLabel("Previzualizare")
        preview_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        preview_title.setStyleSheet("color: white;")
        preview_layout.addWidget(preview_title)

        self.preview_label = QLabel("Selectează o față")
        self.preview_label.setMinimumSize(200, 200)
        self.preview_label.setMaximumSize(300, 300)
        self.preview_label.setStyleSheet("""
            background: rgba(0, 0, 0, 0.3);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        """)
        self.preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_label)

        self.preview_name_label = QLabel("")
        self.preview_name_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.preview_name_label.setStyleSheet("color: #667eea;")
        self.preview_name_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_name_label)

        preview_layout.addStretch()
        splitter.addWidget(preview_widget)

        splitter.setSizes([700, 300])
        faces_layout.addWidget(splitter)

        tabs.addTab(faces_tab, "Fețe Înregistrate")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)

        log_btn_layout = QHBoxLayout()

        refresh_log_btn = QPushButton("Refresh")
        refresh_log_btn.clicked.connect(self.refresh_log)
        log_btn_layout.addWidget(refresh_log_btn)

        log_btn_layout.addStretch()

        clear_log_btn = QPushButton("Șterge Tot Log-ul")
        clear_log_btn.setObjectName("deleteBtn")
        clear_log_btn.clicked.connect(self.clear_log)
        log_btn_layout.addWidget(clear_log_btn)

        log_layout.addLayout(log_btn_layout)

        self.log_table = QTableWidget()
        self.log_table.setColumnCount(5)
        self.log_table.setHorizontalHeaderLabels(["ID", "Nume", "Rezultat", "Încredere", "Data/Ora"])
        self.log_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        log_layout.addWidget(self.log_table)

        tabs.addTab(log_tab, "Istoric Verificări")

        self.current_faces_data = []

    def refresh_all(self):
        self.refresh_faces()
        self.refresh_log()
        self.refresh_stats()

    def refresh_stats(self):
        stats = self.db.get_stats()
        for key, value in stats.items():
            if key in self.stats_labels:
                self.stats_labels[key].setText(str(value))

    def refresh_faces(self):
        self.current_faces_data = self.db.get_all_faces()
        self.faces_table.setRowCount(len(self.current_faces_data))

        for row, face in enumerate(self.current_faces_data):
            face_id, name, image_data, reg_date = face

            self.faces_table.setItem(row, 0, QTableWidgetItem(str(face_id)))
            self.faces_table.setItem(row, 1, QTableWidgetItem(name))

            if isinstance(reg_date, datetime):
                date_str = reg_date.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = str(reg_date)[:16] if reg_date else "N/A"
            self.faces_table.setItem(row, 2, QTableWidgetItem(date_str))

            delete_btn = QPushButton("Șterge")
            delete_btn.setObjectName("deleteBtn")
            delete_btn.setStyleSheet("""
                QPushButton {
                    background: #e74c3c;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background: #c0392b;
                }
            """)
            delete_btn.clicked.connect(lambda checked, fid=face_id: self.delete_face_by_id(fid))
            self.faces_table.setCellWidget(row, 3, delete_btn)

    def refresh_log(self):
        log_data = self.db.get_verification_log()
        self.log_table.setRowCount(len(log_data))

        for row, log_entry in enumerate(log_data):
            log_id, name, result, confidence, timestamp = log_entry

            self.log_table.setItem(row, 0, QTableWidgetItem(str(log_id)))
            self.log_table.setItem(row, 1, QTableWidgetItem(name or "N/A"))

            result_item = QTableWidgetItem(result)
            if result == "VERIFIED":
                result_item.setForeground(QColor("#2ecc71"))
            elif result == "PHOTO_DETECTED":
                result_item.setForeground(QColor("#e67e22"))
            else:
                result_item.setForeground(QColor("#e74c3c"))
            self.log_table.setItem(row, 2, result_item)

            conf_str = f"{confidence:.1%}" if confidence else "N/A"
            self.log_table.setItem(row, 3, QTableWidgetItem(conf_str))

            if isinstance(timestamp, datetime):
                time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = str(timestamp)[:19] if timestamp else "N/A"
            self.log_table.setItem(row, 4, QTableWidgetItem(time_str))

    def on_face_selected(self):
        selected = self.faces_table.selectedItems()
        if not selected:
            self.preview_label.setText("Selectează o față")
            self.preview_label.setPixmap(QPixmap())
            self.preview_name_label.setText("")
            return

        row = selected[0].row()
        if row < len(self.current_faces_data):
            face_id, name, image_data, reg_date = self.current_faces_data[row]
            self.preview_name_label.setText(name)

            if image_data:
                try:
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = img_rgb.shape
                        bytes_per_line = ch * w
                        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qt_img)
                        scaled = pixmap.scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.preview_label.setPixmap(scaled)
                        return
                except Exception as e:
                    print(f"Error loading image: {e}")

            self.preview_label.setText("Imagine indisponibilă")

    def delete_selected_face(self):
        selected = self.faces_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Atenție", "Selectează o față pentru a o șterge.")
            return

        row = selected[0].row()
        if row < len(self.current_faces_data):
            face_id = self.current_faces_data[row][0]
            name = self.current_faces_data[row][1]
            self.delete_face_by_id(face_id, name)

    def delete_face_by_id(self, face_id, name=None):
        msg = f"Ștergi față ID {face_id}"
        if name:
            msg += f" ({name})"
        msg += "?"

        reply = QMessageBox.question(self, "Confirmare", msg,
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db.delete_face(face_id)
            self.refresh_all()
            QMessageBox.information(self, "Succes", "Fața a fost ștearsă.")

    def clear_log(self):
        reply = QMessageBox.question(self, "Confirmare",
                                     "Ștergi tot istoricul verificărilor?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db.clear_verification_log()
            self.refresh_all()
            QMessageBox.information(self, "Succes", "Istoricul a fost șters.")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(26, 26, 46))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 50))
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(102, 126, 234))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.Highlight, QColor(102, 126, 234))
    app.setPalette(dark_palette)

    window = DatabaseViewer()
    window.show()

    print("\n" + "=" * 50)
    print("Database Viewer - Face Verification")
    print("=" * 50 + "\n")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
