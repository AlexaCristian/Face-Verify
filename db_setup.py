"""
SQL Server Database Configuration Tool
Standalone PyQt5 application for configuring SQL Server connection.
"""

import sys
import os
import json
import base64
from datetime import datetime

try:
    import pyodbc
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QPushButton, QLineEdit,
                                 QFrame, QCheckBox, QMessageBox, QComboBox)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont, QColor, QIcon
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)

CONFIG_FILE = "db_config.json"


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


class DatabaseSetupApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SQL Server Database Configuration")
        self.setGeometry(100, 100, 500, 600)
        self.setMinimumSize(450, 550)
        self.connection_tested = False
        self.setup_ui()
        self.load_config()

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
            QComboBox {
                background: rgba(255, 255, 255, 0.08);
                border: 2px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 10px 15px;
                color: white;
                font-size: 13px;
            }
            QComboBox:focus {
                border: 2px solid #667eea;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid white;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background: #302b63;
                color: white;
                selection-background-color: #667eea;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 10)
        title_layout.setSpacing(5)

        header = QLabel("Database Configuration")
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setStyleSheet("color: #667eea;")
        header.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(header)

        subtitle = QLabel("SQL Server Connection Setup")
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.5);")
        subtitle.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(subtitle)

        main_layout.addWidget(title_widget)

        form_card = self._create_card()
        form_layout = form_card.layout()

        self.auth_type_label = QLabel("Authentication Type")
        self.auth_type_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px;")
        form_layout.addWidget(self.auth_type_label)

        self.auth_combo = QComboBox()
        self.auth_combo.addItems(["SQL Server Authentication", "Windows Authentication"])
        self.auth_combo.currentIndexChanged.connect(self.on_auth_type_changed)
        form_layout.addWidget(self.auth_combo)

        form_layout.addSpacing(10)

        server_label = QLabel("Server Name")
        server_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px;")
        form_layout.addWidget(server_label)

        self.server_entry = QLineEdit()
        self.server_entry.setPlaceholderText("e.g., localhost\\SQLEXPRESS or 192.168.1.100")
        form_layout.addWidget(self.server_entry)

        form_layout.addSpacing(5)

        db_label = QLabel("Database Name")
        db_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px;")
        form_layout.addWidget(db_label)

        self.database_entry = QLineEdit()
        self.database_entry.setPlaceholderText("e.g., FaceVerification")
        form_layout.addWidget(self.database_entry)

        form_layout.addSpacing(5)

        self.username_label = QLabel("Username")
        self.username_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px;")
        form_layout.addWidget(self.username_label)

        self.username_entry = QLineEdit()
        self.username_entry.setPlaceholderText("SQL Server username")
        form_layout.addWidget(self.username_entry)

        form_layout.addSpacing(5)

        self.password_label = QLabel("Password")
        self.password_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px;")
        form_layout.addWidget(self.password_label)

        self.password_entry = QLineEdit()
        self.password_entry.setPlaceholderText("SQL Server password")
        self.password_entry.setEchoMode(QLineEdit.Password)
        form_layout.addWidget(self.password_entry)

        form_layout.addSpacing(5)

        self.show_password_cb = QCheckBox("Show password")
        self.show_password_cb.stateChanged.connect(self.toggle_password_visibility)
        form_layout.addWidget(self.show_password_cb)

        main_layout.addWidget(form_card)

        status_card = self._create_card()
        status_layout = status_card.layout()

        status_header = QHBoxLayout()
        status_title = QLabel("Connection Status")
        status_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        status_title.setStyleSheet("color: white; border: none; background: transparent;")
        status_header.addWidget(status_title)

        self.status_icon = QLabel("●")
        self.status_icon.setFont(QFont("Segoe UI", 16))
        self.status_icon.setStyleSheet("color: #888;")
        status_header.addWidget(self.status_icon)
        status_header.addStretch()

        status_layout.addLayout(status_header)

        self.status_label = QLabel("Not tested")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        main_layout.addWidget(status_card)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.test_btn = GlowButton("Test Connection", "#3498db")
        self.test_btn.clicked.connect(self.test_connection)
        button_layout.addWidget(self.test_btn)

        self.save_btn = GlowButton("Save Configuration", "#27ae60")
        self.save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_btn)

        main_layout.addLayout(button_layout)

        self.create_tables_cb = QCheckBox("Create tables if they don't exist")
        self.create_tables_cb.setChecked(True)
        main_layout.addWidget(self.create_tables_cb)

        main_layout.addStretch()

        footer = QLabel("Face Verification System • Database Setup")
        footer.setFont(QFont("Segoe UI", 9))
        footer.setStyleSheet("color: rgba(255, 255, 255, 0.3);")
        footer.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer)

    def _create_card(self):
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)
        return card

    def on_auth_type_changed(self, index):
        is_windows_auth = index == 1
        self.username_entry.setEnabled(not is_windows_auth)
        self.password_entry.setEnabled(not is_windows_auth)
        self.username_label.setEnabled(not is_windows_auth)
        self.password_label.setEnabled(not is_windows_auth)
        self.show_password_cb.setEnabled(not is_windows_auth)

        if is_windows_auth:
            self.username_entry.setStyleSheet("""
                QLineEdit {
                    background: rgba(255, 255, 255, 0.03);
                    border: 2px solid rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    padding: 10px 15px;
                    color: #666;
                    font-size: 13px;
                }
            """)
            self.password_entry.setStyleSheet("""
                QLineEdit {
                    background: rgba(255, 255, 255, 0.03);
                    border: 2px solid rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    padding: 10px 15px;
                    color: #666;
                    font-size: 13px;
                }
            """)
        else:
            self.username_entry.setStyleSheet("")
            self.password_entry.setStyleSheet("")

        self.connection_tested = False
        self.update_status("Not tested", "#888", "●")

    def toggle_password_visibility(self, state):
        if state == Qt.Checked:
            self.password_entry.setEchoMode(QLineEdit.Normal)
        else:
            self.password_entry.setEchoMode(QLineEdit.Password)

    def update_status(self, message, color, icon):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_icon.setText(icon)
        self.status_icon.setStyleSheet(f"color: {color};")

    def get_connection_string(self):
        server = self.server_entry.text().strip()
        database = self.database_entry.text().strip()
        is_windows_auth = self.auth_combo.currentIndex() == 1

        if is_windows_auth:
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Trusted_Connection=yes;"
            )
        else:
            username = self.username_entry.text().strip()
            password = self.password_entry.text()
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
            )
        return conn_str

    def test_connection(self):
        server = self.server_entry.text().strip()
        database = self.database_entry.text().strip()

        if not server:
            QMessageBox.warning(self, "Validation Error", "Please enter a server name.")
            return

        if not database:
            QMessageBox.warning(self, "Validation Error", "Please enter a database name.")
            return

        is_windows_auth = self.auth_combo.currentIndex() == 1
        if not is_windows_auth:
            username = self.username_entry.text().strip()
            if not username:
                QMessageBox.warning(self, "Validation Error", "Please enter a username.")
                return

        self.update_status("Testing connection...", "#f39c12", "◌")
        self.test_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            conn_str = self.get_connection_string()
            conn = pyodbc.connect(conn_str, timeout=10)

            if self.create_tables_cb.isChecked():
                self.create_tables(conn)

            conn.close()
            self.connection_tested = True
            self.update_status("Connection successful! ✓", "#27ae60", "✓")
            QMessageBox.information(self, "Success", "Successfully connected to the database!")

        except pyodbc.Error as e:
            self.connection_tested = False
            error_msg = str(e)
            if "Login failed" in error_msg:
                self.update_status("Authentication failed ✗", "#e74c3c", "✗")
            elif "Cannot open database" in error_msg:
                self.update_status("Database not found ✗", "#e74c3c", "✗")
            elif "server was not found" in error_msg or "Network" in error_msg:
                self.update_status("Server not reachable ✗", "#e74c3c", "✗")
            else:
                self.update_status(f"Connection failed ✗", "#e74c3c", "✗")
            QMessageBox.critical(self, "Connection Error", f"Failed to connect:\n\n{error_msg}")

        except Exception as e:
            self.connection_tested = False
            self.update_status("Error occurred ✗", "#e74c3c", "✗")
            QMessageBox.critical(self, "Error", f"An error occurred:\n\n{str(e)}")

        finally:
            self.test_btn.setEnabled(True)

    def create_tables(self, conn):
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

    def encode_password(self, password):
        return base64.b64encode(password.encode('utf-8')).decode('utf-8')

    def decode_password(self, encoded):
        try:
            return base64.b64decode(encoded.encode('utf-8')).decode('utf-8')
        except:
            return ""

    def save_config(self):
        server = self.server_entry.text().strip()
        database = self.database_entry.text().strip()

        if not server or not database:
            QMessageBox.warning(self, "Validation Error", "Please fill in all required fields.")
            return

        if not self.connection_tested:
            reply = QMessageBox.question(
                self, "Connection Not Tested",
                "You haven't tested the connection yet.\n\nDo you want to test it before saving?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self.test_connection()
                if not self.connection_tested:
                    return
            elif reply == QMessageBox.Cancel:
                return

        is_windows_auth = self.auth_combo.currentIndex() == 1

        config = {
            "server": server,
            "database": database,
            "auth_type": "windows" if is_windows_auth else "sql",
            "username": "" if is_windows_auth else self.username_entry.text().strip(),
            "password": "" if is_windows_auth else self.encode_password(self.password_entry.text()),
            "last_updated": datetime.now().isoformat()
        }

        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)

            self.update_status("Configuration saved ✓", "#27ae60", "✓")
            QMessageBox.information(
                self, "Success",
                f"Configuration saved to {CONFIG_FILE}\n\nYou can now use the Face Verification application."
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n\n{str(e)}")

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            return

        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

            self.server_entry.setText(config.get("server", ""))
            self.database_entry.setText(config.get("database", ""))

            auth_type = config.get("auth_type", "sql")
            self.auth_combo.setCurrentIndex(1 if auth_type == "windows" else 0)

            if auth_type != "windows":
                self.username_entry.setText(config.get("username", ""))
                encoded_pwd = config.get("password", "")
                if encoded_pwd:
                    self.password_entry.setText(self.decode_password(encoded_pwd))

            last_updated = config.get("last_updated", "")
            if last_updated:
                self.update_status(f"Config loaded (saved: {last_updated[:10]})", "#3498db", "◆")

        except Exception as e:
            print(f"Error loading config: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = DatabaseSetupApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
