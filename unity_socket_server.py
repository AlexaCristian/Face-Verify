"""
TCP socket server for the Unity smart-home simulation.

Unity sends a simple command, usually "VERIFY", and this server captures one
camera frame, runs the existing face verification pipeline, then returns one
JSON response per line.
"""

import argparse
import json
import socketserver
import threading
import time
from datetime import datetime

import cv2

from face_verification_app import FaceEncoder, create_database


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5055
DEFAULT_CAMERA_INDEX = 0
DEFAULT_THRESHOLD = 0.95


class FaceVerificationService:
    def __init__(self, camera_index=DEFAULT_CAMERA_INDEX, threshold=DEFAULT_THRESHOLD):
        self.camera_index = camera_index
        self.threshold = threshold
        self.encoder = FaceEncoder()
        self.db, self.db_type = create_database()
        self.known_faces = {}
        self.lock = threading.Lock()
        self.reload_faces()

    def reload_faces(self):
        with self.lock:
            self.known_faces = self.db.get_all_faces()
            return sum(len(items) for items in self.known_faces.values())

    def status(self):
        count = sum(len(items) for items in self.known_faces.values())
        return {
            "ok": True,
            "result": "READY",
            "database": self.db_type,
            "registeredFaces": count,
            "threshold": self.threshold,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def verify(self, camera_index=None):
        if camera_index is None:
            camera_index = self.camera_index

        registered_count = self.reload_faces()
        if registered_count == 0:
            return self._response(
                access=False,
                result="NO_REGISTERED_FACES",
                message="No registered faces in database.",
            )

        image = self._capture_frame(camera_index)
        if image is None:
            return self._response(
                access=False,
                result="CAMERA_ERROR",
                message=f"Could not capture image from camera {camera_index}.",
            )

        is_live, liveness_score, liveness_details = self.encoder.check_liveness(image)
        if not is_live:
            self.db.log_verification("SPOOF", "PHOTO_DETECTED", liveness_score)
            return self._response(
                access=False,
                result="PHOTO_DETECTED",
                person="SPOOF",
                liveness=liveness_score,
                message="Static liveness check failed.",
                details=liveness_details,
            )

        current_encoding = self.encoder.get_encoding(image)
        if current_encoding is None:
            return self._response(
                access=False,
                result="NO_FACE_DETECTED",
                liveness=liveness_score,
                message="No face detected in captured frame.",
            )

        best_match = None
        best_similarity = 0.0

        with self.lock:
            faces_snapshot = {
                name: [face_data["encoding"].copy() for face_data in face_list]
                for name, face_list in self.known_faces.items()
            }

        for name, encodings in faces_snapshot.items():
            for known_encoding in encodings:
                similarity = self.encoder.compare_faces(current_encoding, known_encoding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name

        if best_similarity > self.threshold:
            self.db.log_verification(best_match, "VERIFIED", best_similarity)
            return self._response(
                access=True,
                result="VERIFIED",
                person=best_match,
                confidence=best_similarity,
                liveness=liveness_score,
                message="Access granted.",
            )

        self.db.log_verification("UNKNOWN", "UNKNOWN", best_similarity)
        return self._response(
            access=False,
            result="UNKNOWN",
            person=best_match or "UNKNOWN",
            confidence=best_similarity,
            liveness=liveness_score,
            message="Access denied.",
        )

    def _capture_frame(self, camera_index):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            frame = None
            deadline = time.time() + 2.5
            while time.time() < deadline:
                ok, candidate = cap.read()
                if ok and candidate is not None:
                    frame = candidate
                if frame is not None:
                    # Let auto-exposure settle for a few frames.
                    for _ in range(5):
                        cap.read()
                    break
            return frame
        finally:
            cap.release()

    def _response(
        self,
        access,
        result,
        person="",
        confidence=0.0,
        liveness=0.0,
        message="",
        details="",
    ):
        return {
            "ok": True,
            "access": bool(access),
            "result": result,
            "person": person or "",
            "confidence": float(confidence or 0.0),
            "liveness": float(liveness or 0.0),
            "message": message,
            "details": details,
            "database": self.db_type,
            "threshold": self.threshold,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


class UnityRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline(4096).decode("utf-8", errors="replace").strip()
        if not line:
            self._send({"ok": False, "result": "EMPTY_COMMAND", "message": "No command received."})
            return

        print(f"[Unity] {self.client_address[0]}:{self.client_address[1]} -> {line}")
        response = self._dispatch(line)
        self._send(response)

    def _dispatch(self, line):
        service = self.server.verification_service
        command = line
        payload = {}

        if line.startswith("{"):
            try:
                payload = json.loads(line)
                command = payload.get("command", payload.get("cmd", "")).upper()
            except json.JSONDecodeError as exc:
                return {"ok": False, "result": "BAD_JSON", "message": str(exc)}
        else:
            command = line.upper()

        try:
            if command == "PING":
                return {"ok": True, "result": "PONG", "timestamp": datetime.now().isoformat(timespec="seconds")}
            if command == "STATUS":
                return service.status()
            if command == "RELOAD":
                count = service.reload_faces()
                response = service.status()
                response["result"] = "RELOADED"
                response["registeredFaces"] = count
                return response
            if command == "VERIFY":
                camera_index = payload.get("cameraIndex", None)
                return service.verify(camera_index=camera_index)
            return {"ok": False, "result": "UNKNOWN_COMMAND", "message": f"Unknown command: {command}"}
        except Exception as exc:
            return {"ok": False, "result": "SERVER_ERROR", "message": str(exc)}

    def _send(self, response):
        data = json.dumps(response, ensure_ascii=False) + "\n"
        self.wfile.write(data.encode("utf-8"))


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def run_server(host, port, camera_index, threshold):
    service = FaceVerificationService(camera_index=camera_index, threshold=threshold)

    with ThreadedTCPServer((host, port), UnityRequestHandler) as server:
        server.verification_service = service
        print(f"[Server] Listening on {host}:{port}")
        print(f"[Server] Database: {service.db_type}, registered faces: {service.status()['registeredFaces']}")
        print("[Server] Commands: PING, STATUS, RELOAD, VERIFY")
        server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Face verification TCP server for Unity.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_server(args.host, args.port, args.camera, args.threshold)
