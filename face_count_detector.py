# ============================================
#   FACE COUNT DETECTOR - Solo Conteo
# ============================================

import os, warnings, time, cv2, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import mediapipe as mp
from typing import List, Tuple

# -------------------------
# Config
# -------------------------
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "media/logs/prueba01.mp4")  # "0" para webcam
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "face_detection_output")

MAX_WIDTH    = int(os.getenv("MAX_WIDTH", "640"))
PRINT_EVERY  = int(os.getenv("PRINT_EVERY", "50"))
DETECTION_SKIP = int(os.getenv("DETECTION_SKIP", "0"))

# Guardados
ALERTS_DIR  = os.path.join(OUTPUT_DIR, "alerts")

MIN_FACES_ALERT = int(os.getenv("MIN_FACES_ALERT", "2"))
SAVE_FRAMES_WITH_MULTIPLE = os.getenv("SAVE_FRAMES_WITH_MULTIPLE", "1") == "1"
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "0") == "1"

# -------------------------
# Utils
# -------------------------
def safe_makedirs(p: str): os.makedirs(p, exist_ok=True)
def now_str() -> str: return time.strftime("%Y%m%d_%H%M%S")

# -------------------------
# Detector (MediaPipe)
# -------------------------
class MediaPipeDetector:
    def __init__(self, min_conf: float = 0.5, model_selection: int = 1):
        self.face = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection, min_detection_confidence=min_conf
        )

    def detect(self, frame_bgr: np.ndarray, max_width: int = 640) -> List[Tuple[int, int, int, int]]:
        ih, iw = frame_bgr.shape[:2]
        scale = 1.0
        resized = frame_bgr
        if iw > max_width:
            scale = max_width / float(iw)
            resized = cv2.resize(frame_bgr, (int(iw * scale), int(ih * scale)))

        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = self.face.process(img_rgb)

        faces = []
        if results.detections:
            ih_r, iw_r = resized.shape[:2]
            inv = 1.0 / scale
            for d in results.detections:
                b = d.location_data.relative_bounding_box
                x_r, y_r = int(b.xmin * iw_r), int(b.ymin * ih_r)
                w_r, h_r = int(b.width * iw_r), int(b.height * ih_r)
                x, y, w, h = int(x_r * inv), int(y_r * inv), int(w_r * inv), int(h_r * inv)
                if w > 5 and h > 5:
                    faces.append((x, y, w, h))
        return faces

    def close(self): self.face.close()

# -------------------------
# Main
# -------------------------
def main():
    safe_makedirs(OUTPUT_DIR)
    safe_makedirs(ALERTS_DIR)

    detector = MediaPipeDetector()

    src = 0 if VIDEO_SOURCE == "0" else VIDEO_SOURCE
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente: {VIDEO_SOURCE}")
        return

    frame_idx = 0
    alert_count = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1

            # Opcional: saltar frames para subir FPS
            if DETECTION_SKIP > 0 and (frame_idx % (DETECTION_SKIP + 1)) != 1:
                if SHOW_WINDOW:
                    cv2.imshow("Face Counter", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue

            faces = detector.detect(frame, MAX_WIDTH)
            if frame_idx % PRINT_EVERY == 0:
                print(f"[INFO] frame={frame_idx} faces={len(faces)}")

            # Dibujar rectángulos en todas las caras detectadas
            for (x, y, w, h) in faces:
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Mostrar contador de caras
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Guardar alerta si hay múltiples caras
            if len(faces) >= MIN_FACES_ALERT and SAVE_FRAMES_WITH_MULTIPLE:
                alert_count += 1
                cv2.imwrite(os.path.join(ALERTS_DIR, f"alert_multi_{now_str()}_f{frame_idx}_n{len(faces)}.jpg"), frame)

            if SHOW_WINDOW:
                cv2.imshow("Face Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario.")
    finally:
        cap.release()
        if SHOW_WINDOW: cv2.destroyAllWindows()
        detector.close()
        dt = time.time() - t0
        print(f"\nProcesados {frame_idx} frames en {dt:.1f}s ({frame_idx/dt:.1f} fps)")
        print(f"Total de alertas (>= {MIN_FACES_ALERT} rostros): {alert_count}")
        print(f"Salida guardada en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()