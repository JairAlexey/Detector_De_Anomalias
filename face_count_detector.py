# ============================================
#   FACE COUNT DETECTOR - Asignación Estable 1-a-1
# ============================================

import os, warnings, time, cv2, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import onnxruntime as ort
import mediapipe as mp
from typing import List, Tuple, Dict, Optional

# -------------------------
# Config
# -------------------------
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "media/logs/prueba01.mp4")  # "0" para webcam
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "face_detection_output")
MODEL_PATH   = os.getenv("MODEL_PATH", "models/w600k_r50.onnx")

MAX_WIDTH    = int(os.getenv("MAX_WIDTH", "640"))
PRINT_EVERY  = int(os.getenv("PRINT_EVERY", "50"))
DETECTION_SKIP = int(os.getenv("DETECTION_SKIP", "0"))

# Guardados
SAVE_DETECTED_FACES = os.getenv("SAVE_DETECTED_FACES", "1") == "1"
FACES_DIR   = os.path.join(OUTPUT_DIR, "faces")
PERSONS_DIR = os.path.join(OUTPUT_DIR, "persons")
ALERTS_DIR  = os.path.join(OUTPUT_DIR, "alerts")

MIN_FACES_ALERT = int(os.getenv("MIN_FACES_ALERT", "2"))
SAVE_FRAMES_WITH_MULTIPLE = os.getenv("SAVE_FRAMES_WITH_MULTIPLE", "1") == "1"

# Umbrales (para ArcFace / w600k_r50)
SAME_PERSON_COS_SIM = float(os.getenv("SAME_PERSON_COS_SIM", "0.62"))  # más estricto para evitar fusiones
REUSE_WINDOW_SEC    = float(os.getenv("REUSE_WINDOW_SEC", "2.5"))      # estabilidad temporal

# Calidad mínima del rostro (evitar actualizar centroides con blur extremo)
MIN_FACE_LAPLACE_VAR = float(os.getenv("MIN_FACE_LAPLACE_VAR", "80"))

SAVE_INTERVAL_S = float(os.getenv("SAVE_INTERVAL_S", "10"))
USE_REFERENCE_CANDIDATE = os.getenv("USE_REFERENCE_CANDIDATE", "1") == "1"
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "0") == "1"

# -------------------------
# Utils
# -------------------------
def safe_makedirs(p: str): os.makedirs(p, exist_ok=True)
def now_str() -> str: return time.strftime("%Y%m%d_%H%M%S")

def l2_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def preprocess_face_for_onnx(face_bgr: np.ndarray, size=(112, 112)) -> np.ndarray:
    img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)

def laplace_var(img_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

# -------------------------
# Embeddings (ONNX)
# -------------------------
class ONNXEmbedder:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo ONNX no encontrado: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            inp = preprocess_face_for_onnx(face_bgr)
            out = self.session.run([self.output_name], {self.input_name: inp})[0][0]
            return l2_normalize(out.astype(np.float32))
        except Exception:
            return None

# -------------------------
# Tracker con asignación 1-a-1
# -------------------------
class PersonTracker:
    def __init__(self, same_person_threshold: float = 0.62, use_reference_candidate: bool = True):
        self.same_person_threshold = same_person_threshold
        self.use_reference_candidate = use_reference_candidate

        self.centroids: Dict[int, np.ndarray] = {}
        self.counts:    Dict[int, int] = {}
        self.last_seen: Dict[int, float] = {}
        self.last_saved: Dict[int, float] = {}

        self.next_id = 1
        self.candidate_id: Optional[int] = None
        self.alpha = 0.25  # suavizado de centroides

    def _new_id(self) -> int:
        if self.use_reference_candidate and self.candidate_id is None:
            self.candidate_id = 0
            return 0
        pid = self.next_id
        self.next_id += 1
        return pid

    def _update_centroid(self, pid: int, emb: np.ndarray, quality_ok: bool):
        # solo actualiza si la calidad fue razonable
        if pid in self.centroids and quality_ok:
            self.centroids[pid] = l2_normalize((1 - self.alpha) * self.centroids[pid] + self.alpha * emb)
        elif pid not in self.centroids:
            self.centroids[pid] = emb
        self.counts[pid] = self.counts.get(pid, 0) + 1
        self.last_seen[pid] = time.time()

    def assign_batch(self, embs: List[np.ndarray], qualities: List[bool]) -> List[int]:
        """
        Asigna IDs 1-a-1 a una lista de embeddings de un mismo frame.
        - No permite que un ID se use para dos rostros del mismo frame.
        - Reusa ID si similitud >= umbral o si fue visto hace poco (ventana temporal).
        """
        ids = [-1] * len(embs)
        # IDs disponibles para matching este frame
        available = set(self.centroids.keys())

        # Greedy por similitud: primero los embeddings con mejor “mejor coincidencia”
        candidates = []
        now = time.time()
        for i, e in enumerate(embs):
            best_id, best_sim = None, -1.0
            for pid in available:
                sim = cosine_sim(e, self.centroids[pid])
                if sim > best_sim:
                    best_sim, best_id = sim, pid
            candidates.append((i, best_id, best_sim))
        # ordenar por mejor similitud descendente
        candidates.sort(key=lambda x: (x[2] if x[1] is not None else -1.0), reverse=True)

        used_ids = set()
        # Paso 1: asignar a centroides existentes si superan el umbral y no están usados aún
        for i, pid, sim in candidates:
            if pid is not None and sim >= self.same_person_threshold and pid not in used_ids:
                ids[i] = pid
                used_ids.add(pid)
                available.discard(pid)

        # Paso 2: para los no asignados, intentar reusar un ID visto hace poco (ventana temporal)
        for idx, e in enumerate(embs):
            if ids[idx] != -1:
                continue
            # buscar entre IDs no usados aún que hayan sido vistos en ventana temporal
            recent_id = None
            best_sim = -1.0
            for pid in list(available):
                if now - self.last_seen.get(pid, 0) <= REUSE_WINDOW_SEC:
                    sim = cosine_sim(e, self.centroids[pid])
                    if sim > best_sim:
                        best_sim, recent_id = sim, pid
            if recent_id is not None:
                ids[idx] = recent_id
                used_ids.add(recent_id)
                available.discard(recent_id)

        # Paso 3: lo que queda crea nuevos IDs
        for idx, e in enumerate(embs):
            if ids[idx] == -1:
                ids[idx] = self._new_id()

        # Actualizar centroides y last_seen (respetando calidad)
        for i, pid in enumerate(ids):
            self._update_centroid(pid, embs[i], qualities[i])

        return ids

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
    safe_makedirs(OUTPUT_DIR); safe_makedirs(FACES_DIR)
    safe_makedirs(PERSONS_DIR); safe_makedirs(ALERTS_DIR)

    embedder = ONNXEmbedder(MODEL_PATH)
    detector = MediaPipeDetector()
    tracker  = PersonTracker(SAME_PERSON_COS_SIM, USE_REFERENCE_CANDIDATE)

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
                    cv2.imshow("Multi-Face Sorter", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue

            faces = detector.detect(frame, MAX_WIDTH)
            if frame_idx % PRINT_EVERY == 0:
                print(f"[INFO] frame={frame_idx} faces={len(faces)}")

            crops, embs, qualities = [], [], []
            for (x, y, w, h) in faces:
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue

                emb = embedder.embed(crop)
                if emb is None: continue

                crops.append((crop, (x1, y1, x2, y2)))
                embs.append(emb)
                qualities.append(laplace_var(crop) >= MIN_FACE_LAPLACE_VAR)

            if not embs:
                if SHOW_WINDOW:
                    cv2.imshow("Multi-Face Sorter", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            pids = tracker.assign_batch(embs, qualities)

            # Dibujar/guardar
            for (crop, (x1, y1, x2, y2)), pid in zip(crops, pids):
                color = (0, 255, 0) if pid != tracker.candidate_id else (0, 200, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"person_{pid}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if SAVE_DETECTED_FACES:
                    pdir = os.path.join(PERSONS_DIR, f"person_{pid}")
                    safe_makedirs(pdir)
                    avatar = os.path.join(pdir, "avatar.jpg")
                    if not os.path.exists(avatar):
                        cv2.imwrite(avatar, crop)
                    last_ts = tracker.last_saved.get(pid, 0.0)
                    if time.time() - last_ts >= SAVE_INTERVAL_S:
                        fname = f"{now_str()}_f{frame_idx}.jpg"
                        cv2.imwrite(os.path.join(pdir, fname), crop)
                        tracker.last_saved[pid] = time.time()

            cv2.putText(frame, f"Faces: {len(crops)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if len(crops) >= MIN_FACES_ALERT and SAVE_FRAMES_WITH_MULTIPLE:
                alert_count += 1
                cv2.imwrite(os.path.join(ALERTS_DIR, f"alert_multi_{now_str()}_f{frame_idx}_n{len(crops)}.jpg"), frame)

            if SHOW_WINDOW:
                cv2.imshow("Multi-Face Sorter", frame)
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
