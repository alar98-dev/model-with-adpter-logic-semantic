import os
import sys
import json
import time
import base64
import logging
import numpy as np
import redis
import cv2

try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False
    import logging
    logging.getLogger("face_organ").warning("⚠️  [face_organ] face_recognition não instalado. Reconhecimento facial desativado.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '/services/model-with-adpter-logic-semantic')
from swa.db.identity import IdentityController

REDIS_HOST = os.environ.get("WSL_REDIS_HOST", "127.0.0.1")
LEGACY_DB_DIR = "/dev/shm/aeon_identities"

_log = logging.getLogger("face_organ")
_log.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [FACE_ORGAN] %(message)s", "%H:%M:%S"))
_log.addHandler(_handler)


def extract_subtleties(rgb_frame, face_box, landmarks_dict):
    subtleties = {}

    if all(k in landmarks_dict for k in ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_tip']):
        left_brow_top = landmarks_dict['left_eyebrow'][2]
        right_brow_top = landmarks_dict['right_eyebrow'][2]
        norte = np.array([(left_brow_top[0] + right_brow_top[0]) / 2,
                          (left_brow_top[1] + right_brow_top[1]) / 2])
        sul = np.array(landmarks_dict['chin'][8])
        leste = np.array(landmarks_dict['chin'][0])
        oeste = np.array(landmarks_dict['chin'][16])
        centro = np.array(landmarks_dict['nose_tip'][2])

        eixo_NS = np.linalg.norm(norte - sul)
        eixo_LO = np.linalg.norm(leste - oeste)
        subtleties['geometria_cardeal'] = float(eixo_NS / eixo_LO if eixo_LO > 0 else 0)

        grav_sul = np.linalg.norm(centro - sul)
        grav_norte = np.linalg.norm(centro - norte)
        subtleties['geometria_gravitacional'] = float(grav_sul / grav_norte if grav_norte > 0 else 0)

        jaw_width = oeste[0] - leste[0]
        yaw_ratio = 0.5
        if jaw_width > 0:
            yaw_ratio = (centro[0] - leste[0]) / jaw_width

        if yaw_ratio < 0.4:
            subtleties['current_angle'] = "right"
        elif yaw_ratio > 0.6:
            subtleties['current_angle'] = "left"
        else:
            subtleties['current_angle'] = "front"
        subtleties['yaw_ratio'] = float(yaw_ratio)

    mask = np.zeros(rgb_frame.shape[:2], dtype=np.uint8)
    if 'left_eye' in landmarks_dict:
        cv2.fillPoly(mask, [np.array(landmarks_dict['left_eye'])], 255)
    if 'right_eye' in landmarks_dict:
        cv2.fillPoly(mask, [np.array(landmarks_dict['right_eye'])], 255)
    subtleties['eye_color'] = list(cv2.mean(rgb_frame, mask=mask)[:3])

    mask_beard = np.zeros(rgb_frame.shape[:2], dtype=np.uint8)
    if 'chin' in landmarks_dict:
        cv2.fillPoly(mask_beard, [np.array(landmarks_dict['chin'])], 255)
    subtleties['beard_color'] = list(cv2.mean(rgb_frame, mask=mask_beard)[:3])

    return subtleties


def check_subtleties(baseline, current):
    if not baseline or not current:
        return False
    score = 0
    if abs(baseline.get('geometria_cardeal', 0) - current.get('geometria_cardeal', 0)) < 0.45:
        score += 1
    if abs(baseline.get('geometria_gravitacional', 0) - current.get('geometria_gravitacional', 0)) < 0.50:
        score += 1
    eye_c1 = np.array(baseline.get('eye_color', [0, 0, 0]))
    eye_c2 = np.array(current.get('eye_color', [0, 0, 0]))
    if np.linalg.norm(eye_c1 - eye_c2) < 150:
        score += 1
    return score >= 1


class FaceOrgan:
    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=False)
        self.pubsub = self.r.pubsub()
        self.pubsub.subscribe("vision_stream")

        self.accept_threshold = 0.50
        self.MAX_TRUST = 15

        self.trackers = {}

        self.db = IdentityController()
        self._load_identities_from_legacy_db()
        self.identities = self._load_identities_from_db()
        _log.info(f"{len(self.identities)} entidades carregadas do banco.")

    def _load_identities_from_db(self):
        """Load all identities and their face encodings from PostgreSQL."""
        identities = {}
        for ident in self.db.list_identities(ativo_only=False):
            nome = ident.get("apelido") or ident.get("nome")
            faces = self.db.get_face_encodings(ident["id"])
            encodings = {"front": None, "left": None, "right": None}
            subtleties_map = {"front": {}, "left": {}, "right": {}}
            face_ids = {"front": None, "left": None, "right": None}

            for face in faces:
                angle = face.get("angle", "front")
                if angle in encodings:
                    raw_emb = face["embedding_128"]
                    if isinstance(raw_emb, str):
                        raw_emb = json.loads(raw_emb)
                    encodings[angle] = np.array(raw_emb, dtype=np.float64)
                    subtleties_map[angle] = face.get("subtleties", {})
                    if isinstance(subtleties_map[angle], str):
                        subtleties_map[angle] = json.loads(subtleties_map[angle])
                    face_ids[angle] = face.get("id")

            identities[nome] = {
                "identity_id": ident["id"],
                "encodings": encodings,
                "subtleties": subtleties_map,
                "face_ids": face_ids,
            }
        return identities

    def _load_identities_from_legacy_db(self):
        """One-time migration: import identities from /dev/shm into PostgreSQL."""
        if not os.path.exists(LEGACY_DB_DIR):
            return

        files = os.listdir(LEGACY_DB_DIR)
        npy_files = [f for f in files if f.endswith(".npy")]
        if not npy_files:
            return

        _log.info("Migrando identidades legadas de /dev/shm/ para PostgreSQL...")

        temp_identities = {}
        for f in npy_files:
            parts = f.replace(".npy", "").split("_")
            if len(parts) < 2:
                continue
            ang = parts[-1]
            nome = "_".join(parts[:-1])

            if nome not in temp_identities:
                temp_identities[nome] = {"front": None, "left": None, "right": {}}

            enc = np.load(os.path.join(LEGACY_DB_DIR, f))
            temp_identities[nome][ang] = enc

            sub_file = os.path.join(LEGACY_DB_DIR, f"{nome}_{ang}_subtleties.json")
            if os.path.exists(sub_file):
                with open(sub_file) as fp:
                    temp_identities[nome][ang + "_sub"] = json.load(fp)

        existing = {i.get("nome") or i.get("apelido"): i for i in self.db.list_identities()}

        for nome, data in temp_identities.items():
            if nome in existing:
                ident = existing[nome]
                _log.info(f"  Entidade '{nome}' já existe (ID: {ident['id'][:8]}), adicionando ângulos...")
            else:
                ident = self.db.create_identity(
                    nome=nome,
                    apelido=nome,
                    papel="admin" if nome == "Master" else "usuario",
                    metadata={"source": "legacy_migration", "migrated_at": time.time()},
                )
                _log.info(f"  Criada entidade: {nome} (ID: {ident['id'][:8]})")

            for angle in ["front", "left", "right"]:
                enc = data.get(angle)
                if enc is not None:
                    existing_faces = self.db.get_face_encodings(ident["id"], angle=angle)
                    if not existing_faces:
                        sub = data.get(angle + "_sub", {})
                        self.db.add_face_encoding(
                            identity_id=ident["id"],
                            embedding_128=enc.tolist(),
                            angle=angle,
                            subtleties=sub,
                        )
                        _log.info(f"    + Ângulo {angle} registrado")

        _log.info("Migração concluída.")

    def start(self):
        _log.info("Aguardando stream visual (vision_stream)...")
        while True:
            try:
                msg = self.pubsub.get_message(timeout=1.0)
                if msg and msg['type'] == 'message':
                    data = json.loads(msg['data'].decode('utf-8'))
                    self._process_frame(data)
            except Exception as e:
                _log.error(f"Erro no loop principal: {e}")

    def _process_frame(self, data):
        if not HAS_FACE_RECOGNITION:
            return
        frame_id = data.get('frame_id', 0)
        # Assinatura biométrica diferencial 3D (enviada pelo webcam_sensor no Windows)
        signature3d = data.get('signature3d') or {}
        img_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

        boxes_to_send = []
        names_to_send = []
        identities_to_send = []

        for (top, right, bottom, left), face_encoding, landmarks in zip(
            face_locations, face_encodings, face_landmarks_list
        ):
            cx = (left + right) / 2
            cy = (top + bottom) / 2

            current_subtleties = extract_subtleties(rgb_frame, (top, right, bottom, left), landmarks)
            current_angle = current_subtleties.get('current_angle', 'front')
            # Integra a assinatura 3D (malha/iris/pose) como métrica biométrica diferencial
            if signature3d:
                current_subtleties['signature3d'] = signature3d

            result = self.db.find_identity_by_face(
                face_encoding.tolist(), threshold=self.accept_threshold
            )

            temporal_match_name = None
            for t_nome, t_data in self.trackers.items():
                if t_data["trust"] > 0:
                    dist_spatial = np.linalg.norm(
                        np.array([cx, cy]) - np.array(t_data["center"])
                    )
                    if dist_spatial < 150:
                        temporal_match_name = t_nome
                        break

            label = "Desconhecido"
            matched_identity = None
            matched_id = None

            if result:
                matched_identity = result.get("nome") or result.get("apelido")
                matched_id = result.get("identity_id")
                confidence = result.get("similarity", 0)
                label = matched_identity
                self.trackers[matched_identity] = {"center": (cx, cy), "trust": self.MAX_TRUST}

                face_id = result.get("face_id")
                # AUTO-LEARN bidirecional:
                #  - confiança ALTA (≥0.60): captura de alta qualidade -> refina o
                #    embedding com ajuste fino (quanto mais te vê, mais aprende o rosto);
                #  - confiança BAIXA (<0.30): incorpora novo ângulo/variação salva.
                if face_id:
                    if confidence >= 0.60:
                        self.db.update_face_encoding(
                            face_id=face_id,
                            new_embedding=face_encoding.tolist(),
                            subtleties=current_subtleties,
                            blend_factor=0.03,
                        )
                    elif confidence < 0.30:
                        self.db.update_face_encoding(
                            face_id=face_id,
                            new_embedding=face_encoding.tolist(),
                            subtleties=current_subtleties,
                            blend_factor=0.05,
                        )

                self.db.log_recognition(
                    identity_id=matched_id,
                    face_id=face_id or 0,
                    confidence=confidence,
                    angle=current_angle,
                )

                self._publish_identity_event(matched_id, matched_identity, confidence, current_angle)

            elif temporal_match_name is not None:
                matched_identity = temporal_match_name
                self.trackers[matched_identity]["trust"] -= 1
                self.trackers[matched_identity]["center"] = (cx, cy)
                label = matched_identity
            else:
                if len(self.identities) == 0:
                    new_name = "Master"
                else:
                    new_name = f"Entidade_{len(self.identities)}"

                ident = self.db.create_identity(
                    nome=new_name,
                   apelido=new_name,
                    papel="admin" if new_name == "Master" else "usuario",
                    metadata={"auto_created": True, "created_at": time.time()},
                )
                face_record = self.db.add_face_encoding(
                    identity_id=ident["id"],
                    embedding_128=face_encoding.tolist(),
                    angle=current_angle,
                    subtleties=current_subtleties,
                )

                self.identities[new_name] = {
                    "identity_id": ident["id"],
                    "encodings": {a: None for a in ["front", "left", "right"]},
                    "subtleties": {a: {} for a in ["front", "left", "right"]},
                    "face_ids": {a: None for a in ["front", "left", "right"]},
                }
                self.identities[new_name]["encodings"][current_angle] = face_encoding
                self.identities[new_name]["subtleties"][current_angle] = current_subtleties
                self.identities[new_name]["face_ids"][current_angle] = face_record.get("id")

                self.trackers[new_name] = {"center": (cx, cy), "trust": self.MAX_TRUST}
                label = f"{new_name} (Imprinting)"
                _log.info(f"NOVA ENTIDADE: {new_name} (ID: {ident['id']})")

                self._publish_identity_event(
                    ident["id"], new_name, 1.0, current_angle
                )

            boxes_to_send.append([top, right, bottom, left])
            names_to_send.append(label)
            identities_to_send.append(matched_id or "")

        feedback = {
            "frame_id": frame_id,
            "boxes": boxes_to_send,
            "names": names_to_send,
            "identities": identities_to_send,
            "timestamp": time.time(),
        }
        self.r.publish("vision_feedback", json.dumps(feedback))

        # Publish current presence for dashboard
        presence = {
            "faces_detected": len(boxes_to_send),
            "identities": [
                {"name": n, "id": iid}
                for n, iid in zip(names_to_send, identities_to_send)
            ],
            "timestamp": time.time(),
        }
        self.r.set("swa:vision:current_presence", json.dumps(presence), ex=10)

        # Store latest frame + detections for calibrator
        calib_data = {
            "frame_id": frame_id,
            "image": data['image'],
            "boxes": boxes_to_send,
            "names": names_to_send,
            "identities": identities_to_send,
            "embeddings": [e.tolist() for e in face_encodings] if face_encodings else [],
            "timestamp": time.time(),
        }
        self.r.set("swa:vision:latest_for_calibrate", json.dumps(calib_data), ex=15)

    def _publish_identity_event(self, identity_id, nome, confidence, angle):
        """Publish structured identity event to Redis."""
        event = {
            "identity_id": identity_id,
            "nome": nome,
            "confidence": round(confidence, 4),
            "angle": angle,
            "timestamp": time.time(),
        }
        self.r.publish("swa:vision:identity_recognized", json.dumps(event))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    organ = FaceOrgan()
    organ.start()


def init_organ():
    return FaceOrgan()
