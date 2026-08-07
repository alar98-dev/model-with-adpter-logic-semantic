import os
import sys
import json
import time
import urllib.parse
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import base64

sys.path.insert(0, '/services/model-with-adpter-logic-semantic')

from swa.db.identity import IdentityController

_db = IdentityController()


def _compute_embedding_space_3d():
    """Compute 3D PCA projection of all face embeddings."""
    all_points = []
    all_labels = []
    all_colors = []
    all_angles = []
    all_ids = []

    identities = _db.list_identities(ativo_only=False)
    color_palette = [
        "#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b",
        "#cc5de8", "#ff922b", "#20c997", "#748ffc",
        "#f06595", "#a9e34b", "#4dabf7", "#e599f7",
    ]

    for idx, ident in enumerate(identities):
        faces = _db.get_face_encodings(ident["id"])
        color = color_palette[idx % len(color_palette)]
        for face in faces:
            raw_emb = face.get("embedding_128", [])
            if isinstance(raw_emb, str):
                try:
                    raw_emb = json.loads(raw_emb)
                except:
                    raw_emb = []
            if raw_emb and len(raw_emb) == 128:
                all_points.append(raw_emb)
                all_labels.append(ident.get("apelido") or ident.get("nome"))
                all_colors.append(color)
                all_angles.append(face.get("angle", "front"))
                all_ids.append(ident["id"])

    if len(all_points) < 2:
        points_3d = [[0, 0, 0] for _ in all_points]
        return {
            "points": points_3d,
            "labels": all_labels,
            "colors": all_colors,
            "angles": all_angles,
            "identity_ids": all_ids,
            "method": "insufficient_data",
        }

    mat = np.array(all_points, dtype=np.float64)
    mean = mat.mean(axis=0)
    centered = mat - mean

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    top3_idx = np.argsort(eigenvalues)[::-1][:3]
    projection = eigenvectors[:, top3_idx]

    projected = centered @ projection

    explained = eigenvalues[top3_idx] / eigenvalues.sum() * 100

    return {
        "points": projected.tolist(),
        "labels": all_labels,
        "colors": all_colors,
        "angles": all_angles,
        "identity_ids": all_ids,
        "explained_variance": explained.tolist(),
        "method": "PCA",
        "total_embeddings": len(all_points),
        "total_identities": len(identities),
    }


def _cardiac_waveform(embedding, n_points=128):
    """Transform a 128D embedding into a 3D cardiac waveform.
    
    Each dimension of the embedding drives a different aspect of the heartbeat:
    - dims 0-42   → X-axis oscillation (atrial depolarization)
    - dims 43-85  → Y-axis oscillation (ventricular depolarization)  
    - dims 86-127 → Z-axis oscillation (repolarization)
    
    The waveform has a characteristic P-QRS-T shape modulated by the embedding values.
    """
    emb = np.array(embedding, dtype=np.float64)
    if len(emb) < 128:
        emb = np.pad(emb, (0, 128 - len(emb)))
    elif len(emb) > 128:
        emb = emb[:128]

    t = np.linspace(0, 2 * np.pi, n_points)

    x_comp = emb[0:43]
    y_comp = emb[43:86]
    z_comp = emb[86:128]

    def make_signal(comp, t):
        freqs = np.arange(1, len(comp) + 1)
        amps = np.abs(comp) / (freqs ** 0.8)
        phases = comp * np.pi
        signal = np.zeros(n_points)
        for i in range(len(comp)):
            signal += amps[i] * np.sin(freqs[i] * t + phases[i])
        return signal

    x = make_signal(x_comp, t)
    y = make_signal(y_comp, t)
    z = make_signal(z_comp, t)

    cardiac_mask = np.zeros(n_points)
    peak_center = n_points // 2
    peak_width = n_points // 8
    for i in range(n_points):
        dist = abs(i - peak_center)
        if dist < peak_width:
            cardiac_mask[i] = np.exp(-(dist / (peak_width * 0.3)) ** 2)

    qrs_boost = np.mean(np.abs(emb[43:86])) * 0.5
    y += cardiac_mask * qrs_boost * 5.0
    x -= cardiac_mask * np.mean(emb[0:43]) * 0.3
    z += cardiac_mask * np.mean(emb[86:128]) * 0.4

    baseline = np.mean(emb) * 0.1
    x += baseline * np.sin(t * 2)
    y += baseline * np.cos(t * 3)
    z += baseline * np.sin(t * 5) * 0.5

    scale = np.std(emb) * 0.5 + 0.01
    x = (x - x.mean()) / (x.std() + 1e-8) * scale
    y = (y - y.mean()) / (y.std() + 1e-8) * scale
    z = (z - z.mean()) / (z.std() + 1e-8) * scale

    return np.column_stack([x, y, z]).tolist()


def _compute_cardiac_map():
    """Compute 3D cardiac representation for all identity embeddings.
    
    Each identity's embeddings are averaged and transformed into a unique
    cardiac waveform shape in 3D space. Multiple angles of the same person
    produce similar cardiac forms.
    """
    identities = _db.list_identities(ativo_only=False)
    color_palette = [
        "#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b",
        "#cc5de8", "#ff922b", "#20c997", "#748ffc",
        "#f06595", "#a9e34b", "#4dabf7", "#e599f7",
    ]

    cardiac_entities = []
    all_curves = []

    for idx, ident in enumerate(identities):
        faces = _db.get_face_encodings(ident["id"])
        color = color_palette[idx % len(color_palette)]

        embeddings_for_identity = []
        face_details = []

        for face in faces:
            raw_emb = face.get("embedding_128", [])
            if isinstance(raw_emb, str):
                try:
                    raw_emb = json.loads(raw_emb)
                except:
                    raw_emb = []
            if raw_emb and len(raw_emb) == 128:
                embeddings_for_identity.append(raw_emb)
                face_details.append({
                    "angle": face.get("angle", "front"),
                    "face_id": face.get("id"),
                })

        if not embeddings_for_identity:
            continue

        avg_embedding = np.mean(embeddings_for_identity, axis=0)
        cardiac_curve = _cardiac_waveform(avg_embedding.tolist(), n_points=180)

        individual_curves = []
        for emb in embeddings_for_identity:
            individual_curves.append(_cardiac_waveform(emb, n_points=180))

        centroid = np.mean(cardiac_curve, axis=0).tolist()

        heartbeat_amplitude = float(np.std(avg_embedding) * 10)
        heartbeat_phase = float(np.mean(avg_embedding[:43]) * np.pi)

        entity = {
            "identity_id": ident["id"],
            "nome": ident.get("apelido") or ident.get("nome"),
            "color": color,
            "cardiac_curve": cardiac_curve,
            "individual_curves": individual_curves,
            "centroid": centroid,
            "faces_count": len(embeddings_for_identity),
            "face_details": face_details,
            "heartbeat_amplitude": heartbeat_amplitude,
            "heartbeat_phase": heartbeat_phase,
            "embedding_stats": {
                "mean": float(np.mean(avg_embedding)),
                "std": float(np.std(avg_embedding)),
                "min": float(np.min(avg_embedding)),
                "max": float(np.max(avg_embedding)),
            },
        }
        cardiac_entities.append(entity)
        all_curves.append(cardiac_curve)

    total_embeddings = sum(e["faces_count"] for e in cardiac_entities)

    return {
        "entities": cardiac_entities,
        "total_identities": len(cardiac_entities),
        "total_embeddings": total_embeddings,
        "points_per_curve": 180,
    }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _json_response(handler, data, status=200):
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    handler.send_header('Access-Control-Allow-Headers', 'Content-Type')
    handler.end_headers()
    handler.wfile.write(json.dumps(data, default=str).encode('utf-8'))


def _html_response(handler, filepath, content_type='text/html; charset=utf-8'):
    handler.send_response(200)
    handler.send_header('Content-Type', content_type)
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.end_headers()
    with open(filepath, "r", encoding="utf-8") as f:
        handler.wfile.write(f.read().encode("utf-8"))


def _read_body(handler):
    length = int(handler.headers.get('Content-Length', 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode('utf-8'))


class ObserverHandler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip('/')

        # ── Pages ─────────────────────────────────────────────────
        if path == "" or path == "/":
            _html_response(self, "/home/alifer/swa-aeon/cortex/observer/index.html")

        elif path == "/shell":
            _html_response(self, "/home/alifer/swa-aeon/cortex/observer/shell.html")

        elif path == "/identities":
            _html_response(self, "/home/alifer/swa-aeon/cortex/observer/identities.html")

        elif path == "/dashboard":
            _html_response(self, "/home/alifer/swa-aeon/cortex/observer/dashboard.html")

        elif path == "/mycelium":
            _html_response(self, "/home/alifer/swa-aeon/cortex/observer/mycelium.html")

        # ── API: State ───────────────────────────────────────────
        elif path == "/api/state":
            state = {"dna": {}, "chaos_matrix": "", "diary": [], "voice": ""}
            try:
                with open("/var/lib/aeon/aeon_dna.json", "r") as f:
                    state["dna"] = json.load(f)
            except: pass
            shm_path = "/dev/shm/aeon_latent_chaos.dat"
            if os.path.exists(shm_path):
                try:
                    with open(shm_path, "rb") as f:
                        raw_bytes = f.read(16384)
                    if len(raw_bytes) == 16384:
                        state["chaos_matrix"] = base64.b64encode(raw_bytes).decode('utf-8')
                except: pass
            try:
                with open("/dev/shm/aeon_voice.txt", "r", encoding="utf-8") as f:
                    state["voice"] = f.read().strip()
            except: pass
            diary_path = "/mnt/c/Users/alifer.ramos/Desktop/AEON_DIARY.txt"
            if os.path.exists(diary_path):
                try:
                    with open(diary_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        state["diary"] = [l.strip() for l in lines[-10:]]
                except: pass
            _json_response(self, state)

        # ── API: Event Stream (SSE) ──────────────────────────────
        elif path == "/api/event-stream":
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                import redis as _r
                r = _r.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
                while True:
                    # Coleta Chaos Matrix
                    chaos_matrix = ""
                    shm_path = "/dev/shm/aeon_latent_chaos.dat"
                    if os.path.exists(shm_path):
                        try:
                            with open(shm_path, "rb") as f:
                                raw_bytes = f.read(16384)
                            if len(raw_bytes) == 16384:
                                chaos_matrix = base64.b64encode(raw_bytes).decode('utf-8')
                        except: pass
                    
                    # Coleta Presença
                    presence = {}
                    try:
                        raw = r.get("swa:vision:current_presence")
                        if raw:
                            presence = json.loads(raw)
                    except: pass
                    
                    # Log OODA/Diary
                    diary = []
                    diary_path = "/mnt/c/Users/alifer.ramos/Desktop/AEON_DIARY.txt"
                    if os.path.exists(diary_path):
                        try:
                            with open(diary_path, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                                diary = [l.strip() for l in lines[-15:] if l.strip()]
                        except: pass

                    # Broadcast events
                    if chaos_matrix:
                        self.wfile.write(f"event: matrix_update\ndata: {json.dumps({'chaos_matrix': chaos_matrix})}\n\n".encode('utf-8'))
                    
                    self.wfile.write(f"event: presence_change\ndata: {json.dumps(presence)}\n\n".encode('utf-8'))
                    self.wfile.write(f"event: ooda_log\ndata: {json.dumps({'diary': diary})}\n\n".encode('utf-8'))
                    
                    self.wfile.flush()
                    time.sleep(0.3)
            except Exception as e:
                return

        # ── API: List identities ─────────────────────────────────
        elif path == "/api/identities":
            identities = _db.list_identities(ativo_only=False)
            for ident in identities:
                faces = _db.get_face_encodings(ident["id"])
                ident["faces"] = faces
                ident["face_count"] = len(faces)
                ident["angles"] = [f.get("angle") for f in faces]
            _json_response(self, {"identities": identities, "total": len(identities)})

        # ── API: Recognition history ────────────────────────────
        elif path.startswith("/api/identities/") and path.endswith("/history"):
            identity_id = path.split("/")[3]
            history = _db.get_recognition_history(identity_id, limit=20)
            _json_response(self, {"history": history})

        # ── API: Current presence (who is in front of camera) ───
        elif path == "/api/presence":
            presence = {}
            try:
                import redis as _r
                r = _r.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
                raw = r.get("swa:vision:current_presence")
                if raw:
                    presence = json.loads(raw)
            except: pass
            _json_response(self, presence)

        # ── API: Embedding Space 3D ─────────────────────────────
        elif path == "/api/embeddings/3d":
            _json_response(self, _compute_embedding_space_3d())

        # ── API: Cardiac Map 3D ─────────────────────────────────
        elif path == "/api/cardiac/3d":
            _json_response(self, _compute_cardiac_map())

        # ── API: GraphRAG Fractal (Mycelium) ──────────────────────
        elif path.startswith("/api/mycelium"):
            from urllib.parse import urlparse, parse_qs
            parsed_path = urlparse(self.path)
            query = parse_qs(parsed_path.query)
            target_macro_node = query.get("node", [None])[0]
            
            graph = {"nodes": [], "links": []}
            knowledge_path = "/home/alifer/swa-aeon/swa_kernel_knowledge_fractal.json"
            
            if os.path.exists(knowledge_path):
                try:
                    with open(knowledge_path, "r", encoding="utf-8") as f:
                        fractal_db = json.load(f)
                        
                    # MODO 1: MICRO-COSMOS (Zoom In)
                    if target_macro_node and target_macro_node in fractal_db:
                        micro_nodes = fractal_db[target_macro_node].get("micro_nodes", {})
                        keys = list(micro_nodes.keys())
                        
                        # 1. Cria os nós
                        for k in keys:
                            graph["nodes"].append({
                                "id": k,
                                "name": k,
                                "group": 2, # Cor laranja para os micro-nós
                                "val": 1.5, # Menores que os macro
                            })
                            
                        # 2. Calcula as sinapses locais (micro-relações internas)
                        if len(keys) > 1:
                            mat = np.array([micro_nodes[k]["embedding"] for k in keys], dtype=np.float64)
                            norms = np.linalg.norm(mat, axis=1, keepdims=True)
                            mat_norm = np.divide(mat, norms, out=np.zeros_like(mat), where=norms!=0)
                            sim_matrix = np.dot(mat_norm, mat_norm.T)
                            
                            for i in range(len(keys)):
                                for j in range(i + 1, len(keys)):
                                    sim = sim_matrix[i, j]
                                    if sim > 0.20: # Correlação fraca/média para micro-nós
                                        graph["links"].append({
                                            "source": keys[i], "target": keys[j], "value": float(sim)
                                        })

                    # MODO 2: MACRO-COSMOS (Zoom Out)
                    else:
                        keys = list(fractal_db.keys())
                        
                        for k in keys:
                            graph["nodes"].append({
                                "id": k,
                                "name": k,
                                "group": 1, 
                                "val": 3.0,
                            })
                            
                        if len(keys) > 1:
                            mat = np.array([fractal_db[k]["macro_embedding"] for k in keys], dtype=np.float64)
                            norms = np.linalg.norm(mat, axis=1, keepdims=True)
                            mat_norm = np.divide(mat, norms, out=np.zeros_like(mat), where=norms!=0)
                            sim_matrix = np.dot(mat_norm, mat_norm.T)
                            
                            for i in range(len(keys)):
                                for j in range(i + 1, len(keys)):
                                    sim = sim_matrix[i, j]
                                    if sim > 0.15: # Threshold bem baixo para permitir conexões de macro-cosmos distante
                                        graph["links"].append({
                                            "source": keys[i], "target": keys[j], "value": float(sim)
                                        })
                                    
                except Exception as e:
                    print("Erro no calculo do micélio:", e)
                    
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(graph, f)
            except: pass
            
            _json_response(self, graph)

        # ── API: Latest frame for calibrator ────────────────────
        elif path == "/api/vision/latest":
            import redis as _r
            r = _r.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
            raw = r.get("swa:vision:latest_for_calibrate")
            if raw:
                _json_response(self, json.loads(raw))
            else:
                _json_response(self, {"error": "nenhum frame disponível", "image": None})

        # ── Pages ────────────────────────────────────────────────
        elif path == "/embeddings":
            _html_response(self, "/services/model-with-adpter-logic-semantic/cortex/observer/embeddings.html")

        elif path == "/cardiac":
            _html_response(self, "/services/model-with-adpter-logic-semantic/cortex/observer/cardiac.html")

        elif path == "/calibrate":
            _html_response(self, "/services/model-with-adpter-logic-semantic/cortex/observer/calibrate.html")

        elif path == "/blakegravity":
            _html_response(self, "/services/model-with-adpter-logic-semantic/cortex/observer/blakegravity.html")

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path == "/api/identities":
            body = _read_body(self)
            nome = body.get("nome", "").strip()
            if not nome:
                _json_response(self, {"error": "nome é obrigatório"}, 400)
                return
            ident = _db.create_identity(
                nome=nome,
                apelido=body.get("apelido"),
                email=body.get("email"),
                papel=body.get("papel", "usuario"),
                metadata=body.get("metadata", {}),
            )
            _json_response(self, ident, 201)

        elif path.startswith("/api/identities/") and path.endswith("/faces"):
            identity_id = path.split("/")[3]
            body = _read_body(self)
            embedding = body.get("embedding_128")
            if not embedding:
                _json_response(self, {"error": "embedding_128 é obrigatório"}, 400)
                return
            face = _db.add_face_encoding(
                identity_id=identity_id,
                embedding_128=embedding,
                angle=body.get("angle", "front"),
                photo_path=body.get("photo_path"),
                subtleties=body.get("subtleties"),
            )
            _json_response(self, face, 201)

        elif path == "/api/calibrate/save":
            body = _read_body(self)
            embedding = body.get("embedding")
            identity_name = body.get("name", "").strip()
            angle = body.get("angle", "front")
            if not embedding or not identity_name:
                _json_response(self, {"error": "embedding e name são obrigatórios"}, 400)
                return
            identities = _db.list_identities()
            matched = None
            for ident in identities:
                if ident.get("nome") == identity_name or ident.get("apelido") == identity_name:
                    matched = ident
                    break
            if not matched:
                matched = _db.create_identity(
                    nome=identity_name,
                    apelido=identity_name,
                    papel="usuario",
                    metadata={"source": "calibrator"},
                )
            face = _db.add_face_encoding(
                identity_id=matched["id"],
                embedding_128=embedding,
                angle=angle,
                metadata={"source": "calibrator"},
            )
            _json_response(self, {"ok": True, "identity_id": matched["id"], "face_id": face.get("id"), "nome": identity_name}, 201)

        else:
            self.send_response(404)
            self.end_headers()

    def do_PUT(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path.startswith("/api/identities/") and not path.endswith("/faces"):
            identity_id = path.split("/")[3]
            body = _read_body(self)
            ok = _db.update_identity(identity_id, **body)
            if ok:
                _json_response(self, {"ok": True})
            else:
                _json_response(self, {"error": "identidade não encontrada"}, 404)

        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path.startswith("/api/identities/"):
            identity_id = path.split("/")[3]
            ok = _db.delete_identity(identity_id)
            if ok:
                _json_response(self, {"ok": True})
            else:
                _json_response(self, {"error": "identidade não encontrada"}, 404)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    PORT = 20008
    print(f"👁️ Iniciando AEON Observer Endpoint (SSE + Mycelium) na porta {PORT}...")
    
    # --- INTEGRAÇÃO FRACTAL AO FLOW ---
    try:
        print("🌌 Verificando Integridade da Memória Fractal (GraphRAG)...")
        from organs.fractal_memory import FractalMemoryOrgan
        f_organ = FractalMemoryOrgan()
        f_organ.mock_migrate_from_flat()
    except Exception as e:
        print(f"⚠️ Erro ao integrar Memória Fractal no boot: {e}")
    # ----------------------------------
    
    server = ThreadedHTTPServer(('', PORT), ObserverHandler)
    server.serve_forever()
