import os
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ─── ESTRUTURA BASE (DRY) ────────────────────────────────────────────────
@dataclass
class CosmicElement:
    element: str
    name: str
    color: str
    glow: float
    orbit_speed: float
    shape: str
    behavior: str
    quantum_state: str = "collapsed"

    def to_dict(self):
        return {
            "element": self.element,
            "name": self.name,
            "color": self.color,
            "glow": self.glow,
            "orbit_speed": self.orbit_speed,
            "shape": self.shape,
            "behavior": self.behavior,
            "quantum_state": self.quantum_state
        }

BASE_TEMPLATES = {
    "Plasma":   {"element": "Plasma", "glow": 1.5, "orbit_speed": 1.2, "shape": "esfera", "behavior": "pulsante", "quantum_state": "superposition"},
    "Cristal":  {"element": "Cristal", "glow": 1.0, "orbit_speed": 0.8, "shape": "icosaedro", "behavior": "constante", "quantum_state": "collapsed"},
    "Cometa":   {"element": "Cometa", "glow": 1.2, "orbit_speed": 1.8, "shape": "elipse", "behavior": "rastro", "quantum_state": "collapsed"},
    "AnaBranca":{"element": "Anã Branca", "glow": 0.5, "orbit_speed": 0.3, "shape": "denso", "behavior": "superdenso", "quantum_state": "collapsed"},
    "Nebula":   {"element": "Nébula", "glow": 0.6, "orbit_speed": 0.4, "shape": "nuvem", "behavior": "flutuante", "quantum_state": "collapsed"},
    "Estrela":  {"element": "Estrela", "glow": 1.5, "orbit_speed": 0.9, "shape": "estrela", "behavior": "brilhante", "quantum_state": "superposition"},
    "Planeta":  {"element": "Planeta", "glow": 0.9, "orbit_speed": 0.5, "shape": "esfera", "behavior": "orbitante", "quantum_state": "collapsed"},
    "Quasar":   {"element": "Quasar", "glow": 1.0, "orbit_speed": 0.3, "shape": "octaedro", "behavior": "direcional", "quantum_state": "entangled"},
    "Pulsar":   {"element": "Pulsar", "glow": 0.8, "orbit_speed": 0.4, "shape": "dodecaedro", "behavior": "periódico", "quantum_state": "collapsed"},
    "Meteoro":  {"element": "Meteoro", "glow": 0.5, "orbit_speed": 3.0, "shape": "ponto", "behavior": "efêmero", "quantum_state": "collapsed"},
}

def create_element(base_name: str, name: str, color: str, overrides: Dict[str, Any] = None) -> CosmicElement:
    config = BASE_TEMPLATES.get(base_name, {}).copy()
    config.update({"name": name, "color": color})
    if overrides: config.update(overrides)
    return CosmicElement(**config)

ELEMENT_MAP: Dict[str, CosmicElement] = {
    ".jpg":  create_element("Planeta", "Planeta Imagem", "#ff9800"),
    ".png":  create_element("Planeta", "Planeta Imagem", "#29b6f6"),
    ".svg":  create_element("Planeta", "Planeta Vetorial", "#ffb300", {"shape": "icosaedro"}),
    ".mp4":  create_element("Planeta", "Planeta Vídeo", "#9c27b0", {"shape": "anel"}),
    ".mp3":  create_element("Planeta", "Planeta Áudio", "#e91e63", {"shape": "anel"}),
    ".html": create_element("Estrela", "Estrela HTML", "#e34c26"),
    ".css":  create_element("Estrela", "Estrela CSS", "#264de4"),
    ".vue":  create_element("Estrela", "Estrela Vue", "#41b883"),
    ".py":   create_element("Plasma", "Plasma Python", "#3572A5"),
    ".js":   create_element("Plasma", "Plasma JavaScript", "#f1e05a"),
    ".ts":   create_element("Plasma", "Plasma TypeScript", "#2b7489"),
    ".env":  create_element("Quasar", "Quasar Segredo", "#f44336", {"glow": 1.5, "behavior": "oculto", "quantum_state": "entangled"}),
    ".gitignore": create_element("Quasar", "Quasar Git", "#f05033"),
    ".tmp":  create_element("Meteoro", "Meteoro Temp", "#ffeb3b"),
    ".bak":  create_element("Meteoro", "Meteoro Backup", "#f57c00"),
    ".aeon": create_element("Célula", "Célula de Aeon", "#d500f9", {"shape": "nebulosa", "behavior": "pulsante", "glow": 2.0, "quantum_state": "superposition"}),
    "node_modules": create_element("Meteoro", "Meteoro NPM", "#cb3837", {"orbit_speed": 1.0, "glow": 0.2}),
    ".git":  create_element("Meteoro", "Buraco Negro Git", "#000000", {"glow": 0.0, "quantum_state": "collapsed"})
}

DEFAULT_ELEMENT = create_element("Planeta", "Matéria Escura", "#263238", {"element": "Matéria Escura", "glow": 0.2, "shape": "ponto", "behavior": "desconhecido"})

# ─── ALINHAMENTO QUÂNTICO ────────────────────────────────────────────────
def compute_quantum_alignment(filepath: str, element_type: str) -> dict:
    try:
        stat = os.stat(filepath)
        age_hours = (time.time() - stat.st_mtime) / 3600.0
    except:
        age_hours = 999.0
    
    q_state = "collapsed"
    q_energy = 0.1

    if element_type == "Quasar":
        q_state = "entangled"
        q_energy = 1.0
    elif age_hours < 24.0:
        q_state = "superposition"
        q_energy = 0.8 + (1 / (age_hours + 1))
    else:
        q_state = "collapsed"
        q_energy = 0.2
        
    return {"quantum_state": q_state, "quantum_energy": q_energy}

def compute_blake3_mass(filepath: str) -> dict:
    try:
        size = os.path.getsize(filepath)
        mass = max(0.1, min(10.0, size / 10240.0)) # Apenas aproximação
    except:
        mass = 0.1
    return {"mass": mass, "hash": "blake3_mock"}

# ─── SCANNER CORRIGIDO ───────────────────────────────────────────────────
def scan_tree(dirpath: str, depth: int = 0, max_depth: int = 2) -> dict:
    if depth > max_depth:
        return {"mass": 0, "file_count": 0, "dir_count": 0, "nodes": []}

    star_mass = 0
    file_count = 0
    dir_count = 0
    nodes = []

    try:
        with os.scandir(dirpath) as it:
            for entry in it:
                if entry.is_symlink():
                    continue

                if entry.name in ["node_modules", ".git", "__pycache__", "venv", ".venv"]:
                    file_count += 1
                    elem = ELEMENT_MAP.get(entry.name, DEFAULT_ELEMENT)
                    nodes.append({
                        "name": entry.name,
                        "path": entry.path,
                        "type": "directory_void",
                        "mass": 0.1,
                        "element": elem.to_dict(),
                        "quantum": {"quantum_state": "collapsed", "quantum_energy": 0.1}
                    })
                    continue

                if entry.is_dir():
                    dir_count += 1
                    dir_info = scan_tree(entry.path, depth + 1, max_depth)
                    star_mass += dir_info["mass"] + 0.5
                    
                    nodes.append({
                        "name": entry.name,
                        "path": entry.path,
                        "type": "directory",
                        "mass": dir_info["mass"] + 0.5,
                        "children_count": dir_info["file_count"] + dir_info["dir_count"]
                    })
                
                elif entry.is_file():
                    file_count += 1
                    ext = Path(entry.name).suffix.lower()
                    elem_info = ELEMENT_MAP.get(ext, ELEMENT_MAP.get(entry.name, DEFAULT_ELEMENT))
                    
                    mass_info = compute_blake3_mass(entry.path)
                    quantum = compute_quantum_alignment(entry.path, elem_info.element)
                    
                    effective_mass = mass_info["mass"] * quantum["quantum_energy"]

                    target_path = None
                    if ext == ".service":
                        try:
                            with open(entry.path, 'r', encoding='utf-8', errors='ignore') as f:
                                for line in f:
                                    line = line.strip()
                                    if line.startswith("WorkingDirectory="):
                                        target_path = line.split("=")[1].strip()
                                        break
                                    elif line.startswith("ExecStart="):
                                        parts = line.split()
                                        for p in parts:
                                            if p.startswith("/") and not p.startswith("/usr/bin/") and not p.startswith("/bin/"):
                                                target_path = str(Path(p).parent)
                                                break
                        except Exception:
                            pass
                    elif ext == ".aeon":
                        try:
                            with open(entry.path, 'r', encoding='utf-8', errors='ignore') as f:
                                for line in f:
                                    line = line.strip()
                                    # Células de Aeon podem conter portais gravitacionais escritos dentro delas
                                    if line.startswith("Target=") or line.startswith("Gravity=") or line.startswith("Nucleus="):
                                        target_path = line.split("=")[1].strip()
                                        break
                        except Exception:
                            pass

                    nodes.append({
                        "name": entry.name,
                        "path": entry.path,
                        "target_path": target_path,
                        "type": "file",
                        "mass": effective_mass,
                        "element": elem_info.to_dict(),
                        "quantum": quantum
                    })
                    star_mass += effective_mass
                    
    except PermissionError:
        pass

    return {
        "mass": star_mass,
        "file_count": file_count,
        "dir_count": dir_count,
        "nodes": nodes
    }


# ─── SERVIDOR HTTP (PORTA 20009) ──────────────────────────────────────────
class BlakeGravityServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        # Trata as requisições de Preflight (CORS) do navegador
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        
        # Ignora slashes no final e flexibiliza a rota principal
        if not parsed_url.path.strip('/').endswith('api/tree'):
            self.send_response(404)
            self.end_headers()
            return
            
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        query = parse_qs(parsed_url.query)
        target_path = query.get('path', ['.'])[0]
        max_depth = int(query.get('depth', [2])[0])

        tree_data = scan_tree(target_path, max_depth=max_depth)
        
        # Formatando a resposta exatamente como o frontend espera
        response = {
            "star": {
                "name": os.path.basename(target_path) or target_path,
                "mass": tree_data["mass"],
                "file_count": tree_data["file_count"],
                "dir_count": tree_data["dir_count"]
            },
            "nodes": tree_data["nodes"],
            "element_legend": {k: v.to_dict() for k, v in ELEMENT_MAP.items()}
        }
        
        self.wfile.write(json.dumps(response).encode('utf-8'))

def run_server():
    port = 20009
    server = HTTPServer(('0.0.0.0', port), BlakeGravityServer)
    print(f"[BLAKEGRAVITY ASTRO] Servidor Quântico rodando na porta {port}...")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
