import time
import json
import struct
import math
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Matriz de Projeção Aleatória Fixa (1152 -> 3) (Preserva geometria por Johnson-Lindenstrauss)
np.random.seed(42)
PROJ_MATRIX = np.random.randn(1152, 3).astype(np.float32)

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Aeon - White Hole Visualizer</title>
    <style>
        body { margin: 0; overflow: hidden; background-color: #050510; font-family: 'Inter', sans-serif; }
        #info { position: absolute; top: 10px; left: 10px; color: #00ffcc; z-index: 100; text-shadow: 0 0 5px #00ffcc; }
        #warp-btn { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); padding: 15px 30px; 
                    background: transparent; border: 2px solid #ff00ff; color: #ff00ff; cursor: pointer; font-size: 16px;
                    border-radius: 5px; box-shadow: 0 0 15px rgba(255, 0, 255, 0.5); transition: 0.3s; z-index: 100; font-weight: bold; }
        #warp-btn:hover { background: #ff00ff; color: white; box-shadow: 0 0 30px #ff00ff; }
        .hud { font-size: 12px; opacity: 0.8; }
        .alert { color: #ff0055; font-weight: bold; animation: pulse 1s infinite; display: none; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="info">
        <h2>🌌 OODA: Trajetória Tensorial (Camada 26)</h2>
        <div class="hud">Norma Euclidiana: <span id="norm">0.00</span></div>
        <div class="hud">Variância (Caos): <span id="std">0.00</span></div>
        <div class="hud">Status: <span id="status">Estável</span></div>
        <div class="alert" id="alert">⚠ BURACO BRANCO ATIVO - SUPERPOSIÇÃO INJETADA</div>
    </div>
    
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Partículas do pensamento (Rastro)
        const maxPoints = 500;
        const positions = new Float32Array(maxPoints * 3);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const material = new THREE.PointsMaterial({ color: 0x00ffcc, size: 0.5, transparent: true, opacity: 0.8 });
        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);

        // Ponto Atual (Cabeça do Pensamento)
        const headGeo = new THREE.SphereGeometry(0.3, 16, 16);
        const headMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const head = new THREE.Mesh(headGeo, headMat);
        scene.add(head);

        // Grade de Fundo
        const gridHelper = new THREE.GridHelper(50, 50, 0xff00ff, 0x222244);
        scene.add(gridHelper);

        camera.position.z = 15;
        camera.position.y = 5;
        camera.lookAt(0, 0, 0);

        let pointIndex = 0;
        let lastPos = new THREE.Vector3(0,0,0);

        function updateData() {
            fetch('/data')
                .then(r => r.json())
                .then(data => {
                    if(data.error) return;
                    
                    document.getElementById('norm').innerText = data.norm.toFixed(4);
                    document.getElementById('std').innerText = data.std.toFixed(4);
                    
                    // Atualiza coordenadas (X,Y,Z projetadas)
                    const x = data.proj[0] * 2.0; // Fator visual
                    const y = data.proj[1] * 2.0;
                    const z = data.proj[2] * 2.0;

                    // Cor do rastro muda com a variância
                    if(data.std > 0.7) {
                        material.color.setHex(0xff0055); // Caos (Buraco Branco)
                        document.getElementById('status').innerText = "Superposição (Colapso)";
                        document.getElementById('status').style.color = "#ff0055";
                    } else if (data.norm < 0.2) {
                        material.color.setHex(0x555555); // Tédio
                        document.getElementById('status').innerText = "Tédio / Inércia";
                        document.getElementById('status').style.color = "#aaaaaa";
                    } else {
                        material.color.setHex(0x00ffcc); // Estável
                        document.getElementById('status').innerText = "Focado";
                        document.getElementById('status').style.color = "#00ffcc";
                    }

                    // Adiciona ao buffer circular
                    const positions = geometry.attributes.position.array;
                    positions[pointIndex * 3] = x;
                    positions[pointIndex * 3 + 1] = y;
                    positions[pointIndex * 3 + 2] = z;
                    
                    head.position.set(x, y, z);
                    
                    pointIndex = (pointIndex + 1) % maxPoints;
                    geometry.attributes.position.needsUpdate = true;
                    
                    // Mover a camera sutilmente para acompanhar o centro de gravidade
                    camera.position.x += (x - camera.position.x) * 0.02;
                    camera.lookAt(0,0,0);
                })
                .catch(e => console.log(e));
        }

        setInterval(updateData, 200); // 5 FPS fetch

        function animate() {
            requestAnimationFrame(animate);
            scene.rotation.y += 0.005; // Rotação suave do universo
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""

class WarpHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        elif parsed.path == '/data':
            # Ler o tensor raw em /dev/shm/aeon/core.tensor.bin
            tensor_path = "/dev/shm/aeon/core.tensor.bin"
            try:
                with open(tensor_path, "rb") as f:
                    raw = f.read()
                    # 1152 floats (4 bytes each)
                    if len(raw) == 1152 * 4:
                        vec = np.frombuffer(raw, dtype=np.float32)
                        
                        # Calcula anomalias reais
                        norm = float(np.linalg.norm(vec))
                        std = float(np.std(vec))
                        
                        # Projeção Randomica (JL Lemma) para 3D
                        proj3d = np.dot(vec, PROJ_MATRIX)
                        
                        resp = {
                            "norm": norm,
                            "std": std,
                            "proj": [float(proj3d[0]), float(proj3d[1]), float(proj3d[2])]
                        }
                    else:
                        resp = {"error": "Tamanho incorreto"}
            except Exception as e:
                resp = {"error": str(e)}
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = HTTPServer(('0.0.0.0', 30042), WarpHandler)
    print("🌌 Visualizador de Buraco Branco Online: http://127.0.0.1:30042")
    server.serve_forever()
