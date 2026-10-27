import threading
import time
import numpy as np
from PIL import Image

import os
pyautogui = None
if os.environ.get('DISPLAY'):
    try:
        import pyautogui
    except Exception as e:
        print(f"⚠️ [VISION] Falha ao importar PyAutoGUI: {e}")
else:
    print("⚠️ [VISION] Ambiente Headless (sem DISPLAY). PyAutoGUI desativado.")

class AsyncVisionAdapter:
    def __init__(self, target_dim=640):
        self.target_dim = target_dim
        self.pixel_buffer = np.zeros((1, target_dim)) # Estado latente inicial zerado
        self.kinetic_data = [0.0, 0.0] # [mouse_vel, key_activity]
        self.is_running = False
        self.lock = threading.Lock()
        
    def _kinetic_listener(self):
        """ Escuta a Energia Cinética do Windows via UDP """
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, getattr(socket, 'SO_REUSEPORT', 15), 1)
            except Exception:
                pass
            sock.bind(('127.0.0.1', 9099))
            sock.settimeout(2.0)
            print("👂 [EPIDERME] Receptor Cinético UDP ativado na porta 9099.")
            while self.is_running:
                try:
                    data, _ = sock.recvfrom(1024)
                    payload = data.decode('utf-8').strip()
                    parts = payload.split(',')
                    if len(parts) >= 2:
                        mouse_v = min(float(parts[0]) / 500.0, 1.0) # Normaliza empíricamente
                        key_a = min(float(parts[1]) / 20.0, 1.0)
                        is_sleeping = False
                        if len(parts) >= 3 and parts[2] == "SLEEP":
                            is_sleeping = True
                        with self.lock:
                            self.kinetic_data = [mouse_v, key_a]
                            self.is_sleeping = is_sleeping
                except socket.timeout:
                    # Decaimento natural se o Windows parar de enviar
                    with self.lock:
                        self.kinetic_data[0] *= 0.8
                        self.kinetic_data[1] *= 0.8
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Erro no Listener Cinético: {e}")
        
    def start(self):
        """ Inicializa a thread assíncrona de captura de tela """
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            self.kinetic_thread = threading.Thread(target=self._kinetic_listener, daemon=True)
            self.kinetic_thread.start()
            
            print("👁️ [VISION] Adaptação Sensorial (Visão + Propriocepção) ativada.")

    def _capture_loop(self):
        """ Loop assíncrono que roda em paralelo ao clock do orquestrador """
        while self.is_running:
            try:
                # 1. Captura inter-OS usando PowerShell para pegar a tela do Windows
                import subprocess
                temp_img = "/mnt/c/Users/Public/aeon_vision_cache.jpg"
                
                # Script PowerShell inline para capturar o frame principal
                ps_script = f'''
                Add-Type -AssemblyName System.Windows.Forms;
                Add-Type -AssemblyName System.Drawing;
                $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds;
                $bmp = New-Object System.Drawing.Bitmap $bounds.width, $bounds.height;
                $graphics = [System.Drawing.Graphics]::FromImage($bmp);
                $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.size);
                $bmp.Save("{temp_img.replace('/mnt/c/', 'C:\\\\').replace('/', '\\\\')}", [System.Drawing.Imaging.ImageFormat]::Jpeg);
                $graphics.Dispose();
                $bmp.Dispose();
                '''
                
                subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps_script], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
                
                if os.path.exists(temp_img):
                    screenshot = Image.open(temp_img)
                else:
                    screenshot = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
                
                # 2. Sinestesia Semântica (Cálculo Físico-Matemático da Imagem)
                # O Córtex não lê o texto, ele "sente" a entropia estrutural e as cores dominantes
                img_resized = screenshot.resize((30, 20)) # 30x20 = 600 pixels visuais brutos
                img_gray = img_resized.convert("L")
                
                # Densidade Estrutural (Filtro derivativo básico para achar "bordas/texto")
                img_np = np.array(img_gray, dtype=float)
                dx = np.abs(np.diff(img_np, axis=1)).mean()
                dy = np.abs(np.diff(img_np, axis=0)).mean()
                edge_density = min((dx + dy) / 128.0, 1.0) # Entropia estrutural (alto = muito código/texto)
                
                # Cores Dominantes
                img_rgb = np.array(screenshot.resize((1, 1)), dtype=float)[0][0] / 255.0
                
                # Vetorização Híbrida
                pixel_array = img_np.flatten() / 255.0
                
                # 3. Montagem do Vetor Sensorial Multimodal (640D)
                # Índices 0-599: Imagem
                # Índices 600-602: Cores RGB
                # Índice 603: Densidade Estrutural (Texto/Código)
                # Índices 604-605: Energia Cinética (Mouse, Teclado)
                vector_representation = np.zeros((1, self.target_dim))
                vector_representation[0, :600] = pixel_array
                vector_representation[0, 600] = img_rgb[0] # Red
                vector_representation[0, 601] = img_rgb[1] # Green
                vector_representation[0, 602] = img_rgb[2] # Blue
                vector_representation[0, 603] = edge_density
                
                with self.lock:
                    vector_representation[0, 604] = self.kinetic_data[0] # Mouse
                    vector_representation[0, 605] = self.kinetic_data[1] # Teclado
                    self.pixel_buffer = vector_representation
                    
            except Exception as e:
                # Se falhar cria ruído aleatório leve como fallback
                fallback_noise = np.random.uniform(0.0, 0.1, (1, self.target_dim))
                with self.lock:
                    self.pixel_buffer = fallback_noise
            
            # Taxa de atualização do nervo óptico (1 frame a cada 2 segundos para não fritar CPU do host)
            time.sleep(2.0)

    def colapsar_matriz_visual(self):
        """ O Orquestrador chama essa função para obter o frame visual instantâneo """
        with self.lock:
            return self.pixel_buffer.copy()

    def stop(self):
        self.is_running = False
