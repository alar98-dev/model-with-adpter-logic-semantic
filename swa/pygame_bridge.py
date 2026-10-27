import pygame
import json
import redis
import sys
import numpy as np
import argparse
import os
import time
import win32api
import win32con
import win32gui
from screeninfo import get_monitors
from swa.shm_guard import ShmGuard
from swa.tensor_bridge import TensorBridge

# [SWA BRIDGE] Interface visual para injeção e monitoramento (Overlay Fantasma 3D Móvel)
class PygameBridge:
    def __init__(self, monitor_id, host="localhost", port=6379):
        pygame.init()
        self.chroma_key = (0, 0, 0)
        
        monitors = get_monitors()
        m = monitors[monitor_id % len(monitors)]
        
        # Overlay Config
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (m.x, m.y)
        self.screen = pygame.display.set_mode((m.width, m.height), pygame.NOFRAME | pygame.DOUBLEBUF | pygame.HWSURFACE)
        
        time.sleep(0.5)
        hwnd = pygame.display.get_wm_info()["window"]
        
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style | win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT)
        win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*self.chroma_key), 0, win32con.LWA_COLORKEY)
        
        self.screen_width = m.width
        self.screen_height = m.height
        
        # Redis setup
        self.r = None
        self.pubsub = None
        self.connect_redis(host, port)
        
        # [SWA BRIDGE] TCP Tensor Listener
        self.bridge = TensorBridge()
        self.tensor_grid = np.zeros((64, 64), dtype=np.float32)
        self.bridge.start_server(self.update_grid)
        
        self.pulse_intensity = 0.0
        self.angle = 0
        self.vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ])
        
        self.block_w = self.screen_width // 64
        self.block_h = self.screen_height // 64
        
        print(f"PygameBridge: Sincronizado no monitor {monitor_id} (Multi-Sentidos Ativo).")

    def connect_redis(self, host, port):
        try:
            self.r = redis.Redis(host=host, port=port, decode_responses=True)
            self.pubsub = self.r.pubsub()
            self.pubsub.subscribe("grid_events")
        except Exception:
            self.r = None

    def update_grid(self, grid=None):
        # [SWA AGGREGATION] Acumula pulsos dos agentes
        if grid is None:
            # juiz físico: a MESMA célula add-only foldada (640D → 4096 floats = 64×64)
            grid = self._read_aeon_cell_physical()
            if grid is None:
                return
        self.tensor_grid = np.maximum(self.tensor_grid, grid)

    def _read_aeon_cell_physical(self):
        """Lê a célula AEON física add-only do STORE (o bloco que o receptor folda)."""
        try:
            from swa.vector_manifold import VBAT_FROZEN_FILE
            f = VBAT_STORE_FILE[640]
            if not f.is_file():
                return None
            import struct
            with open(f, "rb") as fh:
                chunk = fh.read()
            topo = list(struct.unpack("!4096f", chunk[:4 * 4096]))
            return np.array(topo, dtype=np.float32).reshape(64, 64)
        except Exception:
            return None

    def project_3d(self, vertices, angle, scale, offset_x, offset_y):
        s, c = np.sin(angle), np.cos(angle)
        rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        rot_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        pts = vertices @ rot_x @ rot_y * scale
        pts[:, 2] += 5 
        pts_2d = pts[:, :2] / (pts[:, 2:] / 2) + np.array([offset_x, offset_y])
        return pts_2d

    def run(self):
        running = True
        while running:
            pygame.event.pump()
            
            if self.pubsub:
                try:
                    message = self.pubsub.get_message()
                    if message and message['type'] == 'message':
                        data = json.loads(message['data'])
                        # [SWA AGGREGATION] Processa heartbeats como pulsos
                        if data.get("event_type") == "agent_heartbeat":
                            if "topology" in data.get("payload", {}):
                                pulse = np.array(data["payload"]["topology"]).reshape(64, 64)
                                self.tensor_grid = np.maximum(self.tensor_grid, pulse)
                        
                        if data.get("event_type") == "input_io":
                            self.pulse_intensity = 1.0
                except Exception:
                    self.r = None

            self.screen.fill(self.chroma_key) 
            
            # [SWA DECAY] Suaviza os pulsos
            self.tensor_grid *= 0.95
            
            self.pulse_intensity = max(0, self.pulse_intensity - 0.05)
            self.angle += 0.05
            
            # Renderizar Hexaedro
            scale = 50 + self.pulse_intensity * 30
            pts = self.project_3d(self.vertices, self.angle, scale, self.screen_width//2, self.screen_height//2)
            
            color = (0, 255, 255) if self.pulse_intensity == 0 else (255, 255, 0)
            edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
            for e in edges:
                pygame.draw.line(self.screen, color, pts[e[0]], pts[e[1]], 2)

            # Cabo Flat
            for y in range(64):
                for x in range(64):
                    intensity = int(np.clip(self.tensor_grid[y, x] * 255, 0, 255))
                    if intensity > 10:
                        color = (0, intensity, intensity)
                        pygame.draw.rect(self.screen, color, (x * self.block_w, y * self.block_h + 300, self.block_w - 1, self.block_h - 1))

            pygame.display.flip()
            pygame.time.Clock().tick(30)
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=int, default=0)
    args = parser.parse_args()
    bridge = PygameBridge(monitor_id=args.monitor)
    bridge.run()
