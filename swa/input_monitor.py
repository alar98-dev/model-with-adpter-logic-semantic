from pynput import keyboard, mouse
import redis
import json

class InputMonitor:
    def __init__(self, host="localhost", port=6379):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        print("InputMonitor: Sincronizado com Redis Synapse.")

    def send_event(self, event_type, details):
        data = {"event_type": event_type, "payload": details}
        self.r.publish("grid_events", json.dumps(data))

    def on_press(self, key):
        # Sentido de Comunicação (Teclado)
        self.send_event("input_io", {"type": "keyboard", "key": str(key), "intensity": 0.8})

    def on_move(self, x, y):
        # Sentido de Propriocepção/Espacial (Mouse)
        self.send_event("input_io", {"type": "mouse_move", "coords": [x, y], "intensity": 0.3})

    def start(self):
        with keyboard.Listener(on_press=self.on_press) as k_list, \
             mouse.Listener(on_move=self.on_move) as m_list:
            k_list.join()
            m_list.join()

if __name__ == "__main__":
    monitor = InputMonitor()
    monitor.start()
