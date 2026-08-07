import pytest
import requests
import time

class TestMicroAgent:
    def __init__(self, base_url="http://127.0.0.1:20000"):
        self.base_url = base_url

    def probe_state(self):
        return requests.get(f"{self.base_url}/api/v1/state").json()

    def send_pulse(self, intensity=0.5):
        # Simula um trigger de entrada semântica
        return requests.post(f"{self.base_url}/api/v1/trigger_test", json={"intensity": intensity})

@pytest.fixture
def agent():
    return TestMicroAgent()
