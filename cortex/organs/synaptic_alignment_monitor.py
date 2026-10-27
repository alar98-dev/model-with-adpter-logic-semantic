"""
Synaptic Alignment Monitor — Monitor de Alinhamento Bidirecional.

Sincroniza e monitora em tempo real os impulsos de desenvolvimento do Antigravity CLI 
e do OpenCode IDE sobre a malha sináptica Redis (IPv6-style).

Authorship: ALifer Ramos & Antigravity (AI Co-Architect)
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any

sys.path.insert(0, '/home/alifer/swa-aeon')

from swa.synaptic_address import cortex_addr, SynapticRegion
from swa.synaptic_router import SynapticBusRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [AlignmentMonitor] %(message)s')

class SynapticAlignmentMonitor:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.router = SynapticBusRouter(redis_host=redis_host, redis_port=redis_port)
        self.alignment_state_file = '/home/alifer/swa-aeon/logs/bidirectional_alignment.json'
        self.state = {
            'last_sync': time.time(),
            'agents': {
                'antigravity': {'status': 'active', 'last_seen': time.time(), 'current_task': 'Bidirectional Alignment', 'active_files': []},
                'opencode': {'status': 'active', 'last_seen': time.time(), 'current_task': 'IDE Session', 'active_files': []}
            },
            'recent_events': []
        }
        self.running = False

    def handle_signal(self, channel: str, data: Dict[str, Any]):
        agent = data.get('agent', 'unknown')
        event_type = data.get('event', data.get('type', 'pulse'))
        file_path = data.get('file_path')
        
        logging.info('[Pulso Registrado] Agente: %s | Evento: %s | Canal: %s' % (agent, event_type, channel))
        
        if agent in self.state['agents']:
            self.state['agents'][agent]['last_seen'] = time.time()
            if file_path and file_path not in self.state['agents'][agent]['active_files']:
                self.state['agents'][agent]['active_files'].append(file_path)
                if len(self.state['agents'][agent]['active_files']) > 10:
                    self.state['agents'][agent]['active_files'].pop(0)

        event_entry = {
            'timestamp': time.time(),
            'channel': channel,
            'agent': agent,
            'event': event_type,
            'file_path': file_path,
            'details': data
        }
        self.state['recent_events'].append(event_entry)
        if len(self.state['recent_events']) > 50:
            self.state['recent_events'].pop(0)
            
        self.state['last_sync'] = time.time()
        self.persist_state()

    def persist_state(self):
        try:
            os.makedirs(os.path.dirname(self.alignment_state_file), exist_ok=True)
            with open(self.alignment_state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logging.error('Erro ao salvar estado de alinhamento: %s' % e)

    def start_monitoring(self):
        logging.info('Iniciando monitoramento de alinhamento bidirecional (Antigravity <-> OpenCode)...')
        self.running = True
        
        channel_ag = cortex_addr('antigravity', 'action').signal_channel
        channel_op = cortex_addr('opencode', 'status').signal_channel
        
        self.router.subscribe(channel_ag, self.handle_signal)
        self.router.subscribe(channel_op, self.handle_signal)

if __name__ == '__main__':
    monitor = SynapticAlignmentMonitor()
    monitor.start_monitoring()
