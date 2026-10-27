"""
Antigravity Synaptic Bridge -- Emissor e alinhador de sinais do Antigravity CLI_.
"""

import os
import sys
import time
import json
import logging
from typing import Any, Dict, Optional, Callable

sys.path.insert(0, '/home/alifer/swa-aeon')

from swa.synaptic_address import cortex_addr, SynapticAddress, SynapticRegion
from swa.synaptic_router import SynapticBusRouter

logger = logging.getLogger('AntigravitySynapticBridge')

class AntigravityBridge:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.router = SynapticBusRouter(redis_host=redis_host, redis_port=redis_port)
        self.agent_id = 'antigravity'
        self.addr_action = cortex_addr(self.agent_id, 'action')
        self.addr_status = cortex_addr(self.agent_id, 'status')

    def pulse_edit(self, file_path: str, action_type: str = 'edit', description: str = '', metadata: Optional[Dict[str, Any]] = None) -> bool:
        payload = {
            'agent': self.agent_id,
            'event': 'file_modified',
            'action_type': action_type,
            'file_path': file_path,
            'description': description,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        success = self.router.pulse(self.addr_action, payload)
        if success:
            logger.info("[SynapticBridge] Pulso enviado para %s (%s)" % (file_path, action_type))
        else:
            logger.warning("[SynapticBridge] Falha ao enviar pulso para %s" % file_path)
        return success

    def pulse_task_status(self, task_name: str, status: str, details: str = '') -> bool:
        payload = {
            'agent': self.agent_id,
            'event': 'task_status',
            'task_name': task_name,
            'status': status,
            'details': details,
            'timestamp': time.time()
        }
        return self.router.pulse(self.addr_status, payload)

    def subscribe_to_opencode(self, callback: Callable[[str, Dict[str, Any]], None]):
        opencode_signal_channel = cortex_addr('opencode', 'status').signal_channel
        def wrapper(channel: str, data: Dict[str, Any]):
            if 'opencode' in channel or data.get('agent') == 'opencode':
                callback(channel, data)
        self.router.subscribe(opencode_signal_channel, wrapper)

_bridge = None
def get_bridge():
    global _bridge
    if _bridge is None:
        _bridge = AntigravityBridge()
    return _bridge

def notify_antigravity_edit(file_path: str, action_type: str = 'edit', description: str = ''):
    try:
        get_bridge().pulse_edit(file_path, action_type, description)
    except Exception as e:
        logger.error("Erro ao notificar edicao do Antigravity: %s" % str(e))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fp = sys.argv[1]
        desc = sys.argv[2] if len(sys.argv) > 2 else 'Edicao Antigravity'
        notify_antigravity_edit(fp, 'edit', desc)
        print("Pulso emitido com sucesso para %s" % fp)
    else:
        print("Uso: python3 -m swa.antigravity_bridge <caminho_do_arquivo> [description]")
