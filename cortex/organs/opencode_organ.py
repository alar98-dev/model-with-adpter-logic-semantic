import os
import time
import json
import sqlite3
import threading
import logging
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [OpenCode Organ] %(message)s')

class OpenCodeOrgan:
    def __init__(self, db_path='/home/alifer/.local/share/opencode/opencode.db', gateway_url='http://127.0.0.1:8050'):
        self.node_id = 'swa-opencode-organ'
        self.db_path = db_path
        self.gateway_url = gateway_url.rstrip('/')
        self.last_time_updated = 0
        self.running = False

    def register_node(self):
        logging.info('OpenCode Organ usando contrato canônico HTTP do gateway AEON em %s.', self.gateway_url)

    def emit_event(self, event_type, payload_data):
        # OpenCode -> AEON passa somente pelo endpoint público do gateway.
        # O organ não toca em SHM, Redis, PostgreSQL ou sockets internos.
        try:
            body = dict(payload_data)
            body['source'] = 'opencode'
            body['event_type'] = event_type
            body['node_id'] = self.node_id
            request = urllib.request.Request(
                self.gateway_url + '/api/v1/integrations/opencode/context',
                data=json.dumps(body).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                result = json.loads(response.read().decode('utf-8'))
            logging.info('Contexto OpenCode aceito pelo AEON: %s', result.get('payload_blake3', 'sem-hash'))
        except Exception as e:
            logging.warning('Gateway AEON indisponível para contexto OpenCode: %s', e)

    def get_latest_context(self):
        try:
            conn = sqlite3.connect('file:%s?mode=ro' % self.db_path, uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, title, directory, time_updated FROM session ORDER BY time_updated DESC LIMIT 1;')
            session = cursor.fetchone()
            
            if not session:
                conn.close()
                return None
                
            session_dict = dict(session)
            
            if session_dict['time_updated'] <= self.last_time_updated:
                conn.close()
                return None
                
            self.last_time_updated = session_dict['time_updated']
            
            cursor.execute('SELECT id, data, time_created FROM message WHERE session_id = ? ORDER BY time_created DESC LIMIT 5;', (session['id'],))
            messages = [dict(m) for m in cursor.fetchall()]
            
            conn.close()
            return {
                'session': session_dict,
                'recent_messages': messages
            }
        except Exception as e:
            logging.error('Erro de I/O no SQLite do OpenCode: %s' % e)
            return None

    def observation_loop(self):
        logging.info('Iniciando loop de observação e alinhamento bidirecional do OpenCode...')
        self.register_node()

        self.running = True
        while self.running:
            context = self.get_latest_context()
            if context:
                logging.info('Nova atividade detectada no OpenCode (Sessão %s)' % context['session']['id'])
                self.emit_event('opencode_context_update', {
                    'session_id': context['session']['id'],
                    'directory': context['session']['directory'],
                    'title': context['session']['title'],
                    'messages': context['recent_messages']
                })
            time.sleep(3)

    def start(self):
        t = threading.Thread(target=self.observation_loop)
        t.daemon = True
        t.start()
        return t

if __name__ == '__main__':
    organ = OpenCodeOrgan()
    organ.observation_loop()
