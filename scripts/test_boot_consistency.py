#!/usr/bin/env python3
"""Smoke test de consistencia do boot e dos services do SWA-Aeon.

Objetivos:
- garantir que os units atuais apontam para /services/model-with-adpter-logic-semantic
- verificar a separacao de responsabilidades entre auto, subcortex e manifold
- falhar cedo se aparecer path legado /mnt/PROJECTS ou fallback 4B escondido
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path('/services/model-with-adpter-logic-semantic')
CHECKS = {
    'swa-auto.service': [
        'ExecStart=/services/model-with-adpter-logic-semantic/venv/bin/python cortex/run.py --interactive --broadcast',
        'Environment=MODEL_DIR=models/gemma-3-1b-it',
        'Environment=VECTORIZER_MODEL_DIR=models/gemma-3-270m',
    ],
    'swa-subcortex.service': [
        'ExecStart=/services/model-with-adpter-logic-semantic/venv/bin/python sensors/subcortex.py',
        'Environment=MODEL_DIR=models/gemma-3-270m',
    ],
    'swa-manifold.service': [
        'ExecStart=/services/model-with-adpter-logic-semantic/venv/bin/python3 /services/model-with-adpter-logic-semantic/scripts/rebuild_vector_reality.py --skip-if-real',
        'Environment=VECTORIZER_MODEL_DIR=models/gemma-3-270m',
    ],
    'swa-rest.service': [
        'WorkingDirectory=/services/model-with-adpter-logic-semantic',
        'ExecStart=/services/model-with-adpter-logic-semantic/venv/bin/python swa/rest_server.py',
    ],
}
FORBIDDEN = [
    '/mnt/PROJECTS/',
    'models/gemma-3-4-b',
    'User=alifer.ramos',
]


def main() -> int:
    failures = []
    for name, expected_snippets in CHECKS.items():
        path = ROOT / 'deployments' / 'systemd' / name
        if not path.exists():
            failures.append(f'missing file: {path}')
            continue
        text = path.read_text(encoding='utf-8')
        for snippet in expected_snippets:
            if snippet not in text:
                failures.append(f'{name}: missing -> {snippet}')
        for forbidden in FORBIDDEN:
            if forbidden in text:
                failures.append(f'{name}: forbidden pattern -> {forbidden}')

    # Checks on runtime source
    run_py = ROOT / 'cortex' / 'run.py'
    flux_py = ROOT / 'swa' / 'flux_coupling.py'
    if 'DEFAULT_MODEL_DIR' not in run_py.read_text(encoding='utf-8'):
        failures.append('cortex/run.py: missing DEFAULT_MODEL_DIR')
    if 'DEFAULT_VECTORIZER_MODEL_DIR' not in run_py.read_text(encoding='utf-8'):
        failures.append('cortex/run.py: missing DEFAULT_VECTORIZER_MODEL_DIR')
    if 'VOICE_MODEL_DIR' not in run_py.read_text(encoding='utf-8'):
        failures.append('cortex/run.py: missing VOICE_MODEL_DIR')
    if 'max_new_tokens=_sentiment_token_budget' not in run_py.read_text(encoding='utf-8'):
        failures.append('cortex/run.py: missing dynamic sentiment budget in generation paths')
    if 'max_new_tokens=main_max_new_tokens' not in flux_py.read_text(encoding='utf-8'):
        failures.append('swa/flux_coupling.py: missing dynamic main_max_new_tokens')
    if 'max_new_tokens=voice_max_new_tokens' not in flux_py.read_text(encoding='utf-8'):
        failures.append('swa/flux_coupling.py: missing dynamic voice_max_new_tokens')

    if failures:
        print('BOOT CONSISTENCY: FAIL')
        for item in failures:
            print('-', item)
        return 1

    print('BOOT CONSISTENCY: OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
