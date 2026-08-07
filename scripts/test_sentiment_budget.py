#!/usr/bin/env python3
"""Smoke test do budget dinamico de geracao baseado em sentimento/entropia.

Nao carrega modelos. Apenas valida a logica estrutural que o Cortex passa a obedecer.
"""
from __future__ import annotations

from dataclasses import dataclass
import ast
from pathlib import Path
import sys

RUN_PY = Path('/services/model-with-adpter-logic-semantic/cortex/run.py')
SOURCE = RUN_PY.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE, filename=str(RUN_PY))


@dataclass
class RuntimeStub:
    current_entropy: float = 0.4
    ram_pct: float = 40.0
    cpu_temp_c: float = 55.0
    sentiment_vector: tuple[float, ...] = (0.7, 0.8, 0.6, 0.7)


def main() -> int:
    names = {node.name for node in TREE.body if isinstance(node, ast.FunctionDef)}
    required = {'_sentiment_scalar', '_sentiment_token_budget'}
    missing = required - names
    if missing:
        print('SENTIMENT BUDGET: FAIL')
        for item in sorted(missing):
            print('-', item)
        return 1

    namespace = {'__file__': str(RUN_PY), '__name__': '__test__'}
    exec(compile(TREE, str(RUN_PY), 'exec'), namespace, namespace)
    budget_fn = namespace['_sentiment_token_budget']
    scalar_fn = namespace['_sentiment_scalar']

    runtime_good = RuntimeStub(current_entropy=0.15, ram_pct=35.0, cpu_temp_c=52.0, sentiment_vector=(0.8, 0.9, 0.85))
    runtime_stressed = RuntimeStub(current_entropy=0.9, ram_pct=88.0, cpu_temp_c=82.0, sentiment_vector=(0.1, 0.2, 0.15))

    good_scalar = scalar_fn(runtime_good)
    bad_scalar = scalar_fn(runtime_stressed)
    good_budget = budget_fn(runtime_good, 150, 'hello world', {'cpu_temp_c': 52.0, 'ram_pct': 35.0, 'io_stress': 0.0}, 16, 240)
    bad_budget = budget_fn(runtime_stressed, 150, 'hello world', {'cpu_temp_c': 82.0, 'ram_pct': 88.0, 'io_stress': 0.0}, 16, 240)

    failures = []
    if not (0.0 <= good_scalar <= 1.0):
        failures.append(f'good scalar out of range: {good_scalar}')
    if not (0.0 <= bad_scalar <= 1.0):
        failures.append(f'bad scalar out of range: {bad_scalar}')
    if not (good_budget > bad_budget):
        failures.append(f'expected good budget > bad budget, got {good_budget} <= {bad_budget}')
    if not (16 <= good_budget <= 240 and 16 <= bad_budget <= 240):
        failures.append(f'budgets out of range: good={good_budget}, bad={bad_budget}')

    if failures:
        print('SENTIMENT BUDGET: FAIL')
        for item in failures:
            print('-', item)
        return 1

    print('SENTIMENT BUDGET: OK')
    print(f'good_scalar={good_scalar:.3f} good_budget={good_budget}')
    print(f'bad_scalar={bad_scalar:.3f} bad_budget={bad_budget}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
