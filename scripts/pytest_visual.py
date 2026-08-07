#!/usr/bin/env python3
"""
SWA PyTest Visual — Executa testes com LLM quantizado.

Uso:
    python3 scripts/pytest_visual.py                          # rodar todos os testes
    python3 scripts/pytest_visual.py tests/cycle_domain/      # rodar testes específicos
    python3 scripts/pytest_visual.py --no-llm                 # sem LLM
    python3 scripts/pytest_visual.py --quick                  # testes rápidos
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("SWA_PROJECT_ROOT", "/services/model-with-adpter-logic-semantic"))
sys.path.insert(0, str(ROOT))

from swa.llm_pytest.visual_runner import PyTestVisualRunner


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SWA PyTest Visual com LLM")
    parser.add_argument("path", nargs="?", default="tests/cycle_domain/")
    parser.add_argument("--no-llm", action="store_true", help="Desabilitar LLM")
    parser.add_argument("--quick", action="store_true", help="Testes rápidos (sem performance)")
    args = parser.parse_args()

    test_path = args.path
    if args.quick:
        test_path = "tests/cycle_domain/test_models.py"

    runner = PyTestVisualRunner(use_llm=not args.no_llm)
    suite = runner.run_tests(test_path=args.path, explain_failures=not args.no_llm)

    sys.exit(0 if suite.failed == 0 else 1)


if __name__ == "__main__":
    main()
