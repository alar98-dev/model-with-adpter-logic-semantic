"""
SWA LLM PyTest — Interpretador de testes com LLM quantizado.

Uso:
    from swa.llm_pytest import QuantizedTestInterpreter, PyTestVisualRunner

    # Interpretar falha
    interpreter = QuantizedTestInterpreter()
    interpreter.load()
    explanation = interpreter.explain_failure("test_foo", "AssertionError")

    # Rodar testes visualmente
    runner = PyTestVisualRunner()
    suite = runner.run_tests("tests/cycle_domain/")
"""
from .quantized_interpreter import QuantizedTestInterpreter
from .visual_runner import PyTestVisualRunner, TestResult, TestSuite

__all__ = [
    "QuantizedTestInterpreter",
    "PyTestVisualRunner",
    "TestResult",
    "TestSuite",
]
