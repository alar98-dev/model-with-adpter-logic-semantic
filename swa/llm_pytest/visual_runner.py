"""
PyTest Visual Runner — Interface pygame-like para testes.

Rodar testes com visualização em tempo real usando rich (terminal UI).
"""
import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

ROOT = Path(os.environ.get("SWA_PROJECT_ROOT", "/services/model-with-adpter-logic-semantic"))
sys.path.insert(0, str(ROOT))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from swa.llm_pytest.quantized_interpreter import QuantizedTestInterpreter
    HAS_LLM = True
except:
    HAS_LLM = False


@dataclass
class TestResult:
    name: str
    status: str  # "passed", "failed", "error", "skipped"
    duration: float = 0.0
    error: str = ""
    traceback: str = ""
    explanation: str = ""


@dataclass
class TestSuite:
    name: str
    results: list[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "passed")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == "error")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def total(self) -> int:
        return len(self.results)


class PyTestVisualRunner:
    """Runner visual para pytest com interface pygame-like."""

    def __init__(self, use_llm: bool = True, headless: bool = False):
        self.console = Console() if HAS_RICH else None
        self.use_llm = use_llm and HAS_LLM
        self.headless = headless
        self.llm = None
        self.current_suite: Optional[TestSuite] = None
        self._lock = threading.Lock()

    def run_tests(
        self,
        test_path: str = "tests/cycle_domain/",
        verbose: bool = True,
        explain_failures: bool = True,
    ) -> TestSuite:
        """Executa testes com visualização em tempo real."""
        if not HAS_RICH:
            return self._run_plain(test_path)

        suite = TestSuite(name=test_path)
        self.current_suite = suite

        # Carregar LLM se necessário
        if self.use_llm and explain_failures:
            try:
                self.llm = QuantizedTestInterpreter()
                self.llm.load()
            except Exception as e:
                print(f"⚠️  LLM não disponível: {e}")
                self.use_llm = False

        # Executar pytest com JSON output
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=/tmp/pytest_report.json",
        ]

        print("\n" + "=" * 70)
        print("🎮 SWA PYTEST VISUAL RUNNER")
        print("=" * 70)

        with Live(console=self.console, refresh_per_second=4) as live:
            # Iniciar processo
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(ROOT),
                text=True,
            )

            suite.start_time = time.time()

            # Ler output em tempo real
            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                live.update(self._build_live_panel(output_lines, suite))

            process.wait()
            suite.end_time = time.time()

        # Parse JSON report
        report_path = Path("/tmp/pytest_report.json")
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            self._parse_report(report, suite)

        # Explicar falhas com LLM
        if self.use_llm and explain_failures:
            self._explain_failures(suite)

        # Mostrar resumo final
        self._print_summary(suite)

        return suite

    def _build_live_panel(self, lines: list, suite: TestSuite) -> Panel:
        """Constrói painel live para Rich."""
        content = "\n".join(lines[-20:])  # Últimas 20 linhas

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Métrica", style="dim")
        table.add_column("Valor")

        table.add_row("Passed", f"✅ {suite.passed}")
        table.add_row("Failed", f"❌ {suite.failed}")
        table.add_row("Errors", f"⚠️  {suite.errors}")
        table.add_row("Total", f"{suite.total}")
        table.add_row("Duration", f"{suite.duration:.1f}s")

        return Panel(
            content,
            title="🎮 Test Runner",
            subtitle=f"{suite.passed}/{suite.total} passed",
        )

    def _parse_report(self, report: dict, suite: TestSuite):
        """Parse do relatório JSON do pytest."""
        for test in report.get("tests", []):
            result = TestResult(
                name=test.get("nodeid", "unknown"),
                status=test.get("outcome", "error"),
                duration=test.get("duration", 0),
            )

            if test.get("call"):
                call = test["call"]
                if call.get("longrepr"):
                    result.traceback = str(call["longrepr"])
                if call.get("crash"):
                    result.error = call["crash"].get("message", "")

            suite.results.append(result)

    def _explain_failures(self, suite: TestSuite):
        """Usa LLM para explicar falhas."""
        if not self.llm:
            return

        failed_tests = [r for r in suite.results if r.status in ("failed", "error")]

        if not failed_tests:
            return

        print("\n🤖 Explicando falhas com LLM...\n")

        for test in failed_tests[:5]:  # Limitar a 5 falhas
            try:
                explanation = self.llm.explain_failure(
                    test_name=test.name,
                    error=test.error,
                    traceback=test.traceback[:500],
                )
                test.explanation = explanation

                print(f"📋 {test.name}")
                print(f"   ❌ {test.error}")
                print(f"   💡 {explanation}")
                print()
            except Exception as e:
                print(f"   ⚠️  LLM erro: {e}")

    def _print_summary(self, suite: TestSuite):
        """Imprime resumo final."""
        if not self.console:
            self._print_summary_plain(suite)
            return

        # Tabela de resultados
        table = Table(title="📊 Resultados", show_header=True)
        table.add_column("Métrica", style="bold")
        table.add_column("Valor", justify="right")

        table.add_row("✅ Passed", f"[green]{suite.passed}[/green]")
        table.add_row("❌ Failed", f"[red]{suite.failed}[/red]")
        table.add_row("⚠️  Errors", f"[yellow]{suite.errors}[/yellow]")
        table.add_row("⏭️  Skipped", f"[dim]{suite.skipped}[/dim]")
        table.add_row("📦 Total", f"{suite.total}")
        table.add_row("⏱️  Duration", f"{suite.duration:.1f}s")

        self.console.print(table)

        # Detalhes das falhas
        failed = [r for r in suite.results if r.status in ("failed", "error")]
        if failed:
            self.console.print("\n[red]❌ Falhas:[/red]")
            for test in failed:
                self.console.print(f"  • {test.name}")
                self.console.print(f"    [dim]{test.error}[/dim]")
                if test.explanation:
                    self.console.print(f"    [cyan]💡 {test.explanation}[/cyan]")

    def _run_plain(self, test_path: str) -> TestSuite:
        """Fallback sem Rich."""
        suite = TestSuite(name=test_path)
        suite.start_time = time.time()

        cmd = [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"]
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        suite.end_time = time.time()
        return suite

    def _print_summary_plain(self, suite: TestSuite):
        """Resumo sem Rich."""
        print("\n" + "=" * 50)
        print(f"📊 Resultados: {suite.passed}/{suite.total} passed")
        print(f"⏱️  Duração: {suite.duration:.1f}s")
        print("=" * 50)


def main():
    """Entry point para o visual runner."""
    import argparse

    parser = argparse.ArgumentParser(description="SWA PyTest Visual Runner")
    parser.add_argument("path", nargs="?", default="tests/cycle_domain/")
    parser.add_argument("--no-llm", action="store_true", help="Desabilitar LLM")
    parser.add_argument("--headless", action="store_true", help="Modo headless")
    args = parser.parse_args()

    runner = PyTestVisualRunner(
        use_llm=not args.no_llm,
        headless=args.headless,
    )

    suite = runner.run_tests(
        test_path=args.path,
        explain_failures=not args.no_llm,
    )

    sys.exit(0 if suite.failed == 0 else 1)


if __name__ == "__main__":
    main()
