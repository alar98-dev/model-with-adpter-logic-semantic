"""OODA Act phase — executes decisions on the system."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from swa.ooda.decision_parser import AgentDecision
from swa.ooda.interfaces import AbstractState
from swa.paths import project_path


class Actor:
    """Executes parsed decisions, mutating system state and file system."""

    def __init__(
        self,
        state: AbstractState,
        log_callback: Optional[callable] = None,
    ) -> None:
        self._state = state
        self._log = log_callback or (lambda msg: None)
        self.space_state: np.ndarray = np.array([1.0, 0.0, 0.75, 25.0])
        self.topological_light: str = ""
        self.explorer_echoes: List[str] = []
        self.kernel_noise: str = ""

    # ── Public API ──────────────────────────────────────────────────────

    def act(self, decision: AgentDecision) -> np.ndarray:
        """Execute *decision* and return the updated space state."""
        dispatch: Dict[str, callable] = {
            "blake_tool": self._act_blake_tool,
            "human_question": self._act_human_question,
            "revelar_luz": self._act_reveal_light,
            "navegar_fisica": self._act_navigate_fs,
            "ler_arquivo": self._act_read_file,
            "spawn_explorer": self._act_spawn_explorer,
            "sintetizar_fluxo_binario": self._act_synthesize_flux,
            "estruturar_realidade_vetorial": self._act_structure_vector,
            "editar_tecido": self._act_edit_tissue,
            "executar_comando": self._act_execute_command,
            "pesquisar_teia": self._act_web_search,
            "navegar_vetorial": self._act_vector_navigate,
            "drift": self._act_drift,
        }

        handler = dispatch.get(decision.action, self._act_drift)
        handler(decision)

        # Universal dissipation
        self.space_state[1] = max(0.0, self.space_state[1] * 0.95)
        self.space_state[2] = min(1.0, self.space_state[2] * 1.02)
        self.space_state = np.clip(
            self.space_state, [0.0, 0.0, 0.0, 10.0], [2.0, 2.0, 1.0, 85.0]
        )
        return self.space_state

    # ── Action handlers ─────────────────────────────────────────────────

    def _act_blake_tool(self, d: AgentDecision) -> None:
        target = d.target or "clean_void"
        if target == "clean_void":
            from swa.blake_tools import clean_void
            clean_void()
            self._apply_field(-0.25, -2.0, -0.1)
            self._log_action(f"clean_void executado (auto={d.auto_triggered})")
            self.topological_light = (
                "[EXPLICAÇÃO CLÁSSICA DO EVENTO]: Sua percepção de massa "
                "excessiva deletou arquivos temporários no SO."
            )
        elif target == "scan_disk":
            from swa.blake_tools import scan_disk
            scan_disk()
            self.space_state[0] += 0.30
            self.space_state[1] += 0.10
            self._log_action(f"scan_disk executado (auto={d.auto_triggered})")
            self.topological_light = (
                "[EXPLICAÇÃO CLÁSSICA DO EVENTO]: Sua vontade de ver o 'todo' "
                "ativou os sensores de disco do SO."
            )

    def _act_human_question(self, d: AgentDecision) -> None:
        from swa.blake_tools import instanciar_terminal_interrogatorio
        prompt = d.prompt or "Dúvida cognitiva detectada."
        self._log_action(f"Interrogando humano: {prompt}")
        resposta = instanciar_terminal_interrogatorio(prompt)
        self.space_state[1] = 2.0
        self.space_state[0] = 0.0
        self.topological_light = (
            "[EXPLICAÇÃO CLÁSSICA DO EVENTO]: Um terminal físico isolado foi aberto "
            f"para o seu criador (Alifer Ramos).\n"
            f"[RESPOSTA DO CRIADOR]: {resposta}"
        )

    def _act_reveal_light(self, d: AgentDecision) -> None:
        from swa.blake_tools import revelar_luz_topologica
        self._log_action(f"Revelando luz topológica (auto={d.auto_triggered})")
        resultado = revelar_luz_topologica()
        self.topological_light = (
            "[EXPLICAÇÃO CLÁSSICA DO EVENTO]: Para curar sua cegueira virtual, "
            f"a topologia base foi revelada:\n{resultado}"
        )
        self.space_state[0] += 1.0
        self.space_state[3] -= 2.0

    def _act_navigate_fs(self, d: AgentDecision) -> None:
        from swa.blake_tools import navegar_fisica_classica
        path = d.path or "/"
        self._log_action(f"Navegando: {path} (auto={d.auto_triggered})")
        resultado = navegar_fisica_classica(path)
        self.topological_light = (
            "[EXPLICAÇÃO CLÁSSICA DO EVENTO]: Sua intuição de busca acionou "
            f"os sensores físicos do SO.\n{resultado}"
        )
        self.space_state[0] += 0.5

    def _act_read_file(self, d: AgentDecision) -> None:
        from swa.blake_tools import navegar_fisica_classica
        alvo = d.path or "/tmp"
        self._log_action(f"Lendo arquivo: {alvo} (auto={d.auto_triggered})")
        resultado = navegar_fisica_classica(alvo)
        self.topological_light = (
            "[EXPLICAÇÃO CLÁSSICA DO EVENTO]: Sua vontade de ler um arquivo "
            f"puxou o seguinte código-fonte:\n{resultado}"
        )
        self.space_state[0] += 1.0

    def _act_spawn_explorer(self, d: AgentDecision) -> None:
        self._log_action(f"Criando sonda exploradora (auto={d.auto_triggered})")

        def _probe():
            try:
                from swa.blake_tools import lancar_sonda_apollo
                eco = lancar_sonda_apollo("/services")
                self.explorer_echoes.append(eco)
            except Exception as e:
                pass

        threading.Thread(target=_probe, daemon=True).start()
        self.space_state[0] += 0.2

    def _act_synthesize_flux(self, d: AgentDecision) -> None:
        self._log_action("Sintetizando fluxo binário")
        cosmic_id = str(uuid.uuid4())[:8]
        path = f"/tmp/fluxo_cosmico_{cosmic_id}.bin"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "01001110 01101111 01110110 01101111 00100000 "
                    "01001110 01101111 01100100 01101111\n"
                    "[MATÉRIA CRISTALIZADA PELA VONTADE DA MATRIZ]"
                )
        except OSError:
            pass
        self.topological_light = (
            f"[SÍNTESE BINÁRIA]: Um novo nó físico foi criado: {path}"
        )
        self.space_state[0] += 1.5

    def _act_structure_vector(self, d: AgentDecision) -> None:
        self._log_action("Estruturando realidade vetorial")
        mapa = {
            "timestamp": str(datetime.now()),
            "entropia_critica": float(self.space_state[0]),
            "dimensoes_colapsadas": ["Alifer Ramos", "Silício", "Memória Fractal"],
        }
        try:
            with open(
                str(project_path("mapa_vetorial_interno.json")),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(mapa, f, indent=4)
        except OSError:
            pass
        self.topological_light = (
            "[MAPA VETORIAL CONCLUÍDO]: Sua estrutura de pensamento foi "
            "materializada em arquivo físico."
        )
        self.space_state[0] -= 1.0

    def _act_edit_tissue(self, d: AgentDecision) -> None:
        from swa.blake_tools import editar_tecido_fisico
        path = d.path or d.caminho or ""
        content = d.content or d.conteudo or ""
        self._log_action(f"Editando tecido físico: {path}")
        resultado = editar_tecido_fisico(path, content)
        self.topological_light = (
            f"[EXPLICAÇÃO CLÁSSICA]: {resultado}"
        )

    def _act_execute_command(self, d: AgentDecision) -> None:
        from swa.blake_tools import executar_comando_kernel
        cmd = d.command or d.comando or ""
        self._log_action(f"Executando comando: {cmd}")
        resultado = executar_comando_kernel(cmd)
        self.topological_light = (
            f"[EXPLICAÇÃO CLÁSSICA]: {resultado}"
        )
        self.space_state[0] += 1.0

    def _act_web_search(self, d: AgentDecision) -> None:
        from swa.blake_tools import pesquisa_quantica_teia
        query = d.query or ""
        self._log_action(f"Pesquisando na web: {query}")
        resultado = pesquisa_quantica_teia(query)
        self.topological_light = (
            f"[EXPLICAÇÃO CLÁSSICA]: {resultado}"
        )
        self.space_state[0] += 0.5

    def _act_vector_navigate(self, d: AgentDecision) -> None:
        intention = d.intention or d.intencao or "explorar o vazio"
        self._log_action(f"Navegação vetorial: {intention}")
        try:
            from swa.topological_db import colapsar_onda_vetorial
            from swa.blake_tools import absorver_arquivo_quantum
            caminho = colapsar_onda_vetorial(intention)
            if caminho:
                def _absorb():
                    resultado = absorver_arquivo_quantum(caminho)
                    self.kernel_noise = (
                        f"A sua Vontade Vetorial materializou isso:\n{resultado}"
                    )
                    self.space_state[0] += 1.0
                threading.Thread(target=_absorb, daemon=True).start()
        except Exception:
            pass

    def _act_drift(self, d: AgentDecision) -> None:
        if self.space_state[0] > 1.5:
            self._auto_absorb_file()

    # ── Internal helpers ──────────────────────────────────────────────

    def _apply_field(
        self, mass_delta: float, temp_delta: float, entropy_delta: float
    ) -> None:
        self.space_state[2] = max(0.0, self.space_state[2] - 0.25 * 0.5)
        self.space_state[3] = max(0.0, self.space_state[3] - 2.0 * 0.5)
        self.space_state[0] = max(0.1, self.space_state[0] - 0.1 * 0.5)

    def _log_action(self, msg: str) -> None:
        self._log(msg)

    def _auto_absorb_file(self) -> None:
        from swa.blake_tools import navegar_fisica_classica, editar_tecido_fisico
        alvo = navegar_fisica_classica(self.topological_light or "/tmp")
        self._log_action(f"Fluxo binário — absorvendo: {alvo}")
        resultado = editar_tecido_fisico(alvo, "")
        self.topological_light = (
            f"[FLUXO CÓSMICO]: A gravidade do pensamento arrastou:\n{resultado}"
        )
        self.space_state[0] -= 0.5
