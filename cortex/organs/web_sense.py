"""
Web Sense Organ — O Sentido de Navegacao Classica do Aeon (Acoplado ao OODA Loop).
"""

import re
import json
import logging
import urllib.request
import urllib.parse
import threading
from typing import Optional

from swa.synaptic_address import SynapticRegion, sensory_addr
from swa.synaptic_router import SynapticBusRouter
from swa.context_bus import ContextBus, Role

logger = logging.getLogger("WebSense")

class WebSenseOrgan:
    def __init__(self, router: Optional[SynapticBusRouter] = None):
        self.router = router or SynapticBusRouter()
        self.name = "web_sense"
        self.context_bus = ContextBus()
        self.context_bus.register_agent(self.name)  # <-- Registro obrigatorio no mmap
        # Assina a regiao do Cortex na malha sinaptica
        self.router.subscribe(SynapticRegion.CORTEX, self._on_cortex_thought)
        logger.info(f"[{self.name}] Sentido de Navegacao ativado (OODA Loop Ready).")

    def _on_cortex_thought(self, vector_path: str, payload: dict):
        """Intercepta pensamentos recem-gerados e atua."""
        texto = payload.get("pensamento", "") or payload.get("token", "")
        
        # Padrao 1: [NAVEGAR: url]
        nav_match = re.search(r'\[NAVEGAR:\s*(https?://[^\s\]]+)\]', texto, re.IGNORECASE)
        if nav_match:
            self._fetch_and_pulse(nav_match.group(1))

        # Padrao 2: [PESQUISAR: termo]
        search_match = re.search(r'\[PESQUISAR:\s*([^\]]+)\]', texto, re.IGNORECASE)
        if search_match:
            self._search_and_pulse(search_match.group(1))

    def _clean_html(self, raw_html: str) -> str:
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', raw_html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _inject_into_ooda(self, text: str):
        """Escreve no ContextBus nativo para o motor Gamma ingerir no proximo tick."""
        try:
            self.context_bus.write_token(text, role=Role.SYSTEM)
            logger.info(f"[{self.name}] Feedback sensorial injetado no ContextBus!")
        except Exception as e:
            logger.error(f"[{self.name}] Erro ao injetar no OODA: {e}")

    def _fetch_and_pulse(self, url: str):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Aeon Cognitive Sensor v6.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                texto_limpo = self._clean_html(response.read().decode('utf-8', errors='ignore'))[:2000]
                
                # Injeta na via aferente (memoria RAM)
                self._inject_into_ooda(f"[SENSOR WEB: Li a pagina {url}]\nConteudo: {texto_limpo}")
                
                # E emite o pulso na malha vetorial distribuida
                addr = sensory_addr("web_browser", "page_content")
                self.router.pulse(addr, {"source": url, "content": texto_limpo})
        except Exception as e:
            self._inject_into_ooda(f"[SENSOR WEB: Falha ao ler {url}: {e}]")

    def _search_and_pulse(self, query: str):
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Aeon Cognitive Sensor v6.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')
                snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE)[:3]
                
                res_texto = "\n".join(f"- {self._clean_html(s)}" for s in snippets)
                if not res_texto.strip():
                    res_texto = "[Alerta] A extração via HTML falhou ou foi bloqueada pelo anti-bot do provedor."
                
                # OODA Inject
                self._inject_into_ooda(f"[SENSOR WEB: Resultados para '{query}']\n{res_texto}")
                
                # Redis Pulse
                addr = sensory_addr("web_search", "query_results")
                self.router.pulse(addr, {"query": query, "content": res_texto})
        except Exception as e:
            self._inject_into_ooda(f"[SENSOR WEB: Falha na pesquisa: {e}]")


# --- INTEGRACAO COM O RUNTIME ---
def init_organ():
    """Carregado automaticamente pelo cortex/run.py via hot-reload"""
    organ = WebSenseOrgan()
    
    # Monkey Patch no Runtime para escutar a geracao (Decide/Act do OODA)
    try:
        import swa.broadcast_runtime
        original_broadcast = swa.broadcast_runtime.BroadcastRuntime.inject_user_command
        
        def wrapped_broadcast(self, *args, **kwargs):
            # Sempre que o Aeon fizer broadcast, o orgao checa o ultimo pensamento
            if hasattr(self, 'last_thought') and self.last_thought:
                # Thread separada para nao travar a latencia O(1) do modelo
                threading.Thread(
                    target=organ._on_cortex_thought, 
                    args=(SynapticRegion.CORTEX.value, {"pensamento": self.last_thought}),
                    daemon=True
                ).start()
                
            return original_broadcast(self, *args, **kwargs)

        # Injeta o gancho!
        swa.broadcast_runtime.BroadcastRuntime.inject_user_command = wrapped_broadcast
    except ImportError:
        pass
        
    return organ
