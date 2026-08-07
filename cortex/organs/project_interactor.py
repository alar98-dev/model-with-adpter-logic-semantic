import json
from swa.integrity_shield import verify_integrity
from swa.semantic_protocol import SWAEvent
from swa.kernel_voice import KernelVoice

class ProjectInteractor:
    def __init__(self, engine, knowledge_path):
        self.nome = "ProjectInteractor"
        self.voice = KernelVoice(engine)
        self.knowledge_path = knowledge_path
        self._load_knowledge()

    def _load_knowledge(self):
        try:
            with open(self.knowledge_path, "r") as f:
                self.knowledge = json.load(f)
        except:
            self.knowledge = {"project_summary": "Base de conhecimento não carregada."}

    @verify_integrity
    async def interact(self, prompt: str) -> SWAEvent:
        # Síntese semântica focada no conhecimento do projeto
        context = self.knowledge.get("project_summary", "Sem dados.")
        
        # Estrutura a intenção como um evento do SWA Protocol
        event = SWAEvent(
            sender=self.nome,
            intent="PROJECT_INTERACTION",
            data={"prompt": prompt, "context": context}
        )
        
        # O KernelVoice traduz e faz o broadcast automaticamente
        await self.voice.speak(event)
        
        # Agora o órgão retorna o evento formalizado
        return event
