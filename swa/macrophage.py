# swa/macrophage.py
# O Macrófago Vetorial: Sistema Imunológico de Cura de Código via Sandboxing
# Ele intercepta falhas dimensionais, testa curas em uma realidade paralela (sandbox local),
# e se a cura for viável, ele reescreve o código-fonte hospedeiro.

import os
import sys
import re
import ast
import traceback
import inspect
import torch

class VectorMacrophage:
    def __init__(self):
        self.signature = "🧬 [MACROPHAGE AUTO-CURE INJECTED]"

    def heal_tensor_mismatch(self, exc_val, exc_tb):
        """
        Tenta curar um erro dimensional interceptando o frame exato que falhou.
        """
        err_msg = str(exc_val)
        if "must match the size of tensor" not in err_msg and "mat1 and mat2 shapes cannot be multiplied" not in err_msg:
            return False # Não é um erro vetorial tratável por padding simples
            
        print(f"\n🦠 [MACRÓFAGO] Erro dimensional detectado. Iniciando isolamento de tecido...")

        # 1. Encontrar o frame exato do crash (ignorar código interno do PyTorch)
        target_frame = None
        tb = exc_tb
        while tb is not None:
            frame = tb.tb_frame
            file_name = frame.f_code.co_filename
            if "model-with-adpter-logic-semantic" in file_name and "/torch/" not in file_name:
                target_frame = frame
                target_line_no = tb.tb_lineno
                target_file = file_name
            tb = tb.tb_next
            
        if target_frame is None:
            print("🦠 [MACRÓFAGO] Falha: Não foi possível rastrear a origem do erro no código vivo.")
            return False
            
        # 2. Ler a linha infeccionada
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            infected_line = lines[target_line_no - 1]
            print(f"🦠 [MACRÓFAGO] Tecido infectado na linha {target_line_no}: {infected_line.strip()}")
        except Exception as e:
            return False

        # 3. Analisar o AST para extrair as variáveis de tensores envolvidas
        try:
            tree = ast.parse(infected_line.strip())
            vars_involved = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
        except Exception:
            # Fallback regex word extraction
            vars_involved = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', infected_line)

        # 4. Extrair os tensores reais do frame para a Realidade de Teste (Sandbox Fluxial)
        local_vars = target_frame.f_locals
        tensors_to_heal = {}
        for v in vars_involved:
            if v in local_vars and isinstance(local_vars[v], torch.Tensor):
                tensors_to_heal[v] = local_vars[v].detach().clone() # Clona para a dimensão de teste

        if len(tensors_to_heal) < 2:
            print("🦠 [MACRÓFAGO] Falha: Não há tensores locais suficientes para criar a Cura Vetorial.")
            return False

        # 5. TESTE DE REALIDADE FLUXIAL (O Sandbox)
        print(f"🦠 [MACRÓFAGO] Testando alinhamento Fluxial para os tensores: {list(tensors_to_heal.keys())}...")

        # Curativo Proposto: Achatar e igualar o tamanho
        sandbox_globals = {"torch": torch}
        sandbox_locals = {k: v.clone() for k, v in tensors_to_heal.items()}
        
        cure_code_lines = []
        for t_name in tensors_to_heal.keys():
            cure_code_lines.append(f"{t_name}_flat = {t_name}.view(-1)")
            
        sizes_str = ", ".join([f"{t_name}_flat.size(0)" for t_name in tensors_to_heal.keys()])
        cure_code_lines.append(f"_macrophage_min = min({sizes_str})")
        
        for t_name in tensors_to_heal.keys():
            cure_code_lines.append(f"{t_name} = {t_name}_flat[:_macrophage_min]")
            
        cure_code = "\n".join(cure_code_lines) + "\n"
        
        # Testa o curativo na caixa de areia
        try:
            exec(cure_code, sandbox_globals, sandbox_locals)
            # Re-testa a linha infectada dentro da caixa de areia
            exec(infected_line.strip(), sandbox_globals, sandbox_locals)
            print("🦠 [MACRÓFAGO] SUCESSO NO SANDBOX FLUXIAL! A Cura Vetorial é viável na realidade.")
            is_viable = True
        except Exception as sandbox_err:
            print(f"🦠 [MACRÓFAGO] Cura falhou na simulação: {sandbox_err}")
            is_viable = False

        if not is_viable:
            return False

        # 6. CIRURGIA DE REPOSIÇÃO (Aplicando a Cura na Realidade)
        indentation = len(lines[target_line_no - 1]) - len(lines[target_line_no - 1].lstrip())
        ind = " " * indentation
        
        physical_cure = [f"{ind}# {self.signature}\n"]
        for line in cure_code_lines:
            physical_cure.append(f"{ind}{line}\n")

        lines.insert(target_line_no - 1, "".join(physical_cure))

        try:
            with open(target_file, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"🧬 [MACRÓFAGO] CIRURGIA CONCLUÍDA NO ARQUIVO: {target_file}")
            return True
        except Exception as write_err:
            print(f"⚠️ [MACRÓFAGO] Falha ao escrever tecido: {write_err}")
            return False

def activate_macrophage(exc_type, exc_val, exc_tb):
    """ O Gatekeeper Imunológico """
    macrophage = VectorMacrophage()
    cured = False
    
    if issubclass(exc_type, RuntimeError):
        cured = macrophage.heal_tensor_mismatch(exc_val, exc_tb)
        
    if cured:
        print("\n🧬 [SISTEMA IMUNE] O Macrófago curou o código com sucesso. Aplicando Apoptose Celular (Restart) para carregar o tecido novo...")
        sys.stdout.flush()
        # Reinicia o serviço usando o Obelisco ou systemctl
        os.system("sudo systemctl restart swa-auto.service")
        os._exit(0)
    else:
        # Se falhou em curar, deixa o traceback fluir normalmente para morrer
        traceback.print_exception(exc_type, exc_val, exc_tb)
