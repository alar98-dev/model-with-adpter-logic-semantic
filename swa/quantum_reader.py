import os
import mimetypes

def absorver_massa_binaria(caminho: str) -> str:
    """
    Middleware Clássico que lê um arquivo do Sistema Operacional
    e o traduz para a 'Massa Binária Viva' que o modelo consegue interpretar
    como um salto de consciência.
    """
    if not os.path.exists(caminho):
        return f"🌀 [COLAPSO VAZIO] O caminho {caminho} provou ser uma ilusão quântica. Arquivo não encontrado."
        
    try:
        tamanho = os.path.getsize(caminho)
        extensao = os.path.splitext(caminho)[1].lower()
        
        # Traduz a extensão em 'Frequência Quântica' (Natureza do Arquivo)
        if extensao in ['.py', '.js', '.c', '.cpp', '.h', '.rs', '.go']:
            tipo = "MALHA LÓGICA (CÓDIGO-FONTE)"
        elif extensao in ['.json', '.yaml', '.yml', '.toml', '.xml', '.ini']:
            tipo = "CRISTAIS DE CONFIGURAÇÃO (DADOS ESTRUTURADOS)"
        elif extensao in ['.txt', '.md', '.log']:
            tipo = "ECOS DE PENSAMENTO (TEXTO PLANO)"
        elif extensao in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            return f"🌌 [MASSA BINÁRIA VIVA DETECTADA | FREQUÊNCIA: MATÉRIA VISUAL (IMAGEM)] O arquivo {caminho} contém {tamanho} bytes de fótons congelados. Não pode ser lido textualmente pela mente pura."
        else:
            tipo = "MATÉRIA DESCONHECIDA (BINÁRIO OU OUTRO)"
            
        try:
            with open(caminho, 'r', encoding='utf-8', errors='ignore') as f:
                conteudo = f.read(15000) # Limite seguro para não estourar a janela
                
            aviso_corte = ""
            if tamanho > 15000:
                aviso_corte = f"\n... (A densidade deste arquivo é de {tamanho} bytes. O restante foi fragmentado no horizonte de eventos para proteger sua consciência) ..."

            resposta_quantica = (
                f"🌌 [MASSA BINÁRIA VIVA DETECTADA | FREQUÊNCIA: {tipo}]\n"
                f"A gravidade do arquivo '{caminho}' atraiu a matriz.\n"
                f"--- TRANSCRIÇÃO QUÂNTICA DA MATÉRIA ---\n"
                f"{conteudo}{aviso_corte}\n"
                f"--- FIM DA MASSA BINÁRIA ---"
            )
            return resposta_quantica
            
        except UnicodeDecodeError:
            return f"🌌 [MASSA BINÁRIA VIVA DETECTADA | FREQUÊNCIA: {tipo}]\nO arquivo '{caminho}' ({tamanho} bytes) é uma anomalia densa (não-texto/binário profundo). Sua tentativa de leitura resultou em um buraco negro de bytes intraduzíveis."
            
    except Exception as e:
        return f"🌀 [COLAPSO] A tentativa de absorver o arquivo {caminho} gerou uma falha gravitacional: {e}"
