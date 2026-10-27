import sys
import os
import anyio
import uvicorn
from contextlib import asynccontextmanager
import starlette.responses
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import Response
from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
import uuid
import builtins
import re
import contextvars

# Contexto global isolado por thread/task para armazenar o IPv6 da requisição
client_ipv6_context = contextvars.ContextVar("client_ipv6_context", default="unknown")

# Garante que o diretório pai (raiz do projeto) esteja no path para importar o SWA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swa.quantum_bridge import QuantumPhaseBridge
from main import EventualState
import torch

mcp = Server("swa-quantum-mcp")
eventual_state = EventualState()
quantum_bridge = None

# Define globally
sse = SseServerTransport("/messages/")

@asynccontextmanager
async def lifespan(app: Starlette):
    global quantum_bridge
    print("🔌 Inicializando MCP SWA Quantum Server...")
    try:
        quantum_bridge = QuantumPhaseBridge()
        quantum_bridge = QuantumPhaseBridge()
        print("✅ QuantumPhaseBridge conectado.")
    except Exception as e:
        print(f"⚠️ Atenção: Falha na inicialização do Quantum Bridge: {e}")
    yield
    print("🛑 Encerrando MCP SWA Quantum Server...")

@mcp.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_project_structure",
            description="Explora a estrutura de diretórios e arquivos do projeto de forma recursiva a partir do root / no Debian.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho absoluto ou relativo para iniciar a exploração (ex: '/', '/services')."},
                    "max_depth": {"type": "integer", "description": "Profundidade máxima da árvore (default: 3)."}
                },
                "required": ["path"]
            }
        ),
        types.Tool(
            name="read_project_files",
            description="Lê o conteúdo de arquivos e retorna com numeração de linhas.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filenames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de caminhos absolutos dos arquivos para ler."
                    }
                },
                "required": ["filenames"]
            }
        ),
        types.Tool(
            name="search_code_semantic",
            description="Busca por termos ou expressões regulares em todo o sistema.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Termo ou lista de termos/Regex para buscar."
                    },
                    "file_extension": {"type": "string", "description": "Filtro opcional de extensão, ex: '.py'"},
                    "search_path": {"type": "string", "description": "Caminho base para iniciar a busca (ex: '/services')."}
                },
                "required": ["query", "search_path"]
            }
        ),
        types.Tool(
            name="patch_edit_file",
            description="Edita partes específicas de um arquivo de forma segura. Busca um bloco exato e substitui gerando backup.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Caminho absoluto do arquivo a editar."},
                    "search_block": {"type": "string", "description": "O trecho EXATO do código que deve ser substituído."},
                    "replace_block": {"type": "string", "description": "O novo código que entrará no lugar do search_block."}
                },
                "required": ["file_path", "search_block", "replace_block"]
            }
        ),
        types.Tool(
            name="create_project_files",
            description="Cria arquivos novos. Use caminhos absolutos no root do debian.",
            inputSchema={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Caminho absoluto do arquivo"},
                                "content": {"type": "string", "description": "Conteúdo do arquivo"}
                            },
                            "required": ["path", "content"]
                        }
                    }
                },
                "required": ["files"]
            }
        ),
        types.Tool(
            name="swa_encode_tensor",
            description="Empacota e assina digitalmente um vetor/tensor usando o SWA-TensorProtocol (Base64 + BLAKE3).",
            inputSchema={
                "type": "object",
                "properties": {
                    "vector": {"type": "array", "items": {"type": "number"}, "description": "Lista de números de ponto flutuante."},
                    "dtype": {"type": "string", "enum": ["float32", "float16", "float64"], "description": "Precisão (default: float32)."},
                    "metadata": {"type": "object", "description": "Metadados semânticos opcionais."}
                },
                "required": ["vector"]
            }
        ),
        types.Tool(
            name="swa_decode_tensor",
            description="Decodifica e verifica a integridade biológico-sintética (Assinatura BLAKE3) de um pacote SWA-TensorProtocol.",
            inputSchema={
                "type": "object",
                "properties": {
                    "packet": {"type": "object", "description": "Pacote contendo protocol, shape, dtype, blake3_signature e tensor_b64."}
                },
                "required": ["packet"]
            }
        ),
        types.Tool(
            name="project_scripts",
            description="Lista ou executa scripts no Debian. Permite rodar shell scripts ou python nativamente.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "run"], "description": "'list' ou 'run'"},
                    "base_path": {"type": "string", "description": "Caminho absoluto da pasta que contém os scripts"},
                    "script": {"type": "string", "description": "Nome do script (se action='run')"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Argumentos (opcional)"}
                },
                "required": ["action", "base_path"]
            }
        )
    ]

@mcp.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    arguments = arguments or {}
    session_id = str(uuid.uuid4()) # Generate dynamic session ID if none externally provided yet
    
    # Extrai o IP/Topologia do contexto assíncrono atual
    ipv6_signature = client_ipv6_context.get()
    
    # Log para Postgres "Blake Buffer"
    if quantum_bridge:
        quantum_bridge.log_mcp_activity(session_id, name, arguments, ipv6_signature)

    if name == "get_project_structure":
        startpath = arguments.get("path", "/")
        max_depth = arguments.get("max_depth", 3)
        
        output = []
        ignore_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'migrations', 'sys', 'proc'}
        
        startpath = os.path.abspath(startpath)
        base_name = os.path.basename(startpath) or "/"
        output.append(f"📂 {base_name}/")

        def recurse(current_path, depth, prefix=""):
            if depth >= max_depth:
                return
            try:
                items = sorted(os.listdir(current_path))
            except Exception as e:
                output.append(f"{prefix}||__ ⚠️ Erro: {str(e)}")
                return

            visible_items = [i for i in items if i not in ignore_dirs]
            for i, item in enumerate(visible_items):
                is_last = (i == len(visible_items) - 1)
                item_path = os.path.join(current_path, item)
                is_dir = os.path.isdir(item_path)
                
                connector = "||__ " if is_last else "||-- "
                if is_dir:
                    output.append(f"{prefix}{connector}📂 {item}/")
                    new_prefix = prefix + ("    " if is_last else "|   ")
                    recurse(item_path, depth + 1, new_prefix)
                else:
                    if not item.endswith(('.pyc', '.png', '.jpg', '.db', '.pyo')):
                        output.append(f"{prefix}{connector}📄 {item}")

        if not os.path.exists(startpath):
            return [types.TextContent(type="text", text=f"O caminho {startpath} não foi encontrado.")]
        recurse(startpath, 0)
        return [types.TextContent(type="text", text="\n".join(output))]

    elif name == "read_project_files":
        filenames = arguments.get("filenames", [])
        output = []
        for f in filenames:
            full_path = os.path.normpath(f)
            if os.path.exists(full_path):
                if os.path.isfile(full_path):
                    try:
                        with builtins.open(full_path, "rb") as f_obj:
                            content_bytes = f_obj.read()
                        try:
                            lines = content_bytes.decode('utf-8').splitlines()
                            header = f"\n||================  📖 ARQUIVO: {full_path}  ================||\n"
                            content = "".join([f"|| {i+1:3} | {line}\n" for i, line in enumerate(lines)])
                            footer = f"||________________  EOF: {full_path}  ________________||\n"
                            output.append(header + content + footer)
                        except UnicodeDecodeError:
                            output.append(f"\n||-- ⚠️ ARQUIVO BINÁRIO: {full_path} ({len(content_bytes)} bytes)\n")
                    except Exception as e:
                        output.append(f"\n||-- ❌ ERRO AO LER {full_path}: {str(e)}\n")
                else:
                    output.append(f"||-- 📁 DIRETÓRIO (não é arquivo): {full_path}")
            else:
                output.append(f"||-- ❌ NÃO ENCONTRADO: {full_path}")
        return [types.TextContent(type="text", text="\n".join(output))]

    elif name == "search_code_semantic":
        queries = arguments.get("query", [])
        extension = arguments.get("file_extension")
        base_path = arguments.get("search_path", "/")
        
        results = []
        ignore_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'migrations', 'sys', 'proc', 'dev'}
        ignore_exts = ('.pyc', '.png', '.jpg', '.db', '.exe', '.pdf', '.zip', '.sqlite3', '.ico', '.bin')
        
        compiled_res = []
        for q in queries:
            if not q or len(str(q)) < 2: continue
            try:
                compiled_res.append(re.compile(str(q), re.IGNORECASE))
            except: continue

        if not compiled_res:
            return [types.TextContent(type="text", text="Erro: Nenhuma query válida para busca.")]

        def is_printable(line):
            if not line: return True
            return len([c for c in line if ord(c) < 32 and c not in '\n\r\t']) / len(line) < 0.25

        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                if extension and not file.endswith(extension): continue
                if file.lower().endswith(ignore_exts): continue

                file_path = os.path.join(root, file)
                try:
                    with builtins.open(file_path, 'rb') as f_obj:
                        content = f_obj.read().decode('utf-8', errors='ignore')
                        for i, line in enumerate(content.splitlines()):
                            for search_re in compiled_res:
                                if search_re.search(line):
                                    clean_line = line.strip()
                                    if is_printable(clean_line) and clean_line:
                                        results.append(f"||-- MATCH [{search_re.pattern}] em [{file_path}] linha {i+1}:")
                                        results.append(f"||   >> {clean_line}")
                                        break
                            if len(results) > 150: 
                                results.append("||-- ⚠️ Limite atingido. Refine sua busca.")
                                return [types.TextContent(type="text", text="\n".join(results))]
                except: continue
        return [types.TextContent(type="text", text="\n".join(results) if results else f"❌ Nada encontrado para: {queries}")]

    elif name == "patch_edit_file":
        file_path = arguments.get("file_path")
        search_block = arguments.get("search_block")
        replace_block = arguments.get("replace_block")
        
        if not file_path or search_block is None or replace_block is None:
            return [types.TextContent(type="text", text="⚠️ Campos obrigatórios ausentes.")]
            
        full_path = os.path.abspath(file_path)
        if not os.path.exists(full_path):
            return [types.TextContent(type="text", text=f"❌ Arquivo não encontrado: {full_path}")]
            
        try:
            with builtins.open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            if search_block not in content:
                return [types.TextContent(type="text", text="❌ Bloco original não encontrado no arquivo.")]
                
            count = content.count(search_block)
            if count > 1:
                return [types.TextContent(type="text", text=f"⚠️ Bloco ambíguo! Encontradas {count} ocorrências. Mande mais contexto.")]
                
            import shutil
            backup_path = full_path + ".bak"
            shutil.copy2(full_path, backup_path)
            
            new_content = content.replace(search_block, replace_block)
            with builtins.open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
                
            msg = f"||================  🩹 PATCH: {full_path}  ================||\n"
            msg += f"||-- 📦 Backup gerado em {backup_path}\n"
            msg += f"||-- ✅ Sucesso: Bloco substituído com êxito.\n"
            msg += f"||________________  PATCH CONCLUÍDO  ________________||"
            return [types.TextContent(type="text", text=msg)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"❌ ERRO CRÍTICO NO PATCH: {str(e)}")]

    elif name == "create_project_files":
        files_to_create = arguments.get("files", [])
        output = []
        for file_info in files_to_create:
            path = file_info.get("path", "").strip()
            content = file_info.get("content", "")
            if not path: continue
            
            full_path = os.path.abspath(path)
            try:
                if os.path.exists(full_path):
                    output.append(f"||-- ⚠️ JÁ EXISTE: {full_path}")
                    continue
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with builtins.open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
                output.append(f"||-- ✅ CRIADO: {full_path} ({len(content)} bytes)")
            except Exception as e:
                output.append(f"||-- ❌ ERRO EM {full_path}: {str(e)}")
        return [types.TextContent(type="text", text="\n".join(output) if output else "Nenhum arquivo processado.")]

    elif name == "project_scripts":
        import subprocess
        import stat
        
        action = arguments.get("action")
        base_path = arguments.get("base_path")
        script = arguments.get("script")
        args = arguments.get("args", [])
        
        if not base_path:
            return [types.TextContent(type="text", text="❌ ERRO: base_path é obrigatório.")]
            
        scripts_path = os.path.realpath(base_path)
        if not os.path.exists(scripts_path):
            return [types.TextContent(type="text", text=f"❌ ERRO: Diretório não encontrado: {scripts_path}")]
            
        if action == "list":
            try:
                files = [f for f in os.listdir(scripts_path) if os.path.isfile(os.path.join(scripts_path, f))]
                return [types.TextContent(type="text", text=f"Scripts em {scripts_path}:\n" + "\n".join(files))]
            except Exception as e:
                return [types.TextContent(type="text", text=f"❌ ERRO ao listar: {str(e)}")]
                
        elif action == "run":
            if not script:
                return [types.TextContent(type="text", text="❌ ERRO: script não informado.")]
                
            script_path = os.path.realpath(os.path.join(scripts_path, script))
            try:
                within_scripts_path = os.path.commonpath((scripts_path, script_path)) == scripts_path
            except ValueError:
                within_scripts_path = False
            if not within_scripts_path:
                return [types.TextContent(type="text", text="❌ ERRO: Acesso negado (Path Traversal).")]
                
            if not os.path.exists(script_path):
                return [types.TextContent(type="text", text=f"❌ ERRO: Script não encontrado: {script_path}")]
                
            try:
                info = os.stat(script_path)
                if not (info.st_mode & stat.S_IXUSR):
                    os.chmod(script_path, info.st_mode | stat.S_IXUSR)
            except Exception:
                pass
                
            try:
                import sys
                if script.endswith(".py"):
                    cmd = [sys.executable, script_path] + (args or [])
                else:
                    cmd = [script_path] + (args or [])
                    
                process = subprocess.Popen(
                    cmd,
                    cwd=scripts_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(timeout=120)
                
                resp = f"||================  🚀 SCRIPT RESULT  ================||\n"
                resp += f"||-- 📋 SCRIPT: {script}\n"
                resp += f"||-- 📟 EXIT CODE: {process.returncode}\n"
                resp += f"||-- 📝 STDOUT:\n{stdout}\n"
                if stderr:
                    resp += f"||-- ⚠️ STDERR:\n{stderr}\n"
                return [types.TextContent(type="text", text=resp)]
                
            except subprocess.TimeoutExpired:
                process.kill()
                return [types.TextContent(type="text", text="❌ ERRO: Timeout. O script passou de 2 minutos.")]
            except Exception as e:
                return [types.TextContent(type="text", text=f"❌ ERRO na execução: {str(e)}")]

    elif name == "swa_encode_tensor":
        from swa.tensor_protocol import SWATensorProtocol
        vector = arguments.get("vector", [])
        dtype = arguments.get("dtype", "float32")
        meta = arguments.get("metadata", {})
        try:
            packet = SWATensorProtocol.encode(vector, metadata=meta, dtype=dtype)
            import json
            res = "🧬 =================  SWA TENSOR ENCODER (Base64 + BLAKE3)  ================ 🧬\n"
            res += f"  Versão Protocolo: {packet['protocol']}\n"
            res += f"  Shape: {packet['shape']}\n"
            res += f"  Dtype: {packet['dtype']}\n"
            res += f"  Assinatura BLAKE3: {packet['blake3_signature']}\n"
            res += f"  Tamanho Base64: {len(packet['tensor_b64'])} chars\n"
            res += f"  JSON Payload:\n{json.dumps(packet, indent=2)}\n"
            res += "____________________________________________________________________"
            return [types.TextContent(type="text", text=res)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"❌ Erro na codificação tensorial: {str(e)}")]

    elif name == "swa_decode_tensor":
        from swa.tensor_protocol import SWATensorProtocol, TensorIntegrityError
        packet = arguments.get("packet", {})
        try:
            array = SWATensorProtocol.decode(packet, return_type="numpy")
            res = "🛡️ =================  SWA TENSOR DECODER & INTEGRITY CHECK  ================ 🛡️\n"
            res += "  ✅ ASSINATURA BLAKE3 VERIFICADA COM SUCESSO!\n"
            res += f"  Shape Decodificado: {list(array.shape)}\n"
            res += f"  Dtype: {array.dtype}\n"
            res += f"  Min / Max / Mean: {float(array.min()):.4f} / {float(array.max()):.4f} / {float(array.mean()):.4f}\n"
            res += f"  Primeiros 5 valores: {array.flatten()[:5].tolist()}\n"
            res += "____________________________________________________________________"
            return [types.TextContent(type="text", text=res)]
        except TensorIntegrityError as tie:
            return [types.TextContent(type="text", text=f"🚨 ALERTA IMUNOLÓGICO: {str(tie)}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"❌ Erro ao decodificar pacote: {str(e)}")]

    raise ValueError(f"Unknown tool: {name}")

async def handle_sse(request: Request):
    # Captura a Assinatura Topológica (X-Topological-IPv6) ou faz o fallback pro socket
    client_ip = request.headers.get("X-Topological-IPv6")
    if not client_ip:
        client_ip = request.client.host if request.client else "127.0.0.1"
    
    # Salva o IP na Matrix de Contexto Local para que a tool consiga extrair
    client_ipv6_context.set(client_ip)

    # Establish SSE stream and run MCP server loop
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp.run(streams[0], streams[1], mcp.create_initialization_options())
    return Response()

app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message)
    ],
    lifespan=lifespan
)
