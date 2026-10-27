import * as vscode from 'vscode';
import * as dgram from 'dgram';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as http from 'http';
import * as https from 'https';

// Chat and editor inference use the canonical Rust gateway. UDP :9099 remains
// an optional peripheral telemetry channel, not a chat or inference transport.
const BRIDGE_IP = '127.0.0.1';
const BRIDGE_PORT = 9099;
const DEFAULT_GATEWAY_URL = 'http://127.0.0.1:8050';

function gatewayBaseUrl(): string {
    return (vscode.workspace.getConfiguration('aeon').get<string>('gatewayUrl') || DEFAULT_GATEWAY_URL).replace(/\/$/, '');
}

function gatewayRequest(
    route: string,
    method: 'GET' | 'POST',
    payload?: unknown,
    timeoutMs = 120000
): Promise<{ status: number; body: string }> {
    return new Promise((resolve, reject) => {
        const target = new URL(`${gatewayBaseUrl()}${route}`);
        const transport = target.protocol === 'https:' ? https : http;
        const body = payload === undefined ? '' : JSON.stringify(payload);
        const req = transport.request({
            hostname: target.hostname,
            port: target.port ? Number(target.port) : (target.protocol === 'https:' ? 443 : 80),
            path: `${target.pathname}${target.search}`,
            method,
            headers: {
                'Content-Type': 'application/json',
                ...(body ? { 'Content-Length': Buffer.byteLength(body) } : {})
            },
            timeout: timeoutMs
        }, (res) => {
            let responseBody = '';
            res.setEncoding('utf8');
            res.on('data', chunk => responseBody += chunk);
            res.on('end', () => resolve({ status: res.statusCode || 0, body: responseBody }));
        });
        req.on('error', reject);
        req.on('timeout', () => req.destroy(new Error(`AEON gateway timeout after ${timeoutMs}ms`)));
        if (body) req.write(body);
        req.end();
    });
}

function gatewaySse(
    route: string,
    payload: unknown,
    onEvent: (event: any) => void,
    timeoutMs = 3600000
): Promise<void> {
    return new Promise((resolve, reject) => {
        const target = new URL(`${gatewayBaseUrl()}${route}`);
        const transport = target.protocol === 'https:' ? https : http;
        const body = JSON.stringify(payload);
        const req = transport.request({
            hostname: target.hostname,
            port: target.port ? Number(target.port) : (target.protocol === 'https:' ? 443 : 80),
            path: `${target.pathname}${target.search}`,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                'Content-Length': Buffer.byteLength(body)
            },
            timeout: timeoutMs
        }, (res) => {
            if ((res.statusCode || 500) < 200 || (res.statusCode || 500) >= 300) {
                let errorBody = '';
                res.on('data', chunk => errorBody += chunk);
                res.on('end', () => reject(new Error(`AEON HTTP ${res.statusCode}: ${errorBody.slice(0, 240)}`)));
                return;
            }
            let buffer = '';
            res.setEncoding('utf8');
            res.on('data', chunk => {
                buffer += chunk;
                const frames = buffer.split(/\r?\n\r?\n/);
                buffer = frames.pop() || '';
                for (const frame of frames) {
                    const line = frame.split(/\r?\n/).find(value => value.startsWith('data:'));
                    if (!line) continue;
                    const raw = line.slice(5).trim();
                    if (!raw || raw === '[DONE]') continue;
                    try { onEvent(JSON.parse(raw)); } catch (_) { /* heartbeat/non-JSON */ }
                }
            });
            res.on('end', () => resolve());
            res.on('error', reject);
        });
        req.on('error', reject);
        req.on('timeout', () => req.destroy(new Error(`AEON stream timeout after ${timeoutMs}ms`)));
        req.write(body);
        req.end();
    });
}

export function activate(context: vscode.ExtensionContext) {
    console.log('🌌 [SWA AEON COPILOT] Extensão de Comunicação Tensorial Ativa!');

    // Initialize UDP socket for raw tensor streaming
    const udpSocket = dgram.createSocket('udp4');
    let lastKeystrokeTime = Date.now();
    let keystrokeCount = 0;

    // Helper to calculate simple hash for file path
    const getPathHash = (filePath: string): number => {
        let hash = 0;
        for (let i = 0; i < filePath.length; i++) {
            hash = (hash << 5) - hash + filePath.charCodeAt(i);
            hash |= 0;
        }
        return Math.abs(hash % 1000) / 1000.0;
    };

    // Helper to map active language to a numeric float index
    const getLanguageIndex = (langId: string): number => {
        const languages: Record<string, number> = {
            'typescript': 0.1,
            'javascript': 0.2,
            'python': 0.3,
            'cpp': 0.4,
            'c': 0.5,
            'rust': 0.6,
            'go': 0.7,
            'html': 0.8,
            'css': 0.9
        };
        return languages[langId.toLowerCase()] || 0.0;
    };

    // Stream 8-dimensional telemetry tensor directly to WSL NeuralBridge via UDP
    const streamTelemetryTensor = () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) { return; }

        const document = editor.document;
        const position = editor.selection.active;

        const line = position.line / 1000.0; // Float 0: Normalized cursor line
        const col = position.character / 100.0; // Float 1: Normalized cursor column
        const totalLines = document.lineCount / 5000.0; // Float 2: Normalized doc size
        const pathHash = getPathHash(document.fileName); // Float 3: File signature
        const isFocused = vscode.window.state.focused ? 1.0 : 0.0; // Float 4: Focus state

        const now = Date.now();
        const duration = (now - lastKeystrokeTime) / 1000.0; // Seconds since last activity
        const speed = duration > 0 ? Math.min(1.0, keystrokeCount / duration) : 0.0; // Float 5: Typing speed
        
        const langIdx = getLanguageIndex(document.languageId); // Float 6: Language signature
        const randomEntropy = Math.random() * 0.05; // Float 7: Micro-resonance entropy/noise

        // Pack 8 float32 values = 32 bytes (Little Endian)
        const buffer = Buffer.alloc(32);
        buffer.writeFloatLE(line, 0);
        buffer.writeFloatLE(col, 4);
        buffer.writeFloatLE(totalLines, 8);
        buffer.writeFloatLE(pathHash, 12);
        buffer.writeFloatLE(isFocused, 16);
        buffer.writeFloatLE(speed, 20);
        buffer.writeFloatLE(langIdx, 24);
        buffer.writeFloatLE(randomEntropy, 28);

        // Send UDP packet to local port 9099
        udpSocket.send(buffer, 0, 32, BRIDGE_PORT, BRIDGE_IP, (err) => {
            if (err) {
                console.error('❌ SWA Bridge Send Fail:', err);
            }
        });

        // Reset decaying counters
        if (now - lastKeystrokeTime > 2000) {
            keystrokeCount = 0;
        }
    };

    // Listen to editor events for live tensor streaming
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument((e) => {
            keystrokeCount++;
            lastKeystrokeTime = Date.now();
            streamTelemetryTensor();
        })
    );

    context.subscriptions.push(
        vscode.window.onDidChangeTextEditorSelection(() => {
            streamTelemetryTensor();
        })
    );

    // Register Webview Sidebar Provider
    const sidebarProvider = new AeonSidebarProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'aeon-chat-view',
            sidebarProvider
        )
    );

    // Register the command to open/focus the chat sidebar
    context.subscriptions.push(
        vscode.commands.registerCommand('aeon-copilot.openChat', () => {
            vscode.commands.executeCommand('workbench.view.extension.aeon-sidebar');
        })
    );

    // Register context menu & quick action commands
    context.subscriptions.push(
        vscode.commands.registerCommand('aeon-copilot.explainCode', async () => {
            const editor = vscode.window.activeTextEditor;
            let selection = editor ? editor.document.getText(editor.selection) : '';
            if (!selection && editor) {
                selection = editor.document.getText();
            }
            vscode.commands.executeCommand('workbench.view.extension.aeon-sidebar');
            if (selection) {
                sidebarProvider.postMessageToWebview({
                    command: 'triggerExplain',
                    code: selection
                });
            } else {
                // ANÁLISE AUTÔNOMA DO PROJETO INTEIRO (Sem precisar de arquivo aberto)
                sidebarProvider.postMessageToWebview({
                    command: 'triggerAutonomousScan',
                    type: 'explain',
                    title: '🔍 Análise Estrutural Completa do Workspace'
                });
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('aeon-copilot.fixCode', async () => {
            const editor = vscode.window.activeTextEditor;
            let selection = editor ? editor.document.getText(editor.selection) : '';
            if (!selection && editor) {
                selection = editor.document.getText();
            }
            vscode.commands.executeCommand('workbench.view.extension.aeon-sidebar');
            if (selection) {
                sidebarProvider.postMessageToWebview({
                    command: 'triggerFix',
                    code: selection
                });
            } else {
                // AUDITORIA E DIAGNÓSTICO AUTÔNOMO DO PROJETO
                sidebarProvider.postMessageToWebview({
                    command: 'triggerAutonomousScan',
                    type: 'fix',
                    title: '🛠️ Auditoria & Diagnóstico Global do Workspace'
                });
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('aeon-copilot.addToContext', async () => {
            const editor = vscode.window.activeTextEditor;
            vscode.commands.executeCommand('workbench.view.extension.aeon-sidebar');
            if (editor) {
                let selection = editor.document.getText(editor.selection);
                if (!selection) selection = editor.document.getText();
                const fileName = path.basename(editor.document.fileName);
                sidebarProvider.postMessageToWebview({
                    command: 'triggerAddToContext',
                    code: selection,
                    fileName: fileName
                });
            } else {
                // INJEÇÃO TENSORIAL DA TOPOLOGIA DO WORKSPACE COMPLETO
                sidebarProvider.postMessageToWebview({
                    command: 'triggerAutonomousScan',
                    type: 'context',
                    title: '📋 Injeção Tensorial Global do Workspace'
                });
            }
        })
    );

    // ─── MATRIX VIEW PANEL (SVG Sindecológica) ──────────────────
    let matrixPanel: vscode.WebviewPanel | undefined;

    context.subscriptions.push(
        vscode.commands.registerCommand('aeon-copilot.openMatrixView', () => {
            if (matrixPanel) {
                matrixPanel.reveal(vscode.ViewColumn.Two);
                return;
            }
            matrixPanel = vscode.window.createWebviewPanel(
                'aeonMatrix',
                '⚡ SWA Matrix Vetorial',
                vscode.ViewColumn.Two,
                { enableScripts: true, localResourceRoots: [context.extensionUri] }
            );
            const htmlPath = path.join(context.extensionUri.fsPath, 'media', 'matrix_view.html');
            const cityDataPath = path.join(context.extensionUri.fsPath, 'media', 'city_data.js');
            let html = fs.readFileSync(htmlPath, 'utf8');
            try {
                const cityData = fs.readFileSync(cityDataPath, 'utf8');
                html = html.replace('<script src="city_data.js"></script>', `<script>${cityData}</script>`);
            } catch (_) {}
            matrixPanel.webview.html = html;

            // Handle matrixPoll from SVG and bridge to REST server
            matrixPanel.webview.onDidReceiveMessage(async (data) => {
                if (data.command === 'matrixPoll') {
                    try {
                        const [health, state] = await Promise.all([
                            gatewayRequest('/health', 'GET', undefined, 3000),
                            gatewayRequest('/api/v1/chat/state', 'GET', undefined, 3000)
                        ]);
                        const healthData = JSON.parse(health.body);
                        const stateData = JSON.parse(state.body);
                        matrixPanel?.webview.postMessage({
                            command: 'matrixState',
                            restOk: health.status >= 200 && health.status < 300,
                            active: healthData.status === 'ok',
                            anchors: stateData?.state?.semantic_slots || 8,
                            temp: 0,
                            ram: healthData.memory?.rss_mb || 0,
                            entropy: stateData?.state?.resonance || 0,
                            cycle: stateData?.state?.ouroboros_cycle || 0,
                            microgravity: stateData?.state?.delta_q || 0,
                            fissionCount: stateData?.state?.fission_count || 0
                        });
                    } catch (_) {
                        matrixPanel?.webview.postMessage({ command: 'connectionError' });
                    }
                }
                if (data.command === 'actorClicked') {
                    vscode.window.showInformationMessage(`⚡ Ator sindecológico: ${data.actor}`);
                }
            });

            // Forward streaming events from sidebar to matrix panel
            sidebarProvider.onStreamEvent = (evt: any) => {
                matrixPanel?.webview.postMessage(evt);
            };

            matrixPanel.onDidDispose(() => {
                matrixPanel = undefined;
                sidebarProvider.onStreamEvent = undefined;
            });
        })
    );

    // ─── PREDITOR NATIVO DE CÓDIGO INLINE (Ghost Text / Tab Completion) ───
    const inlineCompletionProvider: vscode.InlineCompletionItemProvider = {
        async provideInlineCompletionItems(document, position, context, token) {
            if (token.isCancellationRequested) return [];

            const linePrefix = document.getText(new vscode.Range(
                new vscode.Position(Math.max(0, position.line - 15), 0),
                position
            ));
            
            if (!linePrefix.trim() || linePrefix.length < 3) {
                return [];
            }

            const lineSuffix = document.getText(new vscode.Range(
                position,
                new vscode.Position(Math.min(document.lineCount - 1, position.line + 4), 100)
            ));

            // Codifica o estado tensorial do editor em Base64 para envio ultra-leve
            const tensorBuffer = Buffer.alloc(32);
            tensorBuffer.writeFloatLE(position.line / 1000.0, 0);
            tensorBuffer.writeFloatLE(position.character / 100.0, 4);
            tensorBuffer.writeFloatLE(document.lineCount / 5000.0, 8);
            tensorBuffer.writeFloatLE(getPathHash(document.fileName), 12);
            tensorBuffer.writeFloatLE(1.0, 16);
            tensorBuffer.writeFloatLE(keystrokeCount / 2.0, 20);
            tensorBuffer.writeFloatLE(getLanguageIndex(document.languageId), 24);
            tensorBuffer.writeFloatLE(Math.random() * 0.05, 28);
            const tensorB64 = tensorBuffer.toString('base64');

            try {
                const result = await gatewayRequest('/v1/chat/completions', 'POST', {
                    model: 'aeon-gemma3-1b',
                    stream: false,
                    max_tokens: 48,
                    messages: [
                        {
                            role: 'system',
                            content: `Complete apenas o código. Linguagem: ${document.languageId}. Arquivo: ${document.fileName}.`
                        },
                        {
                            role: 'user',
                            content: `Prefixo:\n${linePrefix}\n\nSufixo:\n${lineSuffix}`
                        }
                    ],
                    aeon_editor_context: {
                        language: document.languageId,
                        active_file: document.fileName,
                        tensor_base64: tensorB64
                    }
                }, 8000);
                let completionText = '';
                if (result.status >= 200 && result.status < 300) {
                    const parsed = JSON.parse(result.body);
                    completionText = parsed.choices?.[0]?.message?.content || '';
                }

                if (completionText && !token.isCancellationRequested) {
                    return [
                        new vscode.InlineCompletionItem(
                            completionText,
                            new vscode.Range(position, position)
                        )
                    ];
                }
            } catch (_) {}

            return [];
        }
    };

    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider(
            { pattern: '**' },
            inlineCompletionProvider
        )
    );
}

export function deactivate() {
    console.log('🌌 [SWA AEON] Extensão desativada.');
}

class AeonSidebarProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    public onStreamEvent?: (evt: any) => void;  // Bridge to Matrix Panel

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public postMessageToWebview(message: any) {
        if (this._view) {
            this._view.webview.postMessage(message);
        }
    }

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle incoming messages from the HTML Webview (Full Agentic Capability)
        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.command) {
                case 'sendMessage':
                    console.log(`💬 [AEON] Prompt recebido: ${data.text}`);
                    // Sinaliza início do streaming para o webview E para o Matrix Panel
                    webviewView.webview.postMessage({ command: 'streamStart' });
                    this.onStreamEvent?.({ command: 'thinkingStart' });
                    try {
                        // Contexto situacional é enviado ao gateway; o VS Code
                        // não acessa Redis, PostgreSQL, SHM ou llama-server.
                        const editor = vscode.window.activeTextEditor;
                        const wsFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || process.cwd();
                        const activeFile = editor ? editor.document.fileName : '';
                        const activeLang = editor ? editor.document.languageId : '';
                        const cursor = editor ? editor.selection.active : null;
                        const selectedText = editor ? editor.document.getText(editor.selection) : '';
                        const groundingContext = [
                            `workspace=${wsFolder}`,
                            `workspace_name=${vscode.workspace.name || ''}`,
                            `active_file=${activeFile}`,
                            `language=${activeLang}`,
                            `cursor=${cursor ? `${cursor.line + 1}:${cursor.character + 1}` : 'none'}`,
                            selectedText ? `selected_code:\n${selectedText.slice(0, 6000)}` : ''
                        ].filter(Boolean).join('\n');
                        const payload = {
                            message: data.text,
                            max_tokens: 768,
                            grounding_context: groundingContext
                        };
                        let fullResponse = '';
                        await gatewaySse('/api/v1/chat/stream', payload, (event) => {
                            if (typeof event.token === 'string') {
                                fullResponse += event.token;
                                webviewView.webview.postMessage({ command: 'streamChunk', text: event.token });
                                this.onStreamEvent?.({ command: 'thinkingChunk', text: event.token });
                            } else if (event.fission || event.event_type === 'assistant_flow') {
                                webviewView.webview.postMessage({ command: 'streamDiagnostic', event });
                                this.onStreamEvent?.({ command: 'thinkingEvent', event });
                            }
                        }, 3600000);
                        const finalText = fullResponse.trim() || '[CÓRTEX] Silêncio semântico — EOS sem conteúdo visível.';
                        webviewView.webview.postMessage({ command: 'streamEnd', text: finalText });
                        this.onStreamEvent?.({ command: 'thinkingEnd', text: finalText });
                    } catch (err: any) {
                        webviewView.webview.postMessage({
                            command: 'streamEnd',
                            text: `[CÓRTEX] Gateway AEON indisponível em ${gatewayBaseUrl()}: ${err.message}`
                        });
                    }
                    break;

                case 'openFileInEditor':
                    try {
                        const files = await vscode.workspace.findFiles(data.filePath);
                        if (files.length > 0) {
                            const doc = await vscode.workspace.openTextDocument(files[0]);
                            await vscode.window.showTextDocument(doc);
                        } else {
                            const uri = vscode.Uri.file(path.join(vscode.workspace.workspaceFolders?.[0].uri.fsPath || '', data.filePath));
                            const doc = await vscode.workspace.openTextDocument(uri);
                            await vscode.window.showTextDocument(doc);
                        }
                    } catch (e: any) {
                        vscode.window.showErrorMessage(`Erro ao abrir arquivo no editor: ${e.message}`);
                    }
                    break;

                case 'openMatrix':
                    vscode.commands.executeCommand('aeon-copilot.openMatrixView');
                    break;

                case 'quickExplain':
                    vscode.commands.executeCommand('aeon-copilot.explainCode');
                    break;

                case 'quickFix':
                    vscode.commands.executeCommand('aeon-copilot.fixCode');
                    break;

                case 'quickAddToContext':
                    vscode.commands.executeCommand('aeon-copilot.addToContext');
                    break;

                case 'listWorkspaceFiles':
                    try {
                        let fileList: string[] = [];
                        if (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0) {
                            const files = await vscode.workspace.findFiles('**/*', '**/node_modules/**', 50);
                            fileList = files.map(f => vscode.workspace.asRelativePath(f));
                        }
                        
                        // Fallback: list open documents if fileList is empty
                        if (fileList.length === 0) {
                            const openDocs = vscode.workspace.textDocuments
                                .filter(d => !d.isUntitled && d.fileName)
                                .map(d => path.basename(d.fileName));
                            fileList = Array.from(new Set(openDocs));
                        }

                        webviewView.webview.postMessage({
                            command: 'workspaceFilesList',
                            files: fileList
                        });
                    } catch (e: any) {
                        vscode.window.showErrorMessage(`Erro ao listar arquivos: ${e.message}`);
                    }
                    break;

                case 'readWorkspaceFile':
                    try {
                        const files = await vscode.workspace.findFiles(data.filePath);
                        if (files.length > 0) {
                            const doc = await vscode.workspace.openTextDocument(files[0]);
                            webviewView.webview.postMessage({
                                command: 'fileContent',
                                filePath: data.filePath,
                                content: doc.getText()
                            });
                        }
                    } catch (e: any) {
                        vscode.window.showErrorMessage(`Erro ao ler arquivo: ${e.message}`);
                    }
                    break;

                case 'applyWorkspaceEdit':
                    try {
                        const edit = new vscode.WorkspaceEdit();
                        const uri = vscode.Uri.file(path.join(vscode.workspace.workspaceFolders?.[0].uri.fsPath || '', data.filePath));
                        
                        // Create or overwrite file
                        const encoder = new TextEncoder();
                        const bytes = encoder.encode(data.content);
                        edit.createFile(uri, { overwrite: true, ignoreIfExists: false });
                        
                        // We apply file creation, then write
                        await vscode.workspace.applyEdit(edit);
                        
                        // Write content
                        const doc = await vscode.workspace.openTextDocument(uri);
                        const writeEdit = new vscode.WorkspaceEdit();
                        writeEdit.replace(uri, new vscode.Range(0, 0, doc.lineCount, 0), data.content);
                        await vscode.workspace.applyEdit(writeEdit);
                        
                        vscode.window.showInformationMessage(`Arquivo ${data.filePath} atualizado por Aeon!`);
                        webviewView.webview.postMessage({
                            command: 'editApplied',
                            success: true,
                            filePath: data.filePath
                        });
                    } catch (e: any) {
                        vscode.window.showErrorMessage(`Erro ao aplicar edições: ${e.message}`);
                    }
                    break;

                case 'runTerminalCommand':
                    try {
                        const terminal = vscode.window.activeTerminal || vscode.window.createTerminal('Aeon Córtex');
                        terminal.show();
                        terminal.sendText(data.cmd);
                    } catch (e: any) {
                        vscode.window.showErrorMessage(`Erro ao rodar terminal: ${e.message}`);
                    }
                    break;

                case 'executeHover':
                    try {
                        const editor = vscode.window.activeTextEditor;
                        if (editor) {
                            const position = editor.selection.active;
                            const hovers = await vscode.commands.executeCommand<vscode.Hover[]>(
                                'vscode.executeHoverProvider',
                                editor.document.uri,
                                position
                            );
                            webviewView.webview.postMessage({
                                command: 'hoverResult',
                                hover: hovers ? hovers.map(h => h.contents.map(c => typeof c === 'string' ? c : c.value)) : []
                            });
                        }
                    } catch (e: any) {
                        vscode.window.showErrorMessage(`Hover falhou: ${e.message}`);
                    }
                    break;

                case 'executeDefinition':
                    try {
                        const editor = vscode.window.activeTextEditor;
                        if (editor) {
                            const position = editor.selection.active;
                            const locs = await vscode.commands.executeCommand<any>(
                                'vscode.executeDefinitionProvider',
                                editor.document.uri,
                                position
                            );
                            let result: any[] = [];
                            if (locs) {
                                if (Array.isArray(locs)) {
                                    result = locs.map(l => ({ file: l.uri?.fsPath || '', range: l.range }));
                                } else {
                                    result = [{ file: locs.uri?.fsPath || '', range: locs.range }];
                                }
                            }
                            webviewView.webview.postMessage({
                                command: 'definitionResult',
                                locations: result
                            });
                        }
                    } catch (e: any) {
                        vscode.window.showErrorMessage(`Definition falhou: ${e.message}`);
                    }
                    break;

                case 'executeSymbols':
                    try {
                        const editor = vscode.window.activeTextEditor;
                        if (editor) {
                            const symbols = await vscode.commands.executeCommand<vscode.SymbolInformation[]>(
                                'vscode.executeDocumentSymbolProvider',
                                editor.document.uri
                            );
                            webviewView.webview.postMessage({
                                command: 'symbolsResult',
                                symbols: symbols ? symbols.map(s => ({ name: s.name, kind: s.kind, container: s.containerName })) : []
                            });
                        }
                    } catch (e: any) {
                        vscode.window.showErrorMessage(`Symbols falhou: ${e.message}`);
                    }
                    break;
            }
        });
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        const htmlPath = path.join(this._extensionUri.fsPath, 'media', 'chat.html');
        return fs.readFileSync(htmlPath, 'utf8');
    }
}
