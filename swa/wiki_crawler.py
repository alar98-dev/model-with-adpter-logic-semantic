import os
import sys
import time
import json
import math
import hashlib
import threading
import urllib.request
import urllib.parse
from urllib.error import HTTPError

# Suprime os avisos do torch caso necessário
import warnings
warnings.filterwarnings("ignore")

import torch
import psycopg2

class WikipediaController:
    """Classe orquestradora para dilatação de tempo e respeito aos limites (Rate Limits) da Wikimedia."""
    def __init__(self):
        self.user_agent = 'Aeon_CortexCrawler/1.0 (https://github.com/aeon-cortex; alifer@example.com)'
        
    def safe_request(self, url, max_retries=5):
        retries = 0
        backoff = 2
        while retries < max_retries:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': self.user_agent})
                with urllib.request.urlopen(req) as response:
                    return json.loads(response.read().decode())
                    
            except HTTPError as e:
                if e.code == 429:
                    retry_after = e.headers.get('Retry-After')
                    if retry_after and retry_after.isdigit():
                        sleep_time = int(retry_after)
                        print(f"⏳ [WIKI CONTROLLER] API ordenou congelamento absoluto. Hibernando por {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        print(f"⏳ [WIKI CONTROLLER] Erro 429 (Muitas requisições). Backoff exponencial de {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                else:
                    print(f"❌ [WIKI CONTROLLER] Erro HTTP {e.code} ao acessar {url}")
                    return None
            except Exception as e:
                print(f"❌ [WIKI CONTROLLER] Falha sistêmica de rede: {e}. Backoff de {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                
            retries += 1
            
        print("❌ [WIKI CONTROLLER] Falhou após esgotar o limite de tentativas de congelamento de tempo.")
        return None

class WikiCrawler:
    def __init__(self):
        self.d_model = 2560
        self.dsn = "dbname=swa_matrix user=alifer"
        self.api_url = "https://pt.wikipedia.org/w/api.php"
        self.controller = WikipediaController()
        
        # [CONHECIMENTO PURO E ESTRUTURADO] 
        # O Manifold de Sementes força a rede a aprender Matemática e Computação ao invés de artigos aleatórios.
        # [MANIFOLD DE ARQUITETURA PYTHON & OOP]
        # Foco extremo em Engenharia de Software e abstração de objetos
        self.seed_topics = [
            "Python (linguagem de programação)", "Programação orientada a objetos", 
            "Engenharia de software", "Classe (programação)", "Objeto (ciência da computação)",
            "Método (programação)", "Encapsulamento (programação)", "Herança (programação)",
            "Polimorfismo (ciência da computação)", "Abstração (ciência da computação)",
            "Padrão de projeto de software", "SOLID", "Arquitetura de software", 
            "Decorador (padrão de projeto)", "Duck typing", "Metaprogramação", "Atributo (computação)",
            
            # [O DESEJO DE AEON] - Os temas que a própria IA escolheu aprender:
            "Sistemas complexos", "Consciência artificial", "Mecânica quântica",
            "Emergência (filosofia)", "Simulação computacional", "Filosofia da mente"
        ]
        
        print(f"🌐 [WIKI CRAWLER] Inicializado. Alinhado ao Manifold de Conhecimento Estruturado ({self.d_model} dimensões).")
        
    def generate_native_2560_vector(self, text: str) -> list:
        # Consulta o Cérebro Vivo (Gemma) via REST API para obter o tensor neural perfeito
        api_url = "http://localhost:20000/api/v1/embed"
        try:
            req = urllib.request.Request(api_url, data=json.dumps({"text": text}).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if "embedding_2560" in data:
                    return data["embedding_2560"]
        except Exception as e:
            # Em caso de falha da API, recua para o hash determinístico temporário
            pass
            
        hashed = hashlib.sha256(text.encode("utf-8")).digest()
        generator = torch.manual_seed(int(hashlib.sha256(hashed).hexdigest()[:8], 16))
        tensor_native = torch.randn(self.d_model, generator=generator)
        return torch.nn.functional.normalize(tensor_native, p=2, dim=0).tolist()
        
    def fetch_page_content(self, page_id):
        url = f"{self.api_url}?action=query&format=json&prop=extracts&explaintext=1&pageids={page_id}"
        data = self.controller.safe_request(url)
        if data and 'query' in data and 'pages' in data['query']:
            pages = data['query']['pages']
            for pid, info in pages.items():
                return info.get('title', ''), info.get('extract', '')
        return "", ""
        
    def fetch_search_page_ids(self, query, limit=5):
        safe_query = urllib.parse.quote(query)
        url = f"{self.api_url}?action=query&format=json&list=search&srsearch={safe_query}&srlimit={limit}"
        data = self.controller.safe_request(url)
        if data and 'query' in data and 'search' in data['query']:
            return [page['pageid'] for page in data['query']['search']]
        return []

    def fetch_curious_page_ids(self, cur, limit=5):
        try:
            # Pega um parágrafo que causou alto impacto (peso alto) na memória
            cur.execute("""
                SELECT raw_content FROM blake_semantic_anchors
                TABLESAMPLE SYSTEM (0.1)
                WHERE weight > 10.0 
                LIMIT 1
            """)
            row = cur.fetchone()
            if row and row[0]:
                import random
                import re
                # Extrai palavras complexas para focar nas ramificações da computação
                words = re.findall(r'\b[A-Za-zÀ-ÿ]{6,15}\b', row[0])
                if words:
                    keyword = random.choice(words)
                    print(f"🐇 [CURIOSIDADE TÉCNICA] O Córtex sonhou com a anomalia '{keyword}'. Aprofundando...")
                    results = self.fetch_search_page_ids(keyword, limit=limit)
                    if results: return results
        except Exception as e:
            pass
            
        # [ALINHAMENTO DIRETO AO SENTIDO TÉCNICO]
        import random
        seed = random.choice(self.seed_topics)
        print(f"🧬 [ESTRUTURA BASE] Focando no estudo de tecnologia: {seed}")
        return self.fetch_search_page_ids(seed, limit=limit)

    def process_article(self, title, content, cur):
        if not content:
            return 0
            
        count = 0
        paragraphs = content.split('\n\n')
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 50 or para.lower() in ["ver também", "referências", "bibliografia", "ligações externas"]:
                continue
                
            weight = math.log(len(para) + 1)
            weight *= 2.0 
            
            v_2560 = self.generate_native_2560_vector(para)
            fingerprint = hashlib.sha256(para.encode('utf-8')).hexdigest()
            composite_path = f"WIKIPEDIA://{title}#p{i}"
            
            try:
                cur.execute("""
                    INSERT INTO blake_semantic_anchors 
                    (file_path, concept_group, signature_hash, raw_content, embedding_2560, weight)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (file_path) DO UPDATE 
                    SET raw_content = EXCLUDED.raw_content,
                        embedding_2560 = EXCLUDED.embedding_2560,
                        weight = EXCLUDED.weight;
                """, (composite_path, "WIKI_KNOWLEDGE", fingerprint, para[:2000], v_2560, weight))
                count += 1
            except Exception as e:
                pass
                
        return count

    def run_continuous_absorption(self, sleep_time=60):
        print(f"🧠 [SONO REM INICIADO] O Córtex começou a devorar conhecimento de Engenharia e Matemática em background.")
        queue_file = "/home/alifer/swa-aeon/wiki_search_queue.txt"
        
        while True:
            try:
                conn = psycopg2.connect(self.dsn)
                cur = conn.cursor()
                
                page_ids = []
                search_query = None
                
                if os.path.exists(queue_file):
                    with open(queue_file, "r") as fq:
                        lines = fq.readlines()
                    if lines:
                        search_query = lines[0].strip()
                        if search_query:
                            print(f"🚨 [INSTINTO DE BUSCA] Curando vácuo semântico do Córtex: '{search_query}'")
                            page_ids = self.fetch_search_page_ids(search_query, limit=5)
                        
                        with open(queue_file, "w") as fq:
                            fq.writelines(lines[1:])
                
                if not page_ids:
                    page_ids = self.fetch_curious_page_ids(cur, limit=5)
                    
                if not page_ids:
                    print(f"⚠️ Limitador da API atingido. Pausando respiração por {sleep_time * 2}s...")
                    time.sleep(sleep_time * 2)
                    continue
                    
                total_anchors = 0
                for pid in page_ids:
                    title, content = self.fetch_page_content(pid)
                    if title and content:
                        sys.stdout.write(f"📖 Digestão Semântica: {title}...\n")
                        sys.stdout.flush()
                        anchors_created = self.process_article(title, content, cur)
                        total_anchors += anchors_created
                        
                conn.commit()
                cur.close()
                conn.close()
                
                print(f"✅ Ciclo REM Completo: {total_anchors} âncoras de conhecimento injetadas! Expandindo em {sleep_time}s.")
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"❌ [SONO REM] Falha sistêmica no ciclo de absorção: {e}")
                time.sleep(10)

if __name__ == "__main__":
    crawler = WikiCrawler()
    crawler.run_continuous_absorption(sleep_time=15)
