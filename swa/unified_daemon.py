#!/usr/bin/env python3
"""
SWA Unified Daemon — Combina wiki_crawler e shadow_watch em um único processo.

Reduz overhead de Python e simplifica gerenciamento.
"""
import os
import sys
import time
import json
import threading
import urllib.request
import urllib.parse
from urllib.error import HTTPError
from pathlib import Path

from swa.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))

import torch
import psycopg2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class WikipediaCrawler:
    """Crawler da Wikipedia com rate limiting."""

    def __init__(self):
        self.user_agent = 'Aeon_CortexCrawler/1.0 (https://github.com/aeon-cortex; alifer@example.com)'
        self.conn = None
        self.cursor = None
        self._connect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect("dbname=swa_matrix user=alifer connect_timeout=5")
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"⚠️ [WIKI] Falha ao conectar ao PostgreSQL: {e}")

    def safe_request(self, url, max_retries=5):
        retries = 0
        backoff = 2
        while retries < max_retries:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': self.user_agent})
                with urllib.request.urlopen(req, timeout=10) as response:
                    return json.loads(response.read().decode())
            except HTTPError as e:
                if e.code == 429:
                    retry_after = e.headers.get('Retry-After')
                    if retry_after and retry_after.isdigit():
                        sleep_time = int(retry_after)
                        print(f"⏳ [WIKI] API ordenou congelamento. Hibernando por {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        print(f"⏳ [WIKI] Erro 429. Backoff de {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                else:
                    print(f"❌ [WIKI] Erro HTTP {e.code}")
                    return None
            except Exception as e:
                print(f"❌ [WIKI] Falha de rede: {e}. Backoff de {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                retries += 1
        return None

    def fetch_random_article(self):
        """Busca artigo aleatório da Wikipedia."""
        url = "https://pt.wikipedia.org/api/rest_v1/page/random/summary"
        data = self.safe_request(url)
        if not data:
            return None

        title = data.get("title", "")
        extract = data.get("extract", "")
        pageid = data.get("pageid", 0)

        if not title or not extract:
            return None

        return {
            "title": title,
            "extract": extract[:2000],
            "pageid": pageid,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }

    def store_article(self, article):
        """Armazena artigo no PostgreSQL."""
        if not self.conn or not article:
            return False

        try:
            self.cursor.execute("""
                INSERT INTO wiki_articles (title, extract, pageid, url, fetched_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (pageid) DO UPDATE SET
                    title = EXCLUDED.title,
                    extract = EXCLUDED.extract,
                    fetched_at = NOW()
            """, (article["title"], article["extract"], article["pageid"], article["url"]))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"❌ [WIKI] Erro ao armazenar: {e}")
            return False

    def run_cycle(self):
        """Executa um ciclo de crawling."""
        article = self.fetch_random_article()
        if article:
            if self.store_article(article):
                print(f"✅ [WIKI] {article['title']} ({len(article['extract'])} chars)")
                return True
        return False


class FileChangeWatcher:
    """Watcher de mudanças no filesystem."""

    def __init__(self):
        self.conn = None
        self.cursor = None
        self._connect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect("dbname=swa_matrix user=alifer connect_timeout=5")
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"⚠️ [SHADOW] Falha ao conectar ao PostgreSQL: {e}")

    def on_file_modified(self, file_path):
        """Processa mudança em arquivo."""
        if not file_path.endswith('.py'):
            return

        print(f"⚡ [SHADOW] Alteração detectada: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            seed = len(content) % 1000
            torch.manual_seed(seed)
            mutated_tensor = torch.randn(2560, dtype=torch.float32) * 0.05
            vec_list = mutated_tensor.detach().cpu().numpy().tolist()

            self.cursor.execute("""
                UPDATE blake_semantic_anchors 
                SET embedding_2560 = embedding_2560 + %s::vector
                WHERE file_path = %s;
            """, (vec_list, file_path))
            self.conn.commit()
            print(f"🌀 [SHADOW] Massa vetorial atualizada para {file_path}")

        except Exception as e:
            self.conn.rollback()
            print(f"❌ [SHADOW] Erro ao processar mudança: {e}")


class FileEventHandler(FileSystemEventHandler):
    """Handler para eventos do watchdog."""

    def __init__(self, watcher: FileChangeWatcher):
        self.watcher = watcher

    def on_modified(self, event):
        if not event.is_directory:
            self.watcher.on_file_modified(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.watcher.on_file_modified(event.src_path)


class UnifiedDaemon:
    """Daemon unificado que combina wiki_crawler e shadow_watch."""

    def __init__(self):
        self.wiki = WikipediaCrawler()
        self.shadow = FileChangeWatcher()
        self.running = False
        self._wiki_thread = None
        self._shadow_observer = None

    def start(self):
        """Inicia o daemon."""
        self.running = True
        print("🚀 [UNIFIED] Daemon unificado iniciado")

        # Iniciar wiki crawler em thread separada
        self._wiki_thread = threading.Thread(target=self._wiki_loop, daemon=True)
        self._wiki_thread.start()

        # Iniciar file watcher
        event_handler = FileEventHandler(self.shadow)
        self._shadow_observer = Observer()
        self._shadow_observer.schedule(
            event_handler,
            path=str(ROOT),
            recursive=False,
        )
        self._shadow_observer.start()

        # Manter vivo
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _wiki_loop(self):
        """Loop do wiki crawler."""
        while self.running:
            try:
                self.wiki.run_cycle()
                time.sleep(300)  # 5 minutos entre ciclos
            except Exception as e:
                print(f"❌ [WIKI] Erro no loop: {e}")
                time.sleep(60)

    def stop(self):
        """Para o daemon."""
        self.running = False
        if self._shadow_observer:
            self._shadow_observer.stop()
            self._shadow_observer.join(timeout=5)
        print("🛑 [UNIFIED] Daemon unificado encerrado")


def main():
    daemon = UnifiedDaemon()
    daemon.start()


if __name__ == "__main__":
    main()
