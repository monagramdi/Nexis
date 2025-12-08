import requests
import feedparser
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

# Imports Interface
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box
from rich.align import Align

# === IMPORT DE TON MODULE PERSO ===
from sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

# ============================================================
#                 STRUCTURE DES DONNÉES
# ============================================================

class ArticleData:
    def __init__(
        self,
        title: str,
        url: str,
        source: str,
        topic: str,
        published_date: Optional[datetime] = None,
        text: Optional[str] = None,
        # Champs pour le sentiment
        sentiment_score: float = 0.0,
        sentiment_label: str = "neutre"
    ):
        self.title = title
        self.url = url
        self.source = source
        self.topic = topic
        self.published_date = published_date or datetime.now()
        self.text = text or ""
        self.sentiment_score = sentiment_score
        self.sentiment_label = sentiment_label

# ============================================================
#                      SCRAPER RSS
# ============================================================

class RSSScraper:
    RSS_FEEDS = {
        "économie": {
            "lesechos": "https://news.google.com/rss/search?q=site:lesechos.fr+économie&hl=fr&gl=FR&ceid=FR:fr",
            "latribune": "https://news.google.com/rss/search?q=site:latribune.fr+économie&hl=fr&gl=FR&ceid=FR:fr",
        },
        "climat": {
            "lemonde_planete": "https://www.lemonde.fr/planete/rss_full.xml",
            "reporterre": "https://reporterre.net/spip.php?page=backend",
        },
        "politique": {
            "lefigaro_pol": "https://www.lefigaro.fr/rss/figaro_politique.xml",
            "liberation_pol": "https://www.liberation.fr/arc/outboundfeeds/rss/category/politique/",
        },
        "géopolitique": {
            "courrierinter": "https://www.courrierinternational.com/feed/all/rss.xml",
            "diploweb": "https://www.diploweb.com/spip.php?page=backend",
        },
    }

    def __init__(self, max_articles_per_topic: int = 3):
        self.max_articles_per_topic = max_articles_per_topic
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BotActu/1.0"})
        
        # Instanciation de l'analyzer importé
        self.analyzer = SentimentAnalyzer()

    def fetch_article_text(self, url: str) -> str:
        """Récupère le texte brut si possible."""
        try:
            r = self.session.get(url, timeout=4)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # Sélecteurs génériques
            tags = ["article", "main", "div.content", "div.post-content", "div#content"]
            for t in tags:
                content = soup.select_one(t)
                if content:
                    return content.get_text(" ", strip=True)
            return ""
        except Exception:
            return ""

    def scrape_topic(self, topic: str) -> List[ArticleData]:
        articles = []
        feeds = self.RSS_FEEDS.get(topic, {})
        
        for feed_name, feed_url in feeds.items():
            feed = feedparser.parse(feed_url)
            if feed.bozo: continue

            for entry in feed.entries:
                if len(articles) >= self.max_articles_per_topic: break
                
                try:
                    title = entry.title
                    link = entry.link
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])

                    # 1. Récupération du texte
                    full_text = self.fetch_article_text(link)
                    
                    # 2. Préparation du texte à analyser (fallback sur titre si texte vide)
                    text_to_analyze = full_text if len(full_text) > 100 else f"{title} {entry.get('description', '')}"
                    
                    # 3. Appel au module d'analyse externe
                    score, label = self.analyzer.analyze(text_to_analyze)

                    articles.append(ArticleData(
                        title=title, url=link, source=feed_name, topic=topic,
                        published_date=published, text=full_text,
                        sentiment_score=score, sentiment_label=label
                    ))
                except Exception:
                    continue
                    
            if len(articles) >= self.max_articles_per_topic: break
            
        return articles

# ============================================================
#              INTERFACE UTILISATEUR (MAIN)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    console = Console()
    
    scraper = RSSScraper(max_articles_per_topic=3)
    topics = ["économie", "climat", "politique", "géopolitique"]

    # --- Header ---
    console.print(Panel(Align.center("[bold white]🤖 BOT ACTU & SENTIMENT[/]"), border_style="blue"))

    # --- Processus ---
    results = {}
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None, style="blue"),
        TaskProgressColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Analyse...", total=len(topics))
        for topic in topics:
            progress.update(task, description=f"Analyse du sujet : [bold]{topic.upper()}[/]")
            results[topic] = scraper.scrape_topic(topic)
            progress.advance(task)

    # --- Affichage Tableau ---
    console.print("\n")
    for topic, items in results.items():
        if not items: continue

        table = Table(
            title=f"Sujet : [bold cyan]{topic.upper()}[/]",
            box=box.ROUNDED, expand=True, show_lines=True, header_style="bold white on blue"
        )
        table.add_column("Humeur", justify="center", width=12)
        table.add_column("Article", style="white", ratio=4)
        table.add_column("Lien", justify="center", width=8)

        for a in items:
            # Code couleur pour l'humeur
            if a.sentiment_label == "positif":
                mood = f"🟢 [bold green]Positif[/]\n[dim]{a.sentiment_score:.2f}[/]"
            elif a.sentiment_label == "négatif":
                mood = f"🔴 [bold red]Négatif[/]\n[dim]{a.sentiment_score:.2f}[/]"
            else:
                mood = f"⚪ [dim]Neutre[/]\n[dim]{a.sentiment_score:.2f}[/]"

            table.add_row(mood, f"[bold]{a.title}[/]\n[italic green]{a.source}[/]", f"[link={a.url}]Voir ↗[/]")

        console.print(table)
        console.print("\n")