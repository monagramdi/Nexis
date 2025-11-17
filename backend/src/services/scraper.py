"""
Scraper d'articles
"""

import requests
import feedparser
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ============================================================
#                 STRUCTURE DES DONNÉES
# ============================================================

class ArticleData:
    """Structure de données pour un article."""
    
    def __init__(
        self,
        title: str,
        url: str,
        source: str,
        topic: str,
        published_date: Optional[datetime] = None,
        text: Optional[str] = None,
        authors: Optional[List[str]] = None,
    ):
        self.title = title
        self.url = url
        self.source = source
        self.topic = topic
        self.published_date = published_date or datetime.now()
        self.text = text or ""
        self.authors = authors or []
        self.scraped_at = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "topic": self.topic,
            "published_date": self.published_date.isoformat(),
            "text": self.text,
            "authors": self.authors,
            "scraped_at": self.scraped_at.isoformat(),
        }


# ============================================================
#                      SCRAPER RSS
# ============================================================

class RSSScraper:
    
    # Flux RSS par thématique
    RSS_FEEDS = {
        "économie": {
            "lesechos": "https://news.google.com/rss/search?q=site:lesechos.fr+économie&hl=fr&gl=FR&ceid=FR:fr",
            "latribune": "https://news.google.com/rss/search?q=site:latribune.fr+économie&hl=fr&gl=FR&ceid=FR:fr",
            "challenges": "https://news.google.com/rss/search?q=site:challenges.fr+économie&hl=fr&gl=FR&ceid=FR:fr"
        },
        "climat": {
            "lemonde_planete": "https://www.lemonde.fr/planete/rss_full.xml",
            "reporterre": "https://reporterre.net/spip.php?page=backend",
            "liberation_env": "https://www.liberation.fr/arc/outboundfeeds/rss/category/environnement/"
        },

        "politique française": {
            "lemonde_pol": "https://www.lemonde.fr/politique/rss_full.xml",
            "liberation_pol": "https://www.liberation.fr/arc/outboundfeeds/rss/category/politique/",
            "lefigaro_pol": "https://www.lefigaro.fr/rss/figaro_politique.xml",
        },
        "géopolitique": {
            "lemonde_inter": "https://www.lemonde.fr/international/rss_full.xml",
            "courrierinter": "https://www.courrierinternational.com/feed/all/rss.xml",
            "diploweb": "https://www.diploweb.com/spip.php?page=backend",
        },
    }

    def __init__(self, max_articles_per_topic: int = 5):
        self.max_articles_per_topic = max_articles_per_topic
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0"
        })

    # --------------------------------------------------------------
    # Récupération du contenu complet des articles
    # --------------------------------------------------------------
    def fetch_article_text(self, url: str) -> str:
        """Télécharge la page et tente d'extraire le texte principal."""
        try:
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Sélecteurs génériques de contenu
            article_tag = (
                soup.find("article")
                or soup.find("div", class_="article__content")
                or soup.find("div", class_="content")
                or soup.find("div", {"id": "content"})
            )
            
            if article_tag:
                return article_tag.get_text(" ", strip=True)
            
            return ""
        
        except Exception as e:
            logger.error(f"Impossible d'extraire le texte de {url}: {e}")
            return ""

    # --------------------------------------------------------------
    # Parsing d'un flux RSS
    # --------------------------------------------------------------
    def scrape_feed(self, feed_name: str, feed_url: str, topic: str) -> List[ArticleData]:
        logger.info(f"  Lecture RSS: {feed_name} ({feed_url})")

        articles = []
        feed = feedparser.parse(feed_url)

        if feed.bozo:
            logger.error(f"  ⚠️ Problème de parsing RSS pour {feed_url}")
            return articles

        for entry in feed.entries[: self.max_articles_per_topic]:
            try:
                title = entry.title
                link = entry.link
                published = None

                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])

                # Extraire le texte de l'article
                text = self.fetch_article_text(link)

                authors = []
                if "author" in entry:
                    authors = [entry.author]

                articles.append(
                    ArticleData(
                        title=title,
                        url=link,
                        source=feed_name,
                        topic=topic,
                        published_date=published,
                        text=text,
                        authors=authors,
                    )
                )

            except Exception as e:
                logger.error(f"Erreur lors du parsing d'un entry RSS: {e}")

        return articles

    # --------------------------------------------------------------
    # Scraping par thématique
    # --------------------------------------------------------------
    def scrape_topic(self, topic: str) -> List[ArticleData]:
        logger.info(f"\n=== Scraping du topic: {topic} ===")
        topic_feeds = self.RSS_FEEDS.get(topic, {})
        collected = []

        for feed_name, feed_url in topic_feeds.items():
            collected.extend(self.scrape_feed(feed_name, feed_url, topic))

        collected = collected[: self.max_articles_per_topic]
        logger.info(f"  → {len(collected)} articles collectés pour {topic}")
        return collected

    # --------------------------------------------------------------
    # Scraper toutes les thématiques
    # --------------------------------------------------------------
    def scrape_all_topics(self, topics: List[str]) -> Dict[str, List[ArticleData]]:
        results = {}
        for topic in topics:
            results[topic] = self.scrape_topic(topic)
        return results

    # --------------------------------------------------------------
    # Filtrer les articles récents
    # --------------------------------------------------------------
    def get_recent_articles(self, topics: List[str], days_back: int = 1):
        all_articles = self.scrape_all_topics(topics)
        cutoff = datetime.now() - timedelta(days=days_back)

        recent = {
            topic: [
                a for a in articles
                if a.published_date and a.published_date >= cutoff
            ]
            for topic, articles in all_articles.items()
        }
        return recent


# ============================================================
#              Exemple d’utilisation
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scraper = RSSScraper(max_articles_per_topic=3)
    topics = ["économie", "climat", "politique française", "géopolitique"]

    results = scraper.scrape_all_topics(topics)

    for topic, items in results.items():
        print(f"\n=== {topic.upper()} ===")
        for a in items:
            print(f"- {a.title}")
            print(f"  Source: {a.source}")
            print(f"  URL: {a.url}")
