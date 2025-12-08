"""
Module d'analyse de sentiment basé sur Hugging Face Transformers (BERT).
Plus lent que TextBlob, mais beaucoup plus intelligent pour le contexte.
"""
from transformers import pipeline
import logging

# On réduit le bruit des logs de transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

class SentimentAnalyzer:
    """
    Utilise un modèle BERT multilingue pour classer le sentiment.
    Le modèle retourne un score en 'étoiles' (1 star à 5 stars).
    """
    
    def __init__(self):
        print("Chargement du modèle neuronal (cela peut prendre quelques secondes)...")
        # On utilise un modèle spécialisé qui gère le français, l'anglais, etc.
        # Il va être téléchargé automatiquement au premier lancement.
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.pipe = pipeline("sentiment-analysis", model=model_name)

    def analyze(self, text: str) -> tuple[float, str]:
        """
        Retourne :
        - un score normalisé entre -1 (négatif) et 1 (positif)
        - un label ("négatif", "neutre", "positif")
        """
        if not text or len(text.strip()) < 5:
            return 0.0, "neutre"

        try:
            # Les modèles BERT ont une limite de longueur (souvent 512 tokens).
            # On tronque le texte pour éviter les erreurs.
            result = self.pipe(text[:512])[0]
            
            # Le résultat ressemble à : {'label': '1 star', 'score': 0.95}
            label_star = result['label']  # ex: '1 star', '4 stars'
            confidence = result['score']  # certitude de l'IA (0 à 1)

            # --- CONVERSION DU SYSTÈME D'ÉTOILES EN POLARITÉ ---
            # 1 star  = Très Négatif
            # 2 stars = Négatif
            # 3 stars = Neutre
            # 4 stars = Positif
            # 5 stars = Très Positif
            
            stars = int(label_star.split()[0]) # On récupère juste le chiffre
            
            # On mappe 1..5 vers -1..1
            # 1 -> -1.0
            # 2 -> -0.5
            # 3 ->  0.0
            # 4 -> +0.5
            # 5 -> +1.0
            normalized_score = (stars - 3) / 2.0

            # Définition du label textuel pour ton interface
            if stars <= 2:
                final_label = "négatif"
            elif stars == 3:
                final_label = "neutre"
            else:
                final_label = "positif"

            return normalized_score, final_label

        except Exception as e:
            # En cas de pépin, on reste neutre
            print(f"Erreur analyse BERT : {e}")
            return 0.0, "neutre"