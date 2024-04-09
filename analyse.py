import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import re
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import requests
import csv
from datetime import datetime, timedelta

# Hilfsfunktionen für die Textverarbeitung..
def get_date_days_ago(days: int):
    return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

# Restliche Hilfsfunktionen wie zuvor, zum Beispiel: remove_urls, preprocess_text, ...

def download_and_prepare_data():
    # Definiere die API-Parameter
    API_ROOT = "https://fragdenstaat.de/api/v1/request/"
    LIMIT_PER_PAGE = 20  # Anzahl der Ergebnisse pro Seite
    MAX_PAGES = 50  # Maximale Anzahl von Seiten, die verarbeitet werden sollen
    data_list = []  # Temporärer Speicher für die Daten

    # Anpassung der Parameter
    for page in range(MAX_PAGES):
        params = {
            "limit": LIMIT_PER_PAGE,
            "offset": page * LIMIT_PER_PAGE,
            "created_at_after": get_date_days_ago(120),  # Anfragen der letzten 30 Tage
            "category": "kommunales"
        }
        response = requests.get(API_ROOT, params=params)
        
        if response.status_code == 200:
            data = response.json()
            objects = data.get('objects', [])
            if not objects:
                break
            
            for request in objects:
                data_list.append([
                    request.get('id', 'N/A'),
                    request.get('title', 'Kein Titel verfügbar'),
                    request.get('created_at', 'Kein Datum verfügbar'),
                    "https://fragdenstaat.de" + request.get('url', ''),
                    request.get('status', 'Kein Status verfügbar'),
                    request.get('public', 'Nicht spezifiziert'),
                    request.get('law', 'Kein Gesetz verfügbar'),
                    request.get('description', 'Keine Beschreibung verfügbar'),
                    request.get('summary', 'Keine Zusammenfassung verfügbar'),
                    request.get('due_date', 'Kein Fälligkeitsdatum verfügbar'),
                ])
        else:
            print(f"Fehler bei der Anfrage: Statuscode {response.status_code}")
            break
    
    # Erzeuge einen DataFrame aus den gesammelten Daten
    col_names = ["ID", "Titel", "Erstellt am", "URL", "Status", "Öffentlichkeit", "Gesetz", "Beschreibung", "Zusammenfassung", "Fälligkeitsdatum"]
    df = pd.DataFrame(data_list, columns=col_names)
    return df

# Initialisiere SpaCy und NLTK für die deutsche Sprache
nlp = spacy.load("de_core_news_sm")
nltk.download('stopwords')
stop_words = set(stopwords.words('german'))

def create_tfidf(data):
    """Erstellt TF-IDF Vektoren."""
    stop_words_list = list(stop_words)
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words_list, ngram_range=(1,2))
    data_tfidf = vectorizer.fit_transform(data)
    return data_tfidf, vectorizer

def apply_lda(data_vectorized, n_components=10):
    """Anwenden von LDA zur Themenmodellierung."""
    lda = LatentDirichletAllocation(n_components=n_components, random_state=1, max_iter=10, learning_method='online')
    lda_model = lda.fit(data_vectorized)
    return lda_model

def remove_urls(text):
    """Entferne URLs aus Texten."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def preprocess_text(text):
    """Führe eine fortgeschrittene Textvorverarbeitung durch."""
    text = remove_urls(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.lower() not in stop_words]
    return " ".join(tokens)

def create_bow(data):
    """Erstellt Bag-of-Words Vektoren."""
    stop_words_list = list(stop_words)
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words_list, ngram_range=(1, 2))
    data_bow = vectorizer.fit_transform(data)
    return data_bow, vectorizer

def apply_nmf(data_vectorized, n_components=10):
    """Anwenden von NMF zur Themenmodellierung."""
    nmf = NMF(n_components=n_components, random_state=1, max_iter=1000)
    nmf_model = nmf.fit(data_vectorized)
    return nmf_model

def evaluate_model(data_vectorized, model):
    """Bewerten der Effektivität des Modells."""
    labels = model.transform(data_vectorized).argmax(axis=1)
    score = silhouette_score(data_vectorized, labels)
    print(f"Silhouette-Score des Modells: {score:.4f}")

def display_topics(model, feature_names, no_top_words):
    """Zeigt die Top-Wörter für jedes Thema an."""
    for topic_idx, topic in enumerate(model.components_):
        print(f"Thema {topic_idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def main():
    # Daten herunterladen und vorbereiten
    data_df = download_and_prepare_data()
    
    # Vorverarbeitung der Daten
    data_df['Beschreibung_processed'] = data_df['Beschreibung'].apply(preprocess_text)

    # Speichern des DataFrames in eine CSV-Datei
    data_path = "anfragen_data.csv"  # Pfad zur CSV-Datei
    data_df.to_csv(data_path, index=False)  # Exportieren ohne Index

    # Einlesen der zuvor gespeicherten CSV-Datei
    data = pd.read_csv(data_path)
    data['Beschreibung_processed'] = data['Beschreibung'].apply(preprocess_text)
    
    # Anwendung von TF-IDF
    data_tfidf, vectorizer_tfidf = create_tfidf(data['Beschreibung_processed'])
    
    # Anwendung von NMF auf Bag-of-Words Vektoren
    data_bow, vectorizer_bow = create_bow(data['Beschreibung_processed'])
    nmf_model_bow = apply_nmf(data_bow, n_components=10)
    
    # Anwendung von LDA auf TF-IDF Vektoren
    lda_model_tfidf = apply_lda(data_tfidf, n_components=10)

    # Bewertung der Modelle (optional, hier für NMF als Beispiel)
    evaluate_model(data_bow, nmf_model_bow)
    
    # Anzeigen der Themen
    no_top_words = 10
    tf_feature_names_bow = vectorizer_bow.get_feature_names_out()
    tf_feature_names_tfidf = vectorizer_tfidf.get_feature_names_out()
    
    print("NMF Themen (BOW):")
    display_topics(nmf_model_bow, tf_feature_names_bow, no_top_words)
    
    print("\nLDA Themen (TF-IDF):")
    display_topics(lda_model_tfidf, tf_feature_names_tfidf, no_top_words)

if __name__ == '__main__':
    main()