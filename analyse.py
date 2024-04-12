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
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
import numpy as np

nlp = spacy.load("de_core_news_sm")
def remove_urls(text):
    """Entfernt URLs aus den Texten."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def clean_text(text):
    """Entfernt Satzzeichen aus dem Text und erstellt n-Gramme."""
    punctuation = re.compile(r'[-.?!,":;()|0-9]')
    text = punctuation.sub(" ", text)  # Ersetzt die Satzzeichen durch Leerzeichen
    text = re.sub(r'\s+', ' ', text)  # Entfernt mehrfache Leerzeichen
    return text

def get_date_days_ago(days: int):
    return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

def prepare_gensim_data(texts):
    if not texts:
        print("Keine Texte zur Verarbeitung vorhanden.")
        return None, None

    processed_texts = [text.split() for text in texts if isinstance(text, str) and text.strip()]
    if not processed_texts:
        print("Keine verarbeitbaren Texte vorhanden.")
        return None, None

    dictionary = Dictionary(processed_texts)
    if len(dictionary) == 0:
        print("Das Dictionary enthält keine Tokens. Überprüfen Sie die Daten und deren Vorverarbeitung.")
        return None, None

    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    if not corpus:
        print("Das Corpus ist leer. Überprüfen Sie die Textverarbeitungsschritte.")
        return dictionary, None

    return dictionary, corpus
def calculate_coherence_score(dictionary, corpus, texts, n_topics=10):
    if not texts or not dictionary or not corpus:
        print("Texts, Dictionary oder Corpus fehlt oder ist leer.")
        return None

    try:
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=1)
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        if np.isnan(coherence_score):
            print("Coherence Score resultierte in NaN. Überprüfen der Daten notwendig.")
            return None
        print(f"Coherence Score: {coherence_score}")
        return coherence_score
    except ZeroDivisionError:
        print("Fehler: Division durch Null aufgetreten. Überprüfen Sie die Datenvariabilität.")
        return None
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None
def download_and_prepare_data():
    """
    Ruft -Daten über eine API ab und bereitet sie als Pandas DataFrame vor.
    Lädt die Daten von der FragDenStaat-API, extrahiert die erforderlichen Informationen
    und stellt sie in einem strukturierten Format zur Verfügung.
    
    Returns:
        pd.DataFrame: Ein DataFrame, der die abgerufenen Informationen enthält.
    """
    # API-Parameter definieren
    API_ROOT = "https://fragdenstaat.de/api/v1/request/"
    LIMIT_PER_PAGE = 20
    MAX_PAGES = 30
    data_list = []

    for page in range(MAX_PAGES):
        params = {
            "limit": LIMIT_PER_PAGE,
            "offset": page * LIMIT_PER_PAGE,
            "created_at_after": get_date_days_ago(120),  # Beschränkt die Anfrage auf die letzten 120 Tage
            "category": "kommunales"
        }
        response = requests.get(API_ROOT, params=params)
        
        if response.status_code == 200:
            data = response.json()
            objects = data.get('objects', [])
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

    # Erzeuge und gib einen DataFrame aus den gesammelten Daten zurück
    col_names = ["ID", "Titel", "Erstellt am", "URL", "Status", "Öffentlichkeit", "Gesetz", "Beschreibung", "Zusammenfassung", "Fälligkeitsdatum"]
    df = pd.DataFrame(data_list, columns=col_names)
  
    print(df)  # Optional: Ausgabe des DataFrames zur Überprüfung
    
    return df
def create_tfidf(data, stop_words):
    """Erstellt TF-IDF Vektoren."""
    # Umwandlung von Set zu Liste
    stop_words_list = list(stop_words)
    vectorizer_tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words_list, ngram_range=(1, 1))


    data_tfidf = vectorizer_tfidf.fit_transform(data)
    return data_tfidf, vectorizer_tfidf

def apply_lda(data_vectorized, n_components=10, **kwargs):
    """Anwenden von LDA zur Themenmodellierung mit anpassbaren Hyperparametern."""
    lda = LatentDirichletAllocation(n_components=n_components, random_state=1, max_iter=10, learning_method='online', **kwargs)
    lda_model = lda.fit(data_vectorized)
    return lda_model

def remove_urls(text):
    """Entferne URLs aus Texten."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def preprocess_text(text, nlp, stop_words, min_word_count=10):
    """Bereinigt und filtert den Text basierend auf der Mindestwortanzahl."""
    # URLs entfernen
    text = remove_urls(text)
    # Text mittels SpaCy bereinigen und lemmatisieren
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_.lower() not in stop_words]
    # Frühes Ausschließen von kurzen Texten
    if len(tokens) < min_word_count:
        return None
    clean_text = " ".join(tokens)
    return clean_text

def create_bow(data, stop_words):
    """Erstellt Bag-of-Words Vektoren."""
    stop_words_list = list(stop_words) # Umwandlung von Set zu Liste
    vectorizer_bow = CountVectorizer(max_df=0.85, min_df=5, stop_words=stop_words_list, ngram_range=(1, 1))
    data_bow = vectorizer_bow.fit_transform(data)
    return data_bow, vectorizer_bow
def apply_nmf(data_vectorized, n_components=10, max_iter=5000, **kwargs):
    """Anwenden von NMF zur Themenmodellierung mit anpassbaren Hyperparametern."""
    nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=1, **kwargs)
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
def clean_and_filter_text(text, stop_words, min_length=5):
    doc = nlp(text)
    cleaned_tokens = [token.lemma_ for token in doc if token.text.lower() not in stop_words and token.is_alpha]
    cleaned_text = " ".join(cleaned_tokens)
    if len(cleaned_text.split()) >= min_length:
        return cleaned_text
    else:
        return None
def calculate_and_diagnose_coherence(data_df):
    """
    Diagnostiziert und berechnet den Coherence Score basierend auf 'Beschreibung_processed' in data_df.
    :param data_df: DataFrame mit einer Spalte 'Beschreibung_processed', die die Textdaten enthält.
    """
    # Überprüfung, ob 'Beschreibung_processed' im DataFrame vorhanden ist
    if 'Beschreibung_processed' not in data_df.columns:
        print("Die Spalte 'Beschreibung_processed' fehlt im DataFrame.")
        return

    # Vorbereitung der `texts`-Liste
    texts = [text for text in data_df['Beschreibung_processed'] if text is not None]

    if not texts:
        print("Die Textliste ist leer.")
        return

    print(f"Es gibt {len(texts)} Dokumente in der 'texts'-Liste.")

    # Vorbereitung von `dictionary` und `corpus` für Coherence
    dictionary, corpus = prepare_gensim_data(texts)

    # Überprüfung und Ausgabe des Status von `dictionary`
    if not dictionary:
        print("Das Dictionary ist leer.")
    else:
        print(f"Das Dictionary enthält {len(dictionary)} einzigartige Tokens.")
    
    # Überprüfung und Ausgabe des Status von `corpus`
    if not corpus:
        print("Das Corpus ist leer.")
    else:
        print(f"Das Corpus enthält {len(corpus)} Dokumente.")
    
    # Berechnung des Coherence Scores, wenn Daten korrekt vorliegen
    try:
        coherence_score_lda = calculate_coherence_score(dictionary, corpus, texts, n_topics=7)
        print(f"Coherence Score LDA: {coherence_score_lda:.4f}")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten bei der Berechnung des Coherence Scores: {e}")        
def main():
    global nlp
    nlp = spacy.load("de_core_news_sm")
    nltk.download('stopwords')
    stop_words = set(nlp.Defaults.stop_words).union(set(stopwords.words('german')))    
    data_df = download_and_prepare_data()
    print(f"Anzahl der Dokumente vor Filterung: {len(data_df)}")

    # Bereinigung und Vorverarbeitung
    data_df['bereinigte_Beschreibung'] = data_df['Beschreibung'].apply(lambda text: clean_and_filter_text(text, stop_words, min_length=5))
    data_df.dropna(subset=['bereinigte_Beschreibung'], inplace=True)
    
    data_df['Beschreibung_processed'] = data_df['Beschreibung'].apply(lambda text: preprocess_text(text, nlp, stop_words, min_word_count=5))
    data_df = data_df.dropna(subset=['Beschreibung_processed'])
    print(f"Anzahl der Dokumente nach Filterung: {len(data_df)}")

    texts = [text for text in data_df['Beschreibung_processed'] if text is not None and text.strip() != '']
    print(f"Anzahl der Dokumente im 'texts' für die Coherence Berechnung: {len(texts)}")

    # TF-IDF und BoW
    data_tfidf, vectorizer_tfidf = create_tfidf(data_df['Beschreibung_processed'], stop_words)
    data_bow, vectorizer_bow = create_bow(data_df['Beschreibung_processed'], stop_words)

    lda_model_tfidf = apply_lda(data_tfidf, n_components=5)  # Änderung: Sicherstellung, dass lda_model_tfidf definiert wird
    nmf_model_bow = apply_nmf(data_bow, n_components=10)

    # Thema Anzeigen
    no_top_words = 5
    tf_feature_names_bow = vectorizer_bow.get_feature_names_out()
    tf_feature_names_tfidf = vectorizer_tfidf.get_feature_names_out()
    print("NMF Themen (BOW):")
    display_topics(nmf_model_bow, tf_feature_names_bow, no_top_words)
    print("\nLDA Themen (TF-IDF):")
    display_topics(lda_model_tfidf, tf_feature_names_tfidf, no_top_words)

    # Coherence berechnen und anzeigen
    dictionary, corpus = prepare_gensim_data(texts)
    calculate_and_diagnose_coherence(data_df)

if __name__ == '__main__':
    main()