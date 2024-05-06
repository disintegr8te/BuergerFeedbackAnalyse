# Standard-Bibliotheken
import re
from datetime import datetime, timedelta

# Datenverarbeitung und Analyse
import pandas as pd
import numpy as np

# Natural Language Processing
import spacy
import nltk
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

# Sklearn für maschinelles Lernen
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import silhouette_score

# Erweiterte Themenmodellierung und Textembedding
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from gensim import corpora

# Netzwerkanfragen
import requests

# Initialisiere das 'spaCy' Modell für die Deutsche Sprache
nlp = spacy.load("de_core_news_sm")

def remove_urls(text):
    """Entferne URL-Verweise aus einem gegebenen Text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def clean_text(text):
    """Bereinige den Text von Satzzeichen und mehrfachen Leerzeichen."""
    punctuation_pattern = re.compile(r'[-.?!,":;()|0-9]')
    cleaned_text = punctuation_pattern.sub(" ", text)  # Ersetzen der Satzzeichen durch Leerzeichen
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Mehrfache Leerzeichen entfernen
    return cleaned_text

def get_date_days_ago(days):
    """Berechne das Datum, das eine gegebene Anzahl von Tagen zurückliegt, im Y-m-d Format."""
    return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

def print_topics_from_model(model, num_topics=10, num_words=5):
    """Drucke die Top-Themen aus einem Modell."""
    for topic_id, topic in model.show_topics(formatted=False, num_topics=num_topics, num_words=num_words):
        print(f"Thema {topic_id}: {' '.join([word for word, _ in topic])}")

def prepare_gensim_data(texts):
    """Vorbereite Texte für die Gensim Themenmodellierung."""
    if not texts:
        print("Keine Texte zur Verarbeitung vorhanden.")
        return None, None
    
    # Vorverarbeite Texte für Gensim
    processed_texts = [text.split() for text in texts if text]
    if not processed_texts:
        print("Keine verarbeitbaren Texte vorhanden.")
        return None, None

    # Erstelle ein Gensim Dictionary aus den verarbeiteten Texten
    dictionary = Dictionary(processed_texts)
    if not dictionary:
        print("Das Dictionary enthält keine Tokens. Überprüfen Sie die Daten.")
        return None, None

    # Filtere Extremwerte im Dictionary
    dictionary.filter_extremes(no_below=10, no_above=0.4)
    
    # Erstelle ein Corpus für das Modell
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    if not corpus:
        print("Corpus ist leer. Überprüfe die Textverarbeitung.")
        return dictionary, None

    return dictionary, corpus
def calculate_coherence_score(dictionary, corpus, texts, n_topics=10):
    """
    Berechnet den Coherence-Wert für ein LDA-Modell basierend auf den übergebenen Parametern.
    
    Args:
        dictionary (gensim.corpora.Dictionary): Das Gensim-Wörterbuch der Dokumente.
        corpus (list of list of (int, int)): Der Gensim-Korpus.
        texts (list of list of str): Vorverarbeitete Textdaten.
        n_topics (int): Anzahl der Themen für das LDA-Modell.
    
    Returns:
        float: Der Coherence-Wert für das Modell oder None bei einem Fehler.
    """
    if not texts or not dictionary or not corpus:
        print("Texts, Dictionary oder Corpus fehlt oder ist leer.")
        return None

    try:
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=1)
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        
        if np.isnan(coherence_score):
            print(f"Coherence Score ist NaN. Mögliche leere Themen oder sehr wenige Daten: {coherence_score}")
            return None
        
        return coherence_score
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None

def download_and_prepare_data():
    """
    Ruft Daten über die FragDenStaat-API ab und bereitet sie als Pandas DataFrame vor für die Analyse.
    
    Returns:
        pd.DataFrame: Ein DataFrame, der die abgerufenen Informationen enthält.
    """
    API_ROOT = "https://fragdenstaat.de/api/v1/request/"
    LIMIT_PER_PAGE = 20
    MAX_PAGES = 30
    data_list = []

    for page in range(MAX_PAGES):
        params = {
            "limit": LIMIT_PER_PAGE,
            "offset": page * LIMIT_PER_PAGE,
            "created_at_after": get_date_days_ago(120), # Anfragen der letzten 120 Tage
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

    col_names = ["ID", "Titel", "Erstellt am", "URL", "Status", "Öffentlichkeit", "Gesetz", "Beschreibung", "Zusammenfassung", "Fälligkeitsdatum"]
    dataframe = pd.DataFrame(data_list, columns=col_names)
    return dataframe
def create_tfidf(data, stop_words):
    """
    Erstellt und gibt Term Frequency-Inverse Document Frequency (TF-IDF) Vektoren zurück.
    
    Args:
        data (list): Liste von vorverarbeiteten Textdokumenten.
        stop_words (set): Set von Stop-Wörtern, die bei der Vektorisierung ignoriert werden sollen.
    
    Returns:
        tuple: Tuple, bestehend aus einem TF-IDF transformierten Data-Matrix und dem TF-IDF Vektorisierer.
    """
    stop_words_list = list(stop_words)  # Konvertiere Set in Liste für den Vektorisierer
    vectorizer_tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words_list, ngram_range=(1, 1))
    data_tfidf = vectorizer_tfidf.fit_transform(data)
    return data_tfidf, vectorizer_tfidf

def apply_lda(data_vectorized, n_components=10, **kwargs):
    """
    Wendet Latent Dirichlet Allocation (LDA) zur Themenmodellierung auf die vektorisierten Daten an.
    
    Args:
        data_vectorized (array-like, sparse matrix): Vektorisierte Daten.
        n_components (int): Anzahl der Themen.
        **kwargs: Weitere optionale Argumente für LDA.
    
    Returns:
        LatentDirichletAllocation: Das trainierte LDA-Modell.
    """
    lda = LatentDirichletAllocation(n_components=n_components, random_state=1, max_iter=10, learning_method='online', **kwargs)
    lda_model = lda.fit(data_vectorized)
    return lda_model

def remove_urls(text):
    """
    Entfernt URLs aus einem gegebenen Text.
    
    Args:
        text (str): Der Eingabetext, aus dem URLs entfernt werden sollen.
    
    Returns:
        str: Der Text ohne URLs.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def preprocess_text(text, nlp, stop_words, min_word_count=10):
    """
    Bereinigt und filtert Text basierend auf Stop-Wörtern und einer Mindestwortanzahl.

    Args:
        text (str): Der zu bereinigende Text.
        nlp (Language): Eine spaCy Language Objekt für die Textverarbeitung.
        stop_words (set): Ein Set von Stop-Wörtern.
        min_word_count (int): Die minimale Anzahl von Wörtern, die der bereinigte Text enthalten sollte.
    
    Returns:
        str: Der bereinigte und gefilterte Text oder None, wenn der Text weniger Wörter als min_word_count enthält.
    """
    text = remove_urls(text)
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_.lower() not in stop_words]
    
    if len(tokens) < min_word_count:
        return None
    return " ".join(tokens)

def create_bow(data, stop_words):
    """
    Erstellt und gibt Bag-of-Words (BoW) Vektoren zurück.
    
    Args:
        data (list): Liste von vorverarbeiteten Textdokumenten.
        stop_words (set): Set von Stop-Wörtern, die bei der Vektorisierung ignoriert werden sollen.
    
    Returns:
        tuple: Tuple, bestehend aus einem BoW transformierten Data-Matrix und dem BoW Vektorisierer.
    """
    stop_words_list = list(stop_words)  # Konvertiere Set in Liste für den Vektorisierer
    vectorizer_bow = CountVectorizer(max_df=0.85, min_df=5, stop_words=stop_words_list, ngram_range=(1, 1))
    data_bow = vectorizer_bow.fit_transform(data)
    return data_bow, vectorizer_bow
def apply_nmf(data_vectorized, n_components=10, max_iter=5000, **kwargs):
    """
    Anwendet Non-negative Matrix Factorization (NMF) zur Themenmodellierung.
    
    Args:
        data_vectorized (sparse matrix): Vektorisierte Textdaten.
        n_components (int): Anzahl der zu extrahierenden Themen.
        max_iter (int): Maximale Anzahl von Iterationen für den Algorithmus.
        **kwargs: Weitere Schlüsselwortargumente für NMF.
        
    Returns:
        NMF: Das trainierte NMF-Modell.
    """
    nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=1, **kwargs)
    nmf_model = nmf.fit(data_vectorized)
    return nmf_model

def evaluate_model(data_vectorized, model):
    """
    Bewertet ein Modell basierend auf dem Silhouette-Score.
    
    Args:
        data_vectorized (sparse matrix): Vektorisierte Daten.
        model (Model): Ein trainiertes Modell.
        
    Returns:
        float: Der berechnete Silhouette-Score.
    """
    labels = model.transform(data_vectorized).argmax(axis=1)
    score = silhouette_score(data_vectorized, labels)
    print(f"Silhouette-Score des Modells: {score:.4f}")

def display_topics(model, feature_names, no_top_words):
    """
    Zeigt die wichtigsten Wörter für jedes Thema eines Modells an.
    
    Args:
        model: Ein trainiertes Themenmodell.
        feature_names (list): Liste der Feature-Namen aus dem Vektorisierer.
        no_top_words (int): Anzahl der Wörter, die pro Thema angezeigt werden sollen.
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Thema %d:" % (topic_idx + 1))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))



def clean_and_filter_text(text, stop_words, min_length=5):
    """
    Bereinigt und filtert Texte nach Stopwörtern und einer minimalen Wortanzahl.
    
    Args:
        text (str): Der zu reinigende Text.
        stop_words (set): Set von Stop-Wörtern.
        min_length (int): Minimale Länge des Textes nach der Bereinigung.
        
    Returns:
        str oder None: Der gereinigte Text oder None, wenn er zu kurz ist.
    """
    doc = nlp(text)
    cleaned_tokens = [token.lemma_ for token in doc if token.text.lower() not in stop_words and token.is_alpha]
    cleaned_text = " ".join(cleaned_tokens)
    return cleaned_text if len(cleaned_text.split()) >= min_length else None

def apply_bertopic(texts):
    """
    Wendet BERTopic zur Themenmodellierung auf eine Liste von Texten an.
    
    Args:
        texts (list of str): Liste von Texten.
        
    Returns:
        tuple: Ein Tuple aus einem BERTopic-Modell und den resultierenden Themen.
    """
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    bertopic_model = BERTopic(embedding_model=sentence_model, calculate_probabilities=True, verbose=True)
    topics, probabilities = bertopic_model.fit_transform(texts)
    print("Anzahl der Topics:", len(bertopic_model.get_topics()))
   
    for topic_number, topic_content in bertopic_model.get_topics().items():
        print(f"Thema {topic_number}: {topic_content}")
    return bertopic_model, topics

def calculate_bertopic_coherence(bertopic_model, texts):
    """
    Berechnet den Coherence-Wert für ein BERTopic-Modell.
    
    Args:
        bertopic_model (BERTopic): Ein BERTopic-Modell.
        texts (list of str): Liste von Texten, die zur Berechnung verwendet werden.
        
    Returns:
        float oder None: Der Coherence-Wert oder None, wenn keine Themen gefunden wurden.
    """
    topic_words = [[word for word, _ in bertopic_model.get_topic(topic_number)] for topic_number in sorted(bertopic_model.get_topics())]
    
    # Erstellen Sie das dictionary
    texts_tokenized = [text.split() for text in texts]
    dictionary = corpora.Dictionary(texts_tokenized)
    
    coherence_model = CoherenceModel(topics=topic_words, texts=texts_tokenized, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print("Coherence Score:", coherence_score)
    return coherence_score

def check_topics_content(lda_model):
    """
    Überprüft den Inhalt der Themen eines LDA-Modells.
    
    Args:
        lda_model (LdaModel): Ein trainiertes LDA-Modell.
    """
    topics = lda_model.show_topics(formatted=False)
    for topic_id, topic in topics:
        if not topic:
            print(f"Thema {topic_id} hat keinen Inhalt.")
        else:
            print(f"Thema {topic_id}: {[word for word, _ in topic]}")
def calculate_and_diagnose_coherence(data_df):
    """
    Diagnostiziert und berechnet den Coherence Score basierend auf den verarbeiteten Textdaten in einem DataFrame.
    
    Args:
        data_df (pd.DataFrame): DataFrame, der eine Spalte 'Beschreibung_processed' mit den Textdaten enthält.

    Returns:
        float: Der berechnete Coherence Score für ein LDA Model oder None, wenn die Berechnung fehlschlägt oder keine Daten vorhanden sind.
    """
    # Überprüfung, ob 'Beschreibung_processed' im DataFrame vorhanden ist
    if 'Beschreibung_processed' not in data_df.columns:
        print("Die Spalte 'Beschreibung_processed' fehlt im DataFrame. Überprüfen Sie den DataFrame und versuchen Sie es erneut.")
        return None

    # Extrahiere verarbeitete Texte aus dem DataFrame, ignoriere 'None' Werte
    texts = data_df['Beschreibung_processed'].dropna().tolist()
    if not texts:
        print("Die Liste der verarbeiteten Texte ist leer. Stellen Sie sicher, dass die Daten korrekt vorverarbeitet wurden.")
        return None

    print(f"Es gibt {len(texts)} vorverarbeitete Dokumente zur Analyse.")

    # Vorbereiten des Dictionaries und des Korpus für die Coherence-Berechnung
    dictionary, corpus = prepare_gensim_data(texts)
    
    # Überprüfung der Richtigkeit von `dictionary` und `corpus`
    if not dictionary or not corpus:
        print("Fehler bei der Erstellung von Dictionary oder Corpus. Bitte überprüfen Sie die Eingabetexte.")
        return None

    print(f"Dictionary enthält {len(dictionary)} einzigartige Tokens.")
    print(f"Corpus enthält {len(corpus)} Dokumente.")

    # Berechnung des Coherence Scores mit geeigneten Parametern
    try:
        coherence_score = calculate_coherence_score(dictionary, corpus, texts, n_topics=7)
        if coherence_score:
            print(f"Coherence Score LDA: {coherence_score:.4f}")
            return coherence_score
        else:
            print("Coherence Score konnte nicht berechnet werden. Überprüfen Sie die Eingabedaten und -parameter.")
            return None
    except Exception as error:
        print(f"Ein unerwarteter Fehler ist aufgetreten bei der Berechnung des Coherence Scores: {error}")
        return None
def main():
    global nlp
    nlp = spacy.load("de_core_news_sm")
    nltk.download('stopwords')
    stop_words = set(nlp.Defaults.stop_words).union(set(stopwords.words('german')))

    data_df = None
    texts = None
    data_tfidf, vectorizer_tfidf = None, None
    data_bow, vectorizer_bow = None, None
    lda_model_tfidf, nmf_model_bow = None, None
    bertopic_model, topics = None, None
    
    def display_topics(model, feature_names, no_top_words):
        """ Zeigt die wichtigsten Wörter für jedes Thema eines Modells an. """
        for topic_idx, topic in enumerate(model.components_):
            print(f"Thema {topic_idx+1}:")
            print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    while True:
        print("\nMenü:")
        print("1. Daten von FragDenStaat herunterladen und vorbereiten")
        print("2. Daten bereinigen und vorverarbeiten")
        print("3. TF-IDF und Bag-of-Words (BoW) berechnen")
        print("4. LDA und NMF anwenden")
        print("5. Themen anzeigen und Coherence Score berechnen")
        print("6. BERTopic anwenden und Coherence Score berechnen")
        print("7. Alle Schritte automatisch ausführen")
        print("8. Programm beenden")
        auswahl = input("Bitte wählen Sie eine Option: ")

        if auswahl == '1':
            data_df = download_and_prepare_data()
            print(f"Anzahl der Dokumente vor Filterung: {len(data_df)}")
        elif auswahl == '2' and data_df is not None:
            data_df['Beschreibung_processed'] = data_df['Beschreibung'].apply(lambda text: preprocess_text(text, nlp, stop_words, min_word_count=5))
            data_df.dropna(subset=['Beschreibung_processed'], inplace=True)
            texts = data_df['Beschreibung_processed'].tolist()
            print(f"Anzahl der Dokumente nach Filterung: {len(data_df)}")
        elif auswahl == '3' and texts:
            data_tfidf, vectorizer_tfidf = create_tfidf(texts, stop_words)
            data_bow, vectorizer_bow = create_bow(texts, stop_words)
        elif auswahl == '4' and (data_tfidf is not None and data_bow is not None):
            lda_model_tfidf = apply_lda(data_tfidf, n_components=5)
            nmf_model_bow = apply_nmf(data_bow, n_components=10)
        elif auswahl == '5' and lda_model_tfidf and nmf_model_bow:
            print("NMF Themen (BOW):")
            display_topics(nmf_model_bow, vectorizer_bow.get_feature_names_out(), 5)
            print("\nLDA Themen (TF-IDF):")
            display_topics(lda_model_tfidf, vectorizer_tfidf.get_feature_names_out(), 5)
        elif auswahl == '6' and texts:
            bertopic_model, topics = apply_bertopic(texts)
            coherence_score = calculate_bertopic_coherence(bertopic_model, texts)
            print(f"BERTopic Coherence Score: {coherence_score:.4f}")
        elif auswahl == '7':
            print("Automatische Ausführung aller Schritte...")
            # Schritt 1
            print("Daten werden von FragDenStaat heruntergeladen und vorbereitet...")
            data_df = download_and_prepare_data()
            print(f"Anzahl der Dokumente vor Filterung: {len(data_df)}")
            # Schritt 2
            if data_df is not None:
                print("Daten werden gereinigt und vorverarbeitet...")
                data_df['Beschreibung_processed'] = data_df['Beschreibung'].apply(lambda text: preprocess_text(text, nlp, stop_words, min_word_count=5))
                data_df.dropna(subset=['Beschreibung_processed'], inplace=True)
                texts = data_df['Beschreibung_processed'].tolist()
                print(f"Anzahl der Dokumente nach Filterung: {len(data_df)}")
            # Schritt 3
            if texts:
                print("TF-IDF und Bag-of-Words (BoW) werden berechnet...")
                data_tfidf, vectorizer_tfidf = create_tfidf(texts, stop_words)
                data_bow, vectorizer_bow = create_bow(texts, stop_words)
                print("Vektorisierung abgeschlossen.")
            # Schritt 4
            if data_tfidf is not None and data_bow is not None:
                print("LDA und NMF werden auf die vektorisierten Daten angewendet...")
                lda_model_tfidf = apply_lda(data_tfidf, n_components=5)
                nmf_model_bow = apply_nmf(data_bow, n_components=10)
                print("Themenmodelle wurden erstellt.")
            # Schritt 5
            if lda_model_tfidf and nmf_model_bow:
                print("Themen werden angezeigt und der Coherence Score wird berechnet...")
                print("NMF Themen (BOW):")
                display_topics(nmf_model_bow, vectorizer_bow.get_feature_names_out(), 5)
                print("\nLDA Themen (TF-IDF):")
                display_topics(lda_model_tfidf, vectorizer_tfidf.get_feature_names_out(), 5)
                if texts:
                    dictionary, corpus = prepare_gensim_data(texts)
                    calculate_and_diagnose_coherence(data_df)
                print("Analyse der Themen und Coherence Scores abgeschlossen.")
            # Schritt 6
            if texts:
                print("BERTopic wird angewendet und der Coherence Score berechnet...")
                bertopic_model, topics = apply_bertopic(texts)
                coherence_score = calculate_bertopic_coherence(bertopic_model, texts)
                
                print(f"BERTopic Coherence Score: {coherence_score:.4f}")
                print("Top Themen von BERTopic:")
                for topic_number in sorted(bertopic_model.get_topics()):
                    print(f"Thema {topic_number}: {bertopic_model.get_topic(topic_number)}")
                print("Analyse von BERTopic und Coherence Scores abgeschlossen.")
        elif auswahl == '8':
            print("Programm wird beendet.")
            break
        else:
            print("Ungültige Auswahl oder fehlende Daten für die gewählte Operation.")

if __name__ == '__main__':
    main()
