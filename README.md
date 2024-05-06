
# Projektübersicht

Dieses Repository enthält den Code und die Dokumentation für ein Projekt zur Analyse von Bürgerbeschwerden mittels Natural Language Processing (NLP). Ziel des Projekts ist es, aus den über die FragDenStaat-API gesammelten Daten wichtige Themen zu identifizieren, die Hinweise auf häufige Beschwerden und Anliegen der Bürger bezüglich kommunaler Entscheidungen geben.

## Voraussetzungen
- Python 3.8 oder höher
- pip für die Installation von Python-Paketen

## Installation der Abhängigkeiten

Führen Sie die folgenden Befehle in Ihrem Terminal aus, um die notwendigen Bibliotheken zu installieren:

```bash
pip install pandas spacy nltk sklearn gensim matplotlib requests sentence_transformers bertopic
```

Zusätzlich müssen Sie ein deutsches Sprachmodell für SpaCy herunterladen:

```bash
python -m spacy download de_core_news_sm
```

## Datenabfrage und Verarbeitung
- **Datenabruf**: Die Bürgerbeschwerdedaten werden automatisch über die FragDenStaat-API bezogen. Stellen Sie sicher, dass Sie eine gültige API-URL und die notwendigen Berechtigungen haben.
- **Datenaufbereitung**: Führen Sie das Skript zur Datenbereinigung aus, um URLs, Satzzeichen zu entfernen und eine Lemmatisierung durchzuführen.

## Themenmodellierung und Analyse
- **Vektorisierung der Daten**: Verwenden Sie die Skripte zur Erstellung von BoW- und TF-IDF-Vektoren.
- **Themenextraktion**: Anwendung von LDA oder NMF zur Themenextraktion.
- **Ergebnisauswertung**: Erzeugung visueller Darstellungen der Themen und Berechnung vom Coherence Score zur Bewertung der Modellqualität.

## Beispielanwendung
Ein Beispiel zur Durchführung einer Analyse kann gestartet werden durch das Ausführen des folgenden Skripts im Hauptverzeichnis:

```bash
python analyse.py
```

## Struktur des Codes
Der Hauptcode beinhaltet mehrere Funktionen zur Textbereinigung, Datenanfrage und -verarbeitung, Themenmodellierung und Visualisierung. Nachfolgend die wichtigsten Funktionen im Detail:

- `remove_urls(text: str)`: Entfernt URLs aus Texten.
- `clean_text(text: str)`: Bereinigt Text von Satzzeichen und überflüssigen Leerzeichen.
- `prepare_gensim_data(texts: list)`: Bereitet Textdaten für die Analyse mit Gensim vor.
- `calculate_coherence_score(dictionary, corpus, texts)`: Berechnet den Coherence Score eines LDA-Modells.
- `download_and_prepare_data()`: Ruft Daten von FragDenStaat ab und bereitet sie als DataFrame vor.
