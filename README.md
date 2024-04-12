# Projektübersicht

Dieses Repository enthält den Code und die Dokumentation für ein Projekt zur Analyse von Bürgerbeschwerden mittels Natural Language Processing (NLP). Ziel des Projekts ist es, aus den über die FragDenStaat-API gesammelten Daten wichtige Themen herauszufiltern, die Hinweise auf häufige Beschwerden und Anliegen der Bürger bezüglich kommunaler Entscheidungen geben.

## Installationsanleitung

### Voraussetzungen:
- Python 3.8 oder höher
- pip für die Installation von Python-Paketen

### Benötigte Bibliotheken installieren:
```bash
pip install pandas spacy nltk sklearn gensim matplotlib


### SpaCy Deutsch-Sprachmodell herunterladen:
```bash
python -m spacy download de_core_news_sm


## Verwendung
- **Daten abrufen:** Die Daten werden automatisch über die FragDenStaat-API bezogen. Stellen Sie sicher, dass Sie eine gültige API-URL und die notwendigen Berechtigungen haben.
- **Daten vorbereiten:** Zur Vorbereitung der Daten führen Sie das Skript zur Datenbereinigung aus, das URLs, Satzzeichen entfernt und eine Lemmatisierung durchführt.
- **Themenmodellierung durchführen:** Verwenden Sie die Skripte zur Erstellung von BoW- und TF-IDF-Vektoren, und wenden Sie LDA oder NMF zur Themenextraktion an.
- **Ergebnisse auswerten:** Die Skripte zur Auswertung erzeugen visuelle Darstellungen der Themen und berechnen den Coherence Score, um die Qualität der Themen zu bewerten.

## Beispielanwendung
Ein Beispiel zur Durchführung einer Analyse kann durch das Ausführen des `analyse.py` Skripts im Hauptverzeichnis gestartet werden:
