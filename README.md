
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
- **Datenabruf**: Die Bürgerbeschwerdedaten werden automatisch über die FragDenStaat-API bezogen.
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

## Detaillierte Nutzungsinstruktionen

Das Skript `analyse.py` enthält ein menügesteuertes System, das es Benutzern ermöglicht, verschiedene Schritte des Analyseprozesses manuell zu steuern. Hier ist eine Schritt-für-Schritt-Anleitung zur Nutzung des Menüs:

1. **Daten Herunterladen und Vorbereiten**: Starten Sie diesen Schritt, um die neuesten Bürgerbeschwerdedaten über die FragDenStaat-API zu beziehen. Die Daten werden in einem DataFrame formatiert und lokal gespeichert.

2. **Daten Bereinigen und Vorverarbeiten**: Nachdem die Daten heruntergeladen wurden, können sie mit dieser Option gereinigt und für die Analyse vorbereitet werden. Dieser Schritt umfasst das Entfernen von URLs, die Lemmatisierung der Texte und das Filtern nach Wortanzahl.

3. **TF-IDF und Bag-of-Words Berechnen**: Dies erzeugt Vektoren aus den vorverarbeiteten Textdaten, die für die Themenmodellierung notwendig sind.

4. **LDA und NMF Anwenden**: Hier können Sie Latent Dirichlet Allocation (LDA) und Non-negative Matrix Factorization (NMF) anwenden, um Themen aus den vektorisierten Textdaten zu extrahieren.

5. **Themen Anzeigen und Coherence Score Berechnen**: Nach der Anwendung von LDA oder NMF können die resultierenden Themen angezeigt und evaluiert werden. Dieser Schritt berechnet auch den Coherence Score zur Bewertung der Qualität der Themenmodelle.

6. **BERTopic Anwenden und Coherence Score Berechnen**: Hier wird BERTopic zur Themenmodellierung verwendet, und der Coherence Score wird berechnet, um die Modellqualität zu bewerten.

7. **Alle Schritte Automatisch Ausführen**: Diese Option führt alle oben genannten Schritte in der gegebenen Reihenfolge aus, was eine vollständige End-to-End-Analyse ermöglicht.

8. **Programm Beenden**: Beendet das Programm.

Die Nutzung des Menüs erfolgt durch Eingabe der entsprechenden Nummer für den gewünschten Schritt, wenn dazu aufgefordert wird.

## Struktur des Codes
Der Hauptcode beinhaltet mehrere Funktionen zur Textbereinigung, Datenanfrage und -verarbeitung, Themenmodellierung und Visualisierung. Nachfolgend die wichtigsten Funktionen im Detail:

- `remove_urls(text: str)`: Entfernt URLs aus Texten.
- `clean_text(text: str)`: Bereinigt Text von Satzzeichen und überflüssigen Leerzeichen.
- `prepare_gensim_data(texts: list)`: Bereitet Textdaten für die Analyse mit Gensim vor.
- `calculate_coherence_score(dictionary, corpus, texts)`: Berechnet den Coherence Score eines LDA-Modells.
- `download_and_prepare_data()`: Ruft Daten von FragDenStaat ab und bereitet sie als DataFrame vor.
