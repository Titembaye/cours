# Prétraitement des textes

## Objectifs

- Nettoyer des textes bruités (URLs, HTML, hashtags, emails, emojis)
- Tokeniser avec NLTK et SpaCy
- Supprimer les stopwords (union NLTK + SpaCy)
- Comparer stemming (NLTK Snowball) et lemmatisation (SpaCy)
- Construire une pipeline de prétraitement complète

## Dataset

**Allocine** — avis de films en français, étiquetés Positif / Négatif.
Le dataset a été pré-bruité pour simuler des données réelles du web.

- [`allocine_bruite.csv`](allocine_bruite.csv) — à télécharger avant le TP

## TP

- [Prétraitement de données textuelles](TP_Text_Mining.ipynb)
