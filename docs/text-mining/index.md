# Text Mining

Le **Text Mining** extrait des connaissances à partir de textes non structurés.

## Pipeline générale

```
Texte brut → Nettoyage → Tokenisation → Stopwords → Lemmatisation
                                                          ↓
                                              Représentation (BoW / TF-IDF)
                                                          ↓
                                              Modèle (Naïf de Bayes / SVM / LDA)
```

## Chapitres

| Chapitre | Description |
|---|---|
| [Prétraitement](preprocessing/index.md) | Nettoyage regex, tokenisation NLTK/SpaCy, stopwords, stemming, lemmatisation |
| [Topic Modeling (LDA)](lda/index.md) | Découverte automatique de thèmes (non supervisé) |
| [Classification](classification/index.md) | BoW, TF-IDF, Naïf de Bayes, SVM — dataset Banking77 |
| [Analyse de Sentiments](sentiment/index.md) | Allocine — avis positifs/négatifs |
