import json, sys, uuid
sys.stdout.reconfigure(encoding='utf-8')

path = r"C:\Users\TITEMBAYE\Desktop\Projects\cours\docs\text-mining\classification\TP_Classification_Banking77.ipynb"

with open(path, encoding='utf-8') as f:
    nb = json.load(f)

def uid(): return uuid.uuid4().hex[:8]
def get(cell): return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
def md(src): return {"cell_type":"markdown","id":uid(),"metadata":{},"source":src}
def code(src): return {"cell_type":"code","id":uid(),"metadata":{},"execution_count":None,"outputs":[],"source":src}

# ─────────────────────────────────────────────────────────────────────────────
# NOUVELLE ÉTAPE 3 — chaque sous-étape appliquée au DataFrame, observation directe
# ─────────────────────────────────────────────────────────────────────────────
new_etape3 = [

    md("---\n"
       "### Étape 3 — Prétraitement\n\n"
       "Chaque sous-étape est **appliquée au dataset complet** (train + test) "
       "pour observer les transformations sur les vraies données.\n\n"
       "| Sous-étape | Outil | Colonne produite |\n"
       "|---|---|---|\n"
       "| Nettoyage regex | `re` | `text_clean` |\n"
       "| Tokenisation | NLTK `word_tokenize` | `tokens` |\n"
       "| Stopwords | NLTK + SpaCy (union) | `tokens` |\n"
       "| Lemmatisation | SpaCy `en_core_web_sm` | `tokens_lemme` → `text_clean` |"),

    # 3.1 ─────────────────────────────────────────────────────────────────────
    md("#### 3.1 Nettoyage regex\n\n"
       "On retire les chiffres, la ponctuation, et on met en minuscules."),

    code("def nettoyer(texte):\n"
         "    texte = re.sub(r'[^a-zA-Z\\s]', ' ', texte)  # ponctuation et chiffres\n"
         "    texte = re.sub(r'\\s+',          ' ', texte)  # espaces multiples\n"
         "    return texte.strip().lower()"),

    code("# Vérification sur 1 exemple avant d'appliquer\n"
         "print('BRUT    :', df_train['text'].iloc[0])\n"
         "print('NETTOYÉ :', nettoyer(df_train['text'].iloc[0]))"),

    code("# Application sur tout le dataset\n"
         "df_train['text_clean'] = df_train['text'].apply(nettoyer)\n"
         "df_test['text_clean']  = df_test['text'].apply(nettoyer)\n\n"
         "df_train[['text', 'text_clean']].head(3)"),

    # 3.2 ─────────────────────────────────────────────────────────────────────
    md("#### 3.2 Tokenisation — NLTK\n\n"
       "On découpe chaque message nettoyé en liste de mots."),

    code("from nltk.tokenize import word_tokenize\n\n"
         "# Vérification sur 1 exemple\n"
         "ex = df_train['text_clean'].iloc[0]\n"
         "print('Tokens :', word_tokenize(ex))"),

    code("# Application sur tout le dataset + filtre alpha (longueur > 2)\n"
         "def tokeniser(texte):\n"
         "    return [t for t in word_tokenize(texte) if t.isalpha() and len(t) > 2]\n\n"
         "df_train['tokens'] = df_train['text_clean'].apply(tokeniser)\n"
         "df_test['tokens']  = df_test['text_clean'].apply(tokeniser)\n\n"
         "df_train[['text', 'tokens']].head(3)"),

    # 3.3 ─────────────────────────────────────────────────────────────────────
    md("#### 3.3 Stopwords\n\n"
       "Deux bibliothèques pour l'anglais — on fait l'**union** pour une meilleure couverture."),

    code("from nltk.corpus import stopwords\n\n"
         "stop_nltk = set(stopwords.words('english'))\n"
         "print(f'NLTK  : {len(stop_nltk)} stopwords — ex : {sorted(stop_nltk)[:8]}')"),

    code("from spacy.lang.en import stop_words\n\n"
         "stop_spacy = stop_words.STOP_WORDS\n"
         "print(f'SpaCy : {len(stop_spacy)} stopwords — ex : {sorted(stop_spacy)[:8]}')"),

    code("stop_en = stop_nltk | set(stop_spacy)\n"
         "print(f'Union NLTK ∪ SpaCy : {len(stop_en)} stopwords')"),

    code("# Application sur tout le dataset\n"
         "df_train['tokens'] = df_train['tokens'].apply(lambda t: [w for w in t if w not in stop_en])\n"
         "df_test['tokens']  = df_test['tokens'].apply(lambda t: [w for w in t if w not in stop_en])\n\n"
         "df_train[['text', 'tokens']].head(3)"),

    # 3.4 ─────────────────────────────────────────────────────────────────────
    md("#### 3.4 Lemmatisation — SpaCy `en_core_web_sm`\n\n"
       "Réduit chaque token à sa **forme canonique** : "
       "`cards` → `card`, `declined` → `decline`, `payments` → `payment`."),

    code("import spacy\n"
         "nlp = spacy.load('en_core_web_sm')\n"
         "print('SpaCy prêt.')"),

    code("# Démonstration sur des termes bancaires\n"
         "mots_demo = ['cards', 'declined', 'payments', 'transferring',\n"
         "             'transactions', 'accounts', 'blocked', 'fraudulent']\n"
         "doc_demo = nlp(' '.join(mots_demo))\n\n"
         "print(f\"{'Mot original':<20} {'Lemme':<20} {'POS'}\")\n"
         "print('-' * 55)\n"
         "for tok in doc_demo:\n"
         "    print(f\"{tok.text:<20} {tok.lemma_:<20} {tok.pos_}\")"),

    code("# Application sur tout le dataset\n"
         "def lemmatiser(tokens):\n"
         "    doc = nlp(' '.join(tokens))\n"
         "    return [tok.lemma_.lower() for tok in doc\n"
         "            if not tok.is_stop and len(tok.lemma_) > 2]\n\n"
         "print('Lemmatisation en cours (~2 min)...')\n"
         "df_train['tokens_lemme'] = df_train['tokens'].apply(lemmatiser)\n"
         "df_test['tokens_lemme']  = df_test['tokens'].apply(lemmatiser)\n"
         "print('Terminé.')"),

    code("df_train[['text', 'tokens', 'tokens_lemme']].head(3)"),

    # 3.5 ─────────────────────────────────────────────────────────────────────
    md("#### 3.5 Résumé — du texte brut aux lemmes\n\n"
       "Observation des 4 colonnes côte à côte sur 5 exemples aléatoires."),

    code("df_train[['text', 'text_clean', 'tokens', 'tokens_lemme']].sample(5, random_state=42)"),

    # 3.6 ─────────────────────────────────────────────────────────────────────
    md("#### 3.6 Finalisation — `text_clean` prêt pour la vectorisation\n\n"
       "On joint les lemmes en une chaîne de caractères : c'est l'entrée du vectoriseur TF-IDF."),

    code("df_train['text_clean'] = df_train['tokens_lemme'].apply(' '.join)\n"
         "df_test['text_clean']  = df_test['tokens_lemme'].apply(' '.join)\n\n"
         "print('BRUT   :', df_train['text'].iloc[0])\n"
         "print()\n"
         "print('PROPRE :', df_train['text_clean'].iloc[0])"),
]

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 corrigée — utiliser df_train et df_test tels quels (pas de split)
# ─────────────────────────────────────────────────────────────────────────────
new_etape4_intro = [

    md("---\n"
       "### Étape 4 — Vectorisation TF-IDF\n\n"
       "On transforme chaque message en vecteur numérique.\n\n"
       "- `max_features` : 5 000 mots les plus informatifs\n"
       "- `ngram_range=(1,2)` : mots seuls **et** bigrammes (`card declined`, `transfer failed`)\n\n"
       "> **Important** : Banking77 fournit déjà un train set et un test set séparés. "
       "On les utilise directement — pas besoin de `train_test_split`."),

    code("# Utilisation des splits Banking77 officiels\n"
         "X_train = df_train['text_clean']\n"
         "y_train = df_train['classe']\n\n"
         "X_test  = df_test['text_clean']\n"
         "y_test  = df_test['classe']\n\n"
         "print(f'Train : {len(X_train)} messages | Test : {len(X_test)} messages')"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction du notebook
# ─────────────────────────────────────────────────────────────────────────────
cells = nb['cells']

before      = cells[:23]          # 0-22 : intro, théorie, chargement, classes, exemples
etape4_md   = cells[46]           # md "Étape 4" (on le remplace par notre version)
tfidf_cell  = cells[49]           # TF-IDF fit/transform
after_tfidf = cells[50:]          # modèles, évaluation, comparaison, test, bilan

nb['cells'] = before + new_etape3 + new_etape4_intro + [tfidf_cell] + after_tfidf

print(f"Cellules : {len(cells)} → {len(nb['cells'])}")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook mis à jour.")
