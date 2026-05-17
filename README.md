# TP2 : Traitement de Documents PDF et Préparation RAG avec LangChain

Ce projet contient les premières étapes de mise en place d'un système de génération augmentée par récupération (RAG). Il montre comment charger un document PDF local, l'analyser et le découper en segments (*chunks*) optimisés pour un grand modèle de langage (LLM).

---

## Fonctionnalités Actuelles

- **Gestion de l'environnement** : Chargement automatisé des variables d'environnement (clés API).
- **Extraction de données** : Intégration d'un utilitaire de chargement de fichiers PDF (`PyPDFLoader`).
- **Tokenisation avancée** : Initialisation d'un encodeur basé sur `tiktoken` (spécifique au modèle `gpt-4o-mini`).
- **Segmentation intelligente** : Configuration d'un découpeur de texte récursif (`RecursiveCharacterTextSplitter`) basé sur le nombre de tokens plutôt que sur le nombre de caractères bruts.

---

# Analyse Globale du Code

Le notebook `tp2.ipynb` suit le pipeline initial d'un système RAG classique.

## 1. Importation des Dépendances

Le script commence par importer les briques essentielles de LangChain, de l'écosystème OpenAI, ainsi que des outils de gestion de stockage vectoriel (`Chroma`) et d'agents :

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.agents import create_agent
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
```

---

## 2. Configuration et Chargement du PDF

- **Variables d'environnement** : Initialisation avec `load_dotenv()` pour sécuriser les clés d'API nécessaires aux étapes suivantes.
- **Cible** : Spécification du document à indexer : `bipresentation.pdf`.

```python
load_dotenv()

loader = PyPDFLoader("bipresentation.pdf")
```

---

## 3. Stratégie de Chunking (Segmentation)

Pour garantir que les segments textuels s'insèrent correctement dans la fenêtre de contexte du LLM sans être coupés arbitrairement au milieu d'un mot, le découpage s'appuie sur le tokenizer de `gpt-4o-mini`.

### Paramètres utilisés

- **chunk_size** : `1000` tokens maximum par segment.
- **chunk_overlap** : `200` tokens pour conserver le contexte entre les segments.

```python
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=tokenizer.name,
    chunk_size=1000,
    chunk_overlap=200
)
```

---

# ⚠️ Erreur Bloquante Rencontrée & Résolution

Lors de l'exécution de la cellule de découpage :

```python
loader.load_and_split(splitter)
```

une erreur d'importation survient :

```bash
ImportError: pypdf package not found, please install it with pip install pypdf
```

---

## Pourquoi cette erreur ?

Bien que la classe `PyPDFLoader` soit importée depuis `langchain_community`, elle dépend en arrière-plan d'une bibliothèque tierce nommée `pypdf` pour analyser le contenu des fichiers PDF.

Cette bibliothèque n'est pas installée dans l'environnement virtuel actuel (`.venv`).

---

## Comment la résoudre ?

Active ton environnement virtuel si ce n'est pas déjà fait, puis installe la dépendance manquante avec la commande suivante :

```bash
pip install pypdf
```

Une fois l'installation terminée :

1. Redémarre le noyau du notebook (*Restart Kernel*).
2. Réexécute toutes les cellules.

Le découpage du fichier `bipresentation.pdf` fonctionnera alors correctement.

---

# Prochaines Étapes Prévues

D'après les imports déclarés dans la première cellule, la suite logique du projet comprend :

- Générer des embeddings avec `OpenAIEmbeddings`.
- Stocker les vecteurs dans une base de données vectorielle locale avec `Chroma`.
- Créer des outils personnalisés avec `@tool`.
- Configurer un agent conversationnel capable d'interroger intelligemment les documents.
