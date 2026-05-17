# TP2 : Traitement de Documents PDF et Préparation RAG avec LangChain

Ce projet contient les premières étapes de mise en place d'un système de génération augmentée par récupération (RAG). Il montre comment charger un document PDF local, l'analyser et le découper en segments (chunks) optimisés pour un grand modèle de langage (LLM).

##  Fonctionnalités Actuelles

* **Gestion de l'environnement** : Chargement automatisé des variables d'environnement (clés API).
* **Extraction de données** : Intégration d'un utilitaire de chargement de fichiers PDF (`PyPDFLoader`).
* **Tokenisation avancée** : Initialisation d'un encodeur basé sur `tiktoken` (spécifique au modèle `gpt-4o-mini`).
* **Segmentation intelligente** : Configuration d'un découpeur de texte récursif (`RecursiveCharacterTextSplitter`) basé sur le nombre de tokens plutôt que sur le nombre de caractères bruts.

---

##  Analyse Globale du Code

Le notebook `tp2.ipynb` suit le pipeline initial d'un système RAG classique. Voici le détail étape par étape :

### 1. Importation des Dépendances
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

