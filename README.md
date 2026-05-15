# 🤖 HESTIM · Agentic AI — Multi Agent Systems

> **TP Intelligence Artificielle Distribuée**  
> Réalisé dans le cadre du cours d'IA Distribuée à **HESTIM**

---

## 🧠 C'est quoi un Agent AI ?

Un **Agent AI** est un système autonome qui :
- Reçoit des **messages** en entrée
- **Raisonne** grâce à un LLM
- Exécute des **actions** via des outils (tools)
- **Mémorise** les échanges pour maintenir le contexte

```
👤 User  →  🧠 Agent (LLM)  →  🛠️ Tools  →  💬 Réponse
                  ↑                               |
                  └───────── Mémoire ─────────────┘
```

---

## 🚀 Installation

### 1. Installer les dépendances

```bash
pip install langchain langchain-openai langgraph langchain-tavily python-dotenv
```

### 2. Configurer les clés API

Crée un fichier `.env` à la racine du projet :

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Lancer le notebook

```bash
jupyter notebook RAGV2.ipynb
```

---

## 🗂️ Structure du projet

```
TP2AGENTAI/
│
├── RAGV2.ipynb             # Notebook principal — Agents AI
├── app.py                  # Application RAG Streamlit
├── .env                    # Clés API (ne pas partager !)
├── README_TP_AGENTS.md     # Ce fichier
│
├── pdfs/                   # Documents PDF pour le RAG
└── store/                  # Base vectorielle ChromaDB
```

---

## 📋 Contenu du TP

### Partie 1 — Initialisation du LLM

Chargement de la clé OpenAI depuis `.env` et initialisation de `ChatOpenAI` :

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

---

### Partie 2 — Création d'un Agent simple

Création d'un agent basique avec `create_agent` et invocation avec un message utilisateur :

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    system_prompt="You are a helpful assistant"
)

resp = agent.invoke(input={"messages": [
    {"role": "user", "content": "je m'appelle fatima"}
]})
print(resp['messages'][-1].content)
```

---

### Partie 3 — Sélection dynamique de modèle (Middleware)

Sélection automatique du modèle selon l'environnement (`test` ou `production`) :

```python
basic_llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
advanced_llm = ChatOpenAI(model="gpt-4o",      temperature=0)

@wrap_model_call
def dynamic_model_selection(request, handler):
    env = request.runtime.context.get("env", "test")
    model = basic_llm if env == "test" else advanced_llm
    return handler(request.override(model=model))
```

| Environnement | Modèle sélectionné |
|--------------|-------------------|
| `test` | gpt-4o-mini (économique) |
| `production` | gpt-4o (performant) |

---

### Partie 4 — Mémoire conversationnelle

Utilisation de `InMemorySaver` pour maintenir le contexte entre les échanges :

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
agent = create_agent(
    model="openai:gpt-4o",
    system_prompt="you are a helpful assistant",
    checkpointer=memory
)

config = {"configurable": {"thread_id": 1}}

# Tour 1
agent.invoke({"messages": [HumanMessage("je m'appelle fati")]}, config=config)

# Tour 2 — l'agent se souvient du prénom
agent.invoke({"messages": [HumanMessage("c'est quoi mon nom ?")]}, config=config)
# Réponse : "Votre nom est Fati."
```

> 💡 Chaque `thread_id` correspond à une conversation indépendante.

---

### Partie 5 — Agent avec outils (Tools)

Création d'outils personnalisés avec le décorateur `@tool` :

```python
from langchain.tools import tool

@tool
def get_weather(city: str):
    """Get the weather of the given city"""
    return {"city": city, "temperature": 23, "humidity": 88}

@tool
def get_employee_info(employee_name: str):
    """Get infos about the given employee (salary, seniority)"""
    return {"name": employee_name, "salary": 34000, "seniority": 5}

agent4 = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather, get_employee_info],
    checkpointer=memory,
    system_prompt="answer the user question using only provided tools"
)
```

**Exemple d'utilisation :**

```python
# Météo
agent4.invoke({"messages": [HumanMessage("La météo à Casablanca")]}, config=config)
# → utilise get_weather("Casablanca")

# Employé
agent4.invoke({"messages": [HumanMessage("quel est le salaire de hassan")]}, config=config)
# → utilise get_employee_info("hassan")
```

---

### Partie 6 — Recherche Web (Tavily)

Intégration d'un outil de recherche web en temps réel :

```python
from langchain_tavily import TavilySearch

tavily = TavilySearch(max_results=10, search_depth="advanced")

@tool
def search_web(query: str):
    """Search the web for real-time information"""
    return tavily.invoke({"query": query})
```

---

## ⚙️ Technologies utilisées

| Composant | Technologie |
|-----------|-------------|
| LLM | GPT-4o / GPT-4o-mini (OpenAI) |
| Framework Agent | LangChain / LangGraph |
| Mémoire | InMemorySaver (LangGraph) |
| Recherche Web | Tavily Search API |
| Environnement | Python + Jupyter Notebook |

---

## 🔄 Concepts clés

| Concept | Description |
|---------|-------------|
| `create_agent` | Crée un agent ReAct avec LLM + tools |
| `@tool` | Décore une fonction Python en outil utilisable par l'agent |
| `checkpointer` | Sauvegarde la mémoire de conversation |
| `thread_id` | Identifiant unique d'une session de conversation |
| `middleware` | Intercepte les appels pour modifier le comportement |
| `wrap_model_call` | Permet la sélection dynamique du modèle |

---

## 📌 Différence Agent vs RAG

| | RAG | Agent AI |
|--|-----|---------|
| Source de données | Documents PDF indexés | Outils + Web + APIs |
| Mémoire | Aucune (stateless) | Conversation persistante |
| Actions | Réponse uniquement | Peut exécuter des actions |
| Complexité | Modérée | Élevée |

---

## 👨‍🎓 Auteur

fatima ezzahra bougsissa
TP Intelligence Artificielle Distribuée — 2025/2026
