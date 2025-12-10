# LangChain Agents & RAG Examples

This repository contains a collection of AI agents and Retrieval-Augmented Generation (RAG) systems built using LangChain, LangGraph, Google Gemini, and Ollama.

## Files Description

- **`ActAgent.py`**: Implements a LangGraph-based agent equipped with mathematical tools (add, subtract, multiply). It uses `ChatGoogleGenerativeAI` (Gemini) to decide which tools to call.
- **`ai_agent.py`**: A simple conversational agent loop using LangGraph and `ChatGoogleGenerativeAI`. It maintains a history of messages.
- **`local_rag.py`**: A local RAG system that answers questions about a pizza restaurant based on reviews. It uses `ChatOllama` for the LLM and retrieves relevant documents from a Chroma vector store.
- **`vector.py`**: Handles the initialization of the Chroma vector store and Ollama embeddings. It loads review data from `realistic_restaurant_reviews.csv`.
- **`test.py`**: A utility script to list available Google Gemini models using the `google-generativeai` library.
- **`agent1.ipynb`**: A Jupyter notebook containing agent experiments (likely similar to the scripts).

## Prerequisites

- **Python 3.8+**
- **Google API Key**: Required for using Gemini models.
- **Ollama**: Required for the local RAG system (`local_rag.py` and `vector.py`).
  - Ensure Ollama is installed and running.
  - Pull the required models:
    - `qwen3-vl:4b-instruct-q4_K_M` (for generation)
    - `embeddinggemma` (for embeddings)

## Installation

1.  **Clone the repository** (if applicable).

2.  **Install dependencies**:

    ```bash
    pip install langchain langchain-google-genai langchain-ollama langchain-chroma langgraph pandas python-dotenv google-generativeai
    ```

3.  **Environment Setup**:
    Create a `.env` file in the root directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

## Usage

### Math Agent

Run the agent that can perform calculations:

```bash
python ActAgent.py
```

### Conversational Agent

Start a chat with the simple AI agent:

```bash
python ai_agent.py
```

### Local RAG (Restaurant Q&A)

First, ensure the vector store is populated (this happens automatically when running `vector.py` or `local_rag.py` if the DB doesn't exist).

Run the RAG system:

```bash
python local_rag.py
```

You can then enter questions about the restaurant reviews.

### Check Available Models

To list the Gemini models available to your API key:

```bash
python test.py
```
