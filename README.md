# Orchestrate Support Agent - Implementation Guide

This directory contains the core logic for the HackerRank Orchestrate AI Support Triage Agent. The agent is designed to resolve support tickets for HackerRank, Claude, and Visa by grounding its responses in a verified markdown corpus.

## 🏗️ Architecture & Approach

The agent follows a **Single-Agent RAG (Retrieval-Augmented Generation)** architecture, optimized for accuracy and strict policy adherence.

### 1. Hybrid Triage Flow
- **Routing & Inference**: The agent first identifies the relevant organization (HackerRank, Claude, or Visa) from the ticket text.
- **Metadata-Filtered Retrieval (Classical Layer)**: Instead of searching the entire 8,000+ chunk vector space, the system applies a metadata filter to the ChromaDB query. This "Classical" enhancement restricts the search to only documents belonging to the identified company, significantly reducing noise and prevent across-company hallucinations.
- **Structured Synthesis**: The retrieved context is passed to **Gemini 1.5 Flash**, which uses a specialized system prompt to generate a structured Pydantic response.

### 2. Decision Logic (Reply vs. Escalate)
- **Deterministic Escalation**: Explicit rules force escalation for system outages, technical bugs, or sensitive requests (e.g., specific billing reversals) not covered by documentation.
- **Synthesis-Driven Replies**: The agent is trained to synthesize answers from multiple documents (e.g., combining "how to reinvite" with "how to add extra time") to provide a complete solution before deciding to escalate.

---

## 🛠️ Tech Stack

- **LLM**: `gemini-1.5-flash` (High reasoning quality, structured output support).
- **Embeddings**: `all-MiniLM-L6-v2` (Local HuggingFace embeddings) — used to ensure offline stability and avoid API rate limits during bulk processing.
- **Vector Store**: `ChromaDB` (Local persistence).
- **Orchestration**: `LangChain` & `Pydantic`.
- **Data Handling**: `Pandas` for robust CSV processing and NaN handling.

---

## 📈 Performance Results (Sample Data)

The agent achieved the following metrics on the `sample_support_tickets.csv` baseline:

| Metric | Accuracy |
| :--- | :--- |
| **Status Accuracy** (`replied` vs `escalated`) | **100% (10/10)** |
| **Request Type Accuracy** | **100% (10/10)** |
| **Grounding** | 0 Hallucinations detected |

---

## 🧠 Challenges & Solutions

### 1. The "API Quota" Bottleneck
**Challenge**: Initially using remote embeddings (Gemini) led to frequent `RESOURCE_EXHAUSTED` errors when indexing large corpora.
**Solution**: Migrated to **Local HuggingFace Embeddings**. This eliminated network latency for indexing and ensured the agent could run 100% reliably in a terminal environment without hitting rate limits.

### 2. Context Noise & Hallucinations
**Challenge**: A global vector search sometimes returned relevant-sounding HackerRank docs for Visa tickets, causing the LLM to provide incorrect cross-platform advice.
**Solution**: Implemented **Metadata Filtering**. By tagging chunks with their source organization, we forced the retriever to stay within the boundaries of the specific company's policy.

### 3. "The Reinvite-Time Loop"
**Challenge**: Early versions of the agent struggled with tickets that required two separate actions (e.g., adding extra time AND reinviting).
**Solution**: Refined the system prompt to reward **Synthesis Reasoning**. The agent now explicitly checks if a user's multi-part problem can be solved by combining information from different retrieved documents.

---

## 🚀 How to Run

1. **Environment**: Ensure `.env` contains your `GOOGLE_API_KEY`.
2. **Setup**:
   ```bash
   pip install -r requirements.txt  # (or use the provided .venv)
   ```
3. **Execute**:
   ```bash
   python main.py
   ```
   *The script will automatically index the data directory on the first run and produce `support_tickets/output.csv`.*
