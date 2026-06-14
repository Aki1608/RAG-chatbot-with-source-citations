# Multi-Document RAG Chatbot with Source Citations

An enterprise-grade Retrieval-Augmented Generation (RAG) chatbot built to process complex, multi-lingual academic and corporate documents (PDFs). This system features intelligent layout parsing, structural metadata filtering (e.g., "Only search Chapter 3"), and a high-precision Two-Stage Re-ranking pipeline to deliver perfectly grounded answers with exact text citations.

---

## Why Two-Stage Re-ranking? (The Secret to Precision)

Most basic AI chatbots use a simple "Vector Database" to find answers. They cast a wide net, grabbing paragraphs that share mathematical similarities to the user's question. This is fast, but often **inaccurate**. It frequently retrieves junk data that shares similar keywords but lacks the actual answer, leading to AI hallucinations.

This project solves that by breaking the search into two distinct steps:

1. **Stage 1: The Fast Recruiter (Vector Retrieval)** 
   The system searches the entire document database in milliseconds, pulling the top 15-20 paragraphs that seem related based on basic vector math.
2. **Stage 2: The Hiring Manager (Cross-Encoder Re-ranker)** 
   The pipeline runs those 15 candidates through a heavy, highly precise transformer model (`ms-marco-MiniLM-L-6-v2`). Instead of just comparing vectors, this model reads the question and the paragraph *simultaneously*, grading actual semantic relevance on a strict scale. It throws away the false positives and hands only the **absolute best 3 or 4 paragraphs** to the final LLM.

**The Result:** Lightning-fast search speeds combined with deep reading comprehension. The LLM stays focused, API costs drop dramatically, and hallucinations are virtually eliminated.

---

## Core Features

* **Intelligent Document Parsing:** Utilizes Unstructured's high-res vision models to parse complex PDF layouts, intelligently tracking hierarchical section titles to tag text chunks with precise `Chapter` metadata.
* **Multi-Lingual Support:** Native OCR and layout extraction configured for both English and German documents.
* **Two-Stage Re-ranking Pipeline:** Bypasses basic vector-distance flaws by employing a dedicated HuggingFace Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to mathematically re-rank candidate chunks for unparalleled accuracy.
* **Structural Pre-Filtering:** Users can restrict the AI's search scope to specific chapters via a dynamic UI dropdown.
* **Verifiable Citations:** The LLM generates answers alongside explicit citation cards showing the source document name, chapter title, and the exact text snippet used.
* **Modern Gradio Interface:** A clean, dual-column web UI featuring a file ingestion dashboard on the left and a secure chat interface on the right.

---

## System Architecture

1. **Ingestion Layer (`data_processor.py`):** Uses `langchain-unstructured` (`hi_res` strategy) to extract structural elements. It dynamically tracks `Title` elements to inject `{ "chapter": "Chapter Name" }` metadata into all subsequent text chunks before splitting.
2. **Retrieval Layer (`retriever.py`):** * **Stage 1:** Initial broad semantic search using `Chroma` and `BAAI/bge-small-en-v1.5` embeddings.
   * **Stage 2:** Raw `sentence-transformers` cross-encoder execution to score and filter down to the absolute top 4 most relevant chunks, bypassing fragile LangChain abstractions.
3. **Inference Layer:** Uses `ChatGroq` (`llama-3.1-8b-instant` at Temperature 0) for lightning-fast, factually grounded generation.
4. **UI Layer (`app.py`):** Built with `Gradio`.

---

## Prerequisites

**1. Python Version:** This project strictly requires **Python 3.10 or newer** (Python 3.11/3.12 recommended). Older versions like 3.6 are structurally incompatible with modern AI libraries.

**2. System Libraries (Linux/Codespaces/Ubuntu):**
Deep learning vision models require underlying C++ graphics and OCR libraries. Run this before installing Python packages:
    ```bash
    sudo apt-get update
    sudo apt-get install -y libgl1 tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng
    ```

---

## Installation & Setup

### 1. Clone the Repository
    ```bash
    git clone [https://github.com/Aki1608/RAG-chatbot-with-source-citations.git](https://github.com/Aki1608/RAG-chatbot-with-source-citations.git)
    cd RAG-chatbot-with-source-citations
    ```

### 2. Set Up a Virtual Environment
* **Windows (PowerShell):**
    ```powershell
    python -m venv venv
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
    .\venv\Scripts\activate
    ```
* **Linux / macOS / GitHub Codespaces:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies
This project uses a lean, unpinned deployment approach to avoid internal library version conflicts.
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

### 4. Configure Environment Variables
Create a file named `.env` in the root directory of the project and add your Groq API credentials:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

---

## Running the Application

Always start the application via the frontend UI script. **Do not run `retriever.py` directly.**

    ```bash
    python app.py
    ```

Once the backend builds, the terminal will provide a local URL (typically `http://127.0.0.1:7860`). Open this link in your browser to interact with the system.

### First-Time Run Notice:
The first time you upload a document and click **"Build Vector Database"**, the pipeline will download a 217MB layout-detection vision model (`yolox_l0.05.onnx`) from Hugging Face. If your network connection drops or the display progress bar stays stuck at 0%, manually download and cache the model using this command:
    ```bash
    mkdir -p ~/.cache/unstructured/core
    curl -L -o ~/.cache/unstructured/core/yolox_l0.05.onnx [https://huggingface.co/unstructuredio/yolo_x_layout/resolve/main/yolox_l0.05.onnx](https://huggingface.co/unstructuredio/yolo_x_layout/resolve/main/yolox_l0.05.onnx)
    ```

---

## File Structure

* `app.py`: The main entry point. Initializes the Gradio layout configuration and routes user interaction.
* `data_processor.py`: Orchestrates PDF file uploading, document layout analysis, metadata mapping, and text splitting.
* `retriever.py`: Manages vector database extraction, direct execution of the Cross-Encoder re-ranker, context building, and Groq inference.
* `requirements.txt`: The list of libraries used in thie Repo.
