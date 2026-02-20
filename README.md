# VistaVu Rule Extraction Engine

The **VistaVu Rule Extraction Engine** is a comprehensive pipeline designed to ingest, process, and extract structured rules from complex PDF documents, specifically focusing on labour contracts and agreements.

This system leverages advanced Visual Language Models (VLM), Table of Contents (TOC) analysis, and Large Language Models (LLM) to convert unstructured PDF data into a queryable database of rules.

## 🚀 Key Features

*   **Intelligent Document Chunking**: Splits large PDFs into logical sections based on the Table of Contents using a pattern-learning algorithm.
*   **High-Fidelity PDF Conversion**: Uses Docling and custom VLM pipelines to preserve document structure, including tables and headers.
*   **AI-Powered Rule Extraction**: Utilizes Google Gemini 2.5 Pro to identify, categorize, and extract specific labour rules.
*   **Structured Storage**: Stores extracted rules in a standardized SQLite database (`labour_rules.db`) for easy integration and querying.
*   **Traceability**: Integrated with Langfuse for prompt management and execution tracing.

## 📁 Project Architecture

```
Vista_Vu_Project/
├── data/                    # Project datasets (tracked in git)
│   ├── contracts_initial/   # Raw PDF contracts
│   ├── docling_conversion/  # Intermediate Markdown/JSON outputs
│   ├── db/                  # SQLite database (labour_rules.db)
│   └── rules/               # Extracted rule JSON files
├── docs/                    # Project documentation and reports
├── notebooks/               # Jupyter notebooks for data processing workflow
│   ├── chunking.ipynb       # TOC-based document splitting
│   ├── categorization.ipynb # Chunk classification (Labour vs Non-Labour)
│   ├── conversion.ipynb     # PDF to Markdown conversion experiments
│   └── setup_database.ipynb # Database schema setup and loading
├── vistavu/                 # Main Python package
│   ├── dolphin/             # VLM integration module (Visual Language Model)
│   └── rules/               # Core rule extraction logic
├── requirements.txt         # Python dependencies
└── environment.yml          # Conda environment definition
```

## 🛠️ Prerequisites

*   **Python**: 3.10 or higher
*   **Conda**: For environment management (recommended)
*   **API Keys**: You must set up a `.env` file in the project root with the following variables:
    ```env
    GEMINI_API_KEY="your_google_gemini_key"
    LANGFUSE_SECRET_KEY="your_langfuse_secret"
    LANGFUSE_PUBLIC_KEY="your_langfuse_public"
    LANGFUSE_HOST="http://localhost:3000"
    NEO4J_URI="bolt://localhost:7687"
    NEO4J_USER="neo4j"
    NEO4J_PASSWORD="your_neo4j_password"
    ```

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Vista_Vu_Project
    ```

2.  **Create and activate the environment:**
    ```bash
    # using conda (recommended)
    conda env create -f environment.yml
    conda activate vistavu
    ```
    *Alternatively, using pip:*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify Setup:**
    Run the smoke tests to ensure everything is configured correctly:
    ```bash
    python tests/test_import.py
    ```

## 🏃 Quick Start

### 1. Run the Full Extraction Pipeline
The core extraction logic is encapsulated in the `vistavu.rules` module.

```bash
python -m vistavu.rules.extraction
```
This command will:
1.  Connect to the `labour_rules.db` database.
2.  Fetch chunks identified as "LABOUR".
3.  Send them to the Gemini API for rule extraction.
4.  Save the results to `data/rules/extracted_rules/`.

### 2. Run the Notebook Workflow
For step-by-step processing (PDF -> Markdown -> Chunks -> Database), follow the notebooks in the `notebooks/` directory in this order:

1.  **`conversion.ipynb`**: Convert raw PDFs to Markdown.
2.  **`chunking.ipynb`**: Split the Markdown documents into logical chunks based on the TOC.
3.  **`categorization.ipynb`**: Classify each chunk to identify relevant sections.
4.  **`setup_database.ipynb`**: Load the classified chunks into the SQLite database.

