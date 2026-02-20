# Data Processing Notebooks

This directory contains the Jupyter notebooks used for the data processing pipeline. The notebooks are designed to be run sequentially to transform raw PDF contracts into a structured database ready for rule extraction.

## Workflow Sequence

### 1. `conversion.ipynb` (PDF to Markdown)
**Goal**: Convert complex PDF contracts into clean, structured Markdown text.
*   **Comparison**: Tests three methods:
    1.  **MarkItDown**: Standard conversion.
    2.  **Dolphin/GOT-OCR**: VLM-based approach for difficult layouts.
    3.  **Docling**: The preferred solution using IBM's Docling library with custom pipeline tweaks for table structure preservation.
*   **Output**: Markdown files (`.md`) saved to `data/docling_conversion/`.

### 2. `chunking.ipynb` (Intelligent Splitting)
**Goal**: Split the Markdown documents into smaller, logical "chunks" (e.g., individual Articles or Clauses).
*   **Method**: Uses an adaptive pattern learning algorithm that scans the Table of Contents (TOC) to understand the document's specific structure (keywords like "Article", "Section", numbering schemes like "1.0", "I", "One").
*   **Output**: Individual text files for each section, saved to `data/docling_conversion/CHUNK_NEW/`.

### 3. `categorization.ipynb` (Classification)
**Goal**: Classify each chunk to determine if it contains relevant labour rules.
*   **Method**: Sends each text chunk to Google Gemini 2.5 Pro via Langfuse.
*   **Classification**: Assigns three levels of metadata:
    *   **Type**: e.g., `COMPENSATION`
    *   **Level 2**: e.g., `PAYABLE_ENTITLEMENT`
    *   **Level 3**: e.g., `LABOUR` (This is the target class for extraction).
*   **Output**: A JSON file containing classifications for all chunks.

### 4. `setup_database.ipynb` (Database Loading)
**Goal**: Initialize the extraction database.
*   **Schema**: Creates the `chunks` table in SQLite.
*   **Loading**: Reads the classification JSON and the raw text files, then inserts them into `data/db/labour_rules.db`.
*   **Indexing**: Indexes the table for fast retrieval by the rule extraction engine.

## Usage

To run these notebooks:
1.  Ensure you have installed the requirements (`conda env create -f ../environment.yml`).
2.  Start Jupyter Lab or Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open the notebooks in the order listed above.
