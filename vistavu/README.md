# VistaVu Source Code

This package contains the core logic for the VistaVu project, divided into specialized modules for rule extraction and VLM interactions.

## Modules

### 1. `vistavu.rules`
**Primary File:** `extraction.py`

This module handles the interaction with the Large Language Model (Gemini 2.5 Pro) to extract structured rules from text chunks.

**Key Responsibilities:**
*   **Database Interaction**: Fetches text chunks tagged as `LABOUR` from `data/db/labour_rules.db`.
*   **Prompt Management**: Retrieves and compiles prompts dynamically from Langfuse.
*   **Async Processing**: Processes chunks in batches (default 20) with rate limiting to optimize throughput without hitting API quotas.
*   **Retry Logic**: Implements robust retry mechanisms for API timeouts or JSON parsing errors.
*   **Output**: Saves extracted rules as JSON files in `data/rules/`.

**Usage:**
```python
from vistavu.rules.extraction import process_all_chunks_async
import asyncio

asyncio.run(process_all_chunks_async())
```

### 2. `vistavu.dolphin`
**Primary File:** `chat.py`

This module contains the integration code for the Dolphin Visual Language Model (VLM), used for high-fidelity OCR and document parsing.

**Key Components:**
*   **`DOLPHIN` Class**: Wraps the underlying Donut/SwinTransformer models.
*   **Inference**: Provides methods (`chat`) to send images and prompts to the model and receive text responses.
*   **Deployment**: Includes scripts for deploying the model using TensorRT-LLM or vLLM (see `deployment/` subdirectory).

**Usage:**
```python
from vistavu.dolphin.chat import DOLPHIN
from omegaconf import OmegaConf

config = OmegaConf.load("vistavu/dolphin/config/Dolphin.yaml")
model = DOLPHIN(config=config, ckpt_path="path/to/checkpoint")

response = model.chat(
    question="What is the title of this document?",
    image="path/to/document_page.jpg"
)
print(response)
```

## Configuration

Configuration files for the models are located in `vistavu/dolphin/config/`.
API keys and external service configurations are managed via environment variables (see root `README.md`).
