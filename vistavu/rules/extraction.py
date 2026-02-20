"""
Extraction Rules Module

This module handles the extraction of rules from labour law text chunks using Google's Gemini API
and Langfuse for prompt management and tracing. It processes chunks asynchronously to optimize throughput.
"""

import asyncio
import json
import re
import sqlite3
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import google.generativeai as genai
from langfuse import Langfuse

# ==================== CONFIGURATION ====================

# Resolve project root relative to this file: vistavu/rules/extraction.py -> root is ../../
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "db" / "labour_rules.db"
OUTPUT_DIR = PROJECT_ROOT / "data" / "rules" / "extracted_rules"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TODO: Move these sensitive keys to environment variables or a secure configuration file.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not found. API calls will fail.")
if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
    print("⚠️  Langfuse keys not found. Tracing will be disabled or fail.")


# API Configuration - OPTIMIZED FOR 25 RPM WITH ASYNC
MAX_RETRIES = 4
RETRY_DELAY = 5
API_TIMEOUT = 300  # 5 minutes for complex/large chunks
BATCH_SIZE = 20    # Process 20 chunks concurrently (under 25 RPM limit)
BATCH_DELAY = 60   # Wait 60s between batches

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Langfuse
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)


# ==================== GET LABOUR CHUNKS ====================

def get_labour_chunks() -> List[Tuple[Any, ...]]:
    """
    Fetch only LABOUR chunks from the SQLite database.

    Returns:
        List[Tuple[Any, ...]]: A list of tuples containing (chunk_id, chunk_file_name, chunk_text).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            chunk_id,
            chunk_file_name,
            chunk_text
        FROM chunks
        WHERE third_level = 'LABOUR'
        ORDER BY chunk_file_name
    """)

    chunks = cursor.fetchall()
    conn.close()

    return chunks


# ==================== FETCH PROMPT FROM LANGFUSE ====================

def get_prompt_from_langfuse(chunk_text: str, chunk_file_name: str) -> Tuple[Any, Any]:
    """
    Fetch prompt template from Langfuse and compile with variables.

    Args:
        chunk_text (str): The text content of the chunk.
        chunk_file_name (str): The filename of the chunk.

    Returns:
        Tuple[Any, Any]: A tuple containing the compiled prompt and the prompt object.

    Raises:
        Exception: If there is an error fetching or compiling the prompt.
    """
    try:
        prompt = langfuse.get_prompt("rule-extraction-system", label="production")
        compiled_prompt = prompt.compile(
            chunk_text=chunk_text,
            chunk_file_name=chunk_file_name
        )
        return compiled_prompt, prompt
    except Exception as e:
        print(f"❌ Error fetching prompt from Langfuse: {e}")
        raise


# ==================== EXTRACT JSON FROM RESPONSE ====================

def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON array from response text, handling various formats.

    Args:
        response_text (str): The raw text response from the LLM.

    Returns:
        str: The extracted JSON string.
    """
    # Clean JSON if wrapped in code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    # Find JSON array using regex
    json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
    if json_match:
        return json_match.group(0).strip()

    return response_text.strip()


# ==================== ASYNC PROCESS SINGLE CHUNK ====================

async def process_chunk_async(
    chunk_id: Any,
    chunk_file_name: str,
    chunk_text: str,
    model: genai.GenerativeModel,
    prompt_version: int,
    chunk_index: int
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    Process a single chunk asynchronously with retry logic.

    Args:
        chunk_id (Any): The ID of the chunk.
        chunk_file_name (str): The filename of the chunk.
        chunk_text (str): The text content.
        model (genai.GenerativeModel): The Gemini model instance.
        prompt_version (int): The version of the prompt being used.
        chunk_index (int): The index of the chunk in the batch.

    Returns:
        Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
            A tuple containing the list of extracted rules (or None on failure)
            and an error detail dictionary (or None on success).
    """

    print(f"[{chunk_index}] Processing: {chunk_file_name}")
    print(f"   Text length: {len(chunk_text)} characters")

    # Get prompt from Langfuse
    try:
        compiled_prompt, prompt_obj = get_prompt_from_langfuse(chunk_text, chunk_file_name)
    except Exception as e:
        error_msg = f"Prompt fetch error: {e}"
        print(f"   ❌ {error_msg}")
        return None, {
            "file": chunk_file_name,
            "length": len(chunk_text),
            "error": error_msg
        }

    # Retry logic
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start_time = time.time()

            # ASYNC API CALL WITH TIMEOUT
            response = await model.generate_content_async(
                compiled_prompt,
                request_options={'timeout': API_TIMEOUT}  # 5 minutes timeout
            )

            processing_time = time.time() - start_time
            response_text = response.text

            # Extract and parse JSON
            cleaned_json = extract_json_from_response(response_text)
            extracted_rules = json.loads(cleaned_json)

            # Add chunk metadata
            for rule in extracted_rules:
                rule['chunk_id'] = chunk_id
                rule['source_chunk'] = chunk_file_name

            # Success
            if attempt > 1:
                print(f"   ✅ Succeeded on attempt {attempt} ({processing_time:.1f}s)")
            else:
                print(f"   ✅ Extracted {len(extracted_rules)} rules ({processing_time:.1f}s)")

            return extracted_rules, None

        except json.JSONDecodeError as e:
            error_msg = f"JSON Parse Error: {e}"
            if attempt < MAX_RETRIES:
                print(f"   ⚠️ Attempt {attempt} failed: {error_msg}")
                print(f"   🔄 Retrying in {RETRY_DELAY * attempt} seconds...")
                await asyncio.sleep(RETRY_DELAY * attempt)
                continue

            print(f"   ❌ {error_msg}")
            return None, {
                "file": chunk_file_name,
                "length": len(chunk_text),
                "error": error_msg
            }

        except genai.types.generation_types.BlockedPromptException as e:
            error_msg = f"Blocked by safety filters: {e}"
            print(f"   ❌ {error_msg}")
            return None, {
                "file": chunk_file_name,
                "length": len(chunk_text),
                "error": error_msg
            }

        except Exception as e:
            error_msg = str(e)

            # Check if it's a timeout error
            if "504" in error_msg or "499" in error_msg or "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_DELAY * attempt
                    print(f"   ⚠️ Attempt {attempt} - Timeout (will retry)")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Timeout after {MAX_RETRIES} attempts (chunk too complex)"

            print(f"   ❌ {error_msg}")
            return None, {
                "file": chunk_file_name,
                "length": len(chunk_text),
                "error": error_msg
            }

    # All retries exhausted
    return None, {
        "file": chunk_file_name,
        "length": len(chunk_text),
        "error": f"Failed after {MAX_RETRIES} attempts"
    }


# ==================== MAIN ASYNC PROCESSING ====================

async def process_all_chunks_async():
    """Process all chunks in batches with async concurrency"""

    # Initialize model
    model = genai.GenerativeModel('gemini-2.5-pro')

    # Get chunks
    print("📂 Fetching LABOUR chunks from database...")
    labour_chunks = get_labour_chunks()
    print(f"✅ Found {len(labour_chunks)} LABOUR chunks\n")

    if not labour_chunks:
        print("❌ No LABOUR chunks found!")
        return

    # Verify prompt
    print("🔍 Verifying prompt in Langfuse...")
    try:
        test_prompt = langfuse.get_prompt("rule-extraction-system", label="production")
        print(f"✅ Prompt found: version {test_prompt.version}\n")
    except Exception as e:
        print(f"❌ Cannot find prompt 'rule-extraction-system' in Langfuse!")
        print(f"   Error: {e}")
        return

    print(f"⚙️  Configuration (ASYNC with {BATCH_SIZE} concurrent requests):")
    print(f"   - API Timeout: {API_TIMEOUT}s ({API_TIMEOUT//60} minutes)")
    print(f"   - Batch Size: {BATCH_SIZE} chunks at a time")
    print(f"   - Rate Limit: 25 RPM")
    print(f"   - Estimated time: ~{(len(labour_chunks) // BATCH_SIZE + 1) * BATCH_DELAY // 60} minutes\n")

    all_extracted_rules = []
    failed_chunk_details = []
    successful_chunks = 0
    failed_chunks = 0

    # Process in batches to respect rate limits
    total_chunks = len(labour_chunks)
    for batch_start in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_chunks)
        batch = labour_chunks[batch_start:batch_end]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (total_chunks // BATCH_SIZE) + (1 if total_chunks % BATCH_SIZE else 0)

        print(f"{'='*60}")
        print(f"📦 BATCH {batch_num}/{total_batches} (Processing chunks {batch_start+1}-{batch_end})")
        print(f"{'='*60}\n")

        # Create async tasks for this batch
        tasks = []
        for i, (chunk_id, chunk_file_name, chunk_text) in enumerate(batch):
            chunk_index = batch_start + i + 1
            task = process_chunk_async(
                chunk_id, chunk_file_name, chunk_text,
                model, test_prompt.version, chunk_index
            )
            tasks.append(task)

        # Execute all tasks in batch concurrently
        results = await asyncio.gather(*tasks)

        # Process results
        for extracted_rules, error_detail in results:
            if extracted_rules is not None:
                all_extracted_rules.extend(extracted_rules)
                successful_chunks += 1
            else:
                failed_chunks += 1
                if error_detail:
                    failed_chunk_details.append(error_detail)

        print(f"\n✅ Batch {batch_num} complete: {successful_chunks}/{batch_end} successful\n")

        # Wait between batches (except after last batch)
        if batch_end < total_chunks:
            print(f"⏸️  Waiting {BATCH_DELAY}s before next batch (rate limiting)...\n")
            await asyncio.sleep(BATCH_DELAY)

    # Save results
    output_file = OUTPUT_DIR / "labour_rules_extracted.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_rules, f, indent=2, ensure_ascii=False)

    # Filter and save LABOR_WORKED_TIME rules
    labor_worked_time_rules = [
        rule for rule in all_extracted_rules
        if rule.get('primary_topic') == 'LABOR_WORKED_TIME'
    ]
    labor_worked_time_file = OUTPUT_DIR / "LABOR_WORKED_TIME_RULES.json"
    with open(labor_worked_time_file, 'w', encoding='utf-8') as f:
        json.dump(labor_worked_time_rules, f, indent=2, ensure_ascii=False)

    if failed_chunk_details:
        failed_file = OUTPUT_DIR / "failed_chunks_detailed.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_chunk_details, f, indent=2)

    # Flush Langfuse
    langfuse.flush()

    # Print summary
    print("\n" + "="*60)
    print("📊 EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total chunks processed: {total_chunks}")
    print(f"✅ Successful: {successful_chunks}")
    print(f"❌ Failed: {failed_chunks}")
    print(f"📝 Total rules extracted: {len(all_extracted_rules)}")
    print(f"💾 Results saved to: {output_file}")
    print(f"📝 LABOR_WORKED_TIME rules: {len(labor_worked_time_rules)}")
    print(f"💾 Filtered rules saved to: {labor_worked_time_file}")

    if failed_chunk_details:
        print(f"\n⚠️  FAILED CHUNKS:")
        print(f"   Details saved to: {failed_file}")
        print(f"   Largest failed chunks:")
        for detail in sorted(failed_chunk_details, key=lambda x: x['length'], reverse=True)[:5]:
            print(f"   - {detail['file'][:60]}... ({detail['length']:,} chars)")
            print(f"     Error: {detail['error'][:80]}")

    print(f"\n🔗 View in Langfuse: {LANGFUSE_HOST}")
    print("="*60)


# ==================== ENTRY POINT ====================

def main():
    """Entry point - runs async processing"""
    asyncio.run(process_all_chunks_async())


# Run the extraction
if __name__ == "__main__":
    main()
