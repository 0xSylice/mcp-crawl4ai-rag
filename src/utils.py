"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
from urllib.parse import urlparse
import openai
import re
import time
import threading
import requests
from datetime import datetime
import chromadb
from chromadb.config import Settings
import hashlib

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")

# Concurrency and throttling for OpenAI API calls
LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "3"))
LLM_REQUEST_DELAY = float(os.getenv("LLM_REQUEST_DELAY", "0"))
_llm_semaphore = threading.Semaphore(LLM_MAX_CONCURRENCY)

def _with_llm_limits(func, *args, **kwargs):
    """Call OpenAI API with concurrency limits and optional delay."""
    _llm_semaphore.acquire()
    try:
        return func(*args, **kwargs)
    finally:
        _llm_semaphore.release()
        if LLM_REQUEST_DELAY > 0:
            time.sleep(LLM_REQUEST_DELAY)

def check_chroma_server() -> bool:
    """
    Check if Chroma server is running using v2 API heartbeat.

    Returns:
        True if server is accessible, False otherwise
    """
    chroma_host = os.getenv("CHROMA_HOST", "127.0.0.1")
    chroma_port = os.getenv("CHROMA_PORT", "9000")

    try:
        response = requests.get(f"http://{chroma_host}:{chroma_port}/api/v2/heartbeat", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Chroma server health check failed: {e}")
        return False

def check_collections_exist(client: chromadb.ClientAPI) -> Dict[str, bool]:
    """
    Check if required collections exist in Chroma.

    Args:
        client: Chroma client

    Returns:
        Dictionary with collection names and their existence status
    """
    required_collections = ["sources", "crawled_pages", "code_examples"]
    existing_collections = [col.name for col in client.list_collections()]

    return {
        collection: collection in existing_collections
        for collection in required_collections
    }

def verify_collection_schema(client: chromadb.ClientAPI) -> bool:
    """
    Verify that existing collections have the expected schema structure.

    Args:
        client: Chroma client

    Returns:
        True if all collections have valid schema, False otherwise
    """
    try:
        # Check if collections exist
        collections_status = check_collections_exist(client)
        if not all(collections_status.values()):
            return False

        # Verify each collection can be accessed (basic schema check)
        sources_col = client.get_collection("sources")
        crawled_pages_col = client.get_collection("crawled_pages")
        code_examples_col = client.get_collection("code_examples")

        # Check if collections have the expected metadata structure by examining a sample
        # This is a basic check - in a real scenario you might want more thorough validation
        return True
    except Exception as e:
        print(f"Schema verification failed: {e}")
        return False

def create_collections(client: chromadb.ClientAPI) -> bool:
    """
    Create the required collections with proper schema.

    Args:
        client: Chroma client

    Returns:
        True if collections created successfully, False otherwise
    """
    try:
        # Create sources collection (no embeddings, just metadata)
        client.create_collection(
            name="sources",
            metadata={"description": "Source information and summaries"}
        )

        # Create crawled_pages collection with embeddings
        client.create_collection(
            name="crawled_pages",
            metadata={"description": "Document chunks with embeddings"}
        )

        # Create code_examples collection with embeddings
        client.create_collection(
            name="code_examples",
            metadata={"description": "Code examples with embeddings and summaries"}
        )

        print("✓ All collections created successfully")
        return True
    except Exception as e:
        print(f"Failed to create collections: {e}")
        return False

def get_chroma_client() -> chromadb.ClientAPI:
    """
    Get a Chroma client with the host and port from environment variables.

    Returns:
        Chroma client instance
    """
    chroma_host = os.getenv("CHROMA_HOST", "127.0.0.1")
    chroma_port = int(os.getenv("CHROMA_PORT", "9000"))

    # Create client that connects to Chroma server
    client = chromadb.HttpClient(
        host=chroma_host,
        port=chroma_port,
        settings=Settings(
            allow_reset=True,
            anonymized_telemetry=False
        )
    )

    return client

def extract_source_id(url: str) -> str:
    """
    Extract source_id from URL with special handling for GitHub repositories.

    For GitHub URLs, returns the full repo path (e.g., 'github.com/user/repo').
    For other URLs, returns the domain.

    Args:
        url: The URL to extract source_id from

    Returns:
        Source ID string
    """
    parsed_url = urlparse(url)

    # Special handling for GitHub URLs
    if parsed_url.netloc == 'github.com':
        # Extract user/repo from path like '/user/repo' or '/user/repo.git'
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) >= 2:
            user = path_parts[0]
            repo = path_parts[1].replace('.git', '')  # Remove .git if present
            return f"github.com/{user}/{repo}"

    # Default behavior for non-GitHub URLs
    return parsed_url.netloc or parsed_url.path

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.

    Args:
        texts: List of texts to create embeddings for

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []

    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay

    for retry in range(max_retries):
        try:
            response = _with_llm_limits(
                openai.embeddings.create,
                model="text-embedding-3-small",  # Hardcoding embedding model for now
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0

                for i, text in enumerate(texts):
                    try:
                        individual_response = _with_llm_limits(
                            openai.embeddings.create,
                            model="text-embedding-3-small",
                            input=[text]
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except Exception as individual_error:
                        print(f"Failed to create embedding for text {i}: {individual_error}")
                        # Add zero embedding as fallback
                        embeddings.append([0.0] * 1536)

                print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                return embeddings

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.

    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")

    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document>
{full_document[:25000]}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = _with_llm_limits(
            openai.chat.completions.create,
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=200
        )

        # Extract the generated context
        context = response.choices[0].message.content.strip()

        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"

        return contextual_text, True

    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (url, content, full_document)

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

def _create_document_id(url: str, chunk_number: int) -> str:
    """
    Create a unique document ID for Chroma collections.

    Args:
        url: Document URL
        chunk_number: Chunk number

    Returns:
        Unique document ID
    """
    # Create a hash to ensure ID is valid and not too long
    combined = f"{url}_{chunk_number}"
    return hashlib.md5(combined.encode()).hexdigest()

def add_documents_to_vecdb(
    client: chromadb.ClientAPI,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """
    Add documents to the Vector database crawled_pages collection in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.

    Args:
        client: Vector database client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    collection = client.get_collection("crawled_pages")

    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))

    # Delete existing records for these URLs
    try:
        # Get all documents to find matching URLs
        all_docs = collection.get()
        ids_to_delete = []

        for i, doc_metadata in enumerate(all_docs["metadatas"]):
            if doc_metadata.get("url") in unique_urls:
                ids_to_delete.append(all_docs["ids"][i])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} existing documents")
    except Exception as e:
        print(f"Warning: Could not delete existing documents: {e}")

    # Check if MODEL_CHOICE is set for contextual embeddings
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")

    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))

        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]

        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))

            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_MAX_CONCURRENCY) as executor:
                # Submit all tasks and collect results
                future_to_idx = {executor.submit(process_chunk_with_context, arg): idx
                                for idx, arg in enumerate(process_args)}

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])

            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                print(f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}")
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents

        # Create embeddings for the entire batch at once
        batch_embeddings = create_embeddings_batch(contextual_contents)

        # Prepare batch data for Chroma
        batch_ids = []
        batch_docs = []
        batch_metas = []
        batch_embeds = []

        for j in range(len(contextual_contents)):
            # Create unique document ID
            doc_id = _create_document_id(batch_urls[j], batch_chunk_numbers[j])

            # Extract source_id from URL
            source_id = extract_source_id(batch_urls[j])

            # Prepare metadata for Chroma
            chunk_metadata = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "source_id": source_id,
                "chunk_size": len(contextual_contents[j]),
                "created_at": datetime.utcnow().isoformat(),
                **batch_metadatas[j]
            }

            batch_ids.append(doc_id)
            batch_docs.append(contextual_contents[j])  # Store contextual content as document
            batch_metas.append(chunk_metadata)
            batch_embeds.append(batch_embeddings[j])

        # Insert batch into Vector database with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay

        for retry in range(max_retries):
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=batch_embeds
                )
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into Vector database (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for k in range(len(batch_ids)):
                        try:
                            collection.add(
                                ids=[batch_ids[k]],
                                documents=[batch_docs[k]],
                                metadatas=[batch_metas[k]],
                                embeddings=[batch_embeds[k]]
                            )
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {batch_metas[k]['url']}: {individual_error}")

                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_ids)} records individually")

def search_documents(
    client: chromadb.ClientAPI,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Vector database using vector similarity.

    Args:
        client: Vector database client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter

    Returns:
        List of matching documents
    """
    try:
        collection = client.get_collection("crawled_pages")

        # Create embedding for the query
        query_embedding = create_embedding(query)

        # Prepare where clause for filtering
        where_clause = {}
        if filter_metadata:
            where_clause.update(filter_metadata)

        # Execute the search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=match_count,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"]
        )

        # Format results to match original interface
        formatted_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                # Calculate similarity from distance (Chroma uses cosine distance)
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity

                formatted_result = {
                    "id": results["ids"][0][i],
                    "url": results["metadatas"][0][i].get("url"),
                    "chunk_number": results["metadatas"][0][i].get("chunk_number"),
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "source_id": results["metadatas"][0][i].get("source_id"),
                    "similarity": similarity
                }
                formatted_results.append(formatted_result)

        return formatted_results
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.

    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)

    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []

    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")

    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3

    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]

        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]

        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and not ' ' in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()

        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue

        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()

        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3:context_end].strip()

        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
        })

        # Move to next pair (skip the closing backtick we just processed)
        i += 2

    return code_blocks

def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.

    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code

    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = os.getenv("MODEL_CHOICE")

    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""

    try:
        response = _with_llm_limits(
            openai.chat.completions.create,
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=100
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."

def add_code_examples_to_vecdb(
    client: chromadb.ClientAPI,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20
):
    """
    Add code examples to the Vector database code_examples collection in batches.

    Args:
        client: Vector database client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return

    collection = client.get_collection("code_examples")

    # Delete existing records for these URLs
    unique_urls = list(set(urls))
    try:
        # Get all documents to find matching URLs
        all_docs = collection.get()
        ids_to_delete = []

        for i, doc_metadata in enumerate(all_docs["metadatas"]):
            if doc_metadata.get("url") in unique_urls:
                ids_to_delete.append(all_docs["ids"][i])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} existing code examples")
    except Exception as e:
        print(f"Warning: Could not delete existing code examples: {e}")

    # Process in batches
    total_items = len(urls)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_texts = []

        # Create combined texts for embedding (code + summary)
        for j in range(i, batch_end):
            combined_text = f"{code_examples[j]}\n\nSummary: {summaries[j]}"
            batch_texts.append(combined_text)

        # Create embeddings for the batch
        embeddings = create_embeddings_batch(batch_texts)

        # Check if embeddings are valid (not all zeros)
        valid_embeddings = []
        for embedding in embeddings:
            if embedding and not all(v == 0.0 for v in embedding):
                valid_embeddings.append(embedding)
            else:
                print(f"Warning: Zero or invalid embedding detected, creating new one...")
                # Try to create a single embedding as fallback
                single_embedding = create_embedding(batch_texts[len(valid_embeddings)])
                valid_embeddings.append(single_embedding)

        # Prepare batch data for Chroma
        batch_ids = []
        batch_docs = []
        batch_metas = []
        batch_embeds = []

        for j, embedding in enumerate(valid_embeddings):
            idx = i + j

            # Create unique document ID
            doc_id = _create_document_id(urls[idx], chunk_numbers[idx])

            # Extract source_id from URL
            source_id = extract_source_id(urls[idx])

            # Prepare metadata for Chroma
            code_metadata = {
                "url": urls[idx],
                "chunk_number": chunk_numbers[idx],
                "source_id": source_id,
                "summary": summaries[idx],
                "created_at": datetime.utcnow().isoformat(),
                **metadatas[idx]
            }

            batch_ids.append(doc_id)
            batch_docs.append(code_examples[idx])  # Store code as document
            batch_metas.append(code_metadata)
            batch_embeds.append(embedding)

        # Insert batch into Vector database with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay

        for retry in range(max_retries):
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=batch_embeds
                )
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into Vector database (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for k in range(len(batch_ids)):
                        try:
                            collection.add(
                                ids=[batch_ids[k]],
                                documents=[batch_docs[k]],
                                metadatas=[batch_metas[k]],
                                embeddings=[batch_embeds[k]]
                            )
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {batch_metas[k]['url']}: {individual_error}")

                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_ids)} records individually")
        print(f"Inserted batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size} code examples")

def update_source_info(client: chromadb.ClientAPI, source_id: str, summary: str, word_count: int):
    """
    Update or insert source information in the sources collection.

    Args:
        client: Vector database client
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    try:
        collection = client.get_collection("sources")

        # Check if source already exists
        try:
            existing = collection.get(ids=[source_id])
            source_exists = len(existing["ids"]) > 0
        except:
            source_exists = False

        current_time = datetime.utcnow().isoformat()

        if source_exists:
            # Update existing source by deleting and re-adding
            collection.delete(ids=[source_id])

        # Add/re-add the source
        collection.add(
            ids=[source_id],
            documents=[source_id],  # Use source_id as document content
            metadatas=[{
                "source_id": source_id,
                "summary": summary,
                "total_word_count": word_count,
                "created_at": current_time if not source_exists else existing["metadatas"][0].get("created_at", current_time),
                "updated_at": current_time
            }]
        )

        action = "Updated" if source_exists else "Created new"
        print(f"{action} source: {source_id}")

    except Exception as e:
        print(f"Error updating source {source_id}: {e}")

def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.

    This function uses the OpenAI API to generate a concise summary of the source content.

    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary

    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"

    if not content or len(content.strip()) == 0:
        return default_summary

    # Get the model choice from environment variables
    model_choice = os.getenv("MODEL_CHOICE")

    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content

    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""

    try:
        # Call the OpenAI API to generate the summary
        response = _with_llm_limits(
            openai.chat.completions.create,
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=150
        )

        # Extract the generated summary
        summary = response.choices[0].message.content.strip()

        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary

    except Exception as e:
        print(f"Error generating summary with LLM for {source_id}: {e}. Using default summary.")
        return default_summary

def search_code_examples(
    client: chromadb.ClientAPI,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples in Vector database using vector similarity.

    Args:
        client: Vector database client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results

    Returns:
        List of matching code examples
    """
    try:
        collection = client.get_collection("code_examples")

        # Create a more descriptive query for better embedding match
        # Since code examples are embedded with their summaries, we should make the query more descriptive
        enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"

        # Create embedding for the enhanced query
        query_embedding = create_embedding(enhanced_query)

        # Prepare where clause for filtering
        where_clause = {}
        if filter_metadata:
            where_clause.update(filter_metadata)
        if source_id:
            where_clause["source_id"] = source_id

        # Execute the search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=match_count,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"]
        )

        # Format results to match original interface
        formatted_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                # Calculate similarity from distance (Chroma uses cosine distance)
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity

                formatted_result = {
                    "id": results["ids"][0][i],
                    "url": results["metadatas"][0][i].get("url"),
                    "chunk_number": results["metadatas"][0][i].get("chunk_number"),
                    "content": results["documents"][0][i],
                    "summary": results["metadatas"][0][i].get("summary"),
                    "metadata": results["metadatas"][0][i],
                    "source_id": results["metadatas"][0][i].get("source_id"),
                    "similarity": similarity
                }
                formatted_results.append(formatted_result)

        return formatted_results
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []

# Compatibility aliases for old function names
add_documents_to_supabase = add_documents_to_vecdb
add_code_examples_to_supabase = add_code_examples_to_vecdb
get_supabase_client = get_chroma_client
