"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import sys
import json
import time
import subprocess
import threading
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import requests
import openai
import chromadb
from chromadb.config import Settings
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")

# Concurrency and throttling for OpenAI API calls
LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "3"))
LLM_REQUEST_DELAY = float(os.getenv("LLM_REQUEST_DELAY", "0"))
_llm_semaphore = threading.Semaphore(LLM_MAX_CONCURRENCY)

# Chroma collection schema definitions matching SQL tables exactly
CHROMA_COLLECTIONS = {
    "sources": {
        "has_embeddings": False,
        "metadata_schema": {
            "source_id": str,      # Primary key equivalent
            "summary": str,        # Source description
            "total_word_count": int,  # Aggregate word count
            "created_at": str,     # ISO timestamp
            "updated_at": str      # ISO timestamp
        },
        "unique_constraint": ["source_id"]  # Equivalent to SQL primary key
    },
    "crawled_pages": {
        "has_embeddings": True,
        "embedding_function": "openai",  # text-embedding-3-small
        "embedding_dimension": 1536,
        "metadata_schema": {
            "url": str,
            "chunk_number": int,
            "content": str,        # Full text content
            "metadata": dict,      # Original JSON metadata
            "source_id": str,      # Foreign key to sources
            "created_at": str
        },
        "unique_constraint": ["url", "chunk_number"]  # Composite unique key
    },
    "code_examples": {
        "has_embeddings": True,
        "embedding_function": "openai",  # text-embedding-3-small
        "embedding_dimension": 1536,
        "metadata_schema": {
            "url": str,
            "chunk_number": int,
            "content": str,        # Code content
            "summary": str,        # Code description
            "metadata": dict,      # Original JSON metadata
            "source_id": str,      # Foreign key to sources
            "created_at": str
        },
        "unique_constraint": ["url", "chunk_number"]  # Composite unique key
    }
}

def _with_llm_limits(func, *args, **kwargs):
    """Call OpenAI API with concurrency limits and optional delay."""
    _llm_semaphore.acquire()
    try:
        return func(*args, **kwargs)
    finally:
        _llm_semaphore.release()
        if LLM_REQUEST_DELAY > 0:
            time.sleep(LLM_REQUEST_DELAY)

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

def check_chroma_server_heartbeat(host: str, port: int, timeout: int = 5) -> bool:
    """
    Check if a Chroma server is running and responsive.
    
    Args:
        host: Chroma server host
        port: Chroma server port
        timeout: Request timeout in seconds
        
    Returns:
        True if server is responsive, False otherwise
    """
    try:
        # Use Chroma v2 API heartbeat endpoint
        url = f"http://{host}:{port}/api/v2/heartbeat"
        response = requests.get(url, timeout=timeout)
        
        # Check if response is successful and contains expected heartbeat data
        if response.status_code == 200:
            # Chroma v2 heartbeat returns nanosecond timestamp
            data = response.json()
            return isinstance(data, dict) and "nanosecond heartbeat" in str(data)
        
        return False
    except Exception as e:
        print(f"Chroma server heartbeat check failed: {e}")
        return False

def prompt_user_start_chroma_server(host: str, port: int, data_dir: str) -> bool:
    """
    Prompt user to start a Chroma server if none is found.
    
    Args:
        host: Chroma server host
        port: Chroma server port
        data_dir: Data directory for Chroma storage
        
    Returns:
        True if user wants to start server, False otherwise
    """
    print(f"\n⚠ No Chroma server found at {host}:{port}")
    print(f"📁 Data will be stored in: {data_dir}")
    print("\nWould you like to start a Chroma server? (Y/N): ", end="", flush=True)
    
    try:
        response = input().strip().upper()
        return response in ['Y', 'YES']
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return False
    except Exception as e:
        print(f"\nError reading user input: {e}")
        return False

def start_chroma_server(host: str, port: int, data_dir: str) -> bool:
    """
    Start a Chroma server in the background.
    
    Args:
        host: Chroma server host
        port: Chroma server port
        data_dir: Data directory for Chroma storage
        
    Returns:
        True if server started successfully, False otherwise
    """
    try:
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Start Chroma server using the chromadb command
        print(f"🚀 Starting Chroma server at {host}:{port}...")
        print(f"📁 Using data directory: {data_dir}")
        
        # Use subprocess to start the server in background
        cmd = [
            sys.executable, "-m", "chromadb.cli.cli", 
            "run", "--host", host, "--port", str(port), 
            "--path", data_dir
        ]
        
        # Start the process in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a few seconds for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        # Check if server is now responsive
        for attempt in range(10):  # Try for 10 seconds
            if check_chroma_server_heartbeat(host, port):
                print(f"✅ Chroma server started successfully at {host}:{port}")
                return True
            time.sleep(1)
        
        # If we get here, server didn't start properly
        print("⚠ Failed to start Chroma server or server not responsive")
        
        # Try to terminate the process if it's still running
        if process.poll() is None:
            process.terminate()
            
        return False
        
    except Exception as e:
        print(f"⚠ Error starting Chroma server: {e}")
        return False

def validate_collection_embedding_dimension(client: ClientAPI, collection_name: str) -> bool:
    """
    Validate that a collection can accept our embedding dimensions.
    
    Args:
        client: Chroma client instance
        collection_name: Name of the collection to validate
        
    Returns:
        True if dimensions are compatible, False otherwise
    """
    try:
        collection = client.get_collection(collection_name)
        
        # Test with a small embedding to see what dimension is expected
        test_embedding = [0.1] * 1536  # 1536-dimensional test embedding
        test_id = "dimension_test_id"
        
        try:
            # Attempt to add a test document
            collection.add(
                ids=[test_id],
                documents=["test"],
                embeddings=[test_embedding]
            )
            
            # If successful, clean up the test document
            collection.delete(ids=[test_id])
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if "dimension" in error_msg:
                print(f"❌ Collection '{collection_name}' has embedding dimension mismatch: {e}")
                return False
            else:
                # Some other error, re-raise it
                raise e
                
    except Exception as e:
        print(f"❌ Error validating collection '{collection_name}': {e}")
        return False

def prompt_user_create_collections(missing_collections: List[str]) -> bool:
    """
    Prompt user to create missing collections.
    
    Args:
        missing_collections: List of missing collection names
        
    Returns:
        True if user wants to create collections, False otherwise
    """
    print(f"\n⚠ Missing required collections: {', '.join(missing_collections)}")
    print("These collections are required for the system to function properly.")
    print("\nWould you like to create the missing collections? (Y/N): ", end="", flush=True)
    
    try:
        response = input().strip().upper()
        return response in ['Y', 'YES']
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return False
    except Exception as e:
        print(f"\nError reading user input: {e}")
        return False

def create_collections(client: ClientAPI, collection_names: List[str]) -> None:
    """
    Create the specified Chroma collections with proper schema and embedding dimensions.
    
    Args:
        client: Chroma client instance
        collection_names: List of collection names to create
    """
    for name in collection_names:
        if name not in CHROMA_COLLECTIONS:
            raise ValueError(f"Unknown collection: {name}")
        
        config = CHROMA_COLLECTIONS[name]
        
        try:
            if config["has_embeddings"]:
                # For collections with embeddings, we create the collection 
                # without an embedding function and provide embeddings manually
                # This ensures we control the exact dimension (1536)
                collection = client.create_collection(
                    name=name,
                    metadata={
                        "description": f"Collection for {name} with embeddings",
                        "embedding_dimension": config.get("embedding_dimension", 1536),
                        "embedding_function": config.get("embedding_function", "openai")
                    }
                    # Note: We don't specify embedding_function parameter because we provide embeddings manually
                )
                print(f"✅ Created collection '{name}' for {config.get('embedding_dimension', 1536)}D embeddings")
            else:
                # Collections without embeddings (sources)
                collection = client.create_collection(
                    name=name,
                    metadata={"description": f"Collection for {name} metadata only"}
                )
                print(f"✅ Created collection '{name}' (metadata only)")
                
        except Exception as e:
            print(f"❌ Error creating collection '{name}': {e}")
            raise

def ensure_collections_exist_with_validation(client: ClientAPI) -> None:
    """
    Enhanced version of ensure_collections_exist that validates embedding dimensions.
    
    Args:
        client: Chroma client instance
        
    Raises:
        RuntimeError: If collections have dimension mismatches
    """
    # Get existing collections
    existing_collections = {col.name for col in client.list_collections()}
    required_collections = set(CHROMA_COLLECTIONS.keys())
    missing_collections = required_collections - existing_collections
    
    # Create missing collections
    if missing_collections:
        if prompt_user_create_collections(list(missing_collections)):
            print("🔨 Creating missing collections with correct embedding dimensions...")
            create_collections(client, list(missing_collections))
            print("✅ All collections created successfully")
        else:
            print("❌ Required collections are missing and user declined to create them.")
            print("The system cannot function without these collections. Shutting down.")
            sys.exit(1)
    
    # Validate embedding dimensions for collections that should have embeddings
    collections_with_embeddings = [name for name, config in CHROMA_COLLECTIONS.items() 
                                 if config.get("has_embeddings", False)]
    
    dimension_issues = []
    for collection_name in collections_with_embeddings:
        if collection_name in existing_collections:
            if not validate_collection_embedding_dimension(client, collection_name):
                dimension_issues.append(collection_name)
    
    # Handle dimension mismatches
    if dimension_issues:
        print(f"\n❌ CRITICAL ERROR: Embedding dimension mismatches found in collections: {dimension_issues}")
        print("Your existing collections expect 384-dimensional embeddings, but this system uses 1536-dimensional embeddings.")
        print("\nTo fix this issue:")
        print("1. Stop this application")
        print("2. Delete your ChromaDB data directory completely")
        print("3. Restart this application to create fresh collections with correct dimensions")
        print("\nAlternatively, use ChromaDB admin tools to delete the problematic collections.")
        print("❌ Cannot proceed with dimension mismatches.")
        sys.exit(1)
    else:
        print("✅ All collections exist with correct embedding dimensions")

def get_chroma_client() -> ClientAPI:
    """
    Get a Chroma client with server management and collection initialization.
    
    Returns:
        Chroma client instance
        
    Raises:
        RuntimeError: If server is not available and user chooses not to start one
        ValueError: If required environment variables are missing
    """
    host = os.getenv("CHROMA_HOST", "127.0.0.1")
    port = int(os.getenv("CHROMA_PORT", "9000"))
    
    if not host or not port:
        raise ValueError("CHROMA_HOST and CHROMA_PORT must be set in environment variables")
    
    # Check if server is already running
    if not check_chroma_server_heartbeat(host, port):
        # Determine data directory (project root + /data)
        project_root = os.path.abspath(".")
        data_dir = os.path.join(project_root, "data")
        
        # Ask user if they want to start a server
        if prompt_user_start_chroma_server(host, port, data_dir):
            if not start_chroma_server(host, port, data_dir):
                raise RuntimeError(f"Failed to start Chroma server at {host}:{port}")
        else:
            print("❌ Chroma server is required for operation. Shutting down.")
            sys.exit(1)
    else:
        print(f"✅ Connected to existing Chroma server at {host}:{port}")
    
    # Create Chroma client
    client = chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(
            allow_reset=True,
            anonymized_telemetry=False
        )
    )

    # Ensure required collections exist with proper validation
    ensure_collections_exist_with_validation(client)
    
    return client

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
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            
            # Validate that all embeddings have the correct dimension
            for i, embedding in enumerate(embeddings):
                if len(embedding) != 1536:
                    print(f"⚠ Warning: Expected 1536-dimensional embedding, got {len(embedding)} for text {i}")
            
            return embeddings
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
                        embedding = individual_response.data[0].embedding
                        
                        # Validate dimension
                        if len(embedding) != 1536:
                            print(f"⚠ Warning: Expected 1536-dimensional embedding, got {len(embedding)} for text {i}")
                            # Pad or truncate to correct size
                            if len(embedding) < 1536:
                                embedding.extend([0.0] * (1536 - len(embedding)))
                            else:
                                embedding = embedding[:1536]
                        
                        embeddings.append(embedding)
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
        List of floats representing the embedding (guaranteed to be 1536-dimensional)
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
            max_tokens=200
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

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
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."

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
            max_tokens=150
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

# Document storage and retrieval functions for Chroma

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

def add_documents_to_chroma(
    client: ClientAPI,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """
    Add documents to the Chroma crawled_pages collection in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    
    Args:
        client: Chroma client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    import concurrent.futures
    from datetime import datetime
    
    # Get the crawled_pages collection
    collection = client.get_collection("crawled_pages")
    
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    # Delete existing records for these URLs
    try:
        if unique_urls:
            # Query for existing documents with these URLs
            for url in unique_urls:
                try:
                    # Get existing documents for this URL
                    results = collection.get(
                        where={"url": url}
                    )
                    
                    # Delete the documents if they exist
                    if results['ids']:
                        collection.delete(ids=results['ids'])
                        print(f"Deleted {len(results['ids'])} existing documents for URL: {url}")
                        
                except Exception as e:
                    print(f"Error deleting records for URL {url}: {e}")
                    # Continue with the next URL even if one fails
                    
    except Exception as e:
        print(f"Error during batch deletion: {e}")
    
    # Check if contextual embeddings are enabled
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
        
        # Apply contextual embedding to each chunk if enabled
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
        
        # Validate embedding dimensions
        for j, embedding in enumerate(batch_embeddings):
            if len(embedding) != 1536:
                print(f"⚠ Warning: Embedding dimension mismatch for batch item {j}: expected 1536, got {len(embedding)}")
                # Fix the dimension
                if len(embedding) < 1536:
                    embedding.extend([0.0] * (1536 - len(embedding)))
                else:
                    batch_embeddings[j] = embedding[:1536]
        
        # Prepare batch data for Chroma
        batch_ids = []
        batch_documents = []
        batch_embeddings_final = []
        batch_metadatas_final = []
        
        for j in range(len(contextual_contents)):
            # Create unique ID combining URL and chunk number
            doc_id = f"{extract_source_id(batch_urls[j])}_chunk_{batch_chunk_numbers[j]}_{hash(batch_urls[j]) % 10000}"
            
            # Extract source_id from URL
            source_id = extract_source_id(batch_urls[j])
            
            # Prepare metadata for Chroma (matching SQL schema)
            metadata = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": batch_contents[j],  # Store original content in metadata
                "metadata": batch_metadatas[j],  # Original JSON metadata
                "source_id": source_id,
                "created_at": datetime.utcnow().isoformat() + "Z"
            }
            
            batch_ids.append(doc_id)
            batch_documents.append(contextual_contents[j])  # Document text for embedding
            batch_embeddings_final.append(batch_embeddings[j])
            batch_metadatas_final.append(metadata)
        
        # Insert batch into Chroma with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    embeddings=batch_embeddings_final,
                    metadatas=batch_metadatas_final
                )
                print(f"Successfully inserted batch {i//batch_size + 1} of {(len(contents) + batch_size - 1)//batch_size}")
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into Chroma (attempt {retry + 1}/{max_retries}): {e}")
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
                                documents=[batch_documents[k]],
                                embeddings=[batch_embeddings_final[k]],
                                metadatas=[batch_metadatas_final[k]]
                            )
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {batch_urls[k]}: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_ids)} records individually")

def search_documents(
    client: ClientAPI,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Chroma using vector similarity.
    
    Args:
        client: Chroma client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        
    Returns:
        List of matching documents
    """
    try:
        # Get the crawled_pages collection
        collection = client.get_collection("crawled_pages")
        
        # Create embedding for the query
        query_embedding = create_embedding(query)
        
        # Validate query embedding dimension
        if len(query_embedding) != 1536:
            print(f"⚠ Warning: Query embedding dimension mismatch: expected 1536, got {len(query_embedding)}")
            # Fix the dimension
            if len(query_embedding) < 1536:
                query_embedding.extend([0.0] * (1536 - len(query_embedding)))
            else:
                query_embedding = query_embedding[:1536]
        
        # Prepare where clause for filtering
        where_clause = None
        if filter_metadata:
            where_clause = filter_metadata
        
        # Execute the search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=match_count,
            where=where_clause
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                # Calculate similarity score (Chroma returns distances, we want similarity)
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance  # Convert distance to similarity
                
                metadata = results['metadatas'][0][i]
                
                formatted_result = {
                    "id": doc_id,
                    "url": metadata.get("url"),
                    "chunk_number": metadata.get("chunk_number"),
                    "content": metadata.get("content"),
                    "metadata": metadata.get("metadata", {}),
                    "source_id": metadata.get("source_id"),
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

def add_code_examples_to_chroma(
    client: ClientAPI,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20
):
    """
    Add code examples to the Chroma code_examples collection in batches.
    
    Args:
        client: Chroma client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    from datetime import datetime
    
    if not urls:
        return
    
    # Get the code_examples collection
    collection = client.get_collection("code_examples")
        
    # Delete existing records for these URLs
    unique_urls = list(set(urls))
    for url in unique_urls:
        try:
            # Get existing documents for this URL
            results = collection.get(
                where={"url": url}
            )
            
            # Delete the documents if they exist
            if results['ids']:
                collection.delete(ids=results['ids'])
                print(f"Deleted {len(results['ids'])} existing code examples for URL: {url}")
                
        except Exception as e:
            print(f"Error deleting existing code examples for {url}: {e}")
    
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
        
        # Check if embeddings are valid (not all zeros) and have correct dimensions
        valid_embeddings = []
        for k, embedding in enumerate(embeddings):
            # Validate dimension
            if len(embedding) != 1536:
                print(f"⚠ Warning: Code example embedding dimension mismatch for item {k}: expected 1536, got {len(embedding)}")
                # Fix the dimension
                if len(embedding) < 1536:
                    embedding.extend([0.0] * (1536 - len(embedding)))
                else:
                    embedding = embedding[:1536]
            
            if embedding and not all(v == 0.0 for v in embedding):
                valid_embeddings.append(embedding)
            else:
                print(f"Warning: Zero or invalid embedding detected, creating new one...")
                # Try to create a single embedding as fallback
                single_embedding = create_embedding(batch_texts[len(valid_embeddings)])
                valid_embeddings.append(single_embedding)
        
        # Prepare batch data for Chroma
        batch_ids = []
        batch_documents = []
        batch_embeddings_final = []
        batch_metadatas_final = []
        
        for j, embedding in enumerate(valid_embeddings):
            idx = i + j
            
            # Create unique ID combining URL and chunk number
            doc_id = f"{extract_source_id(urls[idx])}_code_{chunk_numbers[idx]}_{hash(urls[idx]) % 10000}"
            
            # Extract source_id from URL
            source_id = extract_source_id(urls[idx])
            
            # Prepare metadata for Chroma (matching SQL schema)
            metadata = {
                "url": urls[idx],
                "chunk_number": chunk_numbers[idx],
                "content": code_examples[idx],  # Code content
                "summary": summaries[idx],  # Code description
                "metadata": metadatas[idx],  # Original JSON metadata
                "source_id": source_id,
                "created_at": datetime.utcnow().isoformat() + "Z"
            }
            
            batch_ids.append(doc_id)
            batch_documents.append(batch_texts[j])  # Combined text for embedding
            batch_embeddings_final.append(embedding)
            batch_metadatas_final.append(metadata)
        
        # Insert batch into Chroma with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    embeddings=batch_embeddings_final,
                    metadatas=batch_metadatas_final
                )
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into Chroma (attempt {retry + 1}/{max_retries}): {e}")
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
                                documents=[batch_documents[k]],
                                embeddings=[batch_embeddings_final[k]],
                                metadatas=[batch_metadatas_final[k]]
                            )
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {urls[i + k]}: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_ids)} records individually")
        
        print(f"Inserted batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size} code examples")

def search_code_examples(
    client: ClientAPI,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples in Chroma using vector similarity.
    
    Args:
        client: Chroma client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results
        
    Returns:
        List of matching code examples
    """
    try:
        # Get the code_examples collection
        collection = client.get_collection("code_examples")
        
        # Create a more descriptive query for better embedding match
        # Since code examples are embedded with their summaries, we should make the query more descriptive
        enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
        
        # Create embedding for the enhanced query
        query_embedding = create_embedding(enhanced_query)
        
        # Validate query embedding dimension
        if len(query_embedding) != 1536:
            print(f"⚠ Warning: Query embedding dimension mismatch: expected 1536, got {len(query_embedding)}")
            # Fix the dimension
            if len(query_embedding) < 1536:
                query_embedding.extend([0.0] * (1536 - len(query_embedding)))
            else:
                query_embedding = query_embedding[:1536]
        
        # Prepare where clause for filtering
        where_clause = {}
        if filter_metadata:
            where_clause.update(filter_metadata)
        if source_id:
            where_clause["source_id"] = source_id
        
        # Use where_clause only if it has content
        where_param = where_clause if where_clause else None
        
        # Execute the search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=match_count,
            where=where_param
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                # Calculate similarity score (Chroma returns distances, we want similarity)
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance  # Convert distance to similarity
                
                metadata = results['metadatas'][0][i]
                
                formatted_result = {
                    "id": doc_id,
                    "url": metadata.get("url"),
                    "chunk_number": metadata.get("chunk_number"),
                    "content": metadata.get("content"),
                    "summary": metadata.get("summary"),
                    "metadata": metadata.get("metadata", {}),
                    "source_id": metadata.get("source_id"),
                    "similarity": similarity
                }
                formatted_results.append(formatted_result)
        
        return formatted_results
        
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []

def update_source_info(client: ClientAPI, source_id: str, summary: str, word_count: int):
    """
    Update or insert source information in the sources collection.
    
    Args:
        client: Chroma client
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    from datetime import datetime
    
    try:
        # Get the sources collection
        collection = client.get_collection("sources")
        
        # Check if source already exists
        try:
            existing = collection.get(
                where={"source_id": source_id}
            )
            
            if existing['ids']:
                # Update existing source - delete and re-add
                collection.delete(ids=existing['ids'])
                print(f"Updated existing source: {source_id}")
            else:
                print(f"Created new source: {source_id}")
                
        except Exception as e:
            print(f"Error checking existing source: {e}")
            print(f"Creating new source: {source_id}")
        
        # Create unique ID for the source
        doc_id = f"source_{hash(source_id) % 100000}"
        
        # Prepare metadata
        metadata = {
            "source_id": source_id,
            "summary": summary,
            "total_word_count": word_count,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Add to collection (sources collection doesn't use embeddings)
        collection.add(
            ids=[doc_id],
            documents=[f"Source: {source_id} - {summary}"],  # Document text for potential future search
            metadatas=[metadata]
        )
            
    except Exception as e:
        print(f"Error updating source {source_id}: {e}")

def get_available_sources(client: ClientAPI) -> List[Dict[str, Any]]:
    """
    Get all available sources from the sources collection.
    
    Args:
        client: Chroma client
        
    Returns:
        List of source information dictionaries
    """
    try:
        # Get the sources collection
        collection = client.get_collection("sources")
        
        # Get all sources
        results = collection.get()
        
        # Format the sources with their details
        sources = []
        if results['metadatas']:
            for metadata in results['metadatas']:
                sources.append({
                    "source_id": metadata.get("source_id"),
                    "summary": metadata.get("summary"),
                    "total_words": metadata.get("total_word_count"),
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at")
                })
        
        # Sort by source_id for consistency
        sources.sort(key=lambda x: x.get("source_id", ""))
        
        return sources
        
    except Exception as e:
        print(f"Error getting available sources: {e}")
        return []
