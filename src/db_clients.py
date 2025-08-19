"""
Database client abstraction layer for handling RAG storage.

This module provides a unified interface for interacting with different vector database backends,
allowing the application to switch between Supabase and ChromaDB seamlessly.
"""
import os
import sys
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import chromadb
import requests
from chromadb.config import Settings
from supabase import Client, create_client
from pathlib import Path

# Add project root to path to allow absolute imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import re

def _get_required_collections(sql_file_path: Path) -> List[str]:
    """Parse the SQL file to get table names, which correspond to Chroma collections."""
    if not sql_file_path.exists():
        print(f"Warning: SQL file not found at {sql_file_path}. Cannot determine required collections.")
        return ["crawled_pages", "code_examples", "sources"]
    
    try:
        with open(sql_file_path, 'r') as f:
            content = f.read()
        
        # Regex to find 'create table' statements and extract table names
        table_names = re.findall(r'create table\s+(?:if not exists\s+)?(\w+)', content, re.IGNORECASE)
        
        # Filter out any unwanted matches if necessary, though the regex should be specific enough
        collections = [name for name in table_names if name]
        
        if not collections:
            print(f"Warning: No 'create table' statements found in {sql_file_path}. Using default collections.")
            return ["crawled_pages", "code_examples", "sources"]
            
        return collections
        
    except Exception as e:
        print(f"Error reading or parsing SQL file {sql_file_path}: {e}")
        # Fallback to default list if parsing fails
        return ["crawled_pages", "code_examples", "sources"]

class DBClient(ABC):
    """Abstract base class for a database client."""

    @abstractmethod
    def initialize(self):
        """Initialize the database, ensuring tables/collections exist."""
        pass

    @abstractmethod
    def add_documents(self, urls: List[str], chunk_numbers: List[int], contents: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add documentation chunks to the database."""
        pass

    @abstractmethod
    def add_code_examples(self, urls: List[str], chunk_numbers: List[int], contents: List[str], summaries: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add code examples to the database."""
        pass
    
    @abstractmethod
    def search_documents(self, query_embedding: List[float], match_count: int, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for documents using vector similarity."""
        pass

    @abstractmethod
    def search_code_examples(self, query_embedding: List[float], match_count: int, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for code examples using vector similarity."""
        pass

    @abstractmethod
    def get_available_sources(self) -> List[Dict[str, Any]]:
        """Get all available sources from the sources table/collection."""
        pass
    
    @abstractmethod
    def update_source_info(self, source_id: str, summary: str, word_count: int):
        """Update or insert source information."""
        pass

class SupabaseDBClient(DBClient):
    """Supabase database client."""
    def __init__(self):
        self.client: Client = self._get_client()

    def _get_client(self) -> Client:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set for Supabase engine")
        return create_client(url, key)

    def initialize(self):
        # Supabase initialization is handled by its backend and SQL scripts.
        print("Supabase client initialized. Ensure database schema is created.")

    def _insert_with_fallback(self, table_name: str, batch_data: List[Dict[str, Any]]):
        """Insert a batch with retry logic and fallback to individual inserts."""
        try:
            self.client.table(table_name).insert(batch_data).execute()
        except Exception as e:
            print(f"Error inserting batch into {table_name}: {e}. Retrying individually.")
            for record in batch_data:
                try:
                    self.client.table(table_name).insert(record).execute()
                except Exception as individual_error:
                    print(f"Failed to insert individual record for URL {record.get('url')}: {individual_error}")

    def add_documents(self, urls: List[str], chunk_numbers: List[int], contents: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        unique_urls = list(set(urls))
        if unique_urls:
            try:
                self.client.table("crawled_pages").delete().in_("url", unique_urls).execute()
            except Exception as e:
                print(f"Batch delete failed for crawled_pages: {e}")

        batch_data = []
        for i in range(len(contents)):
            batch_data.append({
                "url": urls[i],
                "chunk_number": chunk_numbers[i],
                "content": contents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i],
                "source_id": metadatas[i].get("source")
            })
        
        self._insert_with_fallback("crawled_pages", batch_data)

    def add_code_examples(self, urls: List[str], chunk_numbers: List[int], contents: List[str], summaries: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        unique_urls = list(set(urls))
        if unique_urls:
            try:
                self.client.table('code_examples').delete().in_('url', unique_urls).execute()
            except Exception as e:
                print(f"Batch delete failed for code_examples: {e}")
                
        batch_data = []
        for i in range(len(contents)):
            batch_data.append({
                'url': urls[i],
                'chunk_number': chunk_numbers[i],
                'content': contents[i],
                'summary': summaries[i],
                'metadata': metadatas[i],
                'source_id': metadatas[i].get("source"),
                'embedding': embeddings[i]
            })
            
        self._insert_with_fallback("code_examples", batch_data)

    def search_documents(self, query_embedding: List[float], match_count: int, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = {'query_embedding': query_embedding, 'match_count': match_count}
        if filter_metadata:
            params['filter'] = filter_metadata
        result = self.client.rpc('match_crawled_pages', params).execute()
        return result.data

    def search_code_examples(self, query_embedding: List[float], match_count: int, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = {'query_embedding': query_embedding, 'match_count': match_count}
        if filter_metadata:
            params['filter'] = filter_metadata
        result = self.client.rpc('match_code_examples', params).execute()
        return result.data
        
    def get_available_sources(self) -> List[Dict[str, Any]]:
        result = self.client.from_('sources').select('*').order('source_id').execute()
        return result.data if result.data else []

    def update_source_info(self, source_id: str, summary: str, word_count: int):
        result = self.client.table('sources').update({
            'summary': summary,
            'total_word_count': word_count,
            'updated_at': 'now()'
        }).eq('source_id', source_id).execute()
        
        if not result.data:
            self.client.table('sources').insert({
                'source_id': source_id,
                'summary': summary,
                'total_word_count': word_count
            }).execute()

class ChromaDBClient(DBClient):
    """ChromaDB client."""
    def __init__(self):
        self.host = os.getenv("CHROMA_HOST", "127.0.0.1")
        self.port = int(os.getenv("CHROMA_PORT", "8000"))
        self.client: Optional[chromadb.HttpClient] = None

    def ensure_server_running(self):
        """Check if the ChromaDB server is running and prompt the user to start it if not."""
        if self._is_server_running():
            self.client = chromadb.HttpClient(host=self.host, port=self.port, settings=Settings(anonymized_telemetry=False))
            print(f"INFO:     Chroma server running at: http://{self.host}:{self.port}")
            return

        prompt = f"ChromaDB server not found at http://{self.host}:{self.port}. Start a local, persistent server? (y/n): "
        choice = input(prompt).lower().strip()
        
        if choice == 'n':
            print("ChromaDB server is required. Shutting down.")
            sys.exit(0)
        
        print("Starting local ChromaDB server.")
        print("DB will be stored in 'data' directory.")
        data_path = str(project_root / "data")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        # This will use a local, file-based ChromaDB instance.
        # Note: A separate process for the server is not started here.
        # For a true HTTP server, the user would need to run `chroma run --path /path/to/data`
        self.client = chromadb.PersistentClient(path=data_path, settings=Settings(anonymized_telemetry=False))

    def initialize(self):
        if not self.client:
            raise ConnectionError("ChromaDB client not initialized. Call ensure_server_running() first.")

        # Ensure collections exist based on the SQL schema
        sql_file_path = project_root / 'crawled_pages.sql'
        required_collections = _get_required_collections(sql_file_path)
        
        try:
            collection_names = [c.name for c in self.client.list_collections()]
        except Exception as e:
            print(f"Error listing ChromaDB collections: {e}")
            print("Please ensure your ChromaDB server is running and accessible.")
            sys.exit(1)
            
        missing_collections = [coll for coll in required_collections if coll not in collection_names]

        if missing_collections:
            print(f"The following required Chroma collections are missing: {', '.join(missing_collections)}")
            prompt = "Would you like to create them now? (y/n): "
            choice = input(prompt).lower().strip()

            if choice == 'y':
                for coll_name in missing_collections:
                    try:
                        self.client.create_collection(name=coll_name)
                        print(f"Chroma collection '{coll_name}' created.")
                    except Exception as e:
                        print(f"Error creating Chroma collection '{coll_name}': {e}")
                        sys.exit(1)
                print("All required Chroma collections have been created.")
            else:
                print("Required Chroma collections are missing. Shutting down.")
                sys.exit(0)
        else:
            pass

    def _is_server_running(self) -> bool:
        url = f"http://{self.host}:{self.port}/api/v2/heartbeat"
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def add_documents(self, urls: List[str], chunk_numbers: List[int], contents: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        collection = self.client.get_collection("crawled_pages")
        ids = [f"{url}_{cn}" for url, cn in zip(urls, chunk_numbers)]
        
        # Delete existing documents before adding new ones
        if ids:
            collection.delete(ids=ids)

        try:
            collection.add(ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas)
        except Exception as e:
            print(f"Chroma batch add failed: {e}. Retrying individually.")
            for i in range(len(ids)):
                try:
                    collection.add(ids=[ids[i]], embeddings=[embeddings[i]], documents=[contents[i]], metadatas=[metadatas[i]])
                except Exception as individual_error:
                    print(f"Failed to insert individual Chroma document {ids[i]}: {individual_error}")

    def add_code_examples(self, urls: List[str], chunk_numbers: List[int], contents: List[str], summaries: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        collection = self.client.get_collection("code_examples")
        ids = [f"{url}_{cn}" for url, cn in zip(urls, chunk_numbers)]

        if ids:
            collection.delete(ids=ids)

        for i, meta in enumerate(metadatas):
            meta['summary'] = summaries[i]
        
        try:
            collection.add(ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas)
        except Exception as e:
            print(f"Chroma batch add for code examples failed: {e}. Retrying individually.")
            for i in range(len(ids)):
                try:
                    collection.add(ids=[ids[i]], embeddings=[embeddings[i]], documents=[contents[i]], metadatas=[metadatas[i]])
                except Exception as individual_error:
                    print(f"Failed to insert individual Chroma code example {ids[i]}: {individual_error}")
    
    def search_documents(self, query_embedding: List[float], match_count: int, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        collection = self.client.get_collection("crawled_pages")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=match_count,
            where=filter_metadata if filter_metadata else {}
        )
        
        # Format results to match Supabase output
        formatted = []
        if results and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                formatted.append({
                    "id": doc_id,
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity": 1 - results['distances'][0][i] # Chroma uses L2 distance, convert to cosine similarity like Supabase
                })
        return formatted

    def search_code_examples(self, query_embedding: List[float], match_count: int, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        collection = self.client.get_collection("code_examples")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=match_count,
            where=filter_metadata if filter_metadata else {}
        )
        # Format results
        formatted = []
        if results and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                meta = results['metadatas'][0][i]
                formatted.append({
                    "id": doc_id,
                    "content": results['documents'][0][i], # code is stored in documents
                    "summary": meta.get('summary', ''),
                    "metadata": meta,
                    "similarity": 1 - results['distances'][0][i]
                })
        return formatted
        
    def get_available_sources(self) -> List[Dict[str, Any]]:
        collection = self.client.get_collection("sources")
        # In Chroma, we get all items. 'limit' can be adjusted if needed.
        results = collection.get()
        
        formatted = []
        if results and results['ids']:
            for i, doc_id in enumerate(results['ids']):
                meta = results['metadatas'][i]
                formatted.append({
                    "source_id": doc_id,
                    "summary": meta.get('summary'),
                    "total_words": meta.get('total_word_count'),
                    "created_at": meta.get('created_at'),
                    "updated_at": meta.get('updated_at'),
                })
        return formatted

    def update_source_info(self, source_id: str, summary: str, word_count: int):
        from datetime import datetime, timezone
        collection = self.client.get_collection("sources")
        
        # Chroma's add/upsert is idempotent. Using 'add' with the same ID will update.
        collection.add(
            ids=[source_id],
            metadatas=[{
                "summary": summary,
                "total_word_count": word_count,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }],
            documents=[f"Source information for {source_id}"] # Document content is required but not used
        )

def get_db_client() -> DBClient:
    """
    Factory function to get the appropriate database client based on environment variables.
    """
    engine = os.getenv("DATABASE_ENGINE", "supabase").lower()
    if engine == "chroma":
        print("Using ChromaDB as the database engine.")
        return ChromaDBClient()
    elif engine == "supabase":
        print("Using Supabase as the database engine.")
        return SupabaseDBClient()
    else:
        raise ValueError(f"Unsupported DATABASE_ENGINE: {engine}. Choose 'supabase' or 'chroma'.")