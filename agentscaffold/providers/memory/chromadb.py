"""Memory provider implementation using ChromaDB."""

import os
import json
import hashlib
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple

# Try to import optional dependencies
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

class ChromaDBMemoryProvider:
    """Memory provider using ChromaDB for vector storage."""
    
    def __init__(
        self,
        collection_name: str = "agent_memory",
        persist_directory: Optional[str] = None,
        embedding_provider: str = "openai",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the ChromaDB memory provider.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist data (defaults to ./.chroma)
            embedding_provider: Provider for embeddings (openai or huggingface)
            api_key: API key for embedding provider
            **kwargs: Additional parameters for ChromaDB
        """
        if chromadb is None:
            raise ImportError("chromadb is required. Install with 'pip install chromadb'")
        
        # Set up the persist directory
        self.persist_directory = persist_directory or os.path.join(os.getcwd(), ".chroma")
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Set up the embedding function based on provider
        self.embedding_provider = embedding_provider
        self.api_key = api_key or os.environ.get(
            "OPENAI_API_KEY" if embedding_provider == "openai" else 
            "HUGGINGFACE_API_KEY" if embedding_provider == "huggingface" else
            None
        )
        
        # Initialize the embedding function
        if self.embedding_provider == "openai":
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name="text-embedding-ada-002"
            )
        elif self.embedding_provider == "huggingface":
            self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
                api_key=self.api_key,
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        else:
            self.embedding_function = None  # Use ChromaDB's default
        
        # Get or create the collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Created new collection: {collection_name}")
    
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a text entry to memory with optional metadata.
        
        Args:
            text: The text to store
            metadata: Optional metadata to store with the text
            
        Returns:
            ID of the stored entry
        """
        # Generate a deterministic ID based on content
        entry_id = hashlib.md5(text.encode()).hexdigest()
        
        # Add the document to the collection
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[entry_id]
        )
        
        return entry_id
    
    def add_many(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add multiple text entries to memory.
        
        Args:
            texts: List of texts to store
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of IDs for the stored entries
        """
        if not texts:
            return []
            
        # Generate IDs for each text
        ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]
        
        # Use empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("Length of texts and metadatas must match")
        
        # Add the documents to the collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar entries in memory.
        
        Args:
            query: Query text to search for
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of results with text and metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filters
        )
        
        # Format the results
        formatted_results = []
        if not results or not results.get('documents'):
            return formatted_results
            
        documents = results['documents'][0]  # First query result
        metadatas = results['metadatas'][0]
        distances = results.get('distances', [[]] * len(documents))[0]
        ids = results['ids'][0]
        
        for i, doc in enumerate(documents):
            formatted_results.append({
                'text': doc,
                'metadata': metadatas[i],
                'distance': distances[i] if i < len(distances) else None,
                'id': ids[i]
            })
            
        return formatted_results
    
    def get_context(self, query: str, n_results: int = 3, as_string: bool = True) -> Union[str, List[Dict[str, Any]]]:
        """
        Get context from memory based on a query.
        
        Args:
            query: Query text
            n_results: Number of results to include
            as_string: Whether to return results as a formatted string
            
        Returns:
            Formatted context string or list of results
        """
        results = self.search(query, n_results=n_results)
        
        if not results:
            return "" if as_string else []
        
        if as_string:
            context = f"Relevant context for query: '{query}'\n\n"
            for i, result in enumerate(results, 1):
                context += f"{i}. {result['text']}\n"
                if result['metadata']:
                    context += f"   Metadata: {json.dumps(result['metadata'])}\n"
                context += "\n"
            return context
        else:
            return results
    
    def delete(self, entry_id: str) -> bool:
        """
        Delete an entry from memory by ID.
        
        Args:
            entry_id: ID of the entry to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[entry_id])
            return True
        except Exception as e:
            print(f"Error deleting entry {entry_id}: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all entries from memory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=None)  # Delete all
            return True
        except Exception as e:
            print(f"Error clearing memory: {e}")
            return False