"""Search provider implementation using Brave Search API."""

import os
import json
from typing import Dict, Any, List, Optional

# Try to import optional dependencies
try:
    import httpx
except ImportError:
    httpx = None

class BraveSearchProvider:
    """Search provider using Brave Search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Brave search provider.
        
        Args:
            api_key: Brave Search API key (optional, defaults to BRAVE_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("Brave API key is required. Set BRAVE_API_KEY environment variable or pass as api_key.")
        
        if httpx is None:
            raise ImportError("httpx is required for BraveSearchProvider. Install with 'pip install httpx'")
        
        self.client = httpx.Client(
            base_url="https://api.search.brave.com/",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }
        )
    
    def search(self, query: str, num_results: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform a search query using Brave Search API.
        
        Args:
            query: The search query
            num_results: Number of results to return (default: 5)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            List of search results
        """
        params = {
            "q": query,
            "count": min(num_results, 20),  # API limit is 20
            **kwargs
        }
        
        try:
            response = self.client.get("/res/v1/web/search", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract and process results
            results = []
            for item in data.get("web", {}).get("results", []):
                result = {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "description": item.get("description"),
                    "source": "brave_search"
                }
                results.append(result)
                
            return results
        except Exception as e:
            print(f"Error performing Brave search: {e}")
            return []
    
    def search_with_snippets(self, query: str, num_results: int = 3) -> str:
        """
        Perform a search and return formatted snippets as a string.
        
        Args:
            query: The search query
            num_results: Number of results to include
            
        Returns:
            Formatted search results as a string
        """
        results = self.search(query, num_results)
        
        if not results:
            return f"No results found for query: {query}"
        
        # Format results as a string
        formatted = f"Search results for: {query}\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   URL: {result['url']}\n"
            formatted += f"   {result['description']}\n\n"
            
        return formatted