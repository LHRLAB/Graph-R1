"""
Search tool implementation for simulating internet searches
"""

import time
import random
from typing import Dict, List, Any, Optional

from agent.tool.tool_base import Tool

# from txtai.embeddings import Embeddings
import faiss
from FlagEmbedding import FlagAutoModel
import json
import requests

class SearchTool(Tool):
    """
    Tool for simulating internet searches using the NeuML/txtai-wikipedia model
    """
    
    def __init__(self):
        """
        Initialize the search tool
        
        Args:
            search_db: Custom search database, if None, use default
        """
        name = "search"
        description = "Search for information on the internet using Wikipedia as a knowledge source."
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                # "limit": {
                #     "type": "integer",
                #     "description": "Maximum number of results to return (default: 5)"
                # }
            },
            "required": ["query"]
        }
        
        super().__init__(name, description, parameters)
        self.search_api_url = "http://localhost:8001/search"
    
    def execute(self, args: Dict) -> str:
        """
        Execute search query
        
        Args:
            args: Tool parameters, containing:
                - "query": search query string
                - "limit": optional int to limit number of results
            
        Returns:
            Formatted search results
        """
        pass
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        queries = [x["query"] for x in args_list]
        results_str = requests.post(self.search_api_url, json={"queries": queries}).json()
        return results_str
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # valid tool call
        if "results" in result:
            return 0.0
        else:
            return 0.0