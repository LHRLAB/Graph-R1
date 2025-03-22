"""
Search tool implementation for simulating internet searches
"""

import time
import random
from typing import Dict, List, Any, Optional
import requests
import os
import json

from agent.tool.tool_base import Tool

class WikiSearchTool(Tool):
    """
    Tool for searching Wikipedia using the wiki_search_api
    """
    
    def __init__(self):
        """
        Initialize the search tool
        """
        name = "search"
        description = "Search for information using Wikipedia as a knowledge source."
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
        
        # API配置
        self.api_url = os.environ.get("WIKI_SEARCH_API_URL", "http://localhost:8000")
        print(f"[DEBUG] Wiki Search API URL: {self.api_url}")
        
        # 禁用代理，避免代理问题
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        
        # 检查API是否可用
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code == 200:
                print("[DEBUG] Wiki Search API is available")
            else:
                print(f"[WARNING] Wiki Search API health check failed: {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Failed to connect to Wiki Search API: {e}")
    
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
        query = args.get("query", "").strip()
        limit = args.get("limit", 5)
        
        try:
            # 调用API进行搜索
            response = requests.get(
                f"{self.api_url}/search",
                params={"query": query, "top_k": limit}
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._format_results(result)
            else:
                error_msg = f"Search API returned error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                print(f"[ERROR] {error_msg}")
                return json.dumps({"error": error_msg})
        except Exception as e:
            error_msg = f"Failed to execute search: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return json.dumps({"error": error_msg})
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute multiple search queries in batch
        
        Args:
            args_list: List of tool parameters
            
        Returns:
            List of formatted search results
        """
        # 提取查询和限制
        queries = [args.get("query", "").strip() for args in args_list]
        limits = [args.get("limit", 5) for args in args_list]
        max_limit = max(limits)  # 使用最大的limit值
        
        try:
            # 调用批量搜索API
            response = requests.post(
                f"{self.api_url}/search",
                json={"queries": queries, "top_k": max_limit}
            )
            
            if response.status_code == 200:
                batch_result = response.json()
                # 为每个查询格式化结果
                results = []
                for i, query_result in enumerate(batch_result["query_results"]):
                    # 限制结果数量为每个查询指定的limit
                    limited_results = {
                        "query": query_result["query"],
                        "results": query_result["results"][:limits[i]]
                    }
                    results.append(self._format_results(limited_results))
                return results
            else:
                error_msg = f"Batch search API returned error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                print(f"[ERROR] {error_msg}")
                return [json.dumps({"error": error_msg}) for _ in queries]
        except Exception as e:
            error_msg = f"Failed to execute batch search: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return [json.dumps({"error": error_msg}) for _ in queries]

    def _format_results(self, api_result) -> str:
        """
        Format API search results for better readability
        
        Args:
            api_result: API response containing search results
            
        Returns:
            Formatted results as a string
        """
        if "error" in api_result:
            return json.dumps(api_result)
        
        if "query_results" in api_result:
            # 单个查询的结果
            if len(api_result["query_results"]) > 0:
                query_result = api_result["query_results"][0]
                results_list = []
                
                for result in query_result["results"]:
                    results_list.append(result["document"])
                
                return json.dumps({"results": results_list})
            else:
                return json.dumps({"results": []})
        elif "results" in api_result:
            # 已经是格式化的结果
            return json.dumps(api_result)
        else:
            # 单个查询的结果
            results_list = []
            for result in api_result.get("results", []):
                results_list.append(result["document"])
            
            return json.dumps({"results": results_list})
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        try:
            result_obj = json.loads(result)
            # 有效的工具调用
            if "results" in result_obj:
                return 0.0
            elif "error" in result_obj:
                return -0.1  # 轻微惩罚错误
            else:
                return 0.0
        except:
            return -0.1  # 无法解析结果