"""
Utility functions for working with tools
"""

import inspect
from typing import Callable, Dict, Any, Type, Optional
from functools import wraps

try:
    from transformers.utils import get_json_schema
except ImportError:
    raise ImportError(
        "The transformers library is required for this functionality. "
        "Please install it with: pip install transformers>=4.35.0"
    )

from agent.tool.tool_base import Tool

def function_to_tool(func: Callable) -> Tool:
    """
    Convert a Python function to a Tool object using transformers.utils.get_json_schema.
    
    The function must have proper type annotations for all parameters and Google-style
    docstrings for the function description and parameter descriptions.
    
    Args:
        func: The Python function to convert to a tool. Must have:
            1. Type annotations for all parameters
            2. Google-style docstring with function description and parameter descriptions
            3. For enum parameters, add (choices: ["value1", "value2"]) at the end of the parameter description
    
    Returns:
        A Tool instance that wraps the provided function
    """
    # Get the JSON schema for the function
    schema = get_json_schema(func)
    
    # Extract the relevant information
    function_data = schema.get("function", {})
    name = function_data.get("name", func.__name__)
    description = function_data.get("description", "")
    parameters = function_data.get("parameters", {})
    
    # Create a tool class for this function
    class FunctionTool(Tool):
        def __init__(self):
            super().__init__(name=name, description=description, parameters=parameters)
            self.func = func
        
        def execute(self, args: Dict[str, Any]) -> str:
            """
            Execute the wrapped function with the provided arguments
            
            Args:
                args: Arguments to pass to the function
                
            Returns:
                Result of the function execution as a string
            """
            # Filter args to only include parameters that exist in the function signature
            sig = inspect.signature(self.func)
            valid_args = {k: v for k, v in args.items() if k in sig.parameters}
            
            try:
                result = self.func(**valid_args)
                # Convert result to string if it's not already
                if not isinstance(result, str):
                    result = str(result)
                return result
            except Exception as e:
                return f"Error executing {self.name}: {str(e)}"
    
    # Return an instance of the new tool class
    return FunctionTool()


# Example usage of function_to_tool:
#
# def search_weather(city: str, units: str = "metric"):
#     """
#     Search for weather information for a city.
#     
#     Args:
#         city: The name of the city to search for
#         units: The units to use for temperature (choices: ["metric", "imperial"])
#     
#     Returns:
#         Weather information for the specified city
#     """
#     # Implementation...
#     
# weather_tool = function_to_tool(search_weather)


def tool_decorator(name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to convert a function into a Tool object.
    
    Args:
        name: Optional custom name for the tool (defaults to function name)
        description: Optional custom description (defaults to function docstring)
        
    Returns:
        A decorator function that converts the decorated function to a Tool
    """
    def decorator(func: Callable) -> Tool:
        tool = function_to_tool(func)
        
        # Override name and description if provided
        if name is not None:
            tool.name = name
        if description is not None:
            tool.description = description
            
        return tool
    
    return decorator


# Example usage of tool_decorator:
#
# @tool_decorator(name="GetWeather")
# def search_weather(city: str, units: str = "metric"):
#     """
#     Search for weather information for a city.
#     
#     Args:
#         city: The name of the city to search for
#         units: The units to use for temperature (choices: ["metric", "imperial"])
#     
#     Returns:
#         Weather information for the specified city
#     """
#     # Implementation... 