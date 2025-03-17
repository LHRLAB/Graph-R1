"""
Base tool class definition, providing fundamental tool interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List

class Tool(ABC):
    """
    Tool base class, defining the basic interface for tools
    Each specific tool should inherit from this class and implement its methods
    """
    
    def __init__(self, name: str, description: str, parameters: Dict = None):
        """
        Initialize the tool
        
        Args:
            name: Tool name
            description: Tool description
            parameters: JSON Schema compliant parameter definition, format as follows:
                {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "Parameter 1 description"},
                        "param2": {"type": "number", "description": "Parameter 2 description"}
                    },
                    "required": ["param1"]
                }
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Ensure parameters contains necessary fields
        if "type" not in self.parameters:
            self.parameters["type"] = "object"
        if "properties" not in self.parameters:
            self.parameters["properties"] = {}
        if "required" not in self.parameters:
            self.parameters["required"] = []
    
    def get_description(self) -> Dict:
        """
        Get the tool description in JSON Schema format
        
        Returns:
            Dictionary containing name, description, and parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def get_simple_description(self) -> str:
        """
        Get a simplified tool description for user display
        
        Returns:
            Formatted tool description string
        """
        desc = f"Tool name: {self.name}\nDescription: {self.description}"
        
        if self.parameters and "properties" in self.parameters:
            properties = self.parameters["properties"]
            required = self.parameters.get("required", [])
            
            if properties:
                desc += "\nParameters:"
                for param_name, param_info in properties.items():
                    param_desc = param_info.get("description", "")
                    param_type = param_info.get("type", "")
                    is_required = "(Required)" if param_name in required else "(Optional)"
                    desc += f"\n  - {param_name} {is_required}: {param_desc}"
                    if "enum" in param_info:
                        desc += f", possible values: {', '.join(map(str, param_info['enum']))}"
        
        return desc
    
    @abstractmethod
    def execute(self, args: Dict) -> str:
        """
        Execute the tool functionality
        
        Args:
            args: Tool parameters
            
        Returns:
            Tool execution result
        """
        pass

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute multiple tool calls in batch
        
        By default, this method falls back to individual execution.
        Override this method for tools that can benefit from batch execution.
        
        Args:
            args_list: List of tool parameters
            
        Returns:
            List of tool execution results
        """
        return [self.execute(args) for args in args_list]
    
    
    def validate_args(self, args: Dict) -> Tuple[bool, str]:
        """
        Validate if the tool parameters are valid
        
        Args:
            args: Tool parameters
            
        Returns:
            (is_valid, error_message)
        """
        # Check if args is a Dict
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary"

        # Check required parameters
        required_params = self.parameters.get("required", [])
        for param in required_params:
            if param not in args:
                return False, f"Missing required parameter: {param}"
        
        # Check parameter types
        properties = self.parameters.get("properties", {})
        for param_name, param_value in args.items():
            if param_name in properties:
                param_schema = properties[param_name]
                
                # Check type
                param_type = param_schema.get("type")
                if param_type and not self._check_type(param_value, param_type):
                    return False, f"Parameter {param_name} has incorrect type, should be {param_type}"
                
                # Check enum values
                if "enum" in param_schema and param_value not in param_schema["enum"]:
                    valid_values = ", ".join(map(str, param_schema["enum"]))
                    return False, f"Parameter {param_name} has invalid value, should be one of: {valid_values}"
        
        return True, "Parameters valid"
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if the value's type matches the expected type
        
        Args:
            value: Value to check
            expected_type: Expected type
            
        Returns:
            Whether the type matches
        """
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        return True  # If unknown type, default to pass
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate the reward for tool execution
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # Default implementation returns zero reward, subclasses can override this method
        return 0.0 