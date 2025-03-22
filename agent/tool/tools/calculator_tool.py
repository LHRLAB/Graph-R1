"""
Calculator tool implementation for performing basic arithmetic operations
"""

from typing import Dict
import json
from agent.tool.tool_base import Tool


class CalculatorTool(Tool):
    """
    Tool for performing basic arithmetic calculations
    """
    
    def __init__(self):
        """
        Initialize the calculator tool
        """
        name = "calculator"
        description = "Perform basic arithmetic calculations (addition, subtraction, multiplication, division)."
        parameters = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate (using operators +, -, *, / and parentheses). Must be in a format compatible with Python's eval function, such as '2 + 3 * (4 - 1)'. Only basic arithmetic operations are supported."
                }
            },
            "required": ["expression"]
        }
        
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute calculator operations
        
        Args:
            args: Tool parameters, containing:
                - "expression": arithmetic expression to evaluate
            
        Returns:
            Result of the calculation
        """
        expression = args.get("expression", "").strip()
        
        if not expression:
            return "No expression provided."
        
        try:
            # Create a safe environment for evaluating arithmetic expressions
            # Only allow basic arithmetic operations
            safe_dict = {
                "abs": abs,  # Keep abs for handling negative numbers
                "float": float,  # Keep float for proper division
            }
            
            # Evaluate the expression in the safe environment
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            result = {"result": result}
            
            return json.dumps(result)
        
        except ZeroDivisionError:
            return json.dumps({"error": "Division by zero is not allowed."})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for calculator action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # Check if computation resulted in an error
        if result.startswith("Error:"):
            return 0.05  # Small reward for trying
        
        if result.startswith("No expression"):
            return 0.0  # No reward for empty expression
        
        # Base reward for successful calculation
        reward = 0.4
        
        # Simple reward logic based on complexity of the expression
        expression = args.get("expression", "")
        complexity = len([c for c in expression if c in "+-*/()"]) + 1
        
        # Additional reward based on complexity (with a cap)
        reward += min(0.1, 0.02 * complexity)
        
        return min(0.5, reward)  # Cap at 0.5
    
if __name__ == "__main__":
    calculator_tool = CalculatorTool()
    print(calculator_tool.execute({"expression": "2 + 3"}))
    print(calculator_tool.calculate_reward({"expression": "2 + 3"}, "Result: 5"))
