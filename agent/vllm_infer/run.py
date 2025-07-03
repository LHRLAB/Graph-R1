import argparse
import json
from openai import OpenAI

from agent.tool.tool_env import ToolEnv
from agent.tool.tools import _default_tools

# ANSI color codes for colored output
COLORS = {
    "user": "\033[1;34m",      # Bold Blue
    "assistant": "\033[1;32m",  # Bold Green
    "tool": "\033[1;33m",       # Bold Yellow
    "tool_call": "\033[1;35m",  # Bold Purple
    "reset": "\033[0m",         # Reset to default
    "bg_user": "\033[44m",      # Blue background
    "bg_assistant": "\033[42m", # Green background
    "bg_tool": "\033[43m",      # Yellow background
    "bg_tool_call": "\033[45m", # Purple background
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run VLLM inference with configurable parameters')
    parser.add_argument('--api-key', type=str, default="EMPTY",
                        help='OpenAI API key')
    parser.add_argument('--api-base', type=str, default="http://localhost:8002/v1",
                        help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default="agent",
                        help='Model name for inference')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--max-turns', type=int, default=10,
                        help='Maximum turns of search')
    parser.add_argument('--question', type=str, default="Which film has the director died first, Watch Your Stern or Requiescant?",
                        help='Question to ask the model')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    return parser.parse_args()

def main():
    args = parse_args()
    use_colors = not args.no_color
    OPENAI_API_KEY = args.api_key
    OPENAI_API_BASE = args.api_base
    MODEL_NAME = args.model
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens
    MAX_TURNS = args.max_turns
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )
    
    # Set up tools
    tools = _default_tools("search")
    env = ToolEnv(tools=tools, max_turns=MAX_TURNS)
    
    # Create message with question
    question_raw = args.question
    messages = [{
        "role": "user",
        "content": '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"name\": \"search\", \"description\": \"Search for information on the internet using Wikipedia as a knowledge source.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"Search query\"}}, \"required\": [\"query\"]}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nAnswer the given question. You can query from knowledge base provided to you to answer the question. You can query knowledge as many times as you want.\nYou must first conduct reasoning inside <think>...</think>. If you need to query knowledge, you can set a query statement between <query>...</query> to query from knowledge base after <think>...</think>.\nWhen you have the final answer, you can output the answer inside <answer>...</answer>.\n\nOutput format for tool call:\n<think>\n...\n</think>\n<query>\n...\n</query>\n\nOutput format for answer:\n<think>\n...\n</think>\n<answer>\n...\n</answer>\nQuestion: '+question_raw+'<|im_end|>\n<|im_start|>assistant\n'
    }]
    
    print(f"Running inference with model: {MODEL_NAME}")
    if use_colors:
        print(f"{COLORS['bg_user']} User {COLORS['reset']} {COLORS['user']}{question_raw}{COLORS['reset']}")
    else:
        print(f"User: {question_raw}")
    
    # Run inference loop
    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        
        # Get the response message
        response_message = response.choices[0].message
        
        # Format the assistant's message properly
        assistant_message = {
            "role": "assistant",
            "content": response_message.content
        }
        
        # Add tool calls if any
        if response_message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in response_message.tool_calls
            ]
        
        # Add the formatted message to the conversation
        messages.append(assistant_message)
        
        # Display assistant's response with color
        if use_colors:
            print(f"\n{COLORS['bg_assistant']} Assistant {COLORS['reset']} {COLORS['assistant']}{response_message.content}{COLORS['reset']}")
        else:
            print(f"\nAssistant: {response_message.content}")
        
        # Check if there are any tool calls
        if response_message.tool_calls:
            # Process each tool call
            for tool_call in response_message.tool_calls:
                # Pretty format the arguments for better readability
                try:
                    args_dict = json.loads(tool_call.function.arguments)
                    formatted_args = json.dumps(args_dict, indent=2)
                except json.JSONDecodeError:
                    formatted_args = tool_call.function.arguments
                
                # Log function call details with color
                if use_colors:
                    print(f"\n{COLORS['bg_tool_call']} Tool Call {COLORS['reset']} {COLORS['tool_call']}Function: {tool_call.function.name}{COLORS['reset']}")
                    print(f"{COLORS['tool_call']}Arguments:{COLORS['reset']}\n{formatted_args}")
                else:
                    print(f"\n[Tool Call] Function: {tool_call.function.name}")
                    print(f"Arguments:\n{formatted_args}")
                
                # Execute the tool
                result = env.tool_map[tool_call.function.name].execute(json.loads(tool_call.function.arguments))
                result = result["content"]
                
                # Display tool result with color
                if use_colors:
                    print(f"\n{COLORS['bg_tool']} Tool {COLORS['reset']} {COLORS['tool']}{result}{COLORS['reset']}")
                else:
                    print(f"\nTool: {result}")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.id
                })
        else:
            # No tool calls, we have reached the final answer
            break

if __name__ == "__main__":
    main() 