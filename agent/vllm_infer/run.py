import argparse
import json
from openai import OpenAI

from agent.tool.tool_env import ToolEnv, step_batch
from agent.tool.tools import _default_tools
import re
import copy

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
    parser.add_argument('--max-turns', type=int, default=20,
                        help='Maximum turns of search')
    parser.add_argument('--question', type=str, default="Which magazine came out first, Tit-Bits or Illustreret Nyhedsblad?",
                        help='Question to ask the model')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    return parser.parse_args()

def process_tool_call(responses_str):

    def process_single_response(resp):
        eos_token = "<|im_end|>"
        tool_call_end ="</query>"
        tool_pattern = r'<query>(.*?)</query>'
        match = re.search(tool_pattern, resp, re.DOTALL)
        
        if not match:
            return resp + eos_token, False  # No tool call found
        
        resp = resp.split(tool_call_end)[0] + tool_call_end
        
        return resp + eos_token, True
    
    # Process each response string
    return [process_single_response(resp)[0] for resp in responses_str], [process_single_response(resp)[1] for resp in responses_str]

def execute_tool_calls_batch(response_strs, env, active_masks):
    tool_custom_response_template = "<|im_start|>user\n<knowledge>\n{tool_response}\n</knowledge><|im_end|>\n<|im_start|>assistant\n<think>"
    active_envs = []
    active_responses = []
    active_indices = []
    
    for i, (resp, active) in enumerate(zip(response_strs, active_masks)):
        if active:
            active_envs.append(env)
            active_responses.append(resp)
            active_indices.append(i)
    
    # Initialize result list with empty strings
    tool_responses = [""] * len(response_strs)
    
    if not active_envs:
        return tool_responses
        
    # Use the independent step_batch function for active environments
    batch_results = step_batch(active_envs, active_responses)
    
    # Map results back to original indices
    for idx, result in zip(active_indices, batch_results):
        if result is None:
            tool_responses[idx] = ""
        else:
            tool_response = result[0]
            tool_responses[idx] = tool_custom_response_template.format(tool_response=tool_response)
    return tool_responses

def colorprint(mode, r_str, t_str, use_colors):
    if not r_str.startswith("<think>\n"):
        r_str = "<think>\n" + r_str
    
    if mode is True:
        think = re.findall(r'<think>(.*?)</think>', r_str, re.DOTALL)[0]
        if not think.endswith("\n"):
            think += "\n"
        query = re.findall(r'<query>\n{\n  "query": "(.*?)"\n}\n</query>', r_str, re.DOTALL)[0]
        knowledge = re.findall(r'<knowledge>(.*?)</knowledge>', t_str, re.DOTALL)[0]
        knowledge_list = json.loads(knowledge)['results']
        knowledge = "\n"
        for k in knowledge_list:
            knowledge += str(k) + "\n"
        
        if use_colors:
            print(f"\n{COLORS['bg_tool_call']} Think {COLORS['reset']} {COLORS['tool_call']}{think}{COLORS['reset']}")
            print(f"{COLORS['tool_call']}Query:{COLORS['reset']}\n{query}{COLORS['reset']}")
            print(f"\n{COLORS['bg_tool']} Knowledge {COLORS['reset']} {COLORS['tool']}{knowledge}{COLORS['reset']}")
        else:
            print(f"\n[Think] {think}")
            print(f"Query:\n{query}")
            print(f"\nKnowledge: {knowledge}") 
    else:
        think = re.findall(r'<think>(.*?)</think>', r_str, re.DOTALL)[0]
        if not think.endswith("\n"):
            think += "\n"

        answer = re.findall(r'<answer>\n(.*?)\n</answer>', r_str, re.DOTALL)[0]
        
        if use_colors:
            print(f"\n{COLORS['bg_tool_call']} Think {COLORS['reset']} {COLORS['tool_call']}{think}{COLORS['reset']}")
            print(f"{COLORS['tool_call']}Answer:{COLORS['reset']}\n{answer}{COLORS['reset']}")
        else:
            print(f"\n[Think] {think}")
            print(f"Answer:\n{answer}")
            
        print("\n")

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
    for step in range(MAX_TURNS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            
            # Get the response message
            response_message = response.choices[0].message
            responses_str = [response_message.content]
            
            responses_str, active_masks = process_tool_call(responses_str)
            
            tool_responses = execute_tool_calls_batch(responses_str, env, active_masks)

            colorprint(active_masks[0], copy.deepcopy(responses_str[0]), copy.deepcopy(tool_responses[0]), use_colors)
            
            if active_masks[0] is True:
                prompt = messages[0]["content"]+responses_str[0]+tool_responses[0]
                messages = [{
                    "role": "user",
                    "content": prompt
                }] 
                # print(messages[0]["content"])
            else:
                prompt = messages[0]["content"]+responses_str[0]
                # print(prompt)
                break

        except:
            # print("Aha...")
            continue

if __name__ == "__main__":
    main() 