# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

def answer_check(solution_str, ground_truth):
    # find the last number in the solution string 
    answer = re.findall(r'-?\d+\.?\d*', solution_str)
    if answer:
        answer = answer[-1]
    else:
        answer = None
    
    if answer == ground_truth:
        return True
    else:
        return False

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def compute_score_format(solution_str):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None:
        return 0.0
    
    try:
        # Perfect format match for the new structure
        # First <|im_start|>assistant should have <think> and possibly <query>
        # Then <|im_start|>tool with <knowledge> (can repeat with assistant/tool pairs)
        # Final <|im_start|>assistant with the answer and <|im_end|>
        
        # Check for basic structure with <|im_start|>assistant and <|im_end|> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        tool_blocks = re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)

        format_reward = 0.0
        
        # If no blocks found, return 0
        if not assistant_blocks:
            return 0.0
        
        # Perfect format requires at least one assistant block and matching tool blocks if tool calls exist
        # Check first assistant block contains <think> tags
        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<query>') == 1 and assistant_block.count('</query>') == 1:
                think_match = re.search(r'^<think>(.*?)</think>\n<query>(.*?)</query>$', assistant_block, re.DOTALL)
                if think_match:
                    tool_block = tool_blocks[i+1]
                    tool_response = re.search(r'<knowledge>(.*?)</knowledge>', tool_block, re.DOTALL)
                    if tool_response:
                        tool_response = tool_response.group(1).strip()
                        if tool_response.count('error') == 0:
                            format_reward += 0.5

        # Check the last assistant block contains <answer> tags
        if assistant_blocks:  # 确保有至少一个assistant块
            last_assistant_block = assistant_blocks[-1]
            think_answer_match = re.search(r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', last_assistant_block, re.DOTALL)
            if think_answer_match:
                format_reward += 0.5
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0
    
    return format_reward


def compute_score_answer(solution_str, ground_truth):
    """The scoring function for exact match (EM) with format reward.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    Returns:
        float: Total reward score (format reward + answer reward)
    """
    if solution_str is None:
        return 0.0
    
    try:
        # Extract answer from <answer> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)

        answer_reward = 0.0
        
        if answer is not None:
            # Check for exact match within <answer>
            if answer_check(answer, ground_truth):
                answer_reward = 1.0
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return 0.0
    
    return answer_reward

def compute_score_format_answer(solution_str, ground_truth):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        format_reward = min(format_reward, 1.0)
        if format_reward == 1.0:
            return -1.0 + format_reward + answer_reward
        else:
            return -1.0 + format_reward
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0
