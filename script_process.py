import os
import datasets
import argparse
import json


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', default='2WikiMultiHopQA')

    args = parser.parse_args()

    data_source = args.data_source
    
    with open(f'datasets/{data_source}/raw/qa_train.json', 'r') as f:
        train_data = json.load(f)
    with open(f'datasets/{data_source}/raw/qa_dev.json', 'r') as f:
        dev_data = json.load(f)
    with open(f'datasets/{data_source}/raw/qa_test.json', 'r') as f:
        test_data = json.load(f)
    
    train_dataset = datasets.Dataset.from_list(train_data)
    dev_dataset = datasets.Dataset.from_list(dev_data)
    test_dataset = datasets.Dataset.from_list(test_data)
    
    instruction_following = """Answer the given question. You can query from knowledge base provided to you to answer the question. You can query knowledge as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to query knowledge, you can set a query statement between <query>...</query> to query from knowledge base after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<query>
...
</query>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""    

    # Process each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('question')
            question = instruction_following + "Question: " + question_raw
            
            answer_raw = example.pop('golden_answers')
            
            # Convert all data to string format to avoid type issues
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "multihop_qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw
                },
                "extra_info": {
                    'split': split,
                    'index': str(idx),
                    'answer': answer_raw,
                    'question': question_raw
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    dev_dataset = dev_dataset.map(function=make_map_fn('dev'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = f'datasets/{data_source}/processed'
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    dev_dataset.to_parquet(os.path.join(local_dir, 'dev.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

