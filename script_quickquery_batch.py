import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import argparse
from graphr1 import GraphR1, QueryParam
from transformers import AutoTokenizer, AutoModel
from graphr1.utils import EmbeddingFunc
from graphr1.llm import hf_embedding, hf_model_complete

import os
import asyncio
from tqdm import tqdm
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='2wikimultihopqa')
args = parser.parse_args()
data_source = args.data_source

rag = GraphR1(
    working_dir=f"expr/{data_source}",  
)

async def process_query(query_text, rag_instance):
    result = await rag_instance.aquery(query_text, param=QueryParam(only_need_context=True, top_k=5))
    return {"query": query_text, "result": result}, None

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def queries_to_results(queries: List[str]) -> List[str]:
    results = []
    loop = always_get_an_event_loop()
    for query_text in tqdm(queries, desc="Processing queries", unit="query"):
        result, error = loop.run_until_complete(
            process_query(query_text, rag)
        )
        results.append(json.dumps({"results": result["result"]}))
    return results
########### PREDEFINE ############


with open(f'datasets/{data_source}/raw/qa_test.json', 'r') as f:
    test_data = json.load(f)
queries = [item['question'] for item in test_data]

results_str = queries_to_results(queries)
print(results_str)