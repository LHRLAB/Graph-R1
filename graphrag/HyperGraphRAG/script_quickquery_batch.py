import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import argparse
from hypergraphrag import HyperGraphRAG
import os
import asyncio
from tqdm import tqdm
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='2wikimultihopqa')
args = parser.parse_args()
data_source = args.data_source

rag = HyperGraphRAG(working_dir=f"expr/{data_source}")

async def process_query(query_text, rag_instance):
    try:
        result = await rag_instance.aquery(query_text)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}

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
        results.append(json.dumps({"results": result}))
    return results
########### PREDEFINE ############


with open(f'../../datasets/{data_source}/raw/qa_test.json', 'r') as f:
    test_data = json.load(f)
queries = [item['question'] for item in test_data]

results_str = queries_to_results(queries)
print(results_str)