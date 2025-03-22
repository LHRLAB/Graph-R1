import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import argparse
from graphr1 import HyperGraphRAG, QueryParam

import os
import asyncio
from tqdm import tqdm
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='2wikimultihopqa')
args = parser.parse_args()
data_source = args.data_source

rag = HyperGraphRAG(
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

# 创建 FastAPI 实例
app = FastAPI(title="Search API", description="An API for document retrieval using FAISS and FlagEmbedding.")

class SearchRequest(BaseModel):
    queries: List[str]

@app.post("/search")
def search(request: SearchRequest):
    results_str = queries_to_results(request.queries)
    return results_str

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)