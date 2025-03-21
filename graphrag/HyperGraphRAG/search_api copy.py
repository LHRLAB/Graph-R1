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

# 创建 FastAPI 实例
app = FastAPI(title="Search API", description="An API for document retrieval using FAISS and FlagEmbedding.")

rag = HyperGraphRAG(working_dir=f"expr/{data_source}")

def queries_to_results(queries: List[str]) -> List[str]:
    results_str = []
    for query_text in queries:
        result = rag.query(query_text)
        results_str.append(json.dumps({"results": result}))
    return results_str

class SearchRequest(BaseModel):
    queries: List[str]

@app.post("/search")
def search(request: SearchRequest):
    results_str = queries_to_results(request.queries)
    return results_str

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)