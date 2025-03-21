import faiss
from FlagEmbedding import FlagAutoModel
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='2wikimultihopqa')
args = parser.parse_args()
data_source = args.data_source

# 创建 FastAPI 实例
app = FastAPI(title="Search API", description="An API for document retrieval using FAISS and FlagEmbedding.")

# 加载 FAISS 索引和 FlagEmbedding 模型
print(f"[DEBUG] LOADING EMBEDDINGS")
index = faiss.read_index(f"datasets/{data_source}/index.bin")
model = FlagAutoModel.from_finetuned(
    'BAAI/bge-large-en-v1.5',
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    devices="cpu",
)
corpus = []
with open(f"datasets/{data_source}/corpus.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        corpus.append(data["contents"])
print("[DEBUG] EMBEDDINGS LOADED")

def _format_results(results: List) -> str:
    """
    Format search results for better readability
    
    Args:
        results: List of search result List
        
    Returns:
        Formatted results as a string
    """
    results_list = []
    
    for i, result in enumerate(results):
        results_list.append(corpus[result])
    
    return json.dumps({"results": results_list})

def queries_to_results(queries: List[str]) -> List[str]:
    embeddings = model.encode_queries(queries)
    _, ids = index.search(embeddings, 5)  # 每个查询返回 5 个结果
    results_str = [_format_results(ids[i]) for i in range(len(ids))]
    return results_str

class SearchRequest(BaseModel):
    queries: List[str]

@app.post("/search")
def search(request: SearchRequest):
    results_str = queries_to_results(request.queries)
    return results_str

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)