import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import faiss
from FlagEmbedding import FlagAutoModel
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

# 加载 FAISS 索引和 FlagEmbedding 模型
print(f"[DEBUG] LOADING EMBEDDINGS")
index = faiss.read_index(f"expr/{data_source}/index.bin")
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

# 加载 FAISS 索引和 FlagEmbedding 模型
print(f"[DEBUG] LOADING EMBEDDINGS")
index_entity = faiss.read_index(f"expr/{data_source}/index_entity.bin")
corpus_entity = []
with open(f"expr/{data_source}/vdb_entities.json") as f:
    entities = json.load(f)
    for entity in entities['data']:
        corpus_entity.append(entity['entity_name'])
print("[DEBUG] EMBEDDINGS LOADED")

# 加载 FAISS 索引和 FlagEmbedding 模型
print(f"[DEBUG] LOADING EMBEDDINGS")
index_hyperedge = faiss.read_index(f"expr/{data_source}/index_hyperedge.bin")
corpus_hyperedge = []
with open(f"expr/{data_source}/vdb_hyperedges.json") as f:
    relations = json.load(f)
    for relation in relations['data']:
        corpus_hyperedge.append(relation['hyperedge_name'])
print("[DEBUG] EMBEDDINGS LOADED")

rag = GraphR1(
    working_dir=f"expr/{data_source}",  
)

async def process_query(query_text, rag_instance, entity_match, hyperedge_match, chunk_match):
    result = await rag_instance.aquery(query_text, param=QueryParam(only_need_context=True, top_k=5), entity_match=entity_match, hyperedge_match=hyperedge_match, chunk_match=chunk_match)
    return {"query": query_text, "result": result}, None

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def _format_results(results: List, corpus) -> str:
    results_list = []
    
    for i, result in enumerate(results):
        results_list.append(corpus[result])
    
    return results_list

def queries_to_results(queries: List[str]) -> List[str]:
    
    embeddings = model.encode_queries(queries)
    _, ids = index_entity.search(embeddings, 5)  # 每个查询返回 5 个结果
    entity_match = {queries[i]:_format_results(ids[i], corpus_entity) for i in range(len(ids))}
    _, ids = index_hyperedge.search(embeddings, 5)  # 每个查询返回 5 个结果
    hyperedge_match = {queries[i]:_format_results(ids[i], corpus_hyperedge) for i in range(len(ids))}
    _, ids = index.search(embeddings, 5)  # 每个查询返回 5 个结果
    chunk_match = {queries[i]:_format_results(ids[i], corpus) for i in range(len(ids))}
    
    
    results = []
    loop = always_get_an_event_loop()
    for query_text in tqdm(queries, desc="Processing queries", unit="query"):
        result, error = loop.run_until_complete(
            process_query(query_text, rag, entity_match[query_text], hyperedge_match[query_text], chunk_match[query_text])
        )
        results.append(json.dumps({"results": result["result"]}))
    return results
########### PREDEFINE ############


with open(f'datasets/{data_source}/raw/qa_test.json', 'r') as f:
    test_data = json.load(f)
queries = [item['question'] for item in test_data]

results_str = queries_to_results(queries)
print(results_str[0])