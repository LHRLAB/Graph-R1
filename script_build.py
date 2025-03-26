import os
import json
import time
from graphr1 import GraphR1
import argparse
import numpy as np
from FlagEmbedding import FlagAutoModel
import faiss
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

def extract_knowledge(rag, unique_contexts):
    print(f"Total insert rounds: {len(unique_contexts)//50 + 1}")
    for i in range(0, len(unique_contexts), 50):
        print(f"This is the {i//50 + 1} round of insertion, remain rounds: {len(unique_contexts)//50 - i//50}")
        retries = 0
        max_retries = 50
        while retries < max_retries:
            try:
                rag.insert(unique_contexts[i:i+50])
                break
            except Exception as e:
                retries += 1
                print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
                time.sleep(10)
        if retries == max_retries:
            print("Insertion failed after exceeding the maximum number of retries")
    
    
    retries = 0
    max_retries = 50
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")
        
def embed_knowledge(data_source):
    corpus = []
    with open(f"expr/{data_source}/kv_store_text_chunks.json") as f:
        texts = json.load(f)
        for item in texts:
            corpus.append(texts[item]['content'])

    corpus_entity = []
    corpus_entity_des = []
    with open(f"expr/{data_source}/kv_store_entities.json") as f:
        entities = json.load(f)
        for item in entities:
            corpus_entity.append(entities[item]['entity_name'])
            corpus_entity_des.append(entities[item]['content'])
            
    corpus_hyperedge = []
    with open(f"expr/{data_source}/kv_store_hyperedges.json") as f:
        hyperedges = json.load(f)
        for item in hyperedges:
            corpus_hyperedge.append(hyperedges[item]['content'])

    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-large-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        # devices="cuda:0",   # if not specified, will use all available gpus or cpu when no gpu available
    )

    embeddings = model.encode_corpus(corpus)
    #save
    np.save(f"expr/{data_source}/corpus.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus.npy")
    dim = corpus_numpy.shape[-1]

    corpus_numpy = corpus_numpy.astype(np.float32)

    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index.bin")

    embeddings = model.encode_corpus(corpus_entity_des)
    #save
    np.save(f"expr/{data_source}/corpus_entity.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus_entity.npy")
    dim = corpus_numpy.shape[-1]

    corpus_numpy = corpus_numpy.astype(np.float32)

    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index_entity.bin")

    embeddings = model.encode_corpus(corpus_hyperedge)
    #save
    np.save(f"expr/{data_source}/corpus_hyperedge.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus_hyperedge.npy")
    dim = corpus_numpy.shape[-1]

    corpus_numpy = corpus_numpy.astype(np.float32)

    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index_hyperedge.bin")

def insert_knowledge(data_source, unique_contexts):
    rag = GraphR1(
        working_dir=f"expr/{data_source}"   
    )    
    extract_knowledge(rag, unique_contexts)
    embed_knowledge(data_source)
    print(f"Knowledge successfully inserted and embedded for {data_source}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, default="2WikiMultiHopQA")
    args = parser.parse_args()
    data_source = args.data_source
    
    unique_contexts = []
    with open(f"datasets/{data_source}/corpus.jsonl") as f:
        for line in f:
            data = json.loads(line)
            unique_contexts.append(data["contents"])
    
    insert_knowledge(data_source, unique_contexts)
    
    






