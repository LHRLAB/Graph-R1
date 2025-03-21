import os
import json
import time
from hypergraphrag import HyperGraphRAG
import argparse
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

def insert_knowledge(rag, unique_contexts):
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

parser = argparse.ArgumentParser()
parser.add_argument("--cls", type=str, default="2wikimultihopqa")
args = parser.parse_args()

rag = HyperGraphRAG(working_dir=f"expr/{args.cls}")

unique_contexts = []

with open(f"../../datasets/{args.cls}/corpus.jsonl") as f:
    for line in f:
        data = json.loads(line)
        unique_contexts.append(data["contents"])
new_contexts = []
temp=""
for c in unique_contexts:
    temp+=(c+'\n')
    if len(temp.replace('\n',' ').split(' '))>10000:
        new_contexts.append(temp)
        temp=""
if temp!="":
    new_contexts.append(temp)
    
insert_knowledge(rag, new_contexts)



