# HyperGraphRAG

## Dependence
```bash
conda create -n hypergraphrag python=3.11
conda activate hypergraphrag
pip install torch==2.3.0
pip install -r requirements.txt
```
You need add ``openai_api_key.txt`` in the root directory, which contains the OpenAI API key.

## HyperGraphRAG Process

### 2WikiMultiHopQA
#### HyperGraph Construction
For the extracted contexts, we insert them into the HyperGraphRAG system.
```bash
nohup python script_insert.py --cls 2wikimultihopqa > result_2wikimultihopqa_insert.log 2>&1 &
```
#### HyperGraph-Guided Generation
For the queries collected, we will query HyperGraphRAG.
```bash
nohup python script_query.py --cls hypertension --level easy >> result_hypertension_easy_query.log 2>&1 &
```