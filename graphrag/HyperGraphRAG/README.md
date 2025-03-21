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
#### 1.HyperGraph Construction
For the extracted contexts, we insert them into the HyperGraphRAG system.
```bash
nohup python script_insert.py --cls 2wikimultihopqa > result_2wikimultihopqa_insert.log 2>&1 &
```

#### 2.Test HyperGraphRAG
For the queries collected, we will query HyperGraphRAG.
```bash
python script_quickquery_batch.py --data_source 2wikimultihopqa
```

#### 3.Set up search server
```bash
nohup python -u search_api.py --data_source 2wikimultihopqa > result_search_api_hypergraphrag_2wikimultihopqa.log 2>&1 &
```

