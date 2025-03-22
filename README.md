# Graph-R1

### Install Environment
```bash
conda create -n graphr1 python==3.11
conda activate graphr1
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
# pip install debugpy==1.8.0
# pip install "ray[default]" debugpy
pip install -r requirements.txt
```

### Quick Start: Graph-R1 on 2WikiMultihopQA
#### 1. Preprocess 2WikiMultihopQA dataset
```bash
python to_parquet.py --data_source 2wikimultihopqa
```

#### 2. Set up search server at 8001 port
For the extracted contexts, we insert them into the GraphR1 system.
```bash
nohup python script_insert.py --cls 2wikimultihopqa > result_2wikimultihopqa_insert.log 2>&1 &
```
Test GraphR1.
```bash
python script_quickquery_batch.py --data_source 2wikimultihopqa
```
Set up search server
```bash
nohup python -u search_api.py --data_source 2wikimultihopqa > result_search_api_2wikimultihopqa.log 2>&1 &
```

#### 3. Run GRPO/REINFORCE++/PPO training with Qwen2.5-1.5B-Instruct
```bash
bash run_grpo_2wikimultihopqa.sh
nohup bash run_grpo_2wikimultihopqa.sh > result_grpo_2wikimultihopqa.log 2>&1 &

bash run_rpp_2wikimultihopqa.sh
nohup bash run_rpp_2wikimultihopqa.sh > result_rpp_2wikimultihopqa.log 2>&1 &

bash run_ppo_2wikimultihopqa.sh
nohup bash run_ppo_2wikimultihopqa.sh > result_ppo_2wikimultihopqa.log 2>&1 &
```

#### 4. Close search server 8001 port
```bash
fuser -k 8001/tcp
```











- Note: 
1. The search server is set up on port 8001 by default. 
2. Other GraphRAG methods are also available at [README_GraphRAG](graphrag/GraphRAG/README.md), [README_LightRAG](graphrag/LightRAG/README.md), [README_PathRAG](graphrag/PathRAG/README.md), and [README_HippoRAG2](graphrag/HippoRAG2/README.md).
3. If you want to use StandardRAG, you can first index the knowledge:
    ```bash
    python to_index.py --data_source 2wikimultihopqa
    ```
    and then run the search server as follows:
    ```bash
    nohup python -u search_api.py --data_source 2wikimultihopqa > result_search_api_2wikimultihopqa.log 2>&1 &
    ```
4. You can only run one search server at a time. If you want to switch between different search servers, you need to close the current search server first.