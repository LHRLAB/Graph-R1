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
# pip3 install FlagEmbedding
# pip3 install faiss-cpu
```

### Quick Start: Graph-R1 on 2WikiMultihopQA
#### 1. Preprocess 2WikiMultihopQA dataset and build search index
```bash
python to_parquet.py --data_source 2wikimultihopqa
python to_index.py --data_source 2wikimultihopqa
```

#### 2. Run PPO/REINFORCE++/GRPO training with Qwen2.5-1.5B-Instruct
```bash
bash run_grpo_2wikimultihopqa.sh
nohup bash run_grpo_2wikimultihopqa.sh > result_grpo_2wikimultihopqa.log 2>&1 &

bash run_rpp_2wikimultihopqa.sh
nohup bash run_rpp_2wikimultihopqa.sh > result_rpp_2wikimultihopqa.log 2>&1 &

bash run_ppo_2wikimultihopqa.sh
nohup bash run_ppo_2wikimultihopqa.sh > result_ppo_2wikimultihopqa.log 2>&1 &
```