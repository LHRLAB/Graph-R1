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
#### 1. Preprocess 2WikiMultihopQA dataset to parquet format
```bash
python script_process.py --data_source 2wikimultihopqa
```

#### 2. Set up retrieve server at 8001 port
For the extracted contexts, we insert them into the Graph-R1 system.
```bash
nohup python -u script_build.py --data_source 2wikimultihopqa > result_2wikimultihopqa_build.log 2>&1 &
```
Set up Graph-R1 retrieve server
```bash
nohup python -u script_api.py --data_source 2wikimultihopqa > result_2wikimultihopqa_api.log 2>&1 &
```

#### 3. Run GRPO/REINFORCE++/PPO training with Qwen2.5-1.5B-Instruct
```bash
bash run_grpo_2wikimultihopqa.sh
nohup bash -u run_grpo_2wikimultihopqa.sh > result_2wikimultihopqa_run_grpo.log 2>&1 &

bash run_rpp_2wikimultihopqa.sh
nohup bash -u run_rpp_2wikimultihopqa.sh > result_2wikimultihopqa_run_rpp.log 2>&1 &

bash run_ppo_2wikimultihopqa.sh
nohup bash -u run_ppo_2wikimultihopqa.sh > result_2wikimultihopqa_run_ppo.log 2>&1 &
```

#### 4. Close search server 8001 port
```bash
fuser -k 8001/tcp
```