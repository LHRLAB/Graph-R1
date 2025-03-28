# Graph-R1: Incentivizing Reasoning-on-Graph Capability in LLM via Reinforcement Learning

### Overview

Recently, the **GraphRAG method effectively addresses the data silos issue**, significantly enhancing the efficiency of knowledge retrieval. Nevertheless, in practical scenarios, **the disconnect between graph-structured knowledge and language modalities continues to constrain the model's performance**. To bridge this gap, we propose **Graph-R1, an end-to-end reinforcement learning (RL) framework** designed to substantially improve the **graph-based reasoning capabilities of large language models (LLMs)**. Specifically, we first **construct a knowledge hypergraph using the n-ary relation extraction techniques from HyperGraphRAG**. Subsequently, we employ an **explicit reward mechanism within an end-to-end RL setup**, encouraging the LLM to iteratively execute a **"think–generate query–retrieve subgraph–rethink" reasoning cycle**. This iterative approach ultimately enables the model to produce **precise, high-quality answers by effectively leveraging graph knowledge**.

### Install Environment
```bash
conda create -n graphr1 python==3.11.11
conda activate graphr1
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install -r requirements.txt
```

### Dataset Preparation
> We conduct experiments on seven datasets: 2WikiMultiHopQA, HotpotQA, Musique, NarrativeQA, NQ, PopQA, and TriviaQA. You can download them from [here](), and set the data path in `datasets/`.

### Quick Start: Graph-R1 on 2WikiMultiHopQA
#### 1. Preprocess 2WikiMultiHopQA dataset to parquet format
```bash
python script_process.py --data_source 2WikiMultiHopQA
```

#### 2. Extract contexts and build Knowledge HyperGraph (Optional)
> We use GPT-4o-mini as extractor, so you should set your openai API key in `openai_api_key_txt`.
```bash
nohup python -u script_build.py --data_source 2WikiMultiHopQA > result_build_2WikiMultiHopQA.log 2>&1 &
```
> You can also skip this step, download the pre-built Knowledge HyperGraph from [here](), and set in `expr/`.

#### 3. Set up retrieve server at 8001 port
```bash
nohup python -u script_api.py --data_source 2WikiMultiHopQA > result_api_2WikiMultiHopQA.log 2>&1 &
```

#### 4. Run GRPO/REINFORCE++/PPO training with Qwen2.5-1.5B-Instruct (Need 4 x 32GB GPUs)
```bash
# GRPO
nohup bash -u run_grpo.sh -p /mnt/hdd2/home/luohaoran/huggingface/Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d 2WikiMultiHopQA > result_run_Qwen2.5-1.5B-Instruct_2WikiMultiHopQA_grpo.log 2>&1 &
# REINFORCE++
nohup bash -u run_rpp.sh -p /mnt/hdd2/home/luohaoran/huggingface/Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d 2WikiMultiHopQA > result_run_Qwen2.5-1.5B-Instruct_2WikiMultiHopQA_rpp.log 2>&1 &
# PPO
nohup bash -u run_ppo.sh -p /mnt/hdd2/home/luohaoran/huggingface/Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d 2WikiMultiHopQA > result_run_Qwen2.5-1.5B-Instruct_2WikiMultiHopQA_ppo.log 2>&1 &
```

#### 5. Close search server 8001 port
```bash
fuser -k 8001/tcp
```


## Acknowledgement

This repo benefits from [Agent-R1](https://github.com/0russwest0/Agent-R1), [HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG), [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), [LightRAG](https://github.com/HKUDS/LightRAG), [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG), [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) and [Search-R1](https://github.com/RUCAIBox/R1-Searcher). Thanks for their wonderful works.