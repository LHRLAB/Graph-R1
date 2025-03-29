# Graph-R1

Graph-R1: Incentivizing Reasoning-on-Graph Capability in LLM via Reinforcement Learning

## Overview

<div align="center">
  <img src="figs/1.png" width="90%"/>
</div>

Recently, the **GraphRAG** method effectively addresses the data silos issue, significantly enhancing knowledge retrieval efficiency. Nevertheless, the disconnect between graph-structured knowledge and language modalities continues to constrain performance. 

To bridge this gap, we propose **Graph-R1**, an **end-to-end reinforcement learning (RL)** framework designed to improve **reasoning-on-graph capabilities** of large language models (LLMs). 

Specifically, we constructs a **knowledge hypergraph** using **n-ary relation extraction** from [HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG). We then employ an explicit reward mechanism within RL, enabling the LLM to iteratively execute a "**think–generate query–retrieve subgraph–rethink**" reasoning cycle. This iterative approach enables the model to effectively leverage graph knowledge to produce high-quality answers. 

By integrating structured knowledge into LLM reasoning more flexibly via reinforcement learning, Graph-R1 holds promise for aplications in **knowledge-intensive fields** such as healthcare, finance, and law.

**Our paper is coming soon.**

## Experimental Results
**Results on Different RL Algorithms:**
<table>
  <tr>
    <td><img src="./figs/1_f1.png" width="100%"/></td>
    <td><img src="./figs/1_em.png" width="100%"/></td>
    <td><img src="./figs/1_res.png" width="100%"/></td>
    <td><img src="./figs/1_turn.png" width="100%"/></td>
  </tr>
</table>

**Results on Different GraphRAG Datasets:**
<table>
  <tr>
    <td><img src="./figs/2_f1.png" width="100%"/></td>
    <td><img src="./figs/2_em.png" width="100%"/></td>
    <td><img src="./figs/2_res.png" width="100%"/></td>
    <td><img src="./figs/2_turn.png" width="100%"/></td>
  </tr>
</table>

**Results on Different Parameter Scale of LLM:**
<table>
  <tr>
    <td><img src="./figs/3_f1.png" width="100%"/></td>
    <td><img src="./figs/3_em.png" width="100%"/></td>
    <td><img src="./figs/3_res.png" width="100%"/></td>
    <td><img src="./figs/3_turn.png" width="100%"/></td>
  </tr>
</table>

## Graph-R1 Implementation

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
nohup bash -u run_grpo.sh -p Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d 2WikiMultiHopQA > result_run_Qwen2.5-1.5B-Instruct_2WikiMultiHopQA_grpo.log 2>&1 &
# REINFORCE++
nohup bash -u run_rpp.sh -p Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d 2WikiMultiHopQA > result_run_Qwen2.5-1.5B-Instruct_2WikiMultiHopQA_rpp.log 2>&1 &
# PPO
nohup bash -u run_ppo.sh -p Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d 2WikiMultiHopQA > result_run_Qwen2.5-1.5B-Instruct_2WikiMultiHopQA_ppo.log 2>&1 &
```

#### 5. Close search server 8001 port
```bash
fuser -k 8001/tcp
```

## BibTex

If you find this work is helpful for your research, please cite:

```bibtex
@misc{luo2025graphr1,
      title={Graph-R1: Incentivizing Reasoning-on-Graph Capability in LLM via Reinforcement Learning}, 
      author={Haoran Luo},
      year={2025},
      organization = {GitHub},
      url={https://github.com/LHRLAB/Graph-R1}, 
}
```

For further questions, please contact: luohaoran@bupt.edu.cn.

## Acknowledgement

This repo benefits from [Agent-R1](https://github.com/0russwest0/Agent-R1), [HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG), [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), [LightRAG](https://github.com/HKUDS/LightRAG), [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG), [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) and [Search-R1](https://github.com/RUCAIBox/R1-Searcher). Thanks for their wonderful works.