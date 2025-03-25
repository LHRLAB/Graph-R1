# Graph-R1

### Install Environment
```bash
conda create -n graphr1 python==3.11.11
conda activate graphr1
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install -r requirements.txt
# pip install debugpy==1.8.0
# pip install "ray[default]" debugpy
```

### Dataset Preparation
> We conduct experiments on seven datasets: 2WikiMultiHopQA, HotpotQA, Musique, NarrativeQA, NQ, PopQA, and TriviaQA. You can download them from [here](), and set the data path in `datasets/`.

### Quick Start: Graph-R1 on 2WikiMultiHopQA
#### 1. Preprocess 2WikiMultiHopQA dataset to parquet format
```bash
python script_process.py --data_source 2WikiMultiHopQA
# python script_process.py --data_source HotpotQA
# python script_process.py --data_source NQ
```

#### 2. Extract contexts and build Knowledge HyperGraph (Optional)
For the extracted contexts, we insert them into the Knowledge HyperGraph.
```bash
nohup python -u script_build.py --data_source 2WikiMultiHopQA > result_build_2WikiMultiHopQA.log 2>&1 &
# nohup python -u script_build.py --data_source HotpotQA > result_build_HotpotQA.log 2>&1 &
# nohup python -u script_build.py --data_source NQ > result_build_NQ.log 2>&1 &
```
> You can also skip this step, download the pre-built Knowledge HyperGraph from [here](), and set in `expr/`.

#### 3. Set up retrieve server at 8001 port
Set up Graph-R1 retrieve server
```bash
nohup python -u script_api.py --data_source 2WikiMultiHopQA > result_api_2WikiMultiHopQA.log 2>&1 &
# nohup python -u script_api.py --data_source NQ > result_api_NQ.log 2>&1 &
```

#### 4. Run GRPO/REINFORCE++/PPO training with Qwen2.5-1.5B-Instruct
```bash
# bash run_grpo.sh -p /mnt/hdd2/home/luohaoran/huggingface/Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d 2WikiMultiHopQA
nohup bash -u run_grpo.sh -p /mnt/hdd2/home/luohaoran/huggingface/Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d 2WikiMultiHopQA > result_run_Qwen2.5-1.5B-Instruct_2WikiMultiHopQA_grpo.log 2>&1 &
# nohup bash -u run_grpo.sh -p /mnt/hdd2/home/luohaoran/huggingface/Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d NQ > result_run_Qwen2.5-1.5B-Instruct_NQ_grpo.log 2>&1 &

nohup bash -u run_rpp.sh -p /mnt/hdd2/home/luohaoran/huggingface/Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d 2WikiMultiHopQA > result_run_Qwen2.5-1.5B-Instruct_2WikiMultiHopQA_rpp.log 2>&1 &

nohup bash -u run_ppo.sh -p /mnt/hdd2/home/luohaoran/huggingface/Qwen/Qwen2.5-1.5B-Instruct -m Qwen2.5-1.5B-Instruct -d 2WikiMultiHopQA > result_run_Qwen2.5-1.5B-Instruct_2WikiMultiHopQA_ppo.log 2>&1 &
```

#### 4. Close search server 8001 port
```bash
fuser -k 8001/tcp
```




## Acknowledgement

This repo benefits from [Agent-R1](https://github.com/0russwest0/Agent-R1), [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), [LightRAG](https://github.com/HKUDS/LightRAG), [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG), [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) and [Search-R1](https://github.com/RUCAIBox/R1-Searcher). Thanks for their wonderful works.