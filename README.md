# Graph-R1

### Installation

```bash
conda create -n graphr1 python=3.11
conda activate graphr1
pip install -e .
```

### Data Preparation
```bash
python prepare_postdata.py
```

### GRPO Training

```bash
bash examples/run_qwen2_5_3b_gsm8k.sh
nohup bash examples/run_qwen2_5_3b_gsm8k.sh >> results_grpo_qwen2_5_3b_gsm8k.log 2>&1 &
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/qwen2_5_3b_math/global_step_45
```

