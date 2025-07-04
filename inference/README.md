# Graph-R1 Inference

To set up Graph-R1 inference, we still use ```Graph-R1``` as the working directory. 

#### 1. First, you need to merge the model weights.
```bash
python3 verl/scripts/model_merger.py --backend fsdp --hf_model_path Qwen/Qwen2.5-3B-Instruct --local_dir checkpoints/Graph-R1/Qwen2.5-3B-Instruct_2WikiMultiHopQA_grpo/global_step_40/actor --target_dir checkpoints/Graph-R1/Qwen2.5-3B-Instruct_2WikiMultiHopQA_grpo/model
```

#### 2. Then, you can start the vLLM server with the merged model weights at 8002 port.
```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve checkpoints/Graph-R1/Qwen2.5-3B-Instruct_2WikiMultiHopQA_grpo/model --served-model-name agent --port 8002 > result_modelapi_Qwen2.5-3B-Instruct_2WikiMultiHopQA_grpo.log 2>&1 &
```

#### 3. Set up retrieve server at 8001 port.
```bash
fuser -k 8001/tcp
nohup python -u script_api.py --data_source 2WikiMultiHopQA > result_api_2WikiMultiHopQA.log 2>&1 &
```

#### 4. Now, you can run the inference script to ask questions.
```bash
python3 agent/vllm_infer/run.py --question "Which film has the director died earlier, Nameless Woman or Handle With Care (1977 Film)?"
```

#### 5. When you finish the inference, you can stop the vLLM and retrieve server by killing port 8002 and 8001.
```bash
pkill -TERM -P $(lsof -t -i :8002); kill -9 $(lsof -t -i :8002)
fuser -k 8001/tcp
```