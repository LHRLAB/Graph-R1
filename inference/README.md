# Graph-R1 Inference

To set up Graph-R1 inference, we still use ```Graph-R1``` as the working directory. 

#### 1. First, you need to merge the model weights and then start the vLLM server.
```bash
python3 verl/scripts/model_merger.py --backend fsdp --hf_model_path Qwen/Qwen2.5-3B-Instruct --local_dir checkpoints/Graph-R1/Qwen2.5-3B-Instruct_2WikiMultiHopQA_grpo/global_step_40/actor --target_dir checkpoints/Graph-R1/Qwen2.5-3B-Instruct_2WikiMultiHopQA_grpo/model
```

#### 2. Then, you can start the vLLM server with the merged model weights.
```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve checkpoints/Graph-R1/Qwen2.5-3B-Instruct_2WikiMultiHopQA_grpo/model --served-model-name agent --port 8002 > result_modelapi_Qwen2.5-3B-Instruct_2WikiMultiHopQA_grpo.log 2>&1 &
```

#### 3. Now, you can run the inference script to ask questions.
```bash
python3 agent/vllm_infer/run.py --question "Which film has the director died first, Watch Your Stern or Requiescant?"
```

#### 4. When you finish the inference, you can stop the vLLM server by killing the process running on port 8002.
```bash
pkill -TERM -P $(lsof -t -i :8002); kill -9 $(lsof -t -i :8002)
```