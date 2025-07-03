# Graph-R1 Evaluation

### Preparation
First, to evaluate Graph-R1, we should use ```evaluation``` as the working directory. 
```bash
cd evaluation
```
Then, we need to set openai api key in ```openai_api_key.txt``` file.

###  Eval for Graph-R1
```bash
python get_remote_score.py --dir  ../expr_results/Qwen2.5-3B-Instruct_2WikiMultiHopQA_grpo
# python get_remote_score.py --dir  ../expr_results/Qwen2.5-3B-Instruct_HotpotQA_grpo
# python get_remote_score.py --dir  ../expr_results/Qwen2.5-3B-Instruct_Musique_grpo
# python get_remote_score.py --dir  ../expr_results/Qwen2.5-3B-Instruct_NQ_grpo
# python get_remote_score.py --dir  ../expr_results/Qwen2.5-3B-Instruct_PopQA_grpo
# python get_remote_score.py --dir  ../expr_results/Qwen2.5-3B-Instruct_TriviaQA_grpo
```