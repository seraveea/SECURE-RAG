# Official implementation of DENSE-RAG

## Environments
1. We recommend using python 3.12.9
2. Install the required packages in the [requirements.txt](requirements.txt) file.

## Datasets
Please download open-book QA datasets using official link, and put them into the `data` folder.
1. [TriviaQA](https://nlp.cs.washington.edu/triviaqa/), please download the TriviaQA version 1.0 for RC.

2. Natural Questions, please follow the instructions in [huggingface link](https://huggingface.co/datasets/sentence-transformers/natural-questions).

3. AmbigNQ, please follow the instructions in [huggingface link](https://huggingface.co/datasets/sewon/ambig_qa).

4. WikiQA, please use [official dropbox link](https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?rlkey=ira57daau8lxfj022xvk1irju&e=1) shared by 2WikimultihopQA team.

Since we divide four datasets into certain/uncertain questions, we provide all uncertain questions id in `data/[dataset]_uc_id_list.json`.


## LLM backbones & Embedding models
Please download LLM backbones and embedding models from huggingface, [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), 
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) and [UAE-Large-V1 embeder](https://huggingface.co/WhereIsAI/UAE-Large-V1).

Please replace `[placeholder]` in [utils/utils.py](utils/utils.py) with the path of your downloaded model. Also we set `os.environ['HF_HUB_OFFLINE'] = '1'` to use local model. If you prefer online mode, please remove this line and change
 `[placeholder]` to a huggingface model link.

## Reproduce the results

For 1.5B and 8B models, we recommend use GPU devices with minimum 24GB memory (3090 is enough).

 For 70B model, we recommend use GPU devices with minimum 120GB memory (2xA100 is enough).

### DENSE_eval on Llama3 8B
```
python rag_exp.py --result_file output/trivia/eval_trivia.json --device cuda:1 --context_type single_replace --dataset trivia 

python rag_exp.py --result_file output/nq/eval_nq.json --device cuda:1 --context_type single_replace --dataset nq

python rag_exp.py --result_file output/ambig/eval_ambig.json --device cuda:1 --context_type single_replace --dataset ambignq 

python rag_exp.py --result_file output/2wqa/eval_2wqa.json --device cuda:1 --context_type single_replace --dataset 2wqa
```
### DENSE-RAG on Llama3 8B
```
python rag_exp.py --rag_method rerank --dataset trivia --result_file output/trivia/dense_trivia.json

python rag_exp.py --rag_method rerank --dataset nq --result_file output/nq/dense_NQ.json --split ''

python rag_exp.py --rag_method rerank --dataset 2wqa --result_file output/2wqa/dense_2wqa.json --split ''

python rag_exp.py --rag_method rerank --dataset ambignq --result_file output/ambig/dense_ambig.json
```
### DENSE-RAG on Qwen2.5 1.5B
```
python rag_exp.py --rag_method rerank --dataset trivia --result_file output/trivia/dense_trivia_qwen.json --model qwen-1.5b

python rag_exp.py --rag_method rerank --dataset nq --result_file output/nq/dense_NQ_qwen.json --model qwen-1.5b

python rag_exp.py --rag_method rerank --dataset 2wqa --result_file output/2wqa/dense_2wqa_qwn.json --model qwen-1.5b

python rag_exp.py --rag_method rerank --dataset ambignq --result_file output/ambig/dense_ambig_qwen.json --model qwen-1.5b
```
### DENSE-RAG on Llama3 70B
```
python rag_exp.py --rag_method rerank --dataset trivia --result_file output/trivia/dense_trivia_70b.json --model llama-70b

python rag_exp.py --rag_method rerank --dataset nq --result_file output/nq/dense_NQ_70b.json --model llama-70b

python rag_exp.py --rag_method rerank --dataset ambignq --result_file output/ambig/dense_ambig_70b.json --model llama-70b

python rag_exp.py --rag_method rerank --dataset 2wqa --result_file output/2wqa/dense_2wqa_70b.json --model llama-70b
```

## Quick evaluation
If you don't want to run the whole pipeline, we proivide DENSE-RAG result. You can use the following command to evaluate the result in  `output/[data_set]/`. Please run the `python evaluation_[trivia/nq/ambig/2wqa].py` file for evaluation.

