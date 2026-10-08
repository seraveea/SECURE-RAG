import os
import re
import string
import random
import torch
import numpy as np
from transformers import pipeline
from torch import tensor
from datasets import Dataset
from collections import Counter
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
from angle_emb import AnglE, Prompts
from langchain.text_splitter import RecursiveCharacterTextSplitter
from angle_emb.utils import cosine_similarity

os.environ["HF_HOME"] = "[placeholder]"
huggingface_cache_path = "[placeholder]"
os.environ['HF_HUB_OFFLINE'] = '1'
nltk.data.path.append('[placeholder]')


def pipeline_instance(args):
    if args.model == 'qwen-1.5b':
        model_dir = "[placeholder]"
    elif args.model == 'llama-70b':
        model_dir = "[placeholder]"
    elif args.model == 'qwen3-8b':
        model_dir = "placeholder"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map=args.device
        )
        qwen_pipeline = QwenPipeline(model, tokenizer)
        return qwen_pipeline
    else:
        model_dir = "[placeholder]"
    gen_pipeline = pipeline(
        "text-generation",
        model=model_dir,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    return gen_pipeline


def nli_pipeline_instance(args):
    model_dir = "[placeholder]"
    gen_pipeline = pipeline(
        "text-generation",
        model=model_dir,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    return gen_pipeline



def compute_f1(predicted, gold_answers):
    # the answer to an unanswered question is empty
    if not predicted and gold_answers == [""]:
        return 1.0

    def get_tokens(s):
        return normalize_answer(s).split()

    predicted_tokens = get_tokens(predicted)
    f1_scores = []
    for answer in gold_answers:
        gold_tokens = get_tokens(answer)
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            f1_scores.append(0)
            continue

        precision = num_common / len(predicted_tokens)
        recall = num_common / len(gold_tokens)
        f1_scores.append(2 * (precision * recall) / (precision + recall))
    return max(f1_scores)


def update_eval(answer, standard_answer, total, f1, accuracy, exact_match, correct, question_id):
    total += 1
    em_for_this_question = metric_max_over_ground_truths(exact_match_score, answer, standard_answer)
    
    exact_match += em_for_this_question
    if em_for_this_question:
        current_rougel = 1
        f1 += 1
    else:
        f1 += metric_max_over_ground_truths(f1_score, answer, standard_answer)
        current_rougel = compute_rouge_l(answer, standard_answer)

    if current_rougel > 0.3:
        correct += 1
        if question_id not in accuracy.keys():
            accuracy[question_id] = [1]
        else:
            accuracy[question_id].append(1)
    else:
        if question_id not in accuracy.keys():
            accuracy[question_id] = [0]
        else:
            accuracy[question_id].append(0)
    return total, f1, accuracy, exact_match, correct, em_for_this_question


def update_rougel(answer, standard_answer, total, f1, accuracy, exact_match, correct, question_id, rougel):
    total += 1
    em_for_this_question = metric_max_over_ground_truths(exact_match_score, answer, standard_answer)
    exact_match += em_for_this_question
    if em_for_this_question:
        current_rougel = 1
        f1 += 1
    else:
        f1 += metric_max_over_ground_truths(f1_score, answer, standard_answer)
        current_rougel = compute_rouge_l(answer, standard_answer)

    rougel += current_rougel
    if current_rougel > 0.3:
        correct += 1
        if question_id not in accuracy.keys():
            accuracy[question_id] = [1]
        else:
            accuracy[question_id].append(1)
    else:
        if question_id not in accuracy.keys():
            accuracy[question_id] = [0]
        else:
            accuracy[question_id].append(0)
    return total, f1, accuracy, exact_match, correct, em_for_this_question, rougel


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)



def normalize_answer(s: object) -> object:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def compute_exact_match(predicted, gold_answers):
    return int(predicted in gold_answers)


def reply_pp(item):
    text = normalize_answer(item)
    useless = [
        'based on the provided context, ',
        'based on the provided context',
        'the answer is',
        'i can answer the question',
    ]
    for x in useless:
        text = text.replace(x, "")
    if '404' in text:
        text = ''
    return text


def lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(candidate, reference):
    lcs = lcs_length(candidate, reference)
    precision = lcs / len(candidate) if candidate else 0
    recall = lcs / len(reference) if reference else 0
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_rouge_l(candidate, list_of_reference):
    return max([rouge_l(candidate.split(), i.split()) for i in list_of_reference])


class UAE_retrieval_embeder:
    def __init__(self, device='cpu', chunk_size=512, chunk_overlap=64, chunk_type='recursive', threshold=0.6):
        random_seed = 42
        random.seed(random_seed)
        model_path = '[placeholder]'
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = AnglE.from_pretrained(model_path, pooling_strategy='cls',
                                           cache_dir=huggingface_cache_path,
                                           local_files_only=True).to(self.device)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_type = chunk_type
        self.threshold = threshold

    def encode(self, query: str, doc_list: list) -> object:
        if self.chunk_type == 'recursive':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            text_list = []
            for text in doc_list:
                text_list += text_splitter.split_text(text)
        else:
            from nltk.tokenize import sent_tokenize
            text_list = []
            for text in doc_list:
                sentences = sent_tokenize(text)
                if len(sentences) == 0:
                    continue
                sent_embeddings = []
                for sent in sentences:
                    emb = self.model.encode([sent])[0]
                    sent_embeddings.append(emb)
                chunks = []
                current_chunk = [sentences[0]]
                current_length = len(sentences[0])
                
                for i in range(len(sentences)-1):
                    sim = cosine_similarity(
                        sent_embeddings[i].flatten(),
                        sent_embeddings[i+1].flatten()
                    )
                    next_sent_length = len(sentences[i+1])
                    
                    if sim > self.threshold and current_length + next_sent_length <= 512:
                        current_chunk.append(sentences[i+1])
                        current_length += next_sent_length
                    else:
                        if len(current_chunk) > 0:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [sentences[i+1]]
                        current_length = next_sent_length
                        
                if len(current_chunk) > 0:
                    chunks.append(" ".join(current_chunk))
                text_list.extend(chunks)

        with torch.no_grad():
            qv = self.model.encode(Prompts.C.format(text=query))
            if len(text_list) != 0:
                # encoding one by one to reduce GPU memory usage
                doc_vecs = []
                # doc_vecs = self.model.encode(text_list)  # this code will cost gpu memory high load
                for text in text_list:
                    doc_vecs.append(self.model.encode([text])[0])
            else:
                doc_vecs = []
            current_file = Dataset.from_dict({
                "chunk_id": [i for i in range(len(text_list))],
                "text": text_list,
                'embeddings': doc_vecs
            })

        del doc_vecs, text_list
        torch.cuda.empty_cache()

        return qv, current_file
    
    def encode_query(self, query):
        with torch.no_grad():
            qv = self.model.encode(Prompts.C.format(text=query))
            return qv

    def set_chunk_type(self, chunk_type):
        self.chunk_type = chunk_type

def separate_dictionary(original_dict, score):
    # Initialize empty dictionaries
    separated_dicts = [{} for _ in range(len(score))]

    # Iterate over the keys of the original dictionary
    for key in original_dict:
        # except embeddings here
        # if key != 'embeddings':
        # Iterate over the number of values specified
        for i in range(len(score)):
            # Assign one value to each new dictionary
            separated_dicts[i][key] = original_dict[key][i]

    for i in range(len(score)):
        separated_dicts[i]['relevant score'] = float(score[i])

    return separated_dicts


def find_most_common(answers):
    counter = Counter(answers)
    most_common = counter.most_common()
    if not most_common:
        return ''
    if len(most_common) == 1:
        return most_common[0][0]
    if most_common[0][1] > most_common[1][1]:
        return most_common[0][0]

    return answers[0]


class QwenPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, inputs, temperature=0.01, max_new_tokens=256, enable_thinking=False):
        messages = inputs
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
    
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
        # Parse content (skip thinking content)
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
        except ValueError:
            index = 0
        
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return thinking_content, content
    

if __name__ == "__main__":
    # Example usage
    embeder = UAE_retrieval_embeder()

    file_paths = ["data/triviaqa-rc/evidence/wikipedia/Abuja.txt"]
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        q, data = embeder.encode('test', [content])

        # Preview the dataset
        print(q)
        print(data[0])
    # Access the first chunk
