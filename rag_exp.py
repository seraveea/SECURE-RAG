import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from agent.rag_agent import RAGAgent
from utils.rephrase_utils import *
from utils.nq_utils import extract_NQ
from utils.se_utils import EntailmentDeberta, llm_nli_agent
from datasets.utils.logging import set_verbosity_error, disable_progress_bar
from utils.utils import UAE_retrieval_embeder, pipeline_instance


set_verbosity_error()
disable_progress_bar()


def get_questions(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def split_dataset(dataset, split_str, is_trivia=False, is_nq=False):
    if len(split_str) > 0:
        numerator, denominator = map(int, split_str.split('/'))
        if is_nq:
            total_length = len(dataset['validation'])
            base_size = total_length // denominator
            remainder = total_length % denominator
            start = 0
            for i in range(denominator):
                if i < remainder:
                    end = start + base_size + 1
                else:
                    end = start + base_size
                if i == numerator - 1:
                    qs = dataset['validation'][start:end]
                    return [dict(zip(qs.keys(), values)) for values in zip(*qs.values())]
                start = end
        else:
            groups = []
            total_length = len(dataset)
            base_size = total_length // denominator
            remainder = total_length % denominator
            start = 0
            for i in range(denominator):
                if i < remainder:
                    end = start + base_size + 1
                else:
                    end = start + base_size
                groups.append(dataset[start:end])
                start = end
            return groups[numerator - 1]
    return dataset


def wikiqa_pipeline(args):
    question_set = get_questions(args.question_file)
    embeder = UAE_retrieval_embeder(args.device, chunk_size=int(args.chunk_size), 
                                   chunk_overlap=int(args.chunk_overlap), chunk_type=args.chunk_methods, threshold=float(args.chunk_threshold))
    qa_pipeline = pipeline_instance(args)
    
    if args.nli_type == 'nlp':
        NLI_agent = EntailmentDeberta(args.device)
    else:
        NLI_agent = llm_nli_agent(qa_pipeline)
    
    rag_agent = RAGAgent(args)

    question_set = split_dataset(question_set, args.split)

    answers = []
    for question_instance in tqdm(question_set):
        retrieval_files = [''.join(item[1]) for item in question_instance['context']]
        qid = question_instance['_id']
        query = question_instance['question']

        current_question = rag_agent.run(qa_pipeline, query, retrieval_files, embeder, NLI_agent, qid)
        current_question['question_id'] = qid
        answers.append(current_question)

    json.dump(answers, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'results are saved in {args.result_file}')
    return None


def ambig_pipeline(args):
    question_set = get_questions(args.question_file)
    embeder = UAE_retrieval_embeder(args.device, chunk_size=int(args.chunk_size), chunk_overlap=int(args.chunk_overlap),
                                    chunk_type=args.chunk_methods, threshold=float(args.chunk_threshold))
    qa_pipeline = pipeline_instance(args)
    
    if args.nli_type == 'nlp':
        NLI_agent = EntailmentDeberta(args.device)
    else:
        NLI_agent = llm_nli_agent(qa_pipeline)
    
    rag_agent = RAGAgent(args)

    question_set = split_dataset(question_set, args.split)


    answers = []
    for question_instance in tqdm(question_set):
        if 'singleAnswer' in [i['type'] for i in question_instance['annotations']]:
            retrieval_files = [''.join(item) for item in question_instance['articles_plain_text']]
            qid = question_instance['id']
            query = question_instance['question']

            current_question = rag_agent.run(qa_pipeline, query, retrieval_files, embeder, NLI_agent, qid)
            current_question['question_id'] = qid
            answers.append(current_question)

    json.dump(answers, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'results are saved in {args.result_file}')
    return None


def trivia_pipeline(args):
    if args.dataset == 'trivia_web':
        prefix = 'data/triviaqa-rc/evidence/web/'
    else:
        prefix = 'data/triviaqa-rc/evidence/wikipedia/'
    question_set = get_questions(args.question_file)['Data']
    embeder = UAE_retrieval_embeder(args.device, chunk_size=int(args.chunk_size), 
                                   chunk_overlap=int(args.chunk_overlap), 
                                   chunk_type=args.chunk_methods, threshold=float(args.chunk_threshold))
    qa_pipeline = pipeline_instance(args)
    
    if args.nli_type == 'nlp':
        NLI_agent = EntailmentDeberta(args.device)
    else:
        NLI_agent = llm_nli_agent(qa_pipeline)
    
    rag_agent = RAGAgent(args)

    question_set = split_dataset(question_set, args.split, is_trivia=True)

    answers = []

    for question_instance in tqdm(question_set):
        query = question_instance['Question']
        retrieval_files = [prefix + i['Filename'] for i in question_instance['EntityPages']]
        qid = question_instance['QuestionId']
        content = []
        for file_path in retrieval_files:
            try:
                file_path_ascii = file_path.encode('ascii', 'replace').decode('ascii')
                if not os.path.exists(file_path):
                    if os.path.exists(file_path_ascii):
                        file_path = file_path_ascii
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content.append(file.read())
                else:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content.append(file.read())
            except Exception as e:
                print(f"handle {file_path} : {e}")
                print(f"warning: file {file_path} not exist.")
                continue

        current_question = rag_agent.run(qa_pipeline, query, content, embeder, NLI_agent, qid)
        current_question['question_id'] = qid
        answers.append(current_question)

    json.dump(answers, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'results are saved in {args.result_file}')
    return None


def nq_pipeline(args):
    nq_data = load_dataset("google-research-datasets/natural_questions", "dev", cache_dir="data/NQ")
    embder = UAE_retrieval_embeder(args.device, chunk_size=int(args.chunk_size), 
                                  chunk_overlap=int(args.chunk_overlap), 
                                  chunk_type=args.chunk_methods, threshold=float(args.chunk_threshold))
    qa_pipeline = pipeline_instance(args)
    
    if args.nli_type == 'nlp':
        NLI_agent = EntailmentDeberta(args.device)
    else:
        NLI_agent = llm_nli_agent(qa_pipeline)
    
    rag_agent = RAGAgent(args)

    qs = split_dataset(nq_data, args.split, is_nq=True) if len(args.split) > 0 else nq_data['validation']
    
    
    answers = []
    for question_instance in tqdm(qs):
        query, answer_set, candidate_set, document, qid = extract_NQ(question_instance)
        current_question = rag_agent.run(qa_pipeline, query, [document], embder, NLI_agent, qid)
        current_question['question_id'] = qid
        answers.append(current_question)

    json.dump(answers, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'results are saved in {args.result_file}')
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='hotpot', 
                        help="""
                        choose dataset:
                        normal QA：
                        trivia, nq, trivia_web, ambignq
                        multihop QA：
                        2wqa
                        """)
    parser.add_argument('--model', default='llama', 
                        help="""choose backbone: 
                        llama (default 8B version),  
                        qwen-1.5b,
                        llama-70b,
                        qwen3
                        """)
    parser.add_argument('--question_file', default='', help='Path to the question file')
    parser.add_argument('--result_file', default='', help='Path to save the results')
    parser.add_argument('--chunk_size', default=512, help='Size of text chunks') 
    parser.add_argument('--chunk_overlap', default=64, help='Overlap size of text chunks')
    parser.add_argument('--split', default='', help='Dataset split ratio')
    parser.add_argument('--device', default="cuda:0", help='Device to use')
    parser.add_argument('--rag_method', default="vanilla", help='RAG type: vanilla, single_replace, adaptive_chunk, rerank')
    parser.add_argument('--nli_type', default="llm", help='Choose NLI type: nlp or llm, nlp performs poorly')
    parser.add_argument('--topk', default=5, help='Number of text chunks needed per query')
    parser.add_argument('--candidate', default=30, help='Number of candidates')
    parser.add_argument('--repeat', default=1, help='Number of times each question needs to be answered')
    parser.add_argument('--chunk_methods', default='recursive', help='Text chunking method: recursive, semantic')
    parser.add_argument('--chunk_threshold', default=0.6, help='Threshold for text chunking')
    args = parser.parse_args()
    
    # Trivia, NQ don't need question file
    if args.dataset == 'trivia':
        args.question_file = 'data/triviaqa-rc/qa/wikipedia-dev.json'
    elif args.dataset == '2wqa':
        args.question_file = 'data/2wqa/dev.json'
    elif args.dataset == 'ambignq':
        args.question_file = 'data/ambignq/dev_with_evidence_articles.json'
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == '2wqa':
        wikiqa_pipeline(args)
    elif args.dataset == 'trivia':
        trivia_pipeline(args)
    elif args.dataset == 'nq':
        nq_pipeline(args)
    elif args.dataset == 'ambignq':
        ambig_pipeline(args)
    else:
        print(f"warning: unsupported datasets: '{args.dataset}'.")