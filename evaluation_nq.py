import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from utils.nq_utils import extract_NQ
from utils.utils import normalize_answer, reply_pp, update_eval, find_most_common


def get_files(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def nq_evaluation(args):
    nq_data = load_dataset("google-research-datasets/natural_questions", "dev", cache_dir="../LLM_QA/data/NQ")
    question_set = nq_data['validation']
    reply_set = []
    for temp in args.result_list:
        reply_set += get_files(temp)

    total, uc_total, c_total, exact_match, uc_exact_match, c_exact_match = 0, 0, 0, 0, 0, 0
    f1, uc_f1, c_f1, correct, uc_correct, c_correct = 0.0, 0.0, 0.0, 0, 0, 0
    q_num, uq_num, cq_num = 0, 0, 0
    accuracy, uc_accuracy, c_accuracy = {}, {}, {}
    uc_wrong_reason, c_wrong_reason, uc_right_reason, c_right_reason= {},{},{},{}


    if args.style == 'split_inference':
        with open(args.id_file, 'r') as f:
            uc_id = json.load(f)
    if args.style == 'split':
        uc_id = []




    for question_instance in tqdm(question_set):
        query, answer_set, candidate_set, document, question_id = extract_NQ(question_instance)
        question_id = str(question_id)
        
        standard_answer = []
        for t in answer_set:
            standard_answer += t['short_answer']
        

        standard_answer = [x for x in standard_answer if x and x != '']
        standard_answer = list(set([normalize_answer(a) for a in standard_answer]))

        if len([d for d in reply_set if d.get("question_id") == question_id]) == 0 or len(standard_answer) == 0:
            continue
        else:
            current_answer_set = [d for d in reply_set if d.get("question_id") == question_id]
            if len(current_answer_set) == 1:
                current_answer = current_answer_set[0]
            else:
                current_answer = current_answer_set[-1]
            if args.style == 'split':
                uc_score = current_answer['uncertainty_score']
            candidate_answer = current_answer['answer']
            candidate_answer = [reply_pp(item) for item in candidate_answer]
            answer_dict = current_answer
            q_num += 1

            most_common_answer = find_most_common(candidate_answer)
            if most_common_answer:
                candidate_answer = [most_common_answer]
            else:
                candidate_answer = [candidate_answer[0]]
    
            for answer in candidate_answer:
                total, f1, accuracy, exact_match, correct, current_em = update_eval(answer, standard_answer, total, f1,
                                                                        accuracy, exact_match, correct, question_id)
                
                


            if args.style == 'split':
                if uc_score > 0.2:
                    uc_id.append(question_id)
                    uq_num += 1
                    for answer in candidate_answer:
                        uc_total, uc_f1, uc_accuracy, uc_exact_match, uc_correct, current_em = update_eval(answer, standard_answer,
                                                                                               uc_total, uc_f1, uc_accuracy,
                                                                                               uc_exact_match, uc_correct,
                                                                                               question_id)
                else:
                    cq_num += 1
                    for answer in candidate_answer:
                        c_total, c_f1, c_accuracy, c_exact_match, c_correct, current_em = update_eval(answer, standard_answer,
                                                                                          c_total, c_f1, c_accuracy,
                                                                                          c_exact_match, c_correct,
                                                                                          question_id)
            if args.style == 'split_inference':
                if question_id in uc_id:
                    uq_num += 1
                    uc_id.append(question_id)
                    for answer in candidate_answer:
                        uc_total, uc_f1, uc_accuracy, uc_exact_match, uc_correct, current_em = update_eval(answer, standard_answer,
                                                                                               uc_total, uc_f1, uc_accuracy,
                                                                                               uc_exact_match, uc_correct,
                                                                                               question_id)
                        if current_em != 1 and 'end_type' in answer_dict:
                            if answer_dict['end_type'] not in uc_wrong_reason:
                                uc_wrong_reason[answer_dict['end_type']] = 1
                            else:
                                uc_wrong_reason[answer_dict['end_type']] += 1
                        if current_em == 1 and 'end_type' in answer_dict:
                            if answer_dict['end_type'] not in uc_right_reason:
                                uc_right_reason[answer_dict['end_type']] = 1
                            else:
                                uc_right_reason[answer_dict['end_type']] += 1
                else:
                    cq_num += 1
                    for answer in candidate_answer:
                        c_total, c_f1, c_accuracy, c_exact_match, c_correct, current_em = update_eval(answer, standard_answer,
                                                                                          c_total, c_f1, c_accuracy,
                                                                                          c_exact_match, c_correct,
                                                                                          question_id)
                        if current_em != 1 and 'end_type' in answer_dict:
                            if answer_dict['end_type'] not in c_wrong_reason:
                                c_wrong_reason[answer_dict['end_type']] = 1
                            else:
                                c_wrong_reason[answer_dict['end_type']] += 1
                        if current_em == 1 and 'end_type' in answer_dict:
                            if answer_dict['end_type'] not in c_right_reason:
                                c_right_reason[answer_dict['end_type']] = 1
                            else:
                                c_right_reason[answer_dict['end_type']] += 1



    exact_match = 100.0*exact_match/total
    f1 = 100.0*f1/total
    correct = 100.0*correct/total
    if args.style == 'split' or args.style == 'split_inference':
        uc_exact_match = 100.0*uc_exact_match/uc_total
        uc_f1 = 100.0*uc_f1/uc_total
        uc_correct = 100.0*uc_correct/uc_total
        if c_total !=0:
            c_exact_match = 100.0*c_exact_match/c_total
            c_f1 = 100.0*c_f1/c_total
            c_correct = 100.0*c_correct/c_total


    if args.style == 'split':
        with open(args.id_file, 'w') as f:
            json.dump(uc_id, f)

    return ({"exact_match": exact_match, "f1": f1, "accuracy": correct,
            "uc_exact_match": uc_exact_match, "uc_f1": uc_f1, "uc_accuracy": uc_correct,
            "c_exact_match": c_exact_match, "c_f1": c_f1, "c_accuracy": c_correct,
            "number of all questions": q_num, "number of uncertain questions": uq_num,
            "number of certain questions": cq_num
            }, accuracy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_file', default='data/nq_uc_id_list.json')
    parser.add_argument('--style', default='split_inference', help='split, split_inference. If swith to split mode will cover current uncertain id list')
    parser.add_argument('--result_list',
                        default=['output/nq/secure_nq.json'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    myargs = parse_args()
    result, accuracy = nq_evaluation(myargs)
    print(result)




