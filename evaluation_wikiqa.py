import os
import json
import argparse
from tqdm import tqdm
from utils.utils import reply_pp, update_eval, find_most_common


def get_questions(path):
    # with open(path, "r", encoding="utf-8") as file:
    #     data = json.load(file)
    # return data
    print(f"\n[DEBUG] Loading file: {path}")
    print(f"[DEBUG] Exists: {os.path.exists(path)}, Size: {os.path.getsize(path)} bytes")

    with open(path, 'rb') as f:
        raw = f.read()
        print(f"[DEBUG] Raw bytes: {raw[:50]!r}")

    try:
        text = raw.decode('utf-8')
        print(f"[DEBUG] Decoded text: {text[:100]!r}")
    except UnicodeDecodeError as e:
        raise ValueError(f"[ERROR] File is not UTF-8: {e}")

    if not text.strip():
        raise ValueError(f"[ERROR] File is empty after decoding: {path}")

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"[ERROR] Invalid JSON in file: {e}")


def wikiqa_evaluation(args):
    question_set = get_questions(args.question_file)
    answer_set = []
    for temp in args.result_list:
        answer_set += get_questions(temp)

    if args.style == 'split_inference':
        with open(args.id_file, 'r') as f:
            uc_id = json.load(f)
    if args.style == 'split':
        uc_id = []

    total, uc_total, c_total, exact_match, uc_exact_match, c_exact_match = 0, 0, 0, 0, 0, 0
    f1, uc_f1, c_f1, correct, uc_correct, c_correct = 0.0, 0.0, 0.0, 0, 0, 0
    accuracy, uc_accuracy, c_accuracy = {}, {}, {}
    uc_wrong_reason, c_wrong_reason, uc_right_reason, c_right_reason = {}, {}, {}, {}
    q_num, uq_num, cq_num = 0, 0, 0

    new_uc_score = []

    for question_instance in tqdm(question_set):
        question_id = question_instance['_id']
        standard_answer = [question_instance['answer']]


        if len([d for d in answer_set if d.get("question_id") == question_id]) == 0:
            continue
        elif [d for d in answer_set if d.get("question_id") == question_id][0]['answer'] == 'ERROR, no documents found in evidence.':
            continue
        else:
            current_answer_set = [d for d in answer_set if d.get("question_id") == question_id]
            if len(current_answer_set) == 0:
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
                    uq_num += 1
                    uc_id.append(question_id)
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

        if c_total != 0:
            c_exact_match = 100.0*c_exact_match/c_total
            c_f1 = 100.0*c_f1/c_total
            c_correct = 100.0*c_correct/c_total

    if args.style == 'split':
        with open(args.id_file, 'w') as f:
            json.dump(uc_id, f)

    return {"exact_match": exact_match, "f1": f1, "accuracy": correct,
            "uc_exact_match": uc_exact_match, "uc_f1": uc_f1, "uc_accuracy": uc_correct,
            "c_exact_match": c_exact_match, "c_f1": c_f1, "c_accuracy": c_correct,
            "new_uc_score": new_uc_score,
            "number of all questions": q_num, "number of uncertain questions": uq_num,
            "number of certain questions": cq_num}, accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='2wqa')
    parser.add_argument('--style', default='split_inference', help='split, split_inference')
    parser.add_argument('--question_file', default='data/2wqa/dev.json')
    parser.add_argument('--id_file', default='data/2wqa_uc_id_list.json')
    parser.add_argument('--result_list', default=['output/2wqa/dense_2wqa.json'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    myargs = parse_args()
    result, accuracy = wikiqa_evaluation(myargs)
    print(result)


