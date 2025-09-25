import re


def remove_extra_spaces(text):
    # Remove spaces before punctuation marks like commas, periods, question marks, etc.
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    # Remove extra spaces inside and outside parentheses
    text = re.sub(r'\(\s+', '(', text)  # Space after the left parenthesis
    text = re.sub(r'\s+\)', ')', text)  # Space before the right parenthesis
    text = re.sub(r'\[\s+', '[', text)  # Space after the left square bracket
    text = re.sub(r'\s+\]', ']', text)  # Space before the right square bracket
    text = re.sub(r'\{\s+', '{', text)  # Space after the left curly brace
    text = re.sub(r'\s+\}', '}', text)  # Space before the right curly brace
    # Remove consecutive spaces, keeping only a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_NQ(sample):
    # There may be cases where no long answer is annotated, and short answers may also be missing
    question = sample['question']['text']
    tokens = sample['document']['tokens']['token']
    is_html = sample['document']['tokens']['is_html']
    question_id = sample['id']

    # document = sample['document_tokens']
    answer_set = []
    annotation_list = [dict(zip(sample['annotations'].keys(), values))
                       for values in zip(*sample['annotations'].values())]
    # annotation_list contains all human annotations
    for anno in annotation_list:
        # Get the long answer
        if anno['long_answer']['start_token'] != -1:
            start = anno['long_answer']['start_token']
            end = anno['long_answer']['end_token']
            long_answer = remove_extra_spaces(' '.join([token for token, html_flag in zip(tokens[start:end], is_html[start:end]) if not html_flag]))
        else:
            long_answer = None

        answer_set.append(
            {'id': anno['id'],
             'long_answer': long_answer,
             'short_answer': anno['short_answers']['text']}
        )

    candidate_set = []
    candidate_list = [dict(zip(sample['long_answer_candidates'].keys(), values)) for values in zip(*sample['long_answer_candidates'].values())]
    for candi in candidate_list:
        text = remove_extra_spaces(' '.join([token for token, html_flag in zip(tokens[candi['start_token']:candi['end_token']], is_html[candi['start_token']:candi['end_token']]) if not html_flag]))
        candidate_set.append(
            {'text': text,
             'top_level': candi['top_level']
             }
        )
    document = remove_extra_spaces(' '.join([token for token, html_flag in zip(tokens, is_html) if not html_flag]))
    # document is the raw document, without human filtering

    return question, answer_set, candidate_set, document, question_id
