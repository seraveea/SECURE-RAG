import random
from itertools import product

def sample_entity(map0, keep_num=10):
    new_map = {}
    if len(map0.keys()) < keep_num:
        return map0
    else:
        selected_keys = random.sample(list(map0.keys()), keep_num)
        for key in selected_keys:
            new_map[key] = map0[key]

    return new_map


def rephrase_article(agent, chunk_list):
    """
    :param agent:
    :param chunk_list:
    :return:
    """
    chunk_container = [[]]
    for context_chunk in chunk_list:
        content = f"""Rewrite the following text at the syntactic level without changing its meaning. 
        Modify the sentence structure, but preserve the original intent and semantic meaning.
        ONLY return the rewritten content!!! Any token else is NOT ALLOWED!!!
        \n Text: {context_chunk}"""
        message = [{"role": "user", "content": content}]
        if agent.tokenizer.convert_tokens_to_ids("<|eot_id|>") is None:
            eos_pair = [agent.tokenizer.eos_token_id, agent.tokenizer.convert_tokens_to_ids("<|endoftext|>")] # qwen2.5-1.5b
        else:
            eos_pair = [agent.tokenizer.eos_token_id, agent.tokenizer.convert_tokens_to_ids("<|eot_id|>")] # llama
        prompt = agent.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        output = agent(prompt, max_new_tokens=len(content),
                       eos_token_id=eos_pair,
                       do_sample=True,
                       temperature=0.5, top_p=0.9, pad_token_id=agent.tokenizer.eos_token_id)
        generated = output[0]["generated_text"][len(prompt):]
        generated = generated.replace("Here is the rewritten text:", "")
        chunk_container[0].append(generated)
    result = [chunk_list] + chunk_container
    return result


def rephrase_one_chunk(agent, chunk):
    """
    :param agent:
    :param chunk_list:
    :return:
    """
    content = f"""Rewrite the following text at the syntactic level without changing its meaning. 
    Modify the sentence structure, but preserve the original intent and semantic meaning.
    ONLY return the rewritten content!!! Any token else is NOT ALLOWED!!!
    \n Text: {chunk}"""
    message = [{"role": "user", "content": content}]
    if agent.tokenizer.convert_tokens_to_ids("<|eot_id|>") is None:
        eos_pair = [agent.tokenizer.eos_token_id, agent.tokenizer.convert_tokens_to_ids("<|endoftext|>")] # qwen2.5-1.5b
    else:
        eos_pair = [agent.tokenizer.eos_token_id, agent.tokenizer.convert_tokens_to_ids("<|eot_id|>")] # llama
    prompt = agent.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    output = agent(prompt, max_new_tokens=len(content),
                    eos_token_id=eos_pair,
                    do_sample=True,
                    temperature=0.5, top_p=0.9, pad_token_id=agent.tokenizer.eos_token_id)
    generated = output[0]["generated_text"][len(prompt):]
    generated = generated.replace("Here is the rewritten text:", "")
    return generated


def build_rephrase(text_list, repeat):
    """
    :param text_list:
    :return:
    """
    standard_num = len(text_list[0])
    num_rephrase = len(text_list)
    combinations = [''.join(map(str, combo)) for combo in product(range(num_rephrase), repeat=standard_num)]
    if len(combinations) < repeat:
        print('number of rephrase is not enough!')
        while len(combinations) < repeat:
            combinations.append('0' * standard_num)
    essential = [s for s in combinations if sum(int(char) for char in s) == 0] + [s for s in combinations if sum(int(char) for char in s) == 1]
    filter_combo = [item for item in combinations if item not in essential]
    combinations = essential + random.sample(filter_combo, 10-len(essential))
    result = []
    adv_index = []
    for c in combinations:
        temp = ''
        for index, loc in enumerate(c):
            temp += "\n\n"+text_list[int(loc)][index].lstrip()
        result.append([temp])
        adv_index.append(c)
        if len(result) == repeat:
            # enough
            break
    assert len(result) == repeat
    return result, adv_index


def build_single_rephrase(text_list):
    """
    :param text_list:
    :return:
    """
    standard_num = len(text_list[0])
    num_rephrase = len(text_list)
    combinations = [''.join(map(str, combo)) for combo in product(range(num_rephrase), repeat=standard_num)]
    essential = [s for s in combinations if sum(int(char) for char in s) == 1] + [s for s in combinations if sum(int(char) for char in s) == 0]
    result = []

    adv_index = []
    for c in essential:
        temp = ''
        for index, loc in enumerate(c):
            temp += "\n\n"+text_list[int(loc)][index].lstrip()
        result.append([temp])
        adv_index.append(c)

    return result[::-1], adv_index[::-1]


def build_ablation_set(text_list):
    """
    :param text_list:
    :return:
    """
    original = text_list[0]
    abl_index = [''.join(['1']*len(original))]
    result = ['\n\n'.join(original)]
    for i in range(len(original)):
        temp = ''
        for j in range(len(original)):
            if i != j:
                temp += '\n\n'+original[j]
        result.append(temp)
        index = ['1'] * len(original)
        index[i] = '0'
        abl_index.append(''.join(index))
    return result, abl_index


if __name__ == "__main__":
    test = [
        ['00', '01', '02', '03', '04'],
        ['10', '11', '12', '13', '14'],
    ]
    # print(build_rephrase(test))

    abl_text, abl_index = build_ablation_set(test)
    print(abl_text)
    print(abl_index)
    print('adv test below')
    adv_text, adv_index = build_single_rephrase(test)
    print(adv_text)
    print(adv_index)

    pop_elements = [0, 2, 4]
    # print(pop_elements)

    for pop_index in sorted(pop_elements, reverse=True):
        test[0].pop(pop_index)
        test[1].pop(pop_index)

    print(test)

    chunks_index = [2, 1, 5, 7, 4]
    print([element for idx, element in enumerate(chunks_index) if idx not in pop_elements])
