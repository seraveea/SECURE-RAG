import numpy as np
from utils.rephrase_utils import *
from utils.utils import separate_dictionary
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RAGAgent:
    def __init__(self, args):
        self.args = args

    def run(self, qa_pipeline, query, retrieval_files, embeder, NLI_agent, qid):
        # confirm retrieval files exists
        if qa_pipeline.tokenizer.eos_token_id is None:
            qa_pipeline.tokenizer.eos_token_id = qa_pipeline.tokenizer.convert_tokens_to_ids("<|endoftext|>")  # qwen2.5-1.5b
        if isinstance(retrieval_files, list):
            if self.args.rag_method == 'vanilla':
                query_embedding, dataset = embeder.encode(query, retrieval_files)
                if len(dataset) == 0:
                    return self.no_file_error(qid)
                dataset = dataset.add_faiss_index(column='embeddings')
                return self.rag_vanilla(qa_pipeline, query, dataset, query_embedding)
            elif self.args.rag_method == 'single_replace':
                # single replace have two version, one is only-NLI: rephrase and check NLI
                query_embedding, dataset = embeder.encode(query, retrieval_files)
                if len(dataset) == 0:
                    return self.no_file_error(qid)
                dataset = dataset.add_faiss_index(column='embeddings')
                return self.rag_single_replace(qa_pipeline, query, dataset, query_embedding, NLI_agent)
            elif self.args.rag_method == 'adaptive_chunk':
                # When selecting adaptive chunk, recursive chunking is used by default
                embeder.set_chunk_type('recursive')
                query_embedding, dataset = embeder.encode(query, retrieval_files)
                if len(dataset) == 0:
                    return self.no_file_error(qid)
                dataset = dataset.add_faiss_index(column='embeddings')
                return self.rag_adaptive_chunk(qa_pipeline, query, dataset, query_embedding, NLI_agent, embeder, retrieval_files, qid, type='no rerank')
            elif self.args.rag_method == 'rerank':
                # During rerank, adaptive chunking is triggered first. If uncertainty is low, do nothing; if high, perform semantic segmentation and then rerank
                embeder.set_chunk_type('recursive')
                query_embedding, dataset = embeder.encode(query, retrieval_files)
                if len(dataset) == 0:
                    return self.no_file_error(qid)
                dataset = dataset.add_faiss_index(column='embeddings')
                return self.rag_adaptive_chunk(qa_pipeline, query, dataset, query_embedding, NLI_agent, embeder, retrieval_files, qid, type='rerank')
            else:
                print('no rag method assigned, will use default one.')
                query_embedding, dataset = embeder.encode(query, retrieval_files)
                if len(dataset) == 0:
                    return self.no_file_error(qid)
                dataset = dataset.add_faiss_index(column='embeddings')
                return self.rag_vanilla(qa_pipeline, query, dataset, query_embedding)
        else:
            return self.no_file_error(qid)
        
    def no_file_error(self,qid):
        print('error, no related documents provided.')
        return {
            'question_id': qid,
            'sample_number': 'N/A',
            'answer': 'ERROR, no documents found in evidence.'
        }

    def rag_vanilla(self, qa_pipeline, query, dataset, query_embedding):
        """
        Vanilla RAG implementation without adversarial example generation
        """
        scores, retrieved_doc = dataset.get_nearest_examples('embeddings', query_embedding, k=min(int(self.args.candidate), len(dataset)))
        candidate_dict = separate_dictionary(retrieved_doc, scores)
        current_question = {'answer': []}
        selected_chunks = []
        initial_chunks = candidate_dict[:min(int(self.args.topk), len(candidate_dict))]
        selected_chunks.extend(initial_chunks)

        content = f'Answer question [{query}] based on provided context, ONLY output a short answer with minimum words. Context:\n\n' + '\n\n'.join([d['text'] for d in selected_chunks])
        message = [{"role": "user", "content": content}]
        for i in range(int(self.args.repeat)):
            result = agent_reply(qa_pipeline, message, self.args.model)
            current_question['answer'].append(result)

        return current_question

    def rag_single_replace(self, qa_pipeline, query, dataset, query_embedding, NLI_agent):
        """
        RAG implementation with adversarial example generation
        """
        scores, retrieved_doc = dataset.get_nearest_examples('embeddings', query_embedding,
                                                            k=min(int(self.args.candidate), len(dataset)))
        candidate_dict = separate_dictionary(retrieved_doc, scores)
        for i in range(len(candidate_dict)):
            candidate_dict[i]['local_id'] = i

        current_question = {'answer': [],
                            'adv_index': [],
                            'uncertainty_score': 1}  # default 1

        initial_chunks_indices = [i for i in range(min(int(self.args.topk), len(candidate_dict)))]
        chunks_index = initial_chunks_indices
        new_added = []
        adv_list = []
        
        # Select chunks based on the current indices
        chunk_dict = [i for i in candidate_dict if i['local_id'] in chunks_index]
        if len(chunk_dict) == 0:
            current_question['end_type'] = 'no chunks'
            return current_question

        if len(adv_list) == 0:
            adv_list = rephrase_article(qa_pipeline, [c['text'] for c in chunk_dict])  # First-time perturbation
        else:
            new_adv = rephrase_article(qa_pipeline, [c['text'] for c in chunk_dict if c['local_id'] in new_added])
            adv_list[0] += new_adv[0]
            adv_list[1] += new_adv[1]

        adv_text, adv_index = build_single_rephrase(adv_list)
        current_question['adv_index'] = adv_index
        temp_answer = []

        for i, context in enumerate(adv_text):
            temp = context[0]
            content = f'Answer question [{query}] based on provided context, ONLY output a short answer with minimum words. Context:{temp}'
            message = [{"role": "user", "content": content}]
            result = agent_reply(qa_pipeline, message, self.args.model)
            temp_answer.append(result)

        NLI_matrix = np.zeros((len(adv_index), len(adv_index)))  # [N+1, N+1]
        for index_i, reply_i in enumerate(temp_answer):
            for index_j, reply_j in enumerate(temp_answer):
                NLI_matrix[index_i][index_j] = NLI_agent.check_implication(reply_i, reply_j, query)

        NLI_matrix = NLI_matrix - np.ones((len(adv_index), len(adv_index)))  # [N+1, N+1]
        NLI_matrix = (NLI_matrix + NLI_matrix.T) / 2
        NLI_degree = np.diag(np.sum(NLI_matrix, axis=1))  # scalar value
        epsilon = 1e-4  # A small value to ensure positivity
        uncertainty_score = -np.sum(np.log((np.diag(NLI_degree) + epsilon) / len(adv_index))) / len(adv_index)
        current_question['uncertainty_score'] = float(uncertainty_score)
        current_question['answer'] = temp_answer
        current_question['end_type'] = 'no further operation required'
        current_question['chunks_index'] = chunks_index
        return current_question

    def rag_adaptive_chunk(self, qa_pipeline, query, dataset, query_embedding, NLI_agent, embeder, retrieval_files, qid, type):
        """
        Adaptive chunking:
        Initially, recursive chunking is used by default. Subsequently, the decision to switch to semantic chunking is made based on the uncertainty score.
        Once switched to semantic chunking, no further recursive operations are performed, and the process directly returns.
        """
        scores, retrieved_doc = dataset.get_nearest_examples('embeddings', query_embedding, k=min(int(self.args.candidate), len(dataset)))
        candidate_dict = separate_dictionary(retrieved_doc, scores)
        for i in range(len(candidate_dict)):
            candidate_dict[i]['local_id'] = i
            candidate_dict[i]['text_length'] = len(candidate_dict[i]['text'])


        current_question = {'answer': [],
                            'adv_index': [],
                            'uncertainty_score': 1}  # default 1

        initial_chunks_indices = [i for i in range(min(int(self.args.topk), len(candidate_dict)))]
        
        # Use adversarial text generation
        chunks_index = initial_chunks_indices
        new_added = []
        adv_list = []
        
        # Select chunks based on the current indices
        chunk_dict = [i for i in candidate_dict if i['local_id'] in chunks_index]
        if len(chunk_dict) == 0:
            current_question['end_type'] = 'no chunks'

        if len(adv_list) == 0:
            adv_list = rephrase_article(qa_pipeline, [c['text'] for c in chunk_dict])  # First-time perturbation
        else:
            new_adv = rephrase_article(qa_pipeline, [c['text'] for c in chunk_dict if c['local_id'] in new_added])
            adv_list[0] += new_adv[0]
            adv_list[1] += new_adv[1]


        # [00000,10000,01000,00100,00010,00001]
        adv_text, adv_index = build_single_rephrase(adv_list)
        current_question['adv_index'] = adv_index
        temp_answer = []

        # Generate answers for the perturbed texts
        for i, context in enumerate(adv_text):
            temp = context[0]
            content = f'Answer question [{query}] based on provided context, ONLY output a short answer with minimum words. Context:{temp}'
            message = [{"role": "user", "content": content}]
            result = agent_reply(qa_pipeline, message, self.args.model)
            temp_answer.append(result)

        NLI_matrix = np.zeros((len(adv_index), len(adv_index)))  # [N+1, N+1]
        for index_i, reply_i in enumerate(temp_answer):
            for index_j, reply_j in enumerate(temp_answer):
                NLI_matrix[index_i][index_j] = NLI_agent.check_implication(reply_i, reply_j, query)

        NLI_matrix = NLI_matrix - np.ones((len(adv_index), len(adv_index)))  # [N+1, N+1]
        NLI_matrix = (NLI_matrix + NLI_matrix.T) / 2
        NLI_degree = np.diag(np.sum(NLI_matrix, axis=1))  # scalar value
        epsilon = 1e-4
        uncertainty_score = -np.sum(np.log((np.diag(NLI_degree) + epsilon) / len(adv_index))) / len(adv_index)
        if embeder.chunk_type == 'semantic':
            current_question['end_type'] = 'adaptivesemantic chunk'
            current_question['chunks_index'] = chunks_index
            current_question['answer'] = temp_answer
            current_question['uncertainty_score'] = uncertainty_score
            return current_question
        if uncertainty_score < 0.2:
            current_question['end_type'] = 'uncertainty lower than 0.2'
            current_question['chunks_index'] = chunks_index
            current_question['answer'] = temp_answer
            current_question['uncertainty_score'] = uncertainty_score
            return current_question
        else:
            embeder.set_chunk_type('semantic')
            query_embedding, dataset = embeder.encode(query, retrieval_files)
            if len(dataset) == 0:
                return self.no_file_error(qid)
            
            dataset = dataset.add_faiss_index(column='embeddings')
            if type == 'rerank':
                return self.rag_rerank(qa_pipeline, query, dataset, query_embedding, NLI_agent)
            else:
                return self.rag_vanilla(qa_pipeline, query, dataset, query_embedding) 
            
    def rag_rerank(self, qa_pipeline, query, dataset, query_embedding, NLI_agent):
        scores, retrieved_doc = dataset.get_nearest_examples('embeddings', query_embedding, k=min(int(self.args.candidate), len(dataset)))
        candidate_dict = separate_dictionary(retrieved_doc, scores)
        for i in range(len(candidate_dict)):
            candidate_dict[i]['local_id'] = i

        min_uncertainty = 100  # Initial uncertainty value
        current_question = {'answer': [], 'adv_index': [], 'uncertainty_score': 1, 'self-feedback':0}  # default 1
        initial_chunks_indices = [i for i in range(min(int(self.args.topk), len(candidate_dict)))]
        
        chunks_index = initial_chunks_indices
        visited_index = initial_chunks_indices.copy()
        new_added = [] 
        adv_list = []

        current_question['rerank_record'] = []
        current_rerank_time = 0

        while len(visited_index) <= len(candidate_dict):
            # Select chunks based on the current indices
            chunk_dict = [i for i in candidate_dict if i['local_id'] in chunks_index]
            if len(chunk_dict) == 0:
                current_question['end_type'] = 'no chunks'
                break  # No more new documents available

            if len(adv_list) == 0:
                adv_list = rephrase_article(qa_pipeline, [c['text'] for c in chunk_dict])  # First-time perturbation
            else:
                # rephrase
                new_adv = rephrase_article(qa_pipeline, [c['text'] for c in chunk_dict if c['local_id'] in new_added])
                adv_list[0] += new_adv[0]
                adv_list[1] += new_adv[1]

            # adv_list[0] is the original text，adv_list[1] is the rewritten text

            # [00000,10000,01000,00100,00010,00001]
            adv_text, adv_index = build_single_rephrase(adv_list)
            current_question['adv_index'] = adv_index
            temp_answer = []  # record current reply

            # Generate answers for the perturbed texts
            for i, context in enumerate(adv_text):
                temp = context[0]
                content = f'Answer question [{query}] based on provided context, ONLY output a short answer with minimum words. Context:{temp}'
                message = [{"role": "user", "content": content}]
                result = agent_reply(qa_pipeline, message, self.args.model)
                temp_answer.append(result)


            NLI_matrix = np.zeros((len(adv_index), len(adv_index)))  # [N+1, N+1]
            for index_i, reply_i in enumerate(temp_answer):
                for index_j, reply_j in enumerate(temp_answer):
                    NLI_matrix[index_i][index_j] = NLI_agent.check_implication(reply_i, reply_j, query)

            NLI_matrix = NLI_matrix - np.ones((len(adv_index), len(adv_index)))  # [N+1, N+1]
            NLI_matrix = (NLI_matrix + NLI_matrix.T) / 2
            # conf_score = np.sum(NLI_matrix, axis=1) / len(adv_index)  # [1, N+1]
            conf_score = NLI_matrix[0] 
            NLI_degree = np.diag(np.sum(NLI_matrix, axis=1))  # scalar value
            epsilon = 1e-4  # A small value to ensure positivity
            uncertainty_score = -np.sum(np.log((np.diag(NLI_degree) + epsilon) / len(adv_index))) / len(adv_index)
            current_question['uncertainty_score'] = float(uncertainty_score)

            # ---------self-evaluation-------------
            self_feedback = llm_self_feedback(query,adv_text[0][0],self.args.model,qa_pipeline)
            # ---------------------------------



            if uncertainty_score <= min_uncertainty:
                if current_question['self-feedback'] == 0:
                    current_question['answer'] = temp_answer
                    min_uncertainty = uncertainty_score
                    current_question['self-feedback'] = self_feedback
                elif self_feedback == 1:
                    current_question['answer'] = temp_answer
                    min_uncertainty = uncertainty_score
                    current_question['self-feedback'] = self_feedback
                else:
                    None

            # 字典记录的uncertainty_score一直是最小的
            current_question['uncertainty_score'] = min_uncertainty

            # Condition 1: Stop if uncertainty is below threshold
            if current_question['uncertainty_score'] < 0.2 and self_feedback == 1:
                current_question['end_type'] = 'uncertainty lower than 0.2 and llm agree'
                break
            
            # Condition 2: Stop if no more files to explore
            if len(visited_index) == len(candidate_dict):
                current_question['end_type'] = 'no more files'
                break
            

            abl_reply = []
            abl_text, abl_index = build_ablation_set(adv_list)  # [11111,01111,10111,11011,11101,11110]
            for i, context in enumerate(abl_text):
                temp = context
                content = f'Answer question [{query}] based on provided context, ONLY output a short answer with minimum words. Context:{temp}'
                message = [{"role": "user", "content": content}]
                result = agent_reply(qa_pipeline, message, self.args.model)
                abl_reply.append(result)
            ABL_matrix = np.zeros((1, len(abl_index)))  # [1, N+1]
            for index_i, reply_i in enumerate(abl_reply):

                ABL_matrix[0][index_i] = NLI_agent.check_implication(reply_i, abl_reply[0], query)


            impact_index = np.where(ABL_matrix != 2)[1]
            no_impact_index = np.where(ABL_matrix == 2)[1]
            certain_index = np.where(conf_score == 1)
            uncertain_index = np.where(conf_score != 1)

            local_index_A = np.intersect1d(impact_index, certain_index)
            local_index_B = np.intersect1d(impact_index, uncertain_index)
            local_index_C = np.intersect1d(no_impact_index, certain_index)
            local_index_D = np.intersect1d(no_impact_index, uncertain_index)
            local_index_unimpact = local_index_D

            pop_elements = [i-1 for i in local_index_unimpact]
            if -1 in pop_elements:
                pop_elements.remove(-1)
            if len(pop_elements) == 0:
                if len(chunk_dict) > int(self.args.topk):
                    current_question['end_type'] = 'files all important'
                    break
                else:
                    None
            

            for pop_index in sorted(pop_elements, reverse=True):
                adv_list[0].pop(pop_index)
                adv_list[1].pop(pop_index)

            index_B = [element for idx, element in enumerate(chunks_index) if idx in [i-1 for i in local_index_B]]
            doc_B = [element for element in candidate_dict if element['local_id'] in index_B]


            chunks_index = [element for idx, element in enumerate(chunks_index) if idx not in pop_elements]
            remaining_chunks = [i for i in candidate_dict if i['local_id'] not in visited_index]

            if len(doc_B) > 0:
                trustworthy_embeddings = [x['embeddings'] for x in doc_B]
                new_added = []
                flag = 0
                for embeddings in trustworthy_embeddings:
                    if flag == 1:
                        break
                    distance_sim = [(item, compute_similarity(item['embeddings'], embeddings), np.float16(item['relevant score'])) for item in
                                    remaining_chunks]
                    distance_sim.sort(key=lambda x: x[1], reverse=True)
                    if len(new_added)+len(chunks_index) > int(self.args.topk):
                        flag = 1
                    new_added.append(distance_sim[0][0]['local_id'])
                chunks_index += list(set(new_added))
                current_rerank_time += 1
                current_question['rerank_record'].append((current_rerank_time,'add chunks according to B type documents'))


            elif len(chunks_index) >= int(self.args.topk):
                current_question['end_type'] = 'no uncertain files'
                break

            if len(chunks_index) < int(self.args.topk):
                new_added = []
                for item in remaining_chunks:
                    if len(new_added)+len(chunks_index) >= int(self.args.topk):
                        break
                    new_added.append(item['local_id'])
                chunks_index += new_added
 
            visited_index.extend(chunks_index)
            visited_index = list(set(visited_index))

        current_question['chunks_index'] = chunks_index
        current_question['raw_chunks'] = chunk_dict
        return current_question



def compute_similarity(embedding, reference_embedding):
    return cosine_similarity(np.atleast_2d(embedding), np.atleast_2d(reference_embedding))[0][0]


def compute_distance(embedding, reference_embedding):
    return np.linalg.norm(np.array(embedding) - np.array(reference_embedding))


def check_yes_no(s):
    s = s.lower()
    yes_index = s.find("yes")
    no_index = s.find("no")

    if yes_index == -1 and no_index == -1:
        return None
    elif yes_index == -1:
        return 0
    elif no_index == -1:
        return 1
    else:

        if yes_index < no_index:
            return 1
        else:
            return 0
    

def agent_reply(qa_pipeline, message):
    
    if qa_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>") is None:
        eos_pair = [qa_pipeline.tokenizer.eos_token_id, qa_pipeline.tokenizer.convert_tokens_to_ids("<|endoftext|>")] # qwen2.5-1.5b
    else:
        eos_pair = [qa_pipeline.tokenizer.eos_token_id, qa_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")] # llama
   
    prompt = qa_pipeline.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    output = qa_pipeline(prompt, max_new_tokens=256, 
                            eos_token_id=eos_pair, 
                            do_sample=False, 
                            temperature=0.1, 
                            top_p=0.9, 
                            pad_token_id=qa_pipeline.tokenizer.eos_token_id)
    
    return output[0]["generated_text"][len(prompt):]


def llm_self_feedback(query,text,model,qa_pipeline):
    content = f'Context:{text}\nQuestion:{query}\nDoes the context contain enough information to answer the question? Only answer yes or no.'
    message = [{"role": "user", "content": content}]
    result = agent_reply(qa_pipeline, message, model)
    return check_yes_no(result)
