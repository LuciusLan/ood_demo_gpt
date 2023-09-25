import os
import json

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from tools import Scorer

LEN_THRES = 9999
domain_list = ['book', 'beauty', 'electronics', 'music']



# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-large", cache_dir='D:\\Dev\\LM')
model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-large", cache_dir='D:\\Dev\\LM')

scorer = Scorer('princeton-nlp/unsup-simcse-roberta-large')



for source_domain in domain_list:
    for target_domain in domain_list:
        if target_domain == source_domain:
            continue

        SOURCE_CORPUS = f'./data/{source_domain}.labeled.sent.txt'
        TARGET_CORPUS = f'./data/{target_domain}.labeled.sent.txt'
        SOURCE_LABEL = f'./data/{source_domain}.labeled.label.txt'
        TARGET_LABEL = f'./data/{target_domain}.labeled.label.txt'

        # Tokenize input texts
        with open (SOURCE_CORPUS, 'r') as f:
            source_texts = f.readlines()

        with open (TARGET_CORPUS, 'r') as f:
            query_texts = f.readlines()

        with open (SOURCE_LABEL, 'r') as f:
            source_labels = f.readlines()

        with open (TARGET_LABEL, 'r') as f:
            query_labels = f.readlines()

        f.close()


        #scorer.build_index(source_texts)

        query_w_demos = []
        for query_id, q_sent in tqdm(enumerate(query_texts), total=len(query_texts)):
            if len(q_sent.split(' ')) > LEN_THRES:
                continue
            temp = {'qid': query_id, 'q_sent': q_sent, 'q_label': query_labels[query_id]}
            #results = scorer.search(q_sent, threshold=0, top_k=-5)
            results = scorer.search_random(q_sent, source_len=len(source_texts), top_k=5)
            source = [{'sid': si, 's_sent': source_texts[si], 's_label': source_labels[si]} for si in results]
            temp.update({'source': source})
            query_w_demos.append(temp)



        with open(f'{source_domain}2{target_domain}_rand.json', 'w') as f:
            json.dump(query_w_demos, f, indent=2)
        f.close()
