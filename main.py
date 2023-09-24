import os


import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from tools import Scorer

LEN_THRES = 100
SOURCE_CORPUS = './data/book.labeled.sent.txt'
TARGET_CORPUS = './data/beauty.labeled.sent.txt'
SOURCE_LABEL = './data/book.labeled.label.txt'
TARGET_LABEL = './data/beauty.labeled.label.txt'

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-large", cache_dir='D:\\Dev\\LM')
model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-large", cache_dir='D:\\Dev\\LM')

scorer = Scorer('princeton-nlp/unsup-simcse-roberta-large')



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


scorer.build_index(source_texts)

query_w_demos = []
for query_id, q_sent in tqdm(enumerate(query_texts), total=len(query_texts)):
    if len(q_sent.split(' ')) > LEN_THRES:
        continue
    temp = {'qid': query_id, 'q_sent': q_sent, 'q_label': query_labels[query_id]}
    results = scorer.search(q_sent, threshold=0, top_k=-5)
    index = []
    for i in results:
        index.append(source_texts.index(i[0]))
        # if i[1] > 0.5:
        #     print (query_sents.index(q_sent),q_sent, i)
    # print (index)
    source = [{'sid': si, 's_sent': source_texts[si], 's_label': source_labels[si]} for si in index]
    temp.update({'source': source})
    query_w_demos.append(temp)


# 0 for neg, 1 for neutral, 2 for pos
print()