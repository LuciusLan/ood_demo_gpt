import os


import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from tools import Scorer

LEN_THRES = 100
SOURCE_CORPUS = './data/small/book/set1_text.txt'
TARGET_CORPUS = './data/small/beauty/set1_text.txt'

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-large")

scorer = Scorer('princeton-nlp/unsup-simcse-roberta-large')



# Tokenize input texts
with open (SOURCE_CORPUS, 'r') as f:
    source_texts = f.readlines()

with open (TARGET_CORPUS, 'r') as f:
    query_texts = f.readlines()


scorer.build_index(source_texts)

query_w_demos = []
for query_id, q_sent in tqdm(enumerate(query_texts), total=len(query_texts)):
    if len(q_sent.split(' ')) > LEN_THRES:
        continue
    temp = {'qid': query_id}
    results = scorer.search(q_sent, threshold=0, top_k=5)
    index = []
    for i in results:
        index.append(source_texts.index(i[0]))
        # if i[1] > 0.5:
        #     print (query_sents.index(q_sent),q_sent, i)
    # print (index)
    temp.update({'sid': index})
    query_w_demos.append(temp)

# 0 for neg, 1 for neutral, 2 for pos
print()