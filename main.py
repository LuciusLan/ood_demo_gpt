import os

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

from tools import Scorer

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-large")

scorer = Scorer('princeton-nlp/unsup-simcse-roberta-large')



# Tokenize input texts
with open ('./data/small/book/set1_text.txt', 'r') as f:
    texts = f.readlines()

scorer.build_index(texts)
scorer.search('very oily and creamy not at all what i expected ordered this to try to highlight and contour and it just looked awful ! ! ! plus took forever to arrive', threshold=0)
# 0 for neg, 1 for neutral, 2 for pos
print()