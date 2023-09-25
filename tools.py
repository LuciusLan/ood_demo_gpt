from typing import Union
import logging
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Scorer():
    """
    A class for embedding sentences, calculating similarities, and retriving sentences by SimCSE.
    """
    def __init__(self, model_name_or_path: str, 
                device: str = None,
                num_cells: int = 100,
                num_cells_in_search: int = 10,
                pooler = None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        if pooler is not None:
            self.pooler = pooler
        elif "unsup" in model_name_or_path:
            logger.info("Use `cls_before_pooler` for unsupervised models. If you want to use other pooling policy, specify `pooler` argument.")
            self.pooler = "cls_before_pooler"
        else:
            self.pooler = "cls"

    def encode(self, sentence: Union[str, list[str]], 
                device: str = None, 
                return_numpy: bool = False,
                normalize_to_unit: bool = True,
                keepdim: bool = False,
                batch_size: int = 64,
                max_length: int = 128,
                is_source: bool = False) -> Union[np.ndarray, torch.Tensor]:

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = [] 
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            if is_source:
                for batch_id in tqdm(range(total_batch)):
                    inputs = self.tokenizer(
                        sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                        padding=True, 
                        truncation=True, 
                        max_length=max_length, 
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(target_device) for k, v in inputs.items()}
                    outputs = self.model(**inputs, return_dict=True)
                    if self.pooler == "cls":
                        embeddings = outputs.pooler_output
                    elif self.pooler == "cls_before_pooler":
                        embeddings = outputs.last_hidden_state[:, 0]
                    else:
                        raise NotImplementedError
                    if normalize_to_unit:
                        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                    embedding_list.append(embeddings.cpu())
            else:
                for batch_id in range(total_batch):
                    inputs = self.tokenizer(
                        sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                        padding=True, 
                        truncation=True, 
                        max_length=max_length, 
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(target_device) for k, v in inputs.items()}
                    outputs = self.model(**inputs, return_dict=True)
                    if self.pooler == "cls":
                        embeddings = outputs.pooler_output
                    elif self.pooler == "cls_before_pooler":
                        embeddings = outputs.last_hidden_state[:, 0]
                    else:
                        raise NotImplementedError
                    if normalize_to_unit:
                        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                    embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)
        
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
        
        if return_numpy and not isinstance(embeddings, np.ndarray):
            return embeddings.numpy()
        return embeddings
    
    def build_index(self, sentences_or_file_path: Union[str, list[str]], 
                        use_faiss: bool = False,
                        faiss_fast: bool = False,
                        device: str = None,
                        batch_size: int = 64):

        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True, is_source=True)

        logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}

        index = embeddings
        self.is_faiss_index = False
        self.index["index"] = index
        logger.info("Finished")

    def similarity(self, queries: Union[str, list[str]], 
                    keys: Union[str, list[str], np.ndarray], 
                    device: str = None) -> Union[float, np.ndarray]:
        
        query_vecs = self.encode(queries, device=device, return_numpy=True) # suppose N queries
        
        if not isinstance(keys, np.ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True) # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1 
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)
        
        # returns an N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)
        
        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])
        
        return similarities

    def search(self, queries: Union[str, list[str]], 
                device: str = None, 
                threshold: float = 0.6,
                top_k: int = 5) -> Union[list[tuple[str, float]], list[list[tuple[str, float]]]]:
        

        if isinstance(queries, list):
            combined_results = []
            for query in queries:
                results = self.search(query, device, threshold, top_k)
                combined_results.append(results)
            return combined_results
        
        similarities = self.similarity(queries, self.index["index"]).tolist()
        id_and_score = []
        for i, s in enumerate(similarities):
            if s >= threshold:
                id_and_score.append((i, s))

        if top_k > 0:
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
        elif top_k < 0:
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[top_k:]
        results = [(self.index["sentences"][idx], score) for idx, score in id_and_score]
        return results

    def search_random(self, queries: Union[str, list[str]], 
                source_len: int,
                device: str = None, 
                top_k: int = 5,
                ) -> Union[list[tuple[str, float]], list[list[tuple[str, float]]]]:
        

        if isinstance(queries, list):
            combined_results = []
            for query in queries:
                results = self.search_random(query, source_len, device, top_k)
                combined_results.append(results)
            return combined_results
        
        '''similarities = self.similarity(queries, self.index["index"]).tolist()
        id_and_score = []
        for i, s in enumerate(similarities):
            if s >= threshold:
                id_and_score.append((i, s))'''

        results = random.sample(list(range(source_len)), top_k)
        return results
