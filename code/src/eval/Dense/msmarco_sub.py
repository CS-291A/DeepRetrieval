import json
import os
import sys

from tqdm import tqdm

sys.path.append('./')

from typing import cast

import torch
from pyserini.search.faiss import AutoQueryEncoder, DenseSearchResult, FaissSearcher

from src.Lucene.utils import ndcg_at_k

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# index_dir = "/home/azureuser/cloudfiles/code/DeepRetrieval/indexes/contriever-msmarco-passage-dense-index"
# query_encoder_name = "facebook/contriever"

index_dir = "/shared/eng/pj20/lmr_model/raw_data/msmarco/indexes/contriever-msmarco-passage-dense-index"
query_encoder_name = "facebook/contriever"

# index_dir = "/home/azureuser/cloudfiles/code/DeepRetrieval/indexes/minilm-msmarco-passage-dense-index"
# query_encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

# index_dir = "/home/azureuser/cloudfiles/code/DeepRetrieval/indexes/mpnet-msmarco-passage-dense-index"
# query_encoder_name = "sentence-transformers/all-mpnet-base-v2"

_searcher = None


def get_searcher(mode='dense'):
    global _searcher
    if _searcher is None and mode == 'dense':
        if not os.path.exists(index_dir):
            # pyserini has no prebuilt FAISS index named 'msmarco-v1-passage' and this fallback passed no
            # query encoder, so it could never build a searcher.
            raise FileNotFoundError(f"dense index not found at {index_dir}; set index_dir to your FAISS index")
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if 'contriever' in query_encoder_name:
                query_encoder = AutoQueryEncoder(query_encoder_name, pooling='mean', device=device)
            else:
                query_encoder = AutoQueryEncoder(query_encoder_name, pooling='cls', device=device)
            _searcher = FaissSearcher(index_dir=index_dir, query_encoder=query_encoder)
    return _searcher


def load_matching_dataset(domain):
    
    data_train = []
    data_test = []
    data_val = []
    
    with open(f'data/raw_data/msmarco/msmarco_{domain}/train.jsonl', 'r') as f:
        for line in f:
            data_train.append(json.loads(line))

    with open(f'data/raw_data/msmarco/msmarco_{domain}/dev.jsonl', 'r') as f:
        for line in f:
            data_test.append(json.loads(line))
            
    with open(f'data/raw_data/msmarco/msmarco_{domain}/dev.jsonl', 'r') as f:
        cnt = 0
        for line in f:
            data_val.append(json.loads(line))
            cnt += 1
            if cnt > 100:
                break
    
    train_data = [{'input': x['question'], 'label': x['docs_id']} for x in data_train]
    test_data = [{'input': x['question'], 'label': x['docs_id']} for x in data_test]
    val_data = [{'input': x['question'], 'label': x['docs_id']} for x in data_val]
    
    return train_data, test_data, val_data

def retriver_items(query, top_k=3000, mode='sparse'):
    """Retrieve items from the search system."""
    searcher = get_searcher(mode=mode)
    if searcher is None or mode != 'dense':  # get_searcher only builds the FAISS searcher
        raise ValueError(f"only mode='dense' is supported, got {mode!r}")
    # search() returns (vectors, hits) only when return_vector=True
    hits = cast(list[DenseSearchResult], searcher.search(query, k=top_k))
    doc_list = [hit.docid for hit in hits]
    return doc_list


if __name__ == '__main__':
    searcher = get_searcher()
    for domain in ['health', 'science', 'tech']:
        print(f"Evaluating {domain} domain")
        ndcg = []
        train_data, test_data, val_data = load_matching_dataset(domain)

        for i in tqdm(range(len(test_data))):
            query = test_data[i]['input']
            label = test_data[i]['label']
            targets = [str(l) for l in label]
            scores = [1 for _ in range(len(targets))]

            retrieved = retriver_items(query, top_k=10, mode='dense')
            ndcg.append(ndcg_at_k(retrieved, targets, 10, rel_scores=scores))
        
        print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")