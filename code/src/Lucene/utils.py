import numpy as np


# def get_predicted_ranking(predicted_list, target_list, true_scores):
#     target_score_dict = {item: score for item, score in zip(target_list, true_scores)}
#     return [target_score_dict.get(item, 0) for item in predicted_list]


def _relevance(target, rel_scores=None):
    """Map each relevant doc id to its relevance score (1 for every target when rel_scores is None)."""
    targets = target if isinstance(target, list) else [target]
    if rel_scores is None:
        return {item: 1 for item in targets}
    assert len(targets) == len(rel_scores)
    return {item: rel_scores[i] for i, item in enumerate(targets)}


def dcg_at_k(retrieved, target, k, rel_scores=None):
    """
    Compute DCG@k (Discounted Cumulative Gain).
    Target is a relevant doc id or a list of them; rel_scores (same order as target) grades them.
    A doc retrieved more than once only gains at its first occurrence; repeats keep their rank slot.
    """
    relevance = _relevance(target, rel_scores)
    seen = set()
    gains = []
    for doc in retrieved[:k]:
        gains.append(0 if doc in seen else relevance.get(doc, 0))
        seen.add(doc)
    gains = np.array(gains, dtype=float)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(retrieved, target, k, rel_scores=None):
    """
    Compute NDCG@k. The ideal ranking puts the relevant docs in descending order of relevance.
    """
    relevance = _relevance(target, rel_scores)
    assert all(score >= 0 for score in relevance.values()), "relevance scores must be non-negative"
    dcg = dcg_at_k(retrieved, target, k, rel_scores)
    ideal_gains = np.array(sorted(relevance.values(), reverse=True)[:k], dtype=float)
    ideal_dcg = np.sum(ideal_gains / np.log2(np.arange(2, len(ideal_gains) + 2)))
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    assert 0.0 <= ndcg <= 1.0 + 1e-9, f"NDCG {ndcg} outside [0, 1]"
    return ndcg


def dcg_at_k_rank(rank, k):
    """
    Compute DCG@k (Discounted Cumulative Gain) for a single relevant document at a specific rank.
    
    Args:
        rank (int): The rank of the first occurrence of the answer (1-indexed)
        k (int): Number of documents to consider
        
    Returns:
        float: DCG score
    """
    if rank > k:
        return 0.0
    
    # Create a relevance array with a single 1 at the position of the first answer
    relevance = np.zeros(k)
    if rank <= k:
        relevance[rank-1] = 1.0
    
    # Calculate discounted gains using log2(i+1) for positions i (0-indexed)
    discounts = np.log2(np.arange(2, k + 2))
    return np.sum(relevance / discounts)


def ideal_dcg_at_k(k):
    """
    Compute ideal DCG@k where the answer is at rank 1.
    
    Args:
        k (int): Number of documents to consider
        
    Returns:
        float: Ideal DCG score
    """
    # Ideal case: answer is at rank 1
    return dcg_at_k_rank(1, k)


def ndcg_for_rank(rank, k):
    """
    Compute NDCG@k for a document at a specific rank.
    
    Args:
        rank (int): The rank of the first occurrence of the answer (1-indexed)
        k (int): Number of documents to consider
        
    Returns:
        float: NDCG score between 0 and 1
    """
    dcg = dcg_at_k_rank(rank, k)
    ideal_dcg = ideal_dcg_at_k(k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0