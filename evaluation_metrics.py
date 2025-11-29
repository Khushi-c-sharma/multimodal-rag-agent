"""
Evaluation Metrics Module
Provides comprehensive metrics for RAG system evaluation:
- Retrieval metrics (Precision, Recall, F1)
- Ranking metrics (MRR, NDCG, MAP)
- Diversity metrics
- Latency tracking
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import time
from functools import wraps


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


# ============================================================================
# Retrieval Metrics
# ============================================================================

def calculate_precision_at_k(retrieved: List[str], 
                             relevant: Set[str], 
                             k: Optional[int] = None) -> float:
    """
    Calculate Precision@K.
    Precision = (# relevant items in top-k) / k
    
    Args:
        retrieved: List of retrieved item IDs
        relevant: Set of relevant item IDs
        k: Top-k items to consider (None = all)
    """
    if k is not None:
        retrieved = retrieved[:k]
    
    if not retrieved:
        return 0.0
    
    relevant_retrieved = sum(1 for item in retrieved if item in relevant)
    return relevant_retrieved / len(retrieved)


def calculate_recall_at_k(retrieved: List[str], 
                          relevant: Set[str], 
                          k: Optional[int] = None) -> float:
    """
    Calculate Recall@K.
    Recall = (# relevant items in top-k) / (total # relevant items)
    """
    if not relevant:
        return 0.0
    
    if k is not None:
        retrieved = retrieved[:k]
    
    relevant_retrieved = sum(1 for item in retrieved if item in relevant)
    return relevant_retrieved / len(relevant)


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_retrieval_metrics(retrieved_items: List[Dict[str, Any]],
                                relevant_ids: Optional[Set[str]] = None,
                                k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
    """
    Calculate comprehensive retrieval metrics.
    
    Args:
        retrieved_items: List of retrieved items with 'content' or 'id'
        relevant_ids: Set of IDs that are considered relevant (for evaluation)
        k_values: List of k values for @k metrics
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Extract IDs from retrieved items
    retrieved_ids = []
    for item in retrieved_items:
        item_id = item.get('id') or item.get('metadata', {}).get('id') or item.get('content', '')[:50]
        retrieved_ids.append(item_id)
    
    # If no ground truth, just return basic stats
    if relevant_ids is None:
        metrics['num_retrieved'] = len(retrieved_ids)
        metrics['unique_retrieved'] = len(set(retrieved_ids))
        return metrics
    
    # Calculate metrics at different k values
    for k in k_values:
        if k <= len(retrieved_ids):
            precision = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
            recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
            f1 = calculate_f1_score(precision, recall)
            
            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall
            metrics[f'f1@{k}'] = f1
    
    return metrics


# ============================================================================
# Ranking Metrics
# ============================================================================

def calculate_mrr(retrieved: List[str], 
                 relevant: Set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    MRR = 1 / (rank of first relevant item)
    
    Returns 0 if no relevant items found.
    """
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def calculate_average_precision(retrieved: List[str], 
                                relevant: Set[str]) -> float:
    """
    Calculate Average Precision (AP).
    AP = (sum of P@k for each relevant item) / (total # relevant items)
    """
    if not relevant:
        return 0.0
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            relevant_count += 1
            precision_at_i = relevant_count / i
            precision_sum += precision_at_i
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / len(relevant)


def calculate_ndcg(retrieved: List[str], 
                  relevant: Set[str], 
                  k: Optional[int] = None,
                  relevance_scores: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    Args:
        retrieved: List of retrieved item IDs
        relevant: Set of relevant item IDs
        k: Top-k items to consider
        relevance_scores: Optional dict mapping item IDs to relevance scores
    """
    if k is not None:
        retrieved = retrieved[:k]
    
    if not retrieved:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            # Use provided relevance score or binary (1 if relevant, 0 otherwise)
            rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
            dcg += rel / np.log2(i + 1)
    
    # Calculate IDCG (ideal DCG)
    if relevance_scores:
        # Sort by relevance scores
        ideal_items = sorted(
            [item for item in relevant],
            key=lambda x: relevance_scores.get(x, 0),
            reverse=True
        )
    else:
        ideal_items = list(relevant)
    
    ideal_items = ideal_items[:len(retrieved)]
    idcg = 0.0
    for i, item in enumerate(ideal_items, 1):
        rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
        idcg += rel / np.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_map(queries_results: List[tuple], 
                 queries_relevant: List[Set[str]]) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.
    
    Args:
        queries_results: List of (query_id, retrieved_items) tuples
        queries_relevant: List of relevant item sets for each query
    """
    if not queries_results or len(queries_results) != len(queries_relevant):
        return 0.0
    
    ap_sum = 0.0
    for retrieved, relevant in zip(queries_results, queries_relevant):
        ap_sum += calculate_average_precision(retrieved, relevant)
    
    return ap_sum / len(queries_results)


# ============================================================================
# Diversity Metrics
# ============================================================================

def calculate_diversity_score(items: List[Dict[str, Any]], 
                              method: str = 'embedding') -> float:
    """
    Calculate diversity score for retrieved items.
    Higher score = more diverse results.
    
    Args:
        items: List of retrieved items
        method: 'embedding' or 'content'
    """
    if len(items) < 2:
        return 0.0
    
    if method == 'embedding':
        # Use embeddings if available
        embeddings = []
        for item in items:
            if 'embedding' in item:
                embeddings.append(np.array(item['embedding']))
            elif 'metadata' in item and 'embedding' in item['metadata']:
                embeddings.append(np.array(item['metadata']['embedding']))
        
        if len(embeddings) >= 2:
            embeddings = np.array(embeddings)
            
            # Calculate average pairwise distance
            distances = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    distances.append(dist)
            
            return float(np.mean(distances))
    
    # Fallback: content-based diversity
    contents = [item.get('content', '') for item in items]
    
    # Calculate unique words ratio
    all_words = []
    for content in contents:
        words = content.lower().split()
        all_words.extend(words)
    
    if not all_words:
        return 0.0
    
    unique_words = len(set(all_words))
    total_words = len(all_words)
    
    return unique_words / total_words


def calculate_coverage(retrieved_items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate coverage metrics.
    
    Returns:
        - type_coverage: Distribution across different types (text/image)
        - source_coverage: Distribution across different sources
    """
    metrics = {}
    
    # Type distribution
    types = [item.get('type', 'unknown') for item in retrieved_items]
    type_counts = Counter(types)
    total = len(retrieved_items)
    
    if total > 0:
        metrics['type_distribution'] = {
            t: count / total for t, count in type_counts.items()
        }
    
    # Source distribution (if available)
    sources = []
    for item in retrieved_items:
        source = item.get('metadata', {}).get('source') or item.get('source')
        if source:
            sources.append(source)
    
    if sources:
        source_counts = Counter(sources)
        metrics['source_distribution'] = {
            s: count / len(sources) for s, count in source_counts.items()
        }
        metrics['unique_sources'] = len(source_counts)
    
    return metrics


def calculate_intra_list_diversity(items: List[Dict[str, Any]]) -> float:
    """
    Calculate Intra-List Diversity (ILD).
    Measures how dissimilar items are within the result list.
    """
    if len(items) < 2:
        return 0.0
    
    # Extract embeddings
    embeddings = []
    for item in items:
        emb = None
        if 'embedding' in item:
            emb = np.array(item['embedding'])
        elif 'metadata' in item and 'embedding' in item['metadata']:
            emb = np.array(item['metadata']['embedding'])
        
        if emb is not None:
            embeddings.append(emb)
    
    if len(embeddings) < 2:
        return 0.0
    
    embeddings = np.array(embeddings)
    
    # Calculate all pairwise distances
    n = len(embeddings)
    total_distance = 0.0
    pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Cosine distance = 1 - cosine_similarity
            cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
            )
            cos_dist = 1 - cos_sim
            total_distance += cos_dist
            pairs += 1
    
    if pairs == 0:
        return 0.0
    
    return total_distance / pairs


# ============================================================================
# Latency Metrics
# ============================================================================

class LatencyTracker:
    """Track and analyze latency metrics."""
    
    def __init__(self):
        self.latencies = []
        self.component_times = {}
    
    def record(self, total_time: float, components: Optional[Dict[str, float]] = None):
        """Record a latency measurement."""
        self.latencies.append(total_time)
        
        if components:
            for component, time_val in components.items():
                if component not in self.component_times:
                    self.component_times[component] = []
                self.component_times[component].append(time_val)
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {}
        
        latencies = np.array(self.latencies)
        
        stats = {
            'mean': float(np.mean(latencies)),
            'median': float(np.median(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
        }
        
        # Component breakdown
        if self.component_times:
            stats['components'] = {}
            for component, times in self.component_times.items():
                times_arr = np.array(times)
                stats['components'][component] = {
                    'mean': float(np.mean(times_arr)),
                    'percentage': float(np.mean(times_arr) / stats['mean'] * 100)
                }
        
        return stats
    
    def reset(self):
        """Reset all tracked metrics."""
        self.latencies = []
        self.component_times = {}


# ============================================================================
# Evaluation Suite
# ============================================================================

def evaluate_retrieval_quality(retrieved_items: List[Dict[str, Any]],
                               ground_truth: Optional[Set[str]] = None,
                               k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
    """
    Comprehensive evaluation of retrieval quality.
    
    Returns all relevant metrics in one call.
    """
    results = {
        'timestamp': time.time(),
        'num_items': len(retrieved_items)
    }
    
    # Basic metrics
    results['retrieval_metrics'] = calculate_retrieval_metrics(
        retrieved_items, ground_truth, k_values
    )
    
    # Diversity metrics
    results['diversity_score'] = calculate_diversity_score(retrieved_items)
    results['intra_list_diversity'] = calculate_intra_list_diversity(retrieved_items)
    results['coverage'] = calculate_coverage(retrieved_items)
    
    # Ranking metrics (if ground truth provided)
    if ground_truth:
        retrieved_ids = [
            item.get('id') or item.get('content', '')[:50] 
            for item in retrieved_items
        ]
        results['mrr'] = calculate_mrr(retrieved_ids, ground_truth)
        results['ndcg'] = calculate_ndcg(retrieved_ids, ground_truth)
        results['average_precision'] = calculate_average_precision(retrieved_ids, ground_truth)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Evaluation Metrics Module")
    print("=" * 60)
    
    # Sample data
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = {'doc1', 'doc3', 'doc5', 'doc7'}
    
    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant}")
    print()
    
    # Calculate metrics
    precision = calculate_precision_at_k(retrieved, relevant, k=5)
    recall = calculate_recall_at_k(retrieved, relevant, k=5)
    f1 = calculate_f1_score(precision, recall)
    mrr = calculate_mrr(retrieved, relevant)
    ndcg = calculate_ndcg(retrieved, relevant, k=5)
    
    print(f"Precision@5: {precision:.3f}")
    print(f"Recall@5: {recall:.3f}")
    print(f"F1@5: {f1:.3f}")
    print(f"MRR: {mrr:.3f}")
    print(f"NDCG@5: {ndcg:.3f}")