"""
Optimized Multimodal Parallel RAG + CLIP-only reranker
Key improvements:
- Async LLM calls
- Reuse embeddings from FAISS retrieval
- Batch processing optimization
- Better error handling
- Reduced redundant encoding
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

from multimodal_indexer import CLIPEmbedding


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY env var.")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Optimized cosine similarity with early returns."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


@lru_cache(maxsize=1)
def get_clip_model(model_name: str) -> SentenceTransformer:
    """Cache CLIP model to avoid reloading."""
    return SentenceTransformer(model_name)


def load_dual_vectorstores(text_tables_path: str,
                           images_path: str,
                           clip_model: str = "clip-ViT-B-32") -> Dict[str, Optional[FAISS]]:
    """Load FAISS indexes with better error handling."""
    logger.info("Loading FAISS indexes using CLIP embeddings...")
    embedder = CLIPEmbedding(model_name=clip_model)

    vs = {"text_tables": None, "images": None}

    try:
        if os.path.exists(text_tables_path):
            vs["text_tables"] = FAISS.load_local(
                text_tables_path, embedder, allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded text_tables index from {text_tables_path}")
    except Exception as e:
        logger.error(f"Failed to load text_tables: {e}")

    try:
        if os.path.exists(images_path):
            vs["images"] = FAISS.load_local(
                images_path, embedder, allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded images index from {images_path}")
    except Exception as e:
        logger.error(f"Failed to load images: {e}")

    if not vs["text_tables"] and not vs["images"]:
        raise RuntimeError("No FAISS indexes loaded successfully!")

    return vs


def create_llm(model: str = "gemini-2.5-flash",  # Fixed model name
               temperature: float = 0.3,
               max_output_tokens: int = 1200) -> ChatGoogleGenerativeAI:
    """Create Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )


def create_qa_chains(vectorstores: Dict[str, FAISS],
                     llm: ChatGoogleGenerativeAI,
                     k: int = 10,
                     fetch_k: int = 50,
                     lambda_mult: float = 0.5) -> Dict[str, RetrievalQA]:
    """
    Create QA chains with MMR retrieval for diversity.
    
    Args:
        fetch_k: Number of candidates to fetch before MMR
        lambda_mult: Diversity factor (0=max diversity, 1=max relevance)
    """
    chains = {}

    if vectorstores.get("text_tables"):
        chains["text_tables"] = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstores["text_tables"].as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                }
            ),
            chain_type="stuff",
            return_source_documents=True,
            verbose=False
        )

    if vectorstores.get("images"):
        chains["images"] = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstores["images"].as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                }
            ),
            chain_type="stuff",
            return_source_documents=True,
            verbose=False
        )

    return chains


class CLIPReranker:
    """
    Optimized reranker using CLIP cosine similarity with MMR-based diversity.
    - Reuses embeddings from FAISS retrieval (stored in metadata)
    - Applies MMR for final diverse ranking
    """

    def __init__(self, clip_model: str = "clip-ViT-B-32", 
                 text_weight: float = 1.0, 
                 img_weight: float = 1.0,
                 lambda_mult: float = 0.7):
        """
        Args:
            lambda_mult: Controls diversity vs relevance tradeoff
                        1.0 = pure relevance, 0.0 = pure diversity
        """
        self.clip = get_clip_model(clip_model)
        self.text_weight = text_weight
        self.img_weight = img_weight
        self.lambda_mult = lambda_mult
        self._query_cache = {}

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Cache query embeddings to avoid recomputation."""
        if query not in self._query_cache:
            self._query_cache[query] = self.clip.encode(
                [query], 
                show_progress_bar=False,
                convert_to_numpy=True
            )[0]
        return self._query_cache[query]

    def _compute_mmr_scores(self, 
                           query_emb: np.ndarray,
                           embeddings: np.ndarray,
                           selected_indices: List[int],
                           lambda_mult: float) -> np.ndarray:
        """
        Compute MMR scores for remaining candidates.
        
        MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
        """
        # Query similarity for all candidates
        query_sims = np.dot(embeddings, query_emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
        )
        
        # If no documents selected yet, use pure relevance
        if not selected_indices:
            return query_sims
        
        # Compute max similarity to already selected documents
        selected_embs = embeddings[selected_indices]
        doc_sims = np.dot(embeddings, selected_embs.T) / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) * 
            np.linalg.norm(selected_embs, axis=1) + 1e-10
        )
        max_doc_sims = np.max(doc_sims, axis=1)
        
        # MMR formula
        mmr_scores = lambda_mult * query_sims - (1 - lambda_mult) * max_doc_sims
        return mmr_scores

    def _extract_or_encode_embeddings(self, 
                                     candidates: List[Any], 
                                     is_image: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Extract embeddings from FAISS metadata or encode if missing.
        FAISS stores embeddings in the index, we can retrieve them.
        """
        embeddings = []
        contents = []
        
        for doc in candidates:
            contents.append(doc.page_content)
            
            # Try to get stored embedding from metadata
            if hasattr(doc, 'metadata') and 'embedding' in doc.metadata:
                emb = np.array(doc.metadata['embedding'])
                embeddings.append(emb)
            else:
                # Fallback: encode on the fly (shouldn't happen if FAISS stores them)
                # Note: We'll batch encode these later for efficiency
                embeddings.append(None)
        
        # Batch encode any missing embeddings
        missing_indices = [i for i, emb in enumerate(embeddings) if emb is None]
        if missing_indices:
            missing_texts = [contents[i] for i in missing_indices]
            encoded = self.clip.encode(
                missing_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=32
            )
            for idx, emb in zip(missing_indices, encoded):
                embeddings[idx] = emb
        
        return np.array(embeddings), contents

    def rerank_with_mmr(self,
                        query: str,
                        text_candidates: List[Any],
                        image_candidates: List[Any],
                        top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank candidates using MMR for diversity while maintaining relevance.
        """
        query_emb = self._get_query_embedding(query)
        
        # Prepare all candidates with embeddings
        all_items = []
        all_embeddings = []
        
        # Process text candidates
        if text_candidates:
            text_embs, text_contents = self._extract_or_encode_embeddings(text_candidates, False)
            for doc, emb, content in zip(text_candidates, text_embs, text_contents):
                all_items.append({
                    "type": "text",
                    "content": content,
                    "metadata": doc.metadata,
                    "embedding": emb
                })
                all_embeddings.append(emb)
        
        # Process image candidates
        if image_candidates:
            img_embs, img_contents = self._extract_or_encode_embeddings(image_candidates, True)
            for doc, emb, content in zip(image_candidates, img_embs, img_contents):
                all_items.append({
                    "type": "image",
                    "content": content,
                    "metadata": doc.metadata,
                    "image_path": doc.metadata.get("image_path"),
                    "embedding": emb
                })
                all_embeddings.append(emb)
        
        if not all_items:
            return []
        
        all_embeddings = np.array(all_embeddings)
        
        # Apply MMR selection
        selected_indices = []
        remaining_indices = list(range(len(all_items)))
        
        for _ in range(min(top_k, len(all_items))):
            if not remaining_indices:
                break
            
            # Compute MMR scores for remaining candidates
            mmr_scores = self._compute_mmr_scores(
                query_emb,
                all_embeddings[remaining_indices],
                [remaining_indices.index(i) if i in remaining_indices else i 
                 for i in selected_indices],
                self.lambda_mult
            )
            
            # Select best candidate
            best_idx = np.argmax(mmr_scores)
            actual_idx = remaining_indices[best_idx]
            selected_indices.append(actual_idx)
            remaining_indices.remove(actual_idx)
        
        # Build final results with scores
        results = []
        for idx in selected_indices:
            item = all_items[idx].copy()
            
            # Compute final relevance score
            sim = cosine_sim(query_emb, item["embedding"])
            weight = self.img_weight if item["type"] == "image" else self.text_weight
            item["score"] = float(sim * weight)
            
            # Remove embedding from result (not needed in output)
            del item["embedding"]
            results.append(item)
        
        return results

    # Keep old method for backward compatibility
    def rerank(self, *args, **kwargs):
        """Backward compatible method - redirects to MMR version."""
        return self.rerank_with_mmr(*args, **kwargs)


class ParallelRAGClipOnly:
    """
    Optimized parallel RAG with:
    - Async LLM calls
    - Better error handling
    - Configurable timeouts
    """

    def __init__(self,
                 qa_text: Optional[RetrievalQA],
                 qa_img: Optional[RetrievalQA],
                 llm: ChatGoogleGenerativeAI,
                 reranker: CLIPReranker,
                 top_k: int = 10,
                 timeout: float = 30.0):
        self.qa_text = qa_text
        self.qa_img = qa_img
        self.llm = llm
        self.reranker = reranker
        self.top_k = top_k
        self.timeout = timeout

    async def _fetch_text(self, q: str) -> List[Any]:
        """Fetch text documents with error handling."""
        if not self.qa_text:
            return []
        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(self.qa_text.invoke, {"query": q}),
                timeout=self.timeout
            )
            return res.get("source_documents", [])
        except asyncio.TimeoutError:
            logger.error(f"Text retrieval timeout after {self.timeout}s")
            return []
        except Exception as e:
            logger.error(f"Text retrieval error: {e}")
            return []

    async def _fetch_images(self, q: str) -> List[Any]:
        """Fetch image documents with error handling."""
        if not self.qa_img:
            return []
        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(self.qa_img.invoke, {"query": q}),
                timeout=self.timeout
            )
            return res.get("source_documents", [])
        except asyncio.TimeoutError:
            logger.error(f"Image retrieval timeout after {self.timeout}s")
            return []
        except Exception as e:
            logger.error(f"Image retrieval error: {e}")
            return []

    async def retrieve_and_rerank(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve and rerank in parallel."""
        text_docs, img_docs = await asyncio.gather(
            self._fetch_text(query),
            self._fetch_images(query),
            return_exceptions=True  # Don't fail if one fails
        )
        
        # Handle exceptions
        if isinstance(text_docs, Exception):
            logger.error(f"Text retrieval failed: {text_docs}")
            text_docs = []
        if isinstance(img_docs, Exception):
            logger.error(f"Image retrieval failed: {img_docs}")
            img_docs = []

        return self.reranker.rerank(query, text_docs, img_docs, top_k=self.top_k)

    async def synthesize_async(self, query: str, top_items: List[Dict[str, Any]]) -> str:
        """Async synthesis with LLM."""
        # Build context more efficiently
        context_parts = []
        for it in top_items:
            if it["type"] == "image":
                context_parts.append(
                    f"[IMAGE] path={it.get('image_path')} "
                    f"caption=\"{it['content']}\" score={it['score']:.3f}"
                )
            else:
                # Truncate text efficiently
                content = it['content'][:300]
                context_parts.append(f"[TEXT] {content}... score={it['score']:.3f}")

        context = "\n".join(context_parts)

        prompt = (
            "You are a multimodal assistant.\n"
            "Use the retrieved text and image context.\n"
            "If images are relevant, mention the image paths.\n\n"
            f"USER QUERY:\n{query}\n\n"
            f"MULTIMODAL CONTEXT:\n{context}\n\n"
            "FINAL ANSWER:"
        )

        # Async LLM call
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(self.llm.invoke, prompt),
                timeout=self.timeout
            )
            return getattr(resp, "content", str(resp))
        except asyncio.TimeoutError:
            return f"Error: LLM synthesis timed out after {self.timeout}s"
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return f"Error during synthesis: {str(e)}"

    async def ask_async(self, query: str) -> Dict[str, Any]:
        """Async query processing."""
        reranked = await self.retrieve_and_rerank(query)
        answer = await self.synthesize_async(query, reranked)

        return {
            "query": query,
            "top_items": reranked,
            "answer": answer
        }

    def ask(self, query: str) -> Dict[str, Any]:
        """Synchronous wrapper for async ask."""
        return asyncio.run(self.ask_async(query))


def setup_system(text_tables_path: str,
                 images_path: str,
                 clip_model: str = "clip-ViT-B-32",
                 top_k: int = 10,
                 retrieval_k: int = 10,
                 fetch_k: int = 50,
                 lambda_mult: float = 0.5,
                 rerank_lambda: float = 0.7) -> ParallelRAGClipOnly:
    """
    Setup the RAG system with MMR and optimized configuration.
    
    Args:
        text_tables_path: Path to text FAISS index
        images_path: Path to images FAISS index
        clip_model: CLIP model name
        top_k: Final number of results to return
        retrieval_k: Number of docs to retrieve per source
        fetch_k: Number of candidates to fetch before MMR in retrieval
        lambda_mult: Diversity parameter for retrieval MMR (0=diverse, 1=relevant)
        rerank_lambda: Diversity parameter for reranking MMR (0=diverse, 1=relevant)
    """
    vectorstores = load_dual_vectorstores(text_tables_path, images_path, clip_model)
    llm = create_llm()
    qa_chains = create_qa_chains(
        vectorstores, 
        llm, 
        k=retrieval_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )
    reranker = CLIPReranker(
        clip_model=clip_model,
        lambda_mult=rerank_lambda
    )

    return ParallelRAGClipOnly(
        qa_text=qa_chains.get("text_tables"),
        qa_img=qa_chains.get("images"),
        llm=llm,
        reranker=reranker,
        top_k=top_k
    )


if __name__ == "__main__":
    TEXT = "faiss_indexes\\text_tables"
    IMAGES = "faiss_indexes\\images"

    # Setup with MMR for diverse results
    # fetch_k=50: Consider 50 candidates initially
    # lambda_mult=0.5: Balance between relevance and diversity in retrieval
    # rerank_lambda=0.7: Slightly favor relevance over diversity in final ranking
    agent = setup_system(
        TEXT, 
        IMAGES, 
        top_k=6,
        retrieval_k=20,  # Retrieve more candidates for MMR
        fetch_k=50,      # Initial candidate pool
        lambda_mult=0.5,  # Retrieval diversity
        rerank_lambda=0.7  # Reranking diversity
    )

    result = agent.ask("Explain Qatar's GDP trend and include relevant images.")
    print("\nANSWER:\n", result["answer"])

    print("\nTOP ITEMS (MMR-ranked for diversity):")
    for i, item in enumerate(result["top_items"], 1):
        print(f"\n{i}. {item['type'].upper()} (score={item['score']:.3f})")
        if item['type'] == 'image':
            print(f"   Path: {item.get('image_path')}")
        print(f"   Content: {item['content'][:150]}...")
