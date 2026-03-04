import os
import re
import math
import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ----------------------------
# Helpers
# ----------------------------
def _simple_tokenize(text: str) -> List[str]:
    # simple + effective for BM25 on guideline text
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-/%\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def _minmax_norm(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if math.isclose(lo, hi):
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    score_vec: Optional[float] = None
    score_bm25: Optional[float] = None
    score_fused: Optional[float] = None
    score_rerank: Optional[float] = None


# ----------------------------
# Build / Load Chroma
# ----------------------------
def load_chroma(
    chroma_dir: str = "./data/chroma_who_iris",
    collection_name: str = "who_iris_guidelines",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Chroma:
    if not os.path.isdir(chroma_dir):
        raise FileNotFoundError(f"Chroma directory not found: {chroma_dir}")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vs = Chroma(
        persist_directory=chroma_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return vs


# ----------------------------
# BM25 index persisted on disk
# ----------------------------
def build_or_load_bm25(
    vs: Chroma,
    bm25_cache_path: str = "./data/bm25_cache.pkl",
) -> Tuple[BM25Okapi, List[str], List[str], List[Dict[str, Any]]]:
    """
    Reads all docs from Chroma and builds BM25.
    Caches tokenized corpus for fast reload.
    Returns: (bm25, ids, docs, metas)
    """
    os.makedirs(os.path.dirname(bm25_cache_path), exist_ok=True)

    if os.path.exists(bm25_cache_path):
        with open(bm25_cache_path, "rb") as f:
            payload = pickle.load(f)
        return payload["bm25"], payload["ids"], payload["docs"], payload["metas"]

    # Pull everything stored in Chroma
    store = vs.get(include=["documents", "metadatas"])
    ids: List[str] = store["ids"]
    docs: List[str] = store["documents"]
    metas: List[Dict[str, Any]] = store["metadatas"]

    tokenized = [_simple_tokenize(d or "") for d in docs]
    bm25 = BM25Okapi(tokenized)

    with open(bm25_cache_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "docs": docs, "metas": metas}, f)

    return bm25, ids, docs, metas


# ----------------------------
# Hybrid retrieve + rerank
# ----------------------------
def hybrid_retrieve_rerank(
    query: str,
    *,
    chroma_dir: str = "./data/chroma_who_iris",
    collection_name: str = "who_iris_guidelines",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    bm25_cache_path: str = "./data/bm25_cache.pkl",
    # candidate generation
    top_k_final: int = 5,
    top_k_vec: int = 30,
    top_k_bm25: int = 30,
    # fusion
    alpha: float = 0.6,  # weight on vector; (1-alpha) on bm25
    # reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top_n: int = 30,  # rerank only top N after fusion
) -> List[RetrievedChunk]:
    """
    Returns reranked chunks with tight relevance.
    Strategy:
      - semantic: Chroma similarity_search_with_score
      - lexical: BM25
      - fuse: weighted normalized scores
      - rerank: cross-encoder over fused shortlist
    """
    if not query or not query.strip():
        return []

    vs = load_chroma(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    # --- Vector candidates ---
    vec_pairs = vs.similarity_search_with_score(query, k=top_k_vec)
    # Chroma returns distance-like scores depending on config; treat lower as better sometimes.
    # We'll convert to a "higher is better" similarity by negating.
    vec_chunks: Dict[str, RetrievedChunk] = {}
    for doc, score in vec_pairs:
        cid = (doc.metadata.get("id") if isinstance(doc.metadata, dict) else None) or ""
        # Prefer Chroma's internal id if metadata lacks it — but LangChain doesn't expose it directly.
        # We'll hash (source,page,content) as a stable key.
        key = cid or f"{doc.metadata.get('source','')}|{doc.metadata.get('page','')}|{hash(doc.page_content)}"
        vec_chunks[key] = RetrievedChunk(
            id=key,
            text=doc.page_content,
            metadata=doc.metadata or {},
            score_vec=float(-score),  # flip sign so higher is better
        )

    # --- BM25 candidates ---
    bm25, ids_all, docs_all, metas_all = build_or_load_bm25(vs, bm25_cache_path=bm25_cache_path)
    q_tokens = _simple_tokenize(query)
    bm25_scores_all = bm25.get_scores(q_tokens)

    # top indices by BM25
    top_idx = sorted(range(len(bm25_scores_all)), key=lambda i: bm25_scores_all[i], reverse=True)[:top_k_bm25]

    bm_chunks: Dict[str, RetrievedChunk] = {}
    for i in top_idx:
        text = docs_all[i] or ""
        meta = metas_all[i] or {}
        # Use Chroma id (ids_all[i]) as stable key
        key = ids_all[i]
        bm_chunks[key] = RetrievedChunk(
            id=key,
            text=text,
            metadata=meta,
            score_bm25=float(bm25_scores_all[i]),
        )

    # --- Fuse (union by id) ---
    union: Dict[str, RetrievedChunk] = {}

    # add BM25 first
    for k, ch in bm_chunks.items():
        union[k] = ch

    # merge vector candidates (may not share exact ids; keep both)
    for k, ch in vec_chunks.items():
        if k in union:
            union[k].score_vec = ch.score_vec
        else:
            union[k] = ch

    # normalize scores separately
    vec_list = [c.score_vec if c.score_vec is not None else float("-inf") for c in union.values()]
    bm_list = [c.score_bm25 if c.score_bm25 is not None else float("-inf") for c in union.values()]

    # replace -inf with min real score so normalization doesn't blow up
    def _sanitize(vals: List[float]) -> List[float]:
        real = [v for v in vals if v != float("-inf")]
        if not real:
            return [0.0] * len(vals)
        mn = min(real)
        return [mn if v == float("-inf") else v for v in vals]

    vec_s = _sanitize(vec_list)
    bm_s = _sanitize(bm_list)

    vec_n = _minmax_norm(vec_s)
    bm_n = _minmax_norm(bm_s)

    # assign fused
    for c, vnorm, bnorm in zip(union.values(), vec_n, bm_n):
        c.score_fused = alpha * vnorm + (1.0 - alpha) * bnorm

    # shortlist for rerank
    fused_sorted = sorted(union.values(), key=lambda x: x.score_fused or 0.0, reverse=True)
    shortlist = fused_sorted[: min(rerank_top_n, len(fused_sorted))]

    # --- Cross-encoder rerank ---
    reranker = CrossEncoder(reranker_model)
    pairs = [(query, c.text) for c in shortlist]
    rr_scores = reranker.predict(pairs)  # higher = better

    for c, s in zip(shortlist, rr_scores):
        c.score_rerank = float(s)

    final_sorted = sorted(shortlist, key=lambda x: x.score_rerank or 0.0, reverse=True)
    return final_sorted[:top_k_final]


def format_results(results: List[RetrievedChunk], max_chars: int = 700) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        md = r.metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page", "NA")
        lines.append(
            f"[{i}] rerank={r.score_rerank:.4f} fused={r.score_fused:.4f} "
            f"(vec={r.score_vec}, bm25={r.score_bm25})\n"
            f"    source={src} page={page}\n"
            f"    {r.text[:max_chars].strip()}\n"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    q = "When are corticosteroids recommended for severe COVID-19?"
    res = hybrid_retrieve_rerank(
        q,
        top_k_final=5,
        top_k_vec=40,
        top_k_bm25=40,
        alpha=0.6,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_n=40,
    )
    print(format_results(res))