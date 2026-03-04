import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from hybrid_retrieval import hybrid_retrieve_rerank, RetrievedChunk


# ----------------------------
# Optional: local LLM via Ollama
# ----------------------------
def _try_load_ollama():
    """
    If you have ollama running (e.g. `ollama serve`) and a model pulled,
    this enables natural language answers.
    """
    try:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama
    except Exception:
        return None


# ----------------------------
# Conversation state
# ----------------------------
@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ChatState:
    history: List[Turn] = field(default_factory=list)
    # "slots" — keep it simple but useful for clinical guideline queries
    slots: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "population": None,     # e.g., adults, children, pregnant people
        "severity": None,       # mild/moderate/severe/critical
        "setting": None,        # outpatient/inpatient/ICU
        "intervention": None,   # e.g., corticosteroids, oxygen, antivirals
        "goal": None,           # treat/prevent/diagnose
    })


# ----------------------------
# Clarifying question logic (lightweight)
# ----------------------------
SEVERITY_WORDS = ["mild", "moderate", "severe", "critical", "icu"]
POP_WORDS = ["adult", "adults", "child", "children", "pregnant", "pregnancy", "neonate", "newborn"]
SETTING_WORDS = ["outpatient", "inpatient", "hospital", "icu", "emergency", "primary care"]
INTERVENTION_HINTS = ["corticosteroid", "dexamethasone", "oxygen", "ventilation", "antiviral", "remdesivir", "mask"]


def update_slots_from_text(state: ChatState, text: str) -> None:
    t = text.lower()

    if any(w in t for w in POP_WORDS) and state.slots["population"] is None:
        # crude extraction
        if "preg" in t:
            state.slots["population"] = "pregnant people"
        elif "child" in t:
            state.slots["population"] = "children"
        else:
            state.slots["population"] = "adults"

    if any(w in t for w in SEVERITY_WORDS) and state.slots["severity"] is None:
        for w in SEVERITY_WORDS:
            if w in t:
                state.slots["severity"] = w.upper() if w != "icu" else "ICU/critical"
                break

    if any(w in t for w in SETTING_WORDS) and state.slots["setting"] is None:
        for w in SETTING_WORDS:
            if w in t:
                state.slots["setting"] = w
                break

    if any(w in t for w in INTERVENTION_HINTS) and state.slots["intervention"] is None:
        # pick first matching term as a hint
        for w in INTERVENTION_HINTS:
            if w in t:
                state.slots["intervention"] = w
                break

    # goal heuristic
    if state.slots["goal"] is None:
        if any(x in t for x in ["treat", "treatment", "therapy", "manage", "management"]):
            state.slots["goal"] = "treatment"
        elif any(x in t for x in ["prevent", "prophylaxis", "vaccin"]):
            state.slots["goal"] = "prevention"
        elif any(x in t for x in ["diagnos", "test", "screen"]):
            state.slots["goal"] = "diagnosis"


def needs_clarification(user_text: str, state: ChatState) -> Optional[str]:
    """
    If query is too broad, ask a targeted clarifying question.
    """
    t = user_text.strip().lower()
    if len(t) < 8:
        return "Can you share a bit more detail—what specifically in the WHO guideline are you looking for?"

    # If it's about treatment but missing severity, that's usually the #1 ambiguity.
    looks_clinical = any(x in t for x in ["recommend", "should", "guideline", "treat", "management", "when"])
    if looks_clinical and state.slots["severity"] is None:
        return "Which severity level are we talking about (mild / moderate / severe / critical/ICU)?"

    # If population matters and it's missing
    if looks_clinical and state.slots["population"] is None:
        return "Which patient group—adults, children, pregnant people, or something else?"

    return None


# ----------------------------
# Retrieval → answer synthesis
# ----------------------------
def build_retrieval_query(user_text: str, state: ChatState) -> str:
    """
    Enrich the query with slot context so retrieval is sharper.
    """
    parts = [user_text.strip()]

    for k in ["population", "severity", "setting", "intervention", "goal"]:
        v = state.slots.get(k)
        if v:
            parts.append(f"{k}:{v}")

    # keep it short; too long hurts
    return " | ".join(parts[:4])


def format_citations(chunks: List[RetrievedChunk], max_items: int = 5) -> str:
    cites = []
    for c in chunks[:max_items]:
        md = c.metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page", "NA")
        cites.append(f"- source: {src} (page {page})")
    return "\n".join(cites)


def extractive_answer(user_text: str, chunks: List[RetrievedChunk]) -> str:
    """
    No-LLM fallback: return the best evidence snippets + citations.
    """
    if not chunks:
        return "I couldn’t find anything relevant in the ingested WHO PDFs. Try rephrasing or ingest more documents."

    top = chunks[:3]
    bullets = []
    for i, c in enumerate(top, 1):
        md = c.metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page", "NA")
        snippet = re.sub(r"\s+", " ", (c.text or "")).strip()
        bullets.append(f"[{i}] {snippet[:650]}...\n    (source={src}, page={page})")

    return "Here are the most relevant guideline passages I found:\n\n" + "\n\n".join(bullets)


def llm_answer_with_citations(
    user_text: str,
    chunks: List[RetrievedChunk],
    chat_history: List[Turn],
    ollama_model: str = "llama3.1:8b",
) -> str:
    """
    Uses a local LLM (Ollama) to synthesize a clean answer grounded in retrieved text.
    If Ollama isn't available, caller should use extractive_answer().
    """
    ChatOllama = _try_load_ollama()
    if ChatOllama is None:
        return extractive_answer(user_text, chunks)

    llm = ChatOllama(model=ollama_model, temperature=0.2)

    # Build context string
    context_blocks = []
    for i, c in enumerate(chunks[:8], 1):
        md = c.metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page", "NA")
        context_blocks.append(
            f"[{i}] (source={src}, page={page})\n{c.text}"
        )
    context = "\n\n".join(context_blocks)

    # Lightweight conversation context (truncate)
    hist = "\n".join([f"{t.role.upper()}: {t.content}" for t in chat_history[-6:]])

    prompt = f"""You are a clinical-guideline assistant.
Answer ONLY using the provided WHO guideline excerpts. If the excerpts do not contain the answer, say so.

Conversation:
{hist}

User question:
{user_text}

WHO guideline excerpts:
{context}

Instructions:
- Give a direct, structured answer.
- If there are conditions (severity, oxygen requirement, contraindications), include them.
- Cite sources inline like [1], [2] corresponding to the excerpt numbers above.
"""

    resp = llm.invoke(prompt)
    text = getattr(resp, "content", None) or str(resp)

    # add explicit source list at end
    text += "\n\nSources:\n" + format_citations(chunks, max_items=8)
    return text


# ----------------------------
# Main chat loop
# ----------------------------
def chat():
    state = ChatState()

    chroma_dir = os.environ.get("CHROMA_DIR", "./data/chroma_who_iris")
    collection_name = os.environ.get("CHROMA_COLLECTION", "who_iris_guidelines")
    embedding_model = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(chroma_dir)

    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    use_llm = os.environ.get("USE_LLM", "1").strip() not in ["0", "false", "False"]

    print("WHO Guideline Chat (hybrid retrieval + rerank)")
    print("Type ':quit' to exit, ':slots' to view extracted context, ':reset' to reset.\n")

    while True:
        user = input("you> ").strip()
        if not user:
            continue
        if user == ":quit":
            break
        if user == ":reset":
            state = ChatState()
            print("assistant> Reset conversation.\n")
            continue
        if user == ":slots":
            print(f"assistant> slots={state.slots}\n")
            continue

        state.history.append(Turn(role="user", content=user))
        update_slots_from_text(state, user)

        clar = needs_clarification(user, state)
        if clar:
            state.history.append(Turn(role="assistant", content=clar))
            print(f"assistant> {clar}\n")
            continue

        retrieval_query = build_retrieval_query(user, state)

        chunks = hybrid_retrieve_rerank(
            retrieval_query,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            top_k_final=6,
            top_k_vec=50,
            top_k_bm25=50,
            alpha=0.6,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_top_n=50,
        )

        if use_llm:
            answer = llm_answer_with_citations(
                user_text=user,
                chunks=chunks,
                chat_history=state.history,
                ollama_model=ollama_model,
            )
        else:
            answer = extractive_answer(user, chunks)

        state.history.append(Turn(role="assistant", content=answer))
        print(f"assistant> {answer}\n")


if __name__ == "__main__":
    chat()