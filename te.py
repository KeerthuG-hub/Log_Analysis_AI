__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import re
import json
import logging
import traceback
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# Optional libs
try:
    import numpy as np
except Exception:
    np = None

try:
    import dateparser
except Exception:
    dateparser = None

# Third-party components (gracefully handled if missing)
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    Chroma = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

try:
    from groq import Groq
except Exception:
    Groq = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
CHROMA_PATH = "chroma_db"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Command descriptions (synonyms / intent words)
COMMAND_DESCRIPTIONS = {
    "rm": "delete remove unlink erase destroy eliminate purge",
    "cp": "copy duplicate replicate backup clone",
    "mv": "move rename relocate transfer shift",
    "chmod": "permission rights access security change mode",
    "touch": "create file timestamp update",
    "mkdir": "create directory folder path",
    "ls": "list show directory files contents",
    "find": "search locate discover",
    "cat": "view read display concatenate",
    "head": "view first lines",
    "tail": "view last lines",
    "grep": "search filter pattern match",
    "sed": "edit stream modify replace substitute",
    "systemctl": "service control start stop restart enable disable",
    "ps": "process list tasks running",
    "kill": "terminate stop end process",
    "df": "disk filesystem space usage",
    "du": "directory size usage",
    "ssh": "remote login shell",
    "scp": "secure copy transfer",
    "rsync": "sync copy backup mirror",
    "wget": "download fetch",
    "curl": "download fetch http request",
    "history": "command history log",
    "auditctl": "audit logging event",
    "sshd": "ssh daemon login auth",
    "truncate": "shrink reduce clear zero",
}

# Intent keywords (for scoring)
INTENT_KEYWORDS = {
    "delete": ["delete", "remove", "rm", "erase", "destroy", "unlink", "purge", "deleted", "erased", "removed"],
    "copy": ["copy", "cp", "duplicate", "backup", "clone", "replicate"],
    "move": ["move", "mv", "rename", "relocate", "transfer", "moved"],
    "view": ["view", "show", "display", "cat", "head", "tail", "seen"],
    "list": ["list", "ls", "enumerate", "all"],
    "search": ["search", "find", "grep", "locate"],
    "create": ["create", "make", "mkdir", "touch", "new", "created"],
    "execute": ["execute", "run", "bash", "python", "script", "executed"],
    "service": ["service", "systemctl", "start", "stop", "restart"],
    "process": ["process", "ps", "kill", "terminate"],
    "permission": ["permission", "chmod", "chown", "rights", "access", "ownership"],
    "network": ["network", "ssh", "scp", "wget", "curl", "download"],
    "system": ["system", "mount", "df", "du", "disk", "free", "memory"],
    "authentication": ["auth", "authentication", "login", "password", "unauthorized", "failed", "accepted"],
}

# Intent -> canonical command candidates
INTENT_TO_COMMANDS = {
    "delete": ["rm", "unlink", "truncate"],
    "copy": ["cp", "scp", "rsync"],
    "move": ["mv"],
    "view": ["cat", "head", "tail"],
    "list": ["ls", "find"],
    "search": ["grep", "find"],
    "create": ["mkdir", "touch"],
    "execute": ["bash", "python3"],
    "service": ["systemctl"],
    "process": ["ps", "kill"],
    "permission": ["chmod", "chown"],
    "network": ["ssh", "scp", "wget", "curl"],
    "system": ["df", "du", "mount", "umount", "free"],
    "authentication": ["auditctl", "sshd"],
}

# -----------------------------
# Utility helpers
# -----------------------------
def normalize_text(s: str) -> str:
    return (s or "").strip().lower()

def safe_get(d: dict, key: str, default=""):
    return d.get(key, default) if isinstance(d, dict) else default

def now_utc():
    return datetime.utcnow()

def parse_time_filter(tf: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Return (start, end) datetimes for a time_filter string if parseable."""
    if not tf:
        return None, None
    tf = tf.strip().lower()
    now = now_utc()

    # direct ISO date
    m = re.match(r"^(\d{4}-\d{2}-\d{2})$", tf)
    if m:
        try:
            d = datetime.fromisoformat(m.group(1))
            start = d.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(seconds=1)
            return start, end
        except Exception:
            pass

    # relative "last N hours/days/weeks"
    m = re.search(r"(last|past|previous)\s+(\d+)\s+(hour|hours|day|days|week|weeks|month|months|year|years)", tf)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        if "hour" in unit:
            return now - timedelta(hours=n), now
        if "day" in unit:
            return now - timedelta(days=n), now
        if "week" in unit:
            return now - timedelta(weeks=n), now
        if "month" in unit:
            return now - timedelta(days=30 * n), now
        if "year" in unit:
            return now - timedelta(days=365 * n), now

    if "yesterday" in tf:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1) - timedelta(seconds=1)
        return start, end

    if "today" in tf:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1) - timedelta(seconds=1)
        return start, end

    if dateparser:
        try:
            parsed = dateparser.parse(tf)
            if parsed:
                return parsed, now
        except Exception:
            pass

    return None, None

def vector_norm(vec):
    if vec is None:
        return None
    if np is not None:
        arr = np.asarray(vec, dtype=float)
        n = np.linalg.norm(arr)
        if n == 0:
            return arr.tolist()
        return (arr / n).tolist()
    else:
        # pure Python fallback
        s = sum(x * x for x in vec)
        if s == 0:
            return vec
        norm = s ** 0.5
        return [x / norm for x in vec]

def cosine_sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    try:
        if np is not None:
            return float(np.dot(np.asarray(a), np.asarray(b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        else:
            num = sum(x * y for x, y in zip(a, b))
            den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5) + 1e-12
            return num / den
    except Exception:
        return 0.0

# -----------------------------
# Main class (kept name as requested)
# -----------------------------
class QueryOnlyLogStore:
    def __init__(self):
        self.db = None
        self.embedding_model = None
        self.sentences_data: List[Tuple[str, Dict[str, Any]]] = []
        self.command_index = defaultdict(list)
        self.user_index = defaultdict(list)
        self.keyword_index = defaultdict(list)
        self.groq_client = None
        self.groq_available = False
        self.command_embeddings = {}  # precomputed embeddings for COMMAND_DESCRIPTIONS

        # Initialize components
        self._init_components()

    # -------------------------
    # Initialization & loading
    # -------------------------
    def _init_components(self):
        """Initialize embedding model, Groq client (optional), and load Chroma DB."""
        # Embedding model
        if HuggingFaceEmbeddings is None:
            logger.warning("HuggingFaceEmbeddings not installed — embeddings disabled.")
            self.embedding_model = None
        else:
            try:
                self.embedding_model = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
                logger.info("✅ Embedding model initialized")
            except Exception as e:
                logger.warning(f"Failed to init embedding model: {e}")
                self.embedding_model = None

        # Groq (AI) client
        api_key = os.environ.get("GROQ_API_KEY")
        if Groq and api_key:
            try:
                self.groq_client = Groq(api_key=api_key)
                self.groq_available = True
                logger.info("✅ Groq client initialized")
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")
                self.groq_client = None
                self.groq_available = False
        else:
            self.groq_available = False
            if not Groq:
                logger.info("Groq library not present — AI features disabled.")
            else:
                logger.info("GROQ_API_KEY not set — AI features disabled.")

        # Load Chroma DB and build indices
        self._load_chroma_database()

        # Precompute command description embeddings (if embedding model available)
        self._precompute_command_embeddings()

    def _load_chroma_database(self):
        """Load Chroma DB and build in-memory indices for fast exact/keyword lookups."""
        if Chroma is None:
            logger.warning("Chroma not installed — semantic DB disabled. Will use keyword/exact only.")
            self.db = None
            return

        if not os.path.exists(CHROMA_PATH):
            logger.warning(f"Chroma DB not found at {CHROMA_PATH} — semantic DB disabled.")
            self.db = None
            return

        try:
            self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_model)
            all_docs = self.db.get()
            ids = all_docs.get("ids", [])
            metadatas = all_docs.get("metadatas", [])
            logger.info(f"✅ Loaded {len(ids)} documents from Chroma DB")

            self.sentences_data = []
            for i, (doc_id, metadata) in enumerate(zip(ids, metadatas)):
                original_content = metadata.get("original_content") or metadata.get("content") or ""
                self.sentences_data.append((original_content, metadata))

                # build command index (lowercased keys)
                base_cmd = (metadata.get("base_command") or "").strip()
                if base_cmd:
                    self.command_index[base_cmd.lower()].append(i)

                # original command variants
                orig_cmd = (metadata.get("command") or "").strip()
                if orig_cmd and orig_cmd.lower() != base_cmd.lower():
                    self.command_index[orig_cmd.lower()].append(i)

                # user index
                user = (metadata.get("user") or "").strip()
                if user:
                    self.user_index[user.lower()].append(i)

                # keyword index (simple)
                enhanced = (metadata.get("enhanced_content") or original_content or "").lower()
                for w in re.findall(r"\b\w+\b", enhanced):
                    if len(w) > 2:
                        self.keyword_index[w].append(i)

            logger.info("✅ Indices built from Chroma metadata")

        except Exception as e:
            logger.exception(f"Failed to load Chroma DB: {e}")
            self.db = None

    def _precompute_command_embeddings(self):
        """Embed command descriptions to allow semantic mapping from query -> command."""
        if not self.embedding_model:
            logger.info("Embedding model unavailable — skipping command embeddings.")
            return

        try:
            commands = list(COMMAND_DESCRIPTIONS.keys())
            descs = [COMMAND_DESCRIPTIONS[c] for c in commands]
            embs = None
            try:
                embs = self.embedding_model.embed_documents(descs)
            except Exception:
                try:
                    embs = [self.embedding_model.embed_query(d) for d in descs]
                except Exception:
                    embs = None

            if embs:
                for cmd, emb in zip(commands, embs):
                    self.command_embeddings[cmd] = vector_norm(emb)
                logger.info("✅ Precomputed command description embeddings")
            else:
                logger.info("Could not compute command embeddings — semantic command detection disabled")
        except Exception as e:
            logger.exception(f"Precompute command embeddings failed: {e}")

    # -------------------------
    # Query restructuring (multi-intent scoring)
    # -------------------------
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query into:
          - ranked intents (list)
          - candidate commands (from intent mapping + COMMAND_DESCRIPTIONS semantic match)
          - user (if present)
          - time_filter (if present)
        """
        q = normalize_text(query)
        intent_scores = defaultdict(float)

        # Score intents by keyword overlap (count occurrences)
        for intent, kws in INTENT_KEYWORDS.items():
            for kw in kws:
                # word boundary match preferred
                if re.search(r"\b" + re.escape(kw) + r"\b", q):
                    intent_scores[intent] += 1.0

        # If no explicit keywords matched, try fuzzy presence of key verbs (fallback)
        if not intent_scores:
            # simple verb presence: choose 'search' as default mild intent
            intent_scores["search"] = 0.1

        # Build ranked intents list
        ranked = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        intents = [i for i, s in ranked if s > 0]

        # Candidate commands from intents
        candidate_cmds = []
        for intent in intents:
            candidate_cmds.extend(INTENT_TO_COMMANDS.get(intent, []))

        # Also add commands whose descriptions contain query words
        q_words = set(re.findall(r"\b\w+\b", q))
        for cmd, desc in COMMAND_DESCRIPTIONS.items():
            desc_words = set(desc.split())
            # intersection heuristic
            if len(q_words.intersection(desc_words)) >= 1:
                candidate_cmds.append(cmd)

        # Semantic mapping: if embeddings available, compute similarity between query and command descs
        if self.embedding_model and self.command_embeddings:
            try:
                q_emb = None
                try:
                    q_emb = self.embedding_model.embed_query(query)
                except Exception:
                    q_emb = None
                q_emb = vector_norm(q_emb)
                if q_emb is not None:
                    sims = []
                    for cmd, emb in self.command_embeddings.items():
                        sim = cosine_sim(q_emb, emb)
                        sims.append((cmd, sim))
                    sims.sort(key=lambda x: x[1], reverse=True)
                    # add top few with meaningful similarity
                    for cmd, sim in sims[:3]:
                        if sim > 0.45:  # threshold; tune as needed
                            candidate_cmds.append(cmd)
            except Exception as e:
                logger.debug(f"Command semantic mapping failed: {e}")

        # Normalize candidate commands (unique)
        candidate_cmds = list({c.lower() for c in candidate_cmds if c})

        # Extract user patterns (user:foo or by foo)
        user = ""
        m = re.search(r"user[:=]?\s*([A-Za-z0-9._-]+)", q)
        if m:
            user = m.group(1)
        else:
            m2 = re.search(r"\bby\s+([A-Za-z0-9._-]+)\b", q)
            if m2:
                user = m2.group(1)

        # Extract time filters (basic)
        time_filter = ""
        tm = re.search(r"\b(yesterday|today|last\s+\d+\s+(?:hours|days|weeks|months)|since\s+\d{4}-\d{2}-\d{2}|\d{4}-\d{2}-\d{2})\b", q)
        if tm:
            time_filter = tm.group(1)

        return {
            "intents": intents,
            "commands": candidate_cmds,
            "user": user,
            "time_filter": time_filter
        }

    # -------------------------
    # Retrieval helpers
    # -------------------------
    def _exact_command_search(self, command: str, k: int = 50) -> List[Dict[str, Any]]:
        """Return docs matching a command index (fast)."""
        cmd_key = (command or "").lower()
        if not cmd_key or cmd_key not in self.command_index:
            return []
        results = []
        seen = set()
        for idx in self.command_index[cmd_key]:
            if idx < len(self.sentences_data):
                content, meta = self.sentences_data[idx]
                if not content or content in seen:
                    continue
                seen.add(content)
                results.append(self._result_from_meta(content, meta, base_score=10.0))
                if len(results) >= k:
                    break
        return results

    def _keyword_search(self, query: str, k: int = 100) -> List[Dict[str, Any]]:
        """Keyword-based search using the keyword index and simple substring matches."""
        words = [w.lower() for w in re.findall(r"\b\w+\b", query) if len(w) > 2]
        doc_scores = defaultdict(float)
        for w in words:
            for idx in self.keyword_index.get(w, []):
                doc_scores[idx] += 1.0
        # substring fallback
        for i, (content, meta) in enumerate(self.sentences_data):
            content_l = (content or "").lower()
            for w in words:
                if w in content_l:
                    doc_scores[i] += 0.5
        # rank
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for idx, score in ranked:
            content, meta = self.sentences_data[idx]
            results.append(self._result_from_meta(content, meta, base_score=score))
        return results

    def _semantic_search(self, query: str, k: int = 150) -> List[Dict[str, Any]]:
        """Semantic search via Chroma (if available)."""
        if not self.db:
            return []
        try:
            search_k = min(k, 150)
            docs = self.db.similarity_search(query, k=search_k)
            results = []
            seen = set()
            for doc in docs:
                meta = getattr(doc, "metadata", {}) or {}
                content = meta.get("original_content") or getattr(doc, "page_content", "") or ""
                if not content or content in seen:
                    continue
                seen.add(content)
                results.append(self._result_from_meta(content, meta, base_score=7.0))
            return results
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    # -------------------------
    # Result utilities & reranking
    # -------------------------
    def _result_from_meta(self, content: str, meta: Dict[str, Any], base_score: float = 1.0) -> Dict[str, Any]:
        return {
            "content": content,
            "command": (meta.get("base_command") or meta.get("command") or "").lower(),
            "user": meta.get("user", ""),
            "full_command": meta.get("full_command", ""),
            "date": meta.get("date", ""),
            "time": meta.get("time", ""),
            "target_type": meta.get("target_type", ""),
            "target_path": meta.get("target_path", ""),
            "event_type": meta.get("event_type", ""),
            "auth_details": meta.get("auth_details", ""),
            "score": float(base_score),
            "metadata": meta
        }

    def _local_rerank(self, query: str, candidates: List[Dict[str, Any]], k: int = 10, command_boosts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Deterministic reranker:
          - semantic similarity (if embeddings available)
          - command match boost
          - keyword overlap
          - recency
        """
        q_emb = None
        if self.embedding_model:
            try:
                q_emb = self.embedding_model.embed_query(query)
                q_emb = vector_norm(q_emb)
            except Exception:
                q_emb = None

        scored = []
        q_words = set(re.findall(r"\b\w+\b", normalize_text(query)))
        for c in candidates:
            # semantic sim
            sem = 0.0
            if q_emb is not None:
                try:
                    c_emb = None
                    # If metadata includes embedding, use it
                    meta = c.get("metadata") or {}
                    if meta and meta.get("embedding"):
                        c_emb = vector_norm(meta.get("embedding"))
                    if c_emb is None:
                        # embed snippet
                        c_emb = vector_norm(self.embedding_model.embed_query((c.get("content") or "")[:1024]))
                    if c_emb is not None:
                        sem = cosine_sim(q_emb, c_emb)
                except Exception:
                    sem = 0.0

            # command match
            cmd = (c.get("command") or "").lower()
            cmd_match = 0.0
            if command_boosts:
                for bc in command_boosts:
                    if not bc:
                        continue
                    if cmd == bc or bc in (c.get("full_command") or "").lower() or bc in cmd:
                        cmd_match = 1.0
                        break

            # keyword overlap
            content_words = set(re.findall(r"\b\w+\b", (c.get("content") or "").lower()))
            overlap = len(q_words.intersection(content_words)) / max(1, len(q_words))

            # recency
            recency = 0.0
            dstr = c.get("date", "") or ""
            try:
                if re.match(r"\d{4}-\d{2}-\d{2}", dstr):
                    dt = datetime.fromisoformat(dstr)
                    days = max(0, (now_utc() - dt).days)
                    recency = 1.0 / (1 + days / 7.0)  # decays weekly
            except Exception:
                recency = 0.0

            # combine with weights
            combined = 0.55 * sem + 0.25 * cmd_match + 0.10 * overlap + 0.05 * recency + 0.05 * (c.get("score", 0) / 10.0)
            scored.append((combined, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[:k]]
        return top

    def _ai_rerank(self, query: str, candidates: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        """Optional AI rerank through Groq; strict validation and safe fallback to local order."""
        if not self.groq_available or not self.groq_client or not candidates:
            # fallback: local rerank with command boosts derived from intents
            return self._local_rerank(query, candidates, k=k, command_boosts=None)

        try:
            # Limit to top N to avoid token explosion
            limited = candidates[:min(len(candidates), 10)]
            docs_text = "\n".join([f"{i+1}. { (d['content'][:300] + '...') if len(d['content'])>300 else d['content'] }" for i, d in enumerate(limited)])
            prompt = f"""
You are given a user query and a list of log entries. Rank the entries by relevance to the query (1 = most relevant).
Query: "{query}"

Entries:
{docs_text}

Return ONLY a JSON array of 1-based indices in the new order, e.g. [1,3,2]. Return exactly {min(k, len(limited))} indices. No other text.
"""
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=200
            )
            raw = response.choices[0].message.content.strip()
            # strip fences
            if "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            # attempt to parse
            for attempt in range(3):
                try:
                    arr = json.loads(raw)
                    break
                except Exception:
                    raw = raw.replace("'", '"')
                    raw = raw.replace("None", "null")
            else:
                arr = None

            if not arr or not isinstance(arr, list) or not all(isinstance(i, int) for i in arr):
                logger.warning("AI rerank returned invalid JSON — falling back to local rerank")
                return self._local_rerank(query, candidates, k=k, command_boosts=None)

            reranked = []
            for idx in arr:
                if 1 <= idx <= len(limited):
                    reranked.append(limited[idx - 1])
            # If returned fewer than k, pad with locally reranked entries
            if len(reranked) < k:
                local = self._local_rerank(query, candidates, k=k, command_boosts=None)
                # append those not already included
                for r in local:
                    if r not in reranked:
                        reranked.append(r)
                        if len(reranked) >= k:
                            break
            return reranked[:k]

        except Exception as e:
            logger.exception(f"AI rerank failed: {e}")
            return self._local_rerank(query, candidates, k=k, command_boosts=None)

    # -------------------------
    # Top-level search API
    # -------------------------
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Full search pipeline:
          1. analyze query (multi-intent)
          2. generate candidate commands
          3. gather candidates (exact command index, semantic, keyword)
          4. flexible filtering by user/time (applied after retrieval but before rerank)
          5. dedupe, local rerank, optional AI rerank
        """
        query = (query or "").strip()
        if not query:
            return []

        logger.info(f"🔍 Search started for query: '{query}'")
        structured = self._analyze_query(query)
        intents = structured.get("intents", [])
        candidate_cmds = structured.get("commands", [])
        user_filter = structured.get("user", "")
        time_filter = structured.get("time_filter", "")

        logger.debug(f"Query structured: intents={intents}, commands={candidate_cmds}, user={user_filter}, time={time_filter}")

        candidates = []

        # 1) Exact command index hits (fast, precise)
        for cmd in candidate_cmds:
            hits = self._exact_command_search(cmd, k=500)
            candidates.extend(hits)

        # 2) Semantic search (high recall)
        sem = self._semantic_search(query, k=max(50, k * 5))
        candidates.extend(sem)

        # 3) Keyword search fallback
        kw = self._keyword_search(query, k=max(50, k * 3))
        candidates.extend(kw)

        # 4) If nothing found and there were explicit commands in structured output, try exact on them (defensive)
        if not candidates and structured.get("commands"):
            for cmd in structured.get("commands"):
                candidates.extend(self._exact_command_search(cmd, k=500))

        # 5) Deduplicate by content, preserve highest score
        unique = {}
        for c in candidates:
            content = (c.get("content") or "").strip()
            if not content:
                continue
            prev = unique.get(content)
            if not prev:
                unique[content] = c
            else:
                # keep the one with higher score
                if (c.get("score", 0) or 0) > (prev.get("score", 0) or 0):
                    unique[content] = c
        deduped = list(unique.values())

        logger.info(f"Candidate pool size after dedupe: {len(deduped)}")

        # 6) Flexible filtering — do not over-filter; allow semantic matches
        start_ts, end_ts = parse_time_filter(time_filter)
        filtered = []
        for c in deduped:
            keep = True
            # user filter
            if user_filter:
                if not c.get("user") or user_filter.lower() not in (c.get("user") or "").lower():
                    keep = False
            # time filter
            if (start_ts or end_ts) and keep:
                dstr = c.get("date", "")
                doc_dt = None
                if dstr:
                    try:
                        doc_dt = datetime.fromisoformat(dstr)
                    except Exception:
                        try:
                            doc_dt = datetime.strptime(dstr, "%Y-%m-%d")
                        except Exception:
                            doc_dt = None
                if doc_dt:
                    if start_ts and doc_dt < start_ts:
                        keep = False
                    if end_ts and doc_dt > end_ts:
                        keep = False
            if keep:
                filtered.append(c)

        pool = filtered if filtered else deduped

        # 7) Build command_boosts from top intents / explicit commands for reranker
        command_boosts = list({c.lower() for c in (candidate_cmds or [])})

        # 8) Local rerank and optional AI rerank
        locally_ranked = self._local_rerank(query, pool, k=max(k, 50), command_boosts=command_boosts)
        final_results = self._ai_rerank(query, locally_ranked, k=k)

        logger.info(f"Search complete. Returning {len(final_results)} results.")
        return final_results

    # -------------------------
    # Display + analysis (refactored)
    # -------------------------
    def display_results(self, results: List[Dict[str, Any]], query: str = ""):
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 120)
        if not results:
            print("❌ No results found")
            print("\n💡 Tips:")
            print("- Broaden your keywords.")
            print("- Include a user or a time filter if known.")
            print("- Try 'deleted', 'rm', 'remove' for deletion queries.")
            return

        for i, res in enumerate(results, 1):
            print(f"\n{i}. {res.get('content')}")
            info = []
            if res.get("command"):
                info.append(f"cmd: {res.get('command')}")
            if res.get("user"):
                info.append(f"user: {res.get('user')}")
            d = res.get("date", "")
            t = res.get("time", "")
            if d or t:
                info.append(f"time: {d} {t}".strip())
            if res.get("target_type"):
                info.append(f"target: {res.get('target_type')}")
            if res.get("event_type"):
                info.append(f"type: {res.get('event_type')}")
            if res.get("auth_details"):
                info.append(f"auth: {res.get('auth_details')}")
            if info:
                print(f"   └─ {' | '.join(info)}")
            target_path = res.get("target_path", "")
            if target_path:
                if len(target_path) > 80:
                    print(f"   └─ path: ...{target_path[-77:]}")
                else:
                    print(f"   └─ path: {target_path}")

        print(f"\n📊 Showing {len(results)} results")

    def analyze_results_with_ai(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Context-aware AI analysis if Groq available, else local summary.
        This function focuses on extracting concise findings from results.
        """
        if not results:
            return "❌ No relevant logs found to analyze."

        # Basic local summary
        users = Counter([r.get("user") or "unknown" for r in results])
        cmds = Counter([ (r.get("command") or "unknown") for r in results])
        top_users = users.most_common(5)
        top_cmds = cmds.most_common(8)

        if self.groq_available and self.groq_client:
            # Prepare a compact prompt with up to 30 logs
            logs_text = "\n".join([f"- {r['content']} (user={r.get('user','')}, cmd={r.get('command','')}, date={r.get('date','')})" for r in results[:30]])
            # Choose analysis style by detecting 'security' cues
            query_context = "general"
            if any(w in normalize_text(query) for w in INTENT_KEYWORDS.get("authentication", [])):
                query_context = "security_focused"
            elif any(w in normalize_text(query) for w in INTENT_KEYWORDS.get("delete", [])):
                query_context = "listing_focused"

            if query_context == "security_focused":
                prompt_intro = "You are a security analyst. Provide a concise threat-focused analysis."
            elif query_context == "listing_focused":
                prompt_intro = "You are summarizing a listing of log entries. Provide a concise summary and counts."
            else:
                prompt_intro = "You are analyzing logs. Provide concise summary and key findings."

            prompt = f"""
{prompt_intro}
Query: "{query}"

Retrieved logs:
{logs_text}

Provide:
1) A one-line summary
2) Key findings (commands frequency, users involved)
3) Any notable suspicious indicators (if present)
4) Suggested next steps for investigation (short)

Return plain text only.
"""
            try:
                resp = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.2,
                    max_tokens=500
                )
                analysis = resp.choices[0].message.content.strip()
                return analysis
            except Exception as e:
                logger.warning(f"AI analysis failed: {e}")

        # Fallback local analysis
        lines = [
            f"📋 Local Analysis (AI unavailable):",
            f"  • Retrieved {len(results)} entries.",
            f"  • Top commands: {', '.join([f'{c}({n})' for c, n in top_cmds[:5]])}",
            f"  • Top users: {', '.join([f'{u}({n})' for u, n in top_users[:5]])}",
            "",
            "Notable entries (first 5):"
        ]
        for r in results[:5]:
            lines.append(f"  - {r.get('content')} (user={r.get('user')}, cmd={r.get('command')}, date={r.get('date')})")
        lines.append("")
        lines.append("Recommendations:")
        lines.append("- If you see authentication failures, investigate source IPs and consider blocking.")
        lines.append("- For suspicious deletions, check backups and file metadata; consider restoring from backup.")
        return "\n".join(lines)

    # -------------------------
    # Utility stats
    # -------------------------
    def get_command_statistics(self) -> Dict[str, int]:
        stats = defaultdict(int)
        for _, meta in self.sentences_data:
            cmd = (meta.get("base_command") or meta.get("command") or "").strip()
            if cmd:
                stats[cmd] += 1
        return dict(stats)

    def get_user_statistics(self) -> Dict[str, int]:
        stats = defaultdict(int)
        for _, meta in self.sentences_data:
            u = (meta.get("user") or "").strip()
            if u:
                stats[u] += 1
        return dict(stats)

# -----------------------------
# CLI main (preserves your original UX)
# -----------------------------
def main():
    print("🚀 Initializing AI-Powered Log Query System (refactored)...")
    try:
        store = QueryOnlyLogStore()
    except Exception as e:
        logger.exception("Initialization failed.")
        print(f"❌ Fatal initialization error: {e}")
        return

    print("\n📊 Dataset Statistics:")
    try:
        cmd_stats = store.get_command_statistics()
        user_stats = store.get_user_statistics()
        print(f"   Total log entries: {len(store.sentences_data)}")
        print(f"   Unique commands: {len(cmd_stats)}")
        print(f"   Unique users: {len(user_stats)}")
        if cmd_stats:
            top_commands = sorted(cmd_stats.items(), key=lambda x: x[1], reverse=True)[:6]
            print(f"   Top commands: {', '.join([f'{cmd}({count})' for cmd, count in top_commands])}")
        if user_stats:
            top_users = sorted(user_stats.items(), key=lambda x: x[1], reverse=True)[:6]
            print(f"   Top users: {', '.join([f'{user}({count})' for user, count in top_users])}")
    except Exception:
        logger.exception("Failed to compute stats.")

    print("\n✅ Query system ready!")
    print("\n💡 Example queries:")
    examples = [
        "list all rm commands",
        "show chmod operations by frank",
        "unauthorized login attempts",
        "failed authentication",
        "files deleted by grace",
        "cp commands on directories",
        "user alice file operations",
        "authentication failures",
        "chmod on build.sh",
        "show deleted files",
        "who removed /etc/passwd yesterday"
    ]
    for ex in examples:
        print("  -", ex)

    while True:
        try:
            query = input("\n💭 Enter your search query (or 'quit' to exit): ").strip()
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 Exiting. Goodbye!")
            break
        if query.lower() in ("stats", "statistics"):
            print("\n📊 Current Statistics:")
            print("Commands:", dict(list(store.get_command_statistics().items())[:10]))
            print("Users:", dict(list(store.get_user_statistics().items())[:10]))
            continue

        try:
            results = store.search(query, k=50)
            store.display_results(results, query=query)
            if results:
                print("\n🤖 AI Analysis:")
                print("=" * 80)
                print(store.analyze_results_with_ai(query, results))
            else:
                print("\n💡 Search tips:")
                print("- Broaden keywords, try synonyms (deleted, remove, erase)")
                print("- Add user or time filters, e.g. 'yesterday' or 'by alice'")
        except Exception as e:
            logger.exception("Error during search")
            print(f"❌ Error during search: {e}")

if __name__ == "__main__":
    main()

