# LogInsightAI

AI-powered log analysis tool that lets you query audit and auth logs in natural language. It uses vector embeddings and an optional Groq LLM to search, rerank, and summarise log events.

## What it does

- Parses Linux audit (`SYSCALL`/`PATH`) and SSH auth logs into readable NL sentences
- Ingests them into a local Chroma vector store (sentence-transformers embeddings)
- Accepts natural-language queries and returns the most relevant log entries
- When `GROQ_API_KEY` is set, uses `llama-3.1-8b-instant` for query restructuring, reranking, and AI analysis
- Falls back gracefully to keyword search and local summary when Groq is unavailable
- Streamlit UI with full-width terminal-style log display

## Architecture

```
simulate_logs.sh
    └─ generates raw audit + auth logs
        └─ enterprise_mnc_audit_sim/logs/aggregate/
parse_logs.py
    └─ parses raw logs → data/events_nl.log
log_ingest.py
    └─ reads data/events_nl.log → chroma_db/ (vector store)
query_engine.py   (QueryOnlyLogStore)
    └─ loads chroma_db/, handles all search + AI analysis
app.py
    └─ Streamlit UI, imports QueryOnlyLogStore from query_engine
```

## Run order

```bash
# 1. Generate raw logs (one-time; logs already committed under enterprise_mnc_audit_sim/)
bash simulate_logs.sh

# 2. Parse raw logs into NL sentences
python parse_logs.py   # outputs data/events_nl.log

# 3. Ingest into vector store
python log_ingest.py   # builds chroma_db/

# 4. Launch the UI
streamlit run app.py
```

Both `parse_logs.py` and `log_ingest.py` accept `--help` for path overrides.

## Environment

```bash
export GROQ_API_KEY=gsk_...   # optional; AI features are disabled if unset
```

Without `GROQ_API_KEY` the app still works: queries use keyword + vector search and the analysis panel shows a local text summary.

## Example queries

```
show failed login attempts
find all rm commands by alice
chmod operations by frank
files deleted by grace
unauthorized access attempts
what did bob do on Aug 5
show ssh logins for carol
list commands by dave
heidi file deletions
```

Users in the dataset: alice, bob, carol, dave, eve, frank, grace, heidi, john, sarah

## Known limitations

- **Time filters are extracted but not applied.** The query parser captures dates and relative terms (e.g. "yesterday", "last 3 days") but the retrieval layer does not filter by timestamp.
- **62 hand-injected anomaly lines.** The dataset was seeded with deliberate suspicious events attributed to users trudy, mallory, oscar, and unknown to exercise security queries.
- **Audit SYSCALL and PATH records are joined by timestamp proximity, not by event serial (serial= field).** This can misattribute the path to the wrong syscall when two processes run simultaneously with close timestamps.
