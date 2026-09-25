# LogInsightAI 🔍

An AI-powered log analysis tool that lets you search and query enterprise audit logs using natural language.

## What it does

- **Natural language search** — ask questions like *"Show failed login attempts"* or *"List files deleted by dev02"*
- **Semantic + keyword search** over a ChromaDB vector store
- **AI query understanding** — extracts intent, commands, users, and time filters via Groq LLM
- **AI reranking** — results ranked by relevance using Groq
- **AI analysis** — auto-generates root cause analysis and recommendations
- **Log simulation** — shell scripts (`logsim.sh`, `mnc_log_sim.sh`) to generate synthetic enterprise logs for testing



## Example Queries

```
Show me all failed login attempts
List files deleted by user dev02
Find all chmod operations last week
Who accessed the finance directory?
```
