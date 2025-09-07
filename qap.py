#!/usr/bin/env python3
"""
semantic_log_searcher_with_ai.py

A semantic log retrieval system with an added AI analysis layer.
Builds on the retrieval/indices you provided and adds:
 - richer temporal parsing
 - reload() to rebuild indices
 - per-query use_ai flag
 - result analysis: summaries, RCA, recommendations
 - graceful fallback when Groq is not available
"""

__author__ = "Generated for you"
__version__ = "1.1"

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sys
import os
import re
import json
import logging
import traceback
from typing import List, Dict, Any, Set, Tuple, Optional, Union
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import dateutil.parser as date_parser

# External deps (same as your script)
try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    # If these imports fail at runtime, the error will be caught in FixedLogStore init
    pass

try:
    from groq import Groq
except Exception:
    Groq = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHROMA_PATH = "chroma_db"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 20

# ---------------------------
# Helper: Temporal parsing
# ---------------------------
def parse_temporal_ref(text: str) -> Dict[str, Any]:
    """
    Parse temporal references such as:
     - today
     - yesterday
     - last 3 days
     - last week
     - between 2024-09-01 and 2024-09-07
     - 2024-09-01
     - 09/01/2024
     - this month
     - last month
    Returns dict with keys potentially: type, start_date, end_date (iso strings).
    """
    if not text:
        return {}

    tl = text.lower()

    now = datetime.now()
    def iso(d): return d.strftime("%Y-%m-%d")

    # direct tokens
    if "today" in tl:
        return {"type": "today", "start_date": iso(now), "end_date": iso(now)}
    if "yesterday" in tl:
        d = now - timedelta(days=1)
        return {"type": "yesterday", "start_date": iso(d), "end_date": iso(d)}
    if "this week" in tl:
        start = (now - timedelta(days=now.weekday()))
        return {"type": "this_week", "start_date": iso(start), "end_date": iso(now)}
    if "last week" in tl:
        start = (now - timedelta(days=now.weekday() + 7))
        end = start + timedelta(days=6)
        return {"type": "last_week", "start_date": iso(start), "end_date": iso(end)}
    if "this month" in tl:
        start = now.replace(day=1)
        return {"type": "this_month", "start_date": iso(start), "end_date": iso(now)}
    if "last month" in tl:
        first = now.replace(day=1)
        last_month_end = first - timedelta(days=1)
        start = last_month_end.replace(day=1)
        return {"type": "last_month", "start_date": iso(start), "end_date": iso(last_month_end)}

    # last N days
    m = re.search(r'last\s+(\d+)\s+days?', tl)
    if m:
        n = int(m.group(1))
        start = now - timedelta(days=n)
        return {"type": f"last_{n}_days", "start_date": iso(start), "end_date": iso(now)}

    # between DATE and DATE
    m = re.search(r'between\s+([0-9/-]+)\s+(?:and|-)\s+([0-9/-]+)', tl)
    if m:
        try:
            d1 = date_parser.parse(m.group(1)).date()
            d2 = date_parser.parse(m.group(2)).date()
            if d1 > d2:
                d1, d2 = d2, d1
            return {"type": "range", "start_date": d1.strftime("%Y-%m-%d"), "end_date": d2.strftime("%Y-%m-%d")}
        except Exception:
            pass

    # absolute date yyyy-mm-dd or mm/dd/yyyy
    m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if m:
        return {"type": "absolute", "start_date": m.group(1), "end_date": m.group(1)}
    m = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
    if m:
        try:
            d = date_parser.parse(m.group(1)).date()
            return {"type": "absolute", "start_date": d.strftime("%Y-%m-%d"), "end_date": d.strftime("%Y-%m-%d")}
        except Exception:
            pass

    return {}

# ---------------------------
# SemanticQueryProcessor (extended)
# ---------------------------
class QueryOnlyLogStore:
    """Handles semantic understanding and query transformation with richer temporal parsing"""

    def __init__(self, groq_client=None):
        self.groq_client = groq_client
        self.groq_available = groq_client is not None

        # Strong semantic mappings - NO ambiguity (as you had)
        self.intent_command_mapping = {
            # File operations
            "list_deleted_files": "rm",
            "show_deleted_files": "rm",
            "files_deleted": "rm",
            "deletions": "rm",
            "removed_files": "rm",

            "list_copied_files": "cp",
            "show_copied_files": "cp",
            "files_copied": "cp",
            "copy_operations": "cp",

            "list_moved_files": "mv",
            "show_moved_files": "mv",
            "files_moved": "mv",
            "move_operations": "mv",
            "renamed_files": "mv",

            "permission_changes": "chmod",
            "chmod_operations": "chmod",
            "access_changes": "chmod",

            # Authentication events (NOT commands)
            "failed_logins": "AUTH_FAILED",
            "login_failures": "AUTH_FAILED",
            "unauthorized_access": "AUTH_FAILED",
            "failed_authentication": "AUTH_FAILED",

            "successful_logins": "AUTH_SUCCESS",
            "login_success": "AUTH_SUCCESS",

            # System operations
            "process_kills": "kill",
            "terminated_processes": "kill",
            "service_operations": "systemctl",
        }

        # Query pattern recognition (expanded synonyms)
        self.query_patterns = [
            (r'\b(?:list|show|display|find)\s+(?:all\s+)?(?:files?\s+)?(?:that\s+were\s+)?(?:deleted|removed|erased|deleted\s+by)', "list_deleted_files"),
            (r'\b(?:list|show|display|find)\s+(?:all\s+)?(?:files?\s+)?(?:that\s+were\s+)?(?:copied|duplicated)', "list_copied_files"),
            (r'\b(?:list|show|display|find)\s+(?:all\s+)?(?:files?\s+)?(?:that\s+were\s+)?(?:moved|renamed|relocated)', "list_moved_files"),
            (r'\b(?:list|show|display|find)\s+(?:all\s+)?(?:chmod|permission|access)', "permission_changes"),
            (r'\bfailed\s+(?:login|authentication|auth|ssh|password)', "failed_logins"),
            (r'\bunauthorized\s+(?:access|login|connection)', "failed_logins"),
            (r'\blogin\s+(?:failure|fail|failed)', "failed_logins"),
            (r'\bsuccessful\s+(?:login|authentication|authenticated)', "successful_logins"),
            (r'\b(?:kill|terminate|killed)\b.*process', "process_kills"),
            (r'\bservice\s+(?:start|stop|restart)\b', "service_operations"),
        ]

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Primary query analysis with strong semantic understanding"""
        query_lower = query.lower().strip()

        # Step 1: Pattern matching for common queries
        for pattern, intent in self.query_patterns:
            if re.search(pattern, query_lower):
                return self._build_query_structure(intent, query)

        # Step 2: AI analysis if pattern matching fails and Groq available
        if self.groq_available:
            ai_result = self._ai_analyze_query(query)
            if ai_result and ai_result.get("confidence", 0) > 0.7:
                ai_result["original_query"] = query
                ai_result["temporal"] = parse_temporal_ref(query)
                return ai_result

        # Step 3: Fallback semantic analysis
        return self._fallback_analysis(query)

    def _build_query_structure(self, semantic_intent: str, original_query: str) -> Dict[str, Any]:
        """Build standardized query structure from semantic intent"""
        target_command = self.intent_command_mapping.get(semantic_intent)
        entities = self._extract_entities(original_query)
        temporal = parse_temporal_ref(original_query)

        if target_command and target_command.startswith("AUTH_"):
            search_type = "authentication_events"
            auth_type = "failed" if "FAILED" in target_command else "successful"
            target_command = None
        else:
            search_type = "command_logs"
            auth_type = None

        return {
            "semantic_intent": semantic_intent,
            "target_command": target_command,
            "search_type": search_type,
            "auth_type": auth_type,
            "entities": entities,
            "temporal": temporal,
            "confidence": 0.95,
            "original_query": original_query
        }

    def _ai_analyze_query(self, query: str) -> Optional[Dict[str, Any]]:
        """AI-powered query analysis with strict output format via Groq"""
        try:
            prompt = f"""
Analyze this log search query with STRICT semantic understanding and return JSON:
Query: "{query}"
Return JSON keys:
semantic_intent, target_command (or null), search_type, auth_type (failed|successful|null), entities, confidence (0-1)

Possible intents: list_deleted_files,list_copied_files,list_moved_files,permission_changes,failed_logins,successful_logins,process_kills,service_operations,general
"""
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300
            )
            content = response.choices[0].message.content.strip()
            # Strip code fences if any
            if "```" in content:
                content = content.split("```")[-1].strip()
            result = json.loads(content)
            # add temporal extraction
            result["temporal"] = parse_temporal_ref(query)
            result["original_query"] = query
            return result
        except Exception as e:
            logger.warning(f"AI query analysis failed: {e}")
            return None

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract users, files, and other entities (expanded)"""
        entities = {"users": [], "files": [], "time_refs": []}

        # User patterns (added more cover)
        user_patterns = [
            r'\bby\s+([a-zA-Z0-9._-]+)\b',
            r'\buser:?\s*([a-zA-Z0-9._-]+)\b',
            r'\b([a-zA-Z0-9._-]+)\s+(?:executed|performed|did)\b',
            r'\bfrom\s+([a-zA-Z0-9._-]+)\b',
            r'\bfor\s+user\s+([a-zA-Z0-9._-]+)\b'
        ]

        for pattern in user_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["users"].extend(matches)

        # File patterns
        file_patterns = [
            r'\b([a-zA-Z0-9._/\-]+\.[a-zA-Z0-9]+)\b',  # filenames with ext
            r'"([^"]+)"',
            r"'([^']+)'",
            r'(/[^ \t\n\r]+)'  # absolute paths
        ]

        for pattern in file_patterns:
            matches = re.findall(pattern, query)
            entities["files"].extend(matches)

        # time refs (simple)
        if any(tok in query.lower() for tok in ["today", "yesterday", "last", "between", "this month", "last month", "days"]):
            entities["time_refs"].append(query)

        # Deduplicate
        entities["users"] = list(dict.fromkeys(entities["users"]))
        entities["files"] = list(dict.fromkeys(entities["files"]))
        return entities

    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis using keyword detection"""
        ql = query.lower()
        if any(word in ql for word in ["delete", "deleted", "remove", "removed", "rm", "erased"]):
            return self._build_query_structure("list_deleted_files", query)
        if any(word in ql for word in ["copy", "copied", "cp", "duplicate"]):
            return self._build_query_structure("list_copied_files", query)
        if any(word in ql for word in ["move", "moved", "rename", "mv", "moved to"]):
            return self._build_query_structure("list_moved_files", query)
        if any(word in ql for word in ["chmod", "permission", "access", "chown"]):
            return self._build_query_structure("permission_changes", query)
        if "failed" in ql and any(word in ql for word in ["login", "auth", "ssh", "password"]):
            return self._build_query_structure("failed_logins", query)

        return {
            "semantic_intent": "general",
            "target_command": None,
            "search_type": "general",
            "auth_type": None,
            "entities": self._extract_entities(query),
            "temporal": parse_temporal_ref(query),
            "confidence": 0.3,
            "original_query": query
        }

# ---------------------------
# SmartLogSearcher (unchanged core) but with small enhacements
# ---------------------------
class SmartLogSearcher:
    """Handles the actual log searching with semantic understanding"""

    def __init__(self, log_store):
        self.store = log_store

    def semantic_search(self, query_analysis: Dict, k: int = DEFAULT_K) -> List[Dict[str, Any]]:
        """Execute search based on semantic analysis"""
        search_type = query_analysis.get("search_type", "general")
        logger.info(f"Executing {search_type} search for intent: {query_analysis.get('semantic_intent')}")

        if search_type == "command_logs":
            return self._search_command_logs(query_analysis, k)
        elif search_type == "authentication_events":
            return self._search_auth_events(query_analysis, k)
        else:
            return self._general_search(query_analysis, k)

    def _search_command_logs(self, query_analysis: Dict, k: int) -> List[Dict[str, Any]]:
        target_command = query_analysis.get("target_command")
        if not target_command:
            logger.info("No explicit target command; performing general search")
            return self._general_search(query_analysis, k)

        if target_command.lower() not in self.store.command_index:
            logger.warning(f"Command {target_command} not found in index")
            return []

        results = []
        cmd_lower = target_command.lower()
        logger.info(f"Searching command index for: {cmd_lower}")

        for idx in self.store.command_index[cmd_lower]:
            if len(results) >= k:
                break
            if idx < len(self.store.sentences_data):
                sentence, metadata = self.store.sentences_data[idx]
                actual_cmd = metadata.get("base_command", "").lower()
                if actual_cmd == cmd_lower:
                    results.append({
                        "content": sentence,
                        "command": metadata.get("base_command", ""),
                        "user": metadata.get("user", ""),
                        "full_command": metadata.get("full_command", ""),
                        "date": metadata.get("date", ""),
                        "time": metadata.get("time", ""),
                        "target_type": metadata.get("target_type", ""),
                        "target_path": metadata.get("target_path", ""),
                        "event_type": metadata.get("event_type", ""),
                        "search_method": f"command_exact_{cmd_lower}",
                        "semantic_match": True,
                        "metadata": metadata
                    })

        results = self._apply_entity_filters(results, query_analysis)
        results = self._apply_temporal_filters(results, query_analysis)
        logger.info(f"Found {len(results)} {target_command} command logs")
        return results

    def _search_auth_events(self, query_analysis: Dict, k: int) -> List[Dict[str, Any]]:
        auth_type = query_analysis.get("auth_type", "failed")
        results = []
        if auth_type == "failed":
            auth_keywords = ["failed", "denied", "invalid", "unauthorized", "rejected", "authentication failure", "failure"]
            exclude_keywords = ["accepted", "successful", "authenticated", "login successful"]
        else:
            auth_keywords = ["accepted", "successful", "authenticated", "logged in"]
            exclude_keywords = ["failed", "denied", "rejected"]

        logger.info(f"Searching for {auth_type} authentication events")
        for idx, (sentence, metadata) in enumerate(self.store.sentences_data):
            if len(results) >= k:
                break
            sentence_lower = sentence.lower()
            if not any(auth_term in sentence_lower for auth_term in ["auth", "login", "ssh", "password", "session", "connection"]):
                continue
            has_target_keywords = any(keyword in sentence_lower for keyword in auth_keywords)
            has_exclude_keywords = any(keyword in sentence_lower for keyword in exclude_keywords)
            if has_target_keywords and not has_exclude_keywords:
                results.append({
                    "content": sentence,
                    "command": metadata.get("base_command", ""),
                    "user": metadata.get("user", ""),
                    "full_command": metadata.get("full_command", ""),
                    "date": metadata.get("date", ""),
                    "time": metadata.get("time", ""),
                    "target_type": metadata.get("target_type", ""),
                    "target_path": metadata.get("target_path", ""),
                    "event_type": metadata.get("event_type", ""),
                    "auth_details": metadata.get("auth_details", ""),
                    "search_method": f"auth_events_{auth_type}",
                    "semantic_match": True,
                    "metadata": metadata
                })

        results = self._apply_entity_filters(results, query_analysis)
        results = self._apply_temporal_filters(results, query_analysis)
        logger.info(f"Found {len(results)} {auth_type} authentication events")
        return results

    def _general_search(self, query_analysis: Dict, k: int) -> List[Dict[str, Any]]:
        original_query = query_analysis.get("original_query", "")
        try:
            if self.store.db:
                semantic_results = self.store.db.similarity_search(original_query, k=k*2)
                results = []
                for doc in semantic_results:
                    meta = doc.metadata
                    content = meta.get("original_content", doc.page_content)
                    results.append({
                        "content": content,
                        "command": meta.get("base_command", ""),
                        "user": meta.get("user", ""),
                        "full_command": meta.get("full_command", ""),
                        "date": meta.get("date", ""),
                        "time": meta.get("time", ""),
                        "target_type": meta.get("target_type", ""),
                        "target_path": meta.get("target_path", ""),
                        "event_type": meta.get("event_type", ""),
                        "search_method": "semantic_vector",
                        "semantic_match": False,
                        "metadata": meta
                    })
                return results[:k]
            else:
                logger.info("No vector DB available; performing text-scan fallback")
                # naive text-scan fallback: filter sentences_data by original_query tokens
                tokens = [t for t in re.findall(r'\w+', original_query.lower()) if len(t) > 2]
                results = []
                for idx, (sentence, metadata) in enumerate(self.store.sentences_data):
                    if len(results) >= k:
                        break
                    s_lower = sentence.lower()
                    if all(tok in s_lower for tok in tokens):
                        results.append({
                            "content": sentence,
                            "command": metadata.get("base_command", ""),
                            "user": metadata.get("user", ""),
                            "full_command": metadata.get("full_command", ""),
                            "date": metadata.get("date", ""),
                            "time": metadata.get("time", ""),
                            "target_type": metadata.get("target_type", ""),
                            "target_path": metadata.get("target_path", ""),
                            "event_type": metadata.get("event_type", ""),
                            "search_method": "text_scan_fallback",
                            "semantic_match": False,
                            "metadata": metadata
                        })
                return results
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    def _apply_entity_filters(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        entities = query_analysis.get("entities", {})
        users = entities.get("users", [])
        files = entities.get("files", [])

        if not users and not files:
            return results

        filtered_results = []
        for result in results:
            include = True
            if users:
                # Accept if result user matches any of requested users (case-insensitive)
                result_user = (result.get("user") or "").lower()
                if not any(u.lower() == result_user for u in users):
                    include = False
            if files and include:
                content_lower = (result.get("content") or "").lower()
                target_path = (result.get("target_path") or "").lower()
                file_match = any(f.lower() in content_lower or f.lower() in target_path for f in files)
                if not file_match:
                    include = False
            if include:
                filtered_results.append(result)
        return filtered_results

    def _apply_temporal_filters(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        temporal = query_analysis.get("temporal", {})
        if not temporal:
            return results

        start = temporal.get("start_date")
        end = temporal.get("end_date", start)
        if not start:
            return results

        filtered = []
        try:
            start_dt = date_parser.parse(start).date()
            end_dt = date_parser.parse(end).date()
        except Exception:
            return results

        for res in results:
            res_date_str = res.get("date", "")
            if not res_date_str:
                # include un-dated items (conservative)
                filtered.append(res)
                continue
            try:
                rd = date_parser.parse(res_date_str).date()
                if start_dt <= rd <= end_dt:
                    filtered.append(res)
            except Exception:
                filtered.append(res)

        # If filtering removed everything, return original results (avoid overly strict)
        return filtered if filtered else results

# ---------------------------
# AI Analysis Layer
# ---------------------------
class ResultAnalyzer:
    """
    Produces:
      - Summaries (counts, top users, top files)
      - RCA-style analysis and recommendations for security events
      - Uses Groq if available; else deterministic heuristics
    """
    def __init__(self, groq_client=None):
        self.groq_client = groq_client
        self.groq_available = groq_client is not None

    def analyze(self, query_analysis: Dict, results: List[Dict], use_ai: bool = True) -> Dict[str, Any]:
        """
        Returns a dict:
          {
            "summary": {...},
            "insights": "natural language",
            "rca": {...},  # optional structured RCA
            "recommendations": [...]
          }
        """
        # Quick stats
        summary = self._build_summary(results)
        intent = query_analysis.get("semantic_intent", "general")

        # If Groq present and user requested AI, offload to Groq for richer text
        if use_ai and self.groq_available:
            try:
                prompt = self._build_groq_prompt(query_analysis, summary, results)
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=400
                )
                content = response.choices[0].message.content.strip()
                # Try to parse JSON block if the model returned structure
                parsed = {}
                try:
                    # extract json if present
                    if "```json" in content:
                        json_text = content.split("```json")[1].split("```")[0].strip()
                        parsed = json.loads(json_text)
                    else:
                        # best-effort: look for a JSON object in the response
                        m = re.search(r'(\{.*\})', content, flags=re.DOTALL)
                        if m:
                            parsed = json.loads(m.group(1))
                        else:
                            parsed = {"insights": content}
                except Exception:
                    parsed = {"insights": content}
                # merge
                output = {"summary": summary}
                output.update(parsed)
                # ensure recommendations key exists
                if "recommendations" not in output:
                    output["recommendations"] = self._default_recommendations(intent, summary, results)
                return output
            except Exception as e:
                logger.warning(f"Groq analysis failed: {e}")
                # fall through to heuristic analysis

        # Heuristic analysis
        rca = {}
        insights = ""
        recommendations = []

        # General listing insights
        if intent in ("list_deleted_files", "list_copied_files", "list_moved_files", "permission_changes"):
            insights = f"Found {summary['total_matches']} events matching intent '{intent}'. Top users: {', '.join(summary['top_users'][:5])}."
            # suspicious pattern detection
            if intent == "list_deleted_files" and summary['top_files_by_count']:
                top_files = summary['top_files_by_count'][:5]
                insights += f" Most affected files/paths: {', '.join([f'{f[0]}({f[1]})' for f in top_files])}."
                if summary['total_matches'] > 10 and len(summary['unique_users']) <= 2:
                    insights += " High volume deletions by few users — suspicious activity possible."
                    rca['suspicion'] = "mass_deletion_by_few_users"
                    recommendations.append("Investigate the users who executed deletions; check for scheduled scripts or compromised credentials.")
            if intent == "permission_changes":
                insights += " Review recent permission escalation changes carefully."

        elif intent in ("failed_logins", "login_failures", "unauthorized_access"):
            insights = f"Found {summary['total_matches']} authentication failure events. Top usernames targeted: {', '.join(summary['top_users'][:5])}."
            # detect brute force-like pattern
            user_counts = summary['user_counts']
            brute_force_candidates = [u for u, c in user_counts.items() if c >= 5]
            if brute_force_candidates:
                insights += f" Possible brute-force on users: {', '.join(brute_force_candidates)}."
                rca['suspicion'] = "possible_bruteforce_attempts"
                recommendations.append("Block or rate-limit the offending IPs; review logs for IP addresses and implement multi-factor authentication for targeted accounts.")
            # suggest immediate containment
            recommendations.append("If suspicious, rotate credentials, check for lateral movement, and increase monitoring on authentication endpoints.")

        else:
            insights = f"General search returned {summary['total_matches']} items. Top users: {', '.join(summary['top_users'][:5])}."
            recommendations.append("If you need an RCA or remediation, re-run query with a more specific intent (e.g., 'failed logins yesterday' or 'list files deleted by alice').")

        # Default recommendations if none were generated
        if not recommendations:
            recommendations = self._default_recommendations(intent, summary, results)

        return {
            "summary": summary,
            "insights": insights,
            "rca": rca,
            "recommendations": recommendations
        }

    def _build_summary(self, results: List[Dict]) -> Dict[str, Any]:
        total = len(results)
        users = [r.get("user") or "unknown" for r in results]
        user_counts = Counter(u for u in users if u)
        top_users = [u for u, _ in user_counts.most_common()]
        files = []
        for r in results:
            p = r.get("target_path") or ""
            if p:
                files.append(p)
            # also try to extract filenames from content
            files_in_content = re.findall(r'([/\w\.-]+\.[a-zA-Z0-9]{1,6})', r.get("content") or "")
            files.extend(files_in_content)
        file_counts = Counter(files)
        top_files = file_counts.most_common()
        unique_users = list(user_counts.keys())

        return {
            "total_matches": total,
            "unique_users": unique_users,
            "user_counts": dict(user_counts),
            "top_users": top_users,
            "top_files_by_count": top_files,
        }

    def _default_recommendations(self, intent: str, summary: Dict[str, Any], results: List[Dict]) -> List[str]:
        recs = []
        if intent in ("failed_logins", "login_failures"):
            recs.extend([
                "Identify source IPs for repeated failures and block/limit them.",
                "Enable MFA and reset passwords for accounts showing repeated failures.",
                "Monitor for successful logins following failures."
            ])
        if intent in ("list_deleted_files",):
            recs.extend([
                "Check backups and retention policies for affected files.",
                "Investigate whether deletions were manual or via script (check timestamps & processes).",
                "If malicious, isolate user accounts and rotate credentials."
            ])
        if not recs:
            recs.append("Review the search results manually and run targeted queries for deeper RCA (e.g., include IPs, process names).")
        return recs

    def _build_groq_prompt(self, query_analysis: Dict, summary: Dict, results: List[Dict]) -> str:
        """Create a prompt to ask Groq to analyze results and return JSON with keys: insights, rca, recommendations"""
        example = json.dumps({
            "insights": "Short English insights summary",
            "rca": {"root_cause": "string", "evidence": ["..."]},
            "recommendations": ["..."]
        }, indent=2)
        sample_context = {
            "query_analysis": query_analysis,
            "summary": summary,
            "top_results_sample": [r.get("content") for r in results[:5]]
        }
        prompt = f"""
You are given a log search analysis task. Use the context below to produce a JSON object with keys:
  - insights: a short textual summary (1-3 sentences)
  - rca: structured root cause analysis with keys: root_cause, evidence (list)
  - recommendations: list of recommended remediation/next steps

Context:
{json.dumps(sample_context, indent=2)}

Return ONLY valid JSON. Example:
{example}
"""
        return prompt

# ---------------------------
# FixedLogStore (main)
# ---------------------------
class FixedLogStore:
    """Main log storage system with fixed semantic search and AI analysis"""

    def __init__(self):
        self.db = None
        self.embedding_model = None
        self.sentences_data: List[Tuple[str, Dict[str, Any]]] = []
        self.command_index: Dict[str, List[int]] = defaultdict(list)
        self.user_index: Dict[str, List[int]] = defaultdict(list)
        self.date_index: Dict[str, List[int]] = defaultdict(list)
        self.groq_client = None
        self.groq_available = False

        self.query_processor: Optional[SemanticQueryProcessor] = None
        self.searcher: Optional[SmartLogSearcher] = None
        self.analyzer: Optional[ResultAnalyzer] = None

        self._init_components()

    def _init_components(self):
        """Initialize all components"""
        # embedding model
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
            logger.info("Embedding model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None

        # Groq client
        try:
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key and Groq is not None:
                self.groq_client = Groq(api_key=api_key)
                self.groq_available = True
                logger.info("Groq client initialized")
            else:
                self.groq_available = False
                logger.info("Groq not available (GROQ_API_KEY missing or SDK missing)")
        except Exception as e:
            logger.error(f"Groq initialization failed: {e}")
            self.groq_available = False

        # processors
        self.query_processor = SemanticQueryProcessor(self.groq_client)
        self.searcher = SmartLogSearcher(self)
        self.analyzer = ResultAnalyzer(self.groq_client)

        # load database
        self._load_database()

    def _load_database(self):
        """Load Chroma database and build indices"""
        try:
            if not os.path.exists(CHROMA_PATH):
                raise FileNotFoundError(f"Chroma database not found at {CHROMA_PATH}")

            # try to load Chroma only if embedding model exists
            if self.embedding_model is None:
                logger.warning("Embedding model not loaded; vector DB will be attempted but may fail.")

            self.db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=self.embedding_model
            )
            all_docs = self.db.get()
            logger.info(f"Loaded {len(all_docs.get('ids', []))} documents from Chroma")
            self._build_indices(all_docs)
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            # allow operation to continue with empty in-memory store
            self.db = None
            self.sentences_data = []
            self.command_index = defaultdict(list)
            self.user_index = defaultdict(list)
            self.date_index = defaultdict(list)
            # re-raise if desired, but keep running
            # raise

    def _build_indices(self, all_docs: Dict[str, Any]):
        """Build search indices"""
        self.sentences_data = []
        ids = all_docs.get('ids', [])
        metadatas = all_docs.get('metadatas', [])
        for i, (doc_id, metadata) in enumerate(zip(ids, metadatas)):
            original_content = metadata.get('original_content', metadata.get("content", ""))
            self.sentences_data.append((original_content, metadata))
            base_cmd = (metadata.get("base_command") or "").lower().strip()
            if base_cmd:
                self.command_index[base_cmd].append(i)
            user = (metadata.get("user") or "").lower().strip()
            if user:
                self.user_index[user].append(i)
            date = (metadata.get("date") or "").strip()
            if date:
                self.date_index[date].append(i)
        logger.info(f"Built indices: {len(self.command_index)} commands, {len(self.user_index)} users")

    def reload(self):
        """Reload the database and rebuild indices (useful when Chroma updates)"""
        logger.info("Reloading database and rebuilding indices...")
        self._load_database()
        logger.info("Reload complete.")

    def search(self, query: str, k: int = DEFAULT_K, use_ai: bool = True) -> Dict[str, Any]:
        """
        Main search function with strong semantic understanding and integrated AI analysis.
        Returns:
          {
            "results": [...],
            "analysis": {...}
          }
        """
        if not query or not query.strip():
            return {"results": [], "analysis": {"summary": {}, "insights": "Empty query"}}

        query = query.strip()
        logger.info(f"=== PROCESSING QUERY: '{query}' ===")
        try:
            query_analysis = self.query_processor.analyze_query(query)
            logger.info(f"Query Analysis: {query_analysis}")

            # Execute semantic search
            results = self.searcher.semantic_search(query_analysis, k)

            # Validate results
            validated = self._validate_results(results, query_analysis)
            logger.info(f"Returning {len(validated)} validated results")

            # Analyze results with AI or heuristics
            analysis = self.analyzer.analyze(query_analysis, validated, use_ai=use_ai and self.groq_available)

            return {"results": validated, "analysis": analysis, "query_analysis": query_analysis}
        except Exception as e:
            logger.error(f"Search failed: {e}")
            logger.error(traceback.format_exc())
            return {"results": [], "analysis": {"error": str(e)}}

    def _validate_results(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        semantic_intent = query_analysis.get("semantic_intent", "")
        target_command = query_analysis.get("target_command")

        if not semantic_intent or semantic_intent == "general":
            return results

        validated = []
        for result in results:
            valid = True
            cmd = (result.get("command") or "").lower()
            content_lower = (result.get("content") or "").lower()

            if semantic_intent == "list_deleted_files":
                if cmd != "rm" and "rm " not in content_lower:
                    valid = False
            elif semantic_intent == "list_copied_files":
                if cmd != "cp" and "cp " not in content_lower:
                    valid = False
            elif semantic_intent == "list_moved_files":
                if cmd != "mv" and "mv " not in content_lower:
                    valid = False
            elif semantic_intent == "permission_changes":
                if cmd != "chmod" and "chmod " not in content_lower:
                    valid = False
            elif semantic_intent == "failed_logins":
                if not ("failed" in content_lower and any(t in content_lower for t in ["auth", "login", "ssh", "password", "session"])):
                    valid = False

            if valid:
                validated.append(result)
        return validated

    def display_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], query: str = ""):
        """Pretty print results and analysis to console"""
        print(f"\n=== SEARCH RESULTS FOR: '{query}' ===")
        print("=" * 80)
        if not results:
            print("No results found.")
            return
        for i, res in enumerate(results, 1):
            print(f"\n{i}. {res.get('content')}")
            info_parts = []
            if res.get("command"):
                info_parts.append(f"cmd: {res.get('command')}")
            if res.get("user"):
                info_parts.append(f"user: {res.get('user')}")
            if res.get("date"):
                info_parts.append(f"date: {res.get('date')}")
            if res.get("search_method"):
                info_parts.append(f"found_via: {res.get('search_method')}")
            if res.get("semantic_match"):
                info_parts.append("semantic_match: ✓")
            if info_parts:
                print(f"   └─ {' | '.join(info_parts)}")
            if res.get("target_path"):
                print(f"   └─ path: {res.get('target_path')}")
        # Analysis
        print("\n--- ANALYSIS ---")
        if analysis:
            summary = analysis.get("summary", {})
            print(f"Total matches: {summary.get('total_matches', 0)}")
            print(f"Top users: {', '.join(summary.get('top_users', [])[:5])}")
            print(f"Top files/paths: {', '.join([f[0] for f in summary.get('top_files_by_count', [])[:5]])}")
            print("\nInsights:")
            print(analysis.get("insights", "No insights"))
            if analysis.get("rca"):
                print("\nRCA:")
                print(json.dumps(analysis.get("rca"), indent=2))
            if analysis.get("recommendations"):
                print("\nRecommendations:")
                for rec in analysis.get("recommendations", []):
                    print(f"- {rec}")
        print(f"\nShowing {len(results)} results")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self.sentences_data),
            "commands": dict(list(self.command_index.items())[:10]),
            "users": dict(list(self.user_index.items())[:5]),
            "groq_available": self.groq_available
        }

# ---------------------------
# CLI main
# ---------------------------
def main():
    try:
        print("=== FIXED LOG QUERY SYSTEM WITH AI ANALYSIS ===")
        store = FixedLogStore()
        stats = store.get_stats()
        print("\nSystem Ready:")
        print(f"- Total entries: {stats['total_entries']}")
        print(f"- Commands indexed: {len(store.command_index)}")
        print(f"- Users indexed: {len(store.user_index)}")
        print(f"- AI (Groq) available: {stats['groq_available']}")
        print("\nExamples:")
        print(" - 'list all files deleted yesterday'")
        print(" - 'show files copied by alice last 7 days'")
        print(" - 'failed login attempts between 2025-08-01 and 2025-08-31'")
        print(" - 'chmod operations yesterday'")

        while True:
            query = input("\nEnter semantic query (or 'quit', 'reload', 'stats'): ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if query.lower() == "reload":
                store.reload()
                continue
            if query.lower() == "stats":
                print(json.dumps(store.get_stats(), indent=2))
                continue

            # parse optional flags like --no-ai or --k=10
            use_ai = True
            k = DEFAULT_K
            # simple inline flags support: append " --no-ai" or " --k=5"
            if "--no-ai" in query:
                use_ai = False
                query = query.replace("--no-ai", "").strip()
            m = re.search(r'--k=(\d+)', query)
            if m:
                k = int(m.group(1))
                query = re.sub(r'--k=\d+', '', query).strip()

            resp = store.search(query, k=k, use_ai=use_ai)
            results = resp.get("results", [])
            analysis = resp.get("analysis", {})
            store.display_results(results, analysis, query)

    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")
    except Exception as e:
        print(f"System initialization failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

