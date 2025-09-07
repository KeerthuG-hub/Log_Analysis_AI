__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import re
import json
import logging
from typing import List, Dict, Any, Set, Tuple, Optional, Union
from collections import defaultdict
import traceback
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
CHROMA_PATH = "chroma_db"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Enhanced command descriptions with synonyms
COMMAND_DESCRIPTIONS = {
    "rm": "delete remove file unlink erase destroy eliminate purge",
    "cp": "copy duplicate file replicate backup clone",
    "mv": "move rename relocate transfer shift change name",
    "chmod": "change file permissions security access rights modify protection",
    "touch": "create new file timestamp update modify time",
    "mkdir": "create directory folder make dir new path",
    "tar": "archive compress backup pack bundle extract unpack",
    "systemctl": "service management system control start stop restart enable disable",
    "bash": "shell execution script command run execute",
    "python3": "python script execution programming run code",
    "grep": "search text pattern file find match filter",
    "sed": "edit modify text stream replace substitute change",
    "ufw": "firewall configure network security block allow deny",
    "auditctl": "audit monitoring logging track record watch",
    "hostnamectl": "system identity configuration hostname set change",
    "dd": "copy disk data file convert transfer write",
    "head": "view read file content display show first lines",
    "tail": "view read file content display show last lines",
    "cat": "view read file content display show concatenate",
    "ls": "list directory files show contents display",
    "find": "search locate file directory path discover",
    "which": "locate command binary path find executable",
    "ps": "process list show running tasks display",
    "kill": "terminate process stop end destroy",
    "df": "disk space usage filesystem capacity free",
    "du": "directory disk usage size space consumption",
    "mount": "filesystem mount attach connect link",
    "umount": "filesystem unmount detach disconnect unlink",
    "chown": "change owner ownership user group modify",
    "ln": "link create symbolic hard connection reference",
    "wget": "download fetch retrieve get file http",
    "curl": "download fetch retrieve get file http request",
    "ssh": "secure shell remote connection access login",
    "scp": "secure copy transfer file remote ssh",
    "rsync": "synchronize copy backup transfer mirror",
    "crontab": "schedule task timer automated job",
    "service": "system service control start stop restart",
    "truncate": "empty file clear reduce size shrink zero",
    "history": "command history view past commands executed",
    "netstat": "network connections statistics ports listening",
    "free": "memory usage display available ram system"
}

# Intent keywords mapping
INTENT_KEYWORDS = {
    "delete": ["delete", "remove", "rm", "erase", "destroy", "unlink"],
    "copy": ["copy", "cp", "duplicate", "backup", "clone"],
    "move": ["move", "mv", "rename", "relocate", "transfer"],
    "view": ["view", "show", "display", "cat", "head", "tail", "ls", "list"],
    "search": ["search", "find", "grep", "locate"],
    "create": ["create", "make", "mkdir", "touch", "new"],
    "execute": ["execute", "run", "bash", "python", "script"],
    "service": ["service", "systemctl", "start", "stop", "restart"],
    "process": ["process", "ps", "kill", "terminate"],
    "permission": ["permission", "chmod", "chown", "access", "rights"],
    "network": ["network", "ssh", "curl", "wget", "download", "netstat"],
    "system": ["system", "mount", "df", "du", "disk", "free", "memory"],
    "authentication": ["authentication", "login", "password", "failed", "auth", "unauth", "unauthorized"]
}

# Query context patterns for different analysis approaches
QUERY_CONTEXTS = {
    "security_focused": [
        "failed", "unauthorized", "unauth", "breach", "attack", "suspicious",
        "anomaly", "intrusion", "hack", "malicious", "threat", "vulnerability"
    ],
    "listing_focused": [
        "list all", "show all", "find all", "get all", "display all",
        "list", "show", "display", "enumerate"
    ],
    "investigative_focused": [
        "why", "how", "what happened", "analyze", "investigate", "explain",
        "cause", "reason", "root cause", "forensic"
    ],
    "audit_focused": [
        "audit", "compliance", "policy", "regulation", "standard",
        "violation", "breach", "unauthorized"
    ]
}

class QueryOnlyLogStore:
    def __init__(self):
        self.db = None
        self.embedding_model = None
        self.sentences_data = []
        self.command_index = defaultdict(list)
        self.user_index = defaultdict(list)
        self.keyword_index = defaultdict(list)
        self.groq_client = None
        self.groq_available = False

        # Initialize components with error handling
        self._init_components()

    def _init_components(self):
        """Initialize all components with proper error handling"""
        try:
            # Initialize embedding model
            self.embedding_model = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
            logger.info("✅ Embedding model initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise

        try:
            # Initialize Groq client
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                self.groq_client = Groq(api_key=api_key)
                self.groq_available = True
                logger.info("✅ Groq client initialized")
            else:
                logger.warning("⚠️ GROQ_API_KEY not found. AI features disabled.")
                self.groq_available = False
        except Exception as e:
            logger.error(f"⚠️ Groq initialization failed: {e}")
            self.groq_available = False

        # Load existing Chroma database
        self._load_chroma_database()

    def _load_chroma_database(self):
        """Load existing Chroma database and build indices"""
        try:
            if not os.path.exists(CHROMA_PATH):
                raise FileNotFoundError(f"❌ Chroma database not found at {CHROMA_PATH}. Please run ingestion first.")

            # Load Chroma database
            self.db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=self.embedding_model
            )

            # Get all documents to build indices
            all_docs = self.db.get()
            logger.info(f"✅ Loaded {len(all_docs['ids'])} documents from Chroma DB")

            # Build indices from metadata
            self.sentences_data = []
            for i, (doc_id, metadata) in enumerate(zip(all_docs['ids'], all_docs['metadatas'])):
                # Reconstruct sentence data from metadata
                original_content = metadata.get('original_content', '')
                self.sentences_data.append((original_content, metadata))

                # Build command index
                cmd = metadata.get("base_command", "")
                if cmd:
                    self.command_index[cmd].append(i)

                orig_cmd = metadata.get("command", "")
                if orig_cmd and orig_cmd != cmd:
                    self.command_index[orig_cmd.lower()].append(i)

                # Build user index
                user = metadata.get("user", "")
                if user:
                    self.user_index[user.lower()].append(i)

                # Build keyword index from enhanced content
                enhanced_content = metadata.get("enhanced_content", original_content)
                for word in enhanced_content.split():
                    if len(word) > 2:
                        self.keyword_index[word].append(i)

            logger.info("✅ Indices built successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load Chroma database: {e}")
            raise

    def _determine_query_context(self, query: str) -> str:
        """Determine the context/intent of the query for appropriate analysis"""
        query_lower = query.lower()

        # Check for security-focused queries first (highest priority)
        for keyword in QUERY_CONTEXTS["security_focused"]:
            if keyword in query_lower:
                return "security_focused"

        # Check for listing queries
        for keyword in QUERY_CONTEXTS["listing_focused"]:
            if keyword in query_lower:
                return "listing_focused"

        # Check for investigative queries
        for keyword in QUERY_CONTEXTS["investigative_focused"]:
            if keyword in query_lower:
                return "investigative_focused"

        # Check for audit queries
        for keyword in QUERY_CONTEXTS["audit_focused"]:
            if keyword in query_lower:
                return "audit_focused"

        return "general"

    def _filter_results_by_query_intent(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results based on query intent to focus AI analysis"""
        if not results:
            return results

        query_lower = query.lower()
        context = self._determine_query_context(query)

        # For unauthorized/failed login queries, filter out successful authentications
        if any(keyword in query_lower for keyword in ["unauth", "unauthorized", "failed", "fail"]):
            filtered_results = []
            for result in results:
                content_lower = result.get("content", "").lower()
                # Only include failed authentication attempts
                if "failed" in content_lower and "authentication" in content_lower:
                    filtered_results.append(result)
                # Also include other suspicious activities but not successful logins
                elif "accepted" not in content_lower or "password" not in content_lower:
                    filtered_results.append(result)
            if filtered_results:  # Only use filtered results if we found relevant ones
                return filtered_results

        # For successful login queries, filter out failed attempts
        elif any(keyword in query_lower for keyword in ["successful", "accepted", "success"]):
            filtered_results = []
            for result in results:
                content_lower = result.get("content", "").lower()
                if "accepted" in content_lower and "authentication" in content_lower:
                    filtered_results.append(result)
            if filtered_results:
                return filtered_results

        # For specific command queries (like "list all rm"), don't filter authentication events
        elif any(cmd in query_lower for cmd in COMMAND_DESCRIPTIONS.keys()) and "list" in query_lower:
            # Keep command-specific results only
            target_commands = [cmd for cmd in COMMAND_DESCRIPTIONS.keys() if cmd in query_lower]
            if target_commands:
                filtered_results = []
                for result in results:
                    result_cmd = result.get("command", "").lower()
                    if result_cmd in [cmd.lower() for cmd in target_commands]:
                        filtered_results.append(result)
                if filtered_results:
                    return filtered_results

        return results

    def _safe_ai_query(self, prompt: str, default_response: Dict[str, Any]) -> Dict[str, Any]:
        """Safely query AI with fallback"""
        if not self.groq_available or not self.groq_client:
            return default_response

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON, with multiple fallback attempts
            for attempt in range(3):
                try:
                    # Clean the content
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()

                    result = json.loads(content)

                    # Validate the structure
                    if isinstance(result, dict):
                        # Ensure all required keys exist with defaults
                        validated = {
                            "intent": str(result.get("intent", "")),
                            "commands": list(result.get("commands", [])) if isinstance(result.get("commands"), list) else [],
                            "user": str(result.get("user", "")),
                            "keywords": list(result.get("keywords", [])) if isinstance(result.get("keywords"), list) else [],
                            "time_filter": str(result.get("time_filter", ""))
                        }
                        return validated

                except json.JSONDecodeError as je:
                    if attempt < 2:
                        # Try to fix common JSON issues
                        content = content.replace("'", '"').replace("True", "true").replace("False", "false")
                    continue

            logger.warning("⚠️ AI response parsing failed, using fallback")
            return default_response

        except Exception as e:
            logger.warning(f"⚠️ AI query failed: {e}")
            return default_response

    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback query analysis using keyword matching"""
        query_lower = query.lower()

        # Extract intent
        intent = ""
        for intent_name, keywords in INTENT_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                intent = intent_name
                break

        # Extract commands
        commands = []
        for cmd in COMMAND_DESCRIPTIONS.keys():
            if cmd in query_lower or any(desc_word in query_lower for desc_word in COMMAND_DESCRIPTIONS[cmd].split()):
                commands.append(cmd)

        # Extract user (enhanced patterns for new format)
        user = ""
        user_patterns = [
            r"user:\s*([A-Za-z0-9._-]+)",
            r"\buser\s+([A-Za-z0-9._-]+)",
            r"\b([A-Za-z0-9._-]+)\s+executed",
            r"\b([A-Za-z0-9._-]+)\s+ran",
            r"\bby\s+([A-Za-z0-9._-]+)",
            r"User\s+([A-Za-z0-9._-]+)\s+executed"  # New pattern for current format
        ]

        for pattern in user_patterns:
            match = re.search(pattern, query_lower)
            if match:
                user = match.group(1)
                break

        # Extract keywords (filter common words)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = re.findall(r'\b\w+\b', query_lower)
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]

        # Extract time filter (enhanced patterns)
        time_filter = ""
        time_patterns = [
            r"\b(yesterday|today|last week|this week|last month)\b",
            r"\b(\d{4}-\d{2}-\d{2})\b",
            r"\b(\d{1,2}:\d{2})\b",
            r"\b(Aug|Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul)\s+\d{1,2}\b"  # New pattern for month-day format
        ]
        for pattern in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                time_filter = match.group(1)
                break

        return {
            "intent": intent,
            "commands": commands,
            "user": user,
            "keywords": keywords,
            "time_filter": time_filter
        }

    def ai_query_restructure(self, query: str) -> Dict[str, Any]:
        """AI-powered query restructuring with fallback"""
        default_response = self._fallback_query_analysis(query)

        if not self.groq_available:
            logger.info("🔄 Using fallback query analysis")
            return default_response

        prompt = f"""
        Analyze this log search query and extract structured information.
        Query: "{query}"

        Available commands: {list(COMMAND_DESCRIPTIONS.keys())}

        Extract and return ONLY valid JSON:
        {{
          "intent": "one of: delete, copy, move, view, search, create, execute, service, process, permission, network, system, authentication, or empty string",
          "commands": ["array of relevant command names from the list above"],
          "user": "username if mentioned, otherwise empty string",
          "keywords": ["array of important keywords from the query"],
          "time_filter": "time reference if mentioned, otherwise empty string"
        }}

        Be precise and only include commands that are actually relevant.
        """

        result = self._safe_ai_query(prompt, default_response)
        logger.info(f"🤖 Query analysis: {result}")
        return result

    def _safe_ai_rerank(self, query: str, candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Safely rerank results with AI, fallback to original order"""
        if not self.groq_available or not candidates:
            return candidates[:k]

        try:
            # Limit candidates to prevent token overflow
            max_candidates = min(len(candidates), 10)
            limited_candidates = candidates[:max_candidates]

            docs_text = "\n".join([
                f"{i+1}. {c['content'][:200]}{'...' if len(c['content']) > 200 else ''}"
                for i, c in enumerate(limited_candidates)
            ])

            prompt = f"""
            Query: "{query}"

            Rank these log entries by relevance to the query (1 = most relevant):
            {docs_text}

            Return ONLY a valid JSON array of integers.
            No text, no code fences, no explanation.
            Example: [1, 3, 2]

            Return exactly {min(k, max_candidates)} indices.
            """

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()
            if "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            indices = json.loads(content)

            if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                reranked = []
                for idx in indices:
                    if 1 <= idx <= len(limited_candidates):
                        reranked.append(limited_candidates[idx-1])
                return reranked[:k]

        except Exception as e:
            logger.warning(f"⚠️ AI reranking failed: {e}")

        return candidates[:k]

    def _exact_command_search(self, command: str, k: int) -> List[Dict[str, Any]]:
        """Perform exact command search using command index"""
        try:
            if command not in self.command_index:
                return []

            results = []
            seen_content = set()  # Prevent duplicates

            for doc_idx in self.command_index[command]:
                if doc_idx < len(self.sentences_data):
                    sentence, metadata = self.sentences_data[doc_idx]

                    # Skip duplicates
                    if sentence in seen_content:
                        continue
                    seen_content.add(sentence)

                    # Only include if the base command actually matches
                    if metadata.get("base_command", "").lower() == command.lower():
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
                            "score": 10.0  # High score for exact matches
                        })

            # Sort by time if possible for consistent ordering
            results.sort(key=lambda x: (x.get("date", ""), x.get("time", "")))
            return results[:k]

        except Exception as e:
            logger.warning(f"Exact command search failed: {e}")
            return []

    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Enhanced keyword-based search for new format"""
        try:
            query_words = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 2]

            # Score documents based on keyword matches
            doc_scores = defaultdict(float)

            for word in query_words:
                if word in self.keyword_index:
                    for doc_idx in self.keyword_index[word]:
                        doc_scores[doc_idx] += 1.0

                # Check command matches (enhanced)
                for cmd, desc in COMMAND_DESCRIPTIONS.items():
                    if word in desc.split() or word == cmd:
                        if cmd in self.command_index:
                            for doc_idx in self.command_index[cmd]:
                                doc_scores[doc_idx] += 2.0  # Higher weight for command matches

            # Sort by score and return top k
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

            results = []
            for doc_idx, score in sorted_docs:
                if doc_idx < len(self.sentences_data):
                    sentence, metadata = self.sentences_data[doc_idx]
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
                        "score": score
                    })

            return results

        except Exception as e:
            logger.warning(f"⚠️ Keyword search failed: {e}")
            return []

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Main search function with multiple fallback strategies"""
        if not query or not query.strip():
            return []

        query = query.strip()
        logger.info(f"🔍 Searching for: '{query}'")

        try:
            # Step 1: AI-powered query restructuring (with fallback)
            structured = self.ai_query_restructure(query)

            # Step 2: Check for exact command searches first
            candidates = []
            if structured and structured.get("commands"):
                # If specific commands are identified, do direct command index search
                target_commands = structured["commands"]
                if len(target_commands) == 1 and any(cmd in query.lower() for cmd in ["list all", "show all", "find all"]):
                    cmd = target_commands[0]
                    logger.info(f"Using exact command search for: {cmd}")
                    command_results = self._exact_command_search(cmd, k)

                    if command_results:
                        logger.info(f"Found {len(command_results)} entries for command '{cmd}'")
                        return command_results

            # Step 3: Vector search (if exact command search didn't work)
            if not candidates and self.db:
                try:
                    logger.info("Using semantic search")
                    # Increase k for better reranking
                    search_k = min(k * 10, 150)  # Cap to prevent excessive results

                    # Use the proper Chroma similarity_search method
                    semantic_results = self.db.similarity_search(query, k=search_k)
                    seen_content = set()  # Prevent duplicates

                    # semantic_results is a list of Document objects
                    for doc in semantic_results:
                        # Extract metadata from the Document object
                        meta = doc.metadata
                        content = meta.get("original_content", doc.page_content)

                        if content not in seen_content:  # Check for duplicates
                            seen_content.add(content)
                            candidates.append({
                                "content": content,
                                "command": meta.get("base_command", ""),
                                "user": meta.get("user", ""),
                                "full_command": meta.get("full_command", ""),
                                "date": meta.get("date", ""),
                                "time": meta.get("time", ""),
                                "target_type": meta.get("target_type", ""),
                                "target_path": meta.get("target_path", ""),
                                "event_type": meta.get("event_type", ""),
                                "auth_details": meta.get("auth_details", "")
                            })

                except Exception as e:
                    logger.warning(f"⚠️ Semantic search failed: {e}")
                    candidates = []

            # Step 4: Fallback to keyword search if vector search fails
            if not candidates:
                logger.info("🔄 Using keyword search fallback")
                candidates = self._keyword_search(query, k * 2)

            # Step 5: Apply strict filtering based on structured query
            if structured and candidates:
                filtered_candidates = []

                for candidate in candidates:
                    include = True

                    # STRICT command filtering - this was the main issue
                    if structured.get("commands"):
                        candidate_cmd = candidate.get("command", "").lower()
                        cmd_match = any(cmd.lower() == candidate_cmd for cmd in structured["commands"])
                        if not cmd_match:
                            # Also check if command appears in full_command
                            full_cmd = candidate.get("full_command", "").lower()
                            cmd_in_full = any(cmd.lower() in full_cmd for cmd in structured["commands"])
                            if not cmd_in_full:
                                include = False

                    # Filter by user if specified
                    if structured.get("user") and include:
                        user_match = structured["user"].lower() in candidate.get("user", "").lower()
                        if not user_match:
                            include = False

                    # Filter by intent-specific keywords (relaxed for delete intent)
                    if structured.get("intent") and include:
                        intent = structured["intent"]
                        if intent == "delete":
                            # For delete intent, prioritize rm commands
                            if candidate.get("command", "").lower() != "rm":
                                # Only include non-rm commands if they contain delete-related keywords
                                content_lower = candidate.get("content", "").lower()
                                delete_keywords = ["delete", "remove", "rm", "erase", "destroy"]
                                delete_match = any(kw in content_lower for kw in delete_keywords)
                                if not delete_match:
                                    include = False
                        elif intent in INTENT_KEYWORDS:
                            intent_keywords = INTENT_KEYWORDS[intent]
                            content_lower = candidate.get("content", "").lower()
                            intent_match = any(kw in content_lower for kw in intent_keywords)
                            if not intent_match:
                                # Check if command matches intent
                                cmd = candidate.get("command", "").lower()
                                cmd_match = any(kw == cmd for kw in intent_keywords)
                                if not cmd_match:
                                    include = False

                    if include:
                        filtered_candidates.append(candidate)

                # Use filtered results if we got good matches
                if filtered_candidates:
                    candidates = filtered_candidates

            # Step 6: Remove remaining duplicates (final safety check)
            final_candidates = []
            seen_contents = set()
            for candidate in candidates:
                content = candidate.get("content", "")
                if content and content not in seen_contents:
                    seen_contents.add(content)
                    final_candidates.append(candidate)

            candidates = final_candidates

            # Step 7: AI reranking (with fallback to original order)
            final_results = self._safe_ai_rerank(query, candidates, k)

            # Step 8: Final fallback - return best candidates if all else fails
            if not final_results and candidates:
                final_results = candidates[:k]

            logger.info(f"Found {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            logger.error(traceback.format_exc())

            # Emergency fallback: simple keyword search
            try:
                return self._keyword_search(query, k)
            except:
                return []

    def display_results(self, results: List[Dict[str, Any]], query: str = ""):
        """Enhanced display for new log format"""
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 80)

        if not results:
            print("❌ No results found")
            print("\n💡 Tips:")
            print("- Try using different keywords")
            print("- Check command names (rm, cp, mv, chmod, ls, etc.)")
            print("- Include user names if searching by user")
            print("- Try specific file/directory names")
            print("- Use terms like 'failed', 'authentication', 'delete', etc.")
            return

        for i, res in enumerate(results, 1):
            print(f"\n{i}. {res['content']}")

            # Enhanced metadata display
            info_parts = []

            if res.get("command"):
                info_parts.append(f"cmd: {res['command']}")

            if res.get("user"):
                info_parts.append(f"user: {res['user']}")

            if res.get("date") or res.get("time"):
                time_str = f"{res.get('date', '')} {res.get('time', '')}".strip()
                if time_str:
                    info_parts.append(f"time: {time_str}")

            if res.get("target_type"):
                info_parts.append(f"target: {res['target_type']}")

            if res.get("event_type"):
                info_parts.append(f"type: {res['event_type']}")

            if res.get("auth_details"):
                info_parts.append(f"auth: {res['auth_details']}")

            if info_parts:
                print(f"   └─ {' | '.join(info_parts)}")

            # Show target path if available and not too long
            target_path = res.get("target_path", "")
            if target_path and len(target_path) > 0:
                if len(target_path) > 60:
                    path_display = f"...{target_path[-57:]}"
                else:
                    path_display = target_path
                print(f"   └─ path: {path_display}")

        print(f"\n📊 Showing {len(results)} results")

    def analyze_results_with_ai(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Enhanced AI analysis with query-aware filtering and context-sensitive insights"""
        if not results:
            return "❌ No relevant logs found to analyze."

        # Step 1: Filter results based on query intent BEFORE analysis
        filtered_results = self._filter_results_by_query_intent(query, results)
        analysis_results = filtered_results if filtered_results else results

        # Step 2: Determine query context for appropriate analysis depth
        query_context = self._determine_query_context(query)

        # Step 3: Format logs for analysis
        logs_text = "\n".join([
            f"- {r['content']}\n  └─ user={r.get('user','')}, cmd={r.get('command','')}, type={r.get('event_type','')}, time={r.get('date','')} {r.get('time','')}"
            for r in analysis_results
        ])

        # Fallback if Groq not available
        if not self.groq_available:
            return f"📋 Retrieved {len(analysis_results)} relevant logs (AI analysis disabled):\n\n{logs_text}"

        try:
            # Step 4: Create context-aware analysis prompt
            if query_context == "listing_focused":
                prompt = f"""
                You are analyzing logs for a LISTING query: "{query}"

                Retrieved logs:
                {logs_text}

                Provide a CONCISE summary focused on:
                1. **Summary**: Brief overview of what was found ({len(analysis_results)} entries)
                2. **Key Findings**:
                   - Command breakdown and frequency
                   - Users involved
                   - Any notable patterns ONLY if genuinely suspicious or anomalous
                3. **Notable Observations**: Only mention if there are clear security concerns or unusual patterns

                Keep the analysis brief and factual. Don't speculate about security issues unless there are clear indicators.
                """

            elif query_context == "security_focused":
                prompt = f"""
                You are a security analyst examining potentially suspicious logs for: "{query}"

                Retrieved logs:
                {logs_text}

                Provide a SECURITY-FOCUSED analysis:
                1. **Threat Assessment**: What security concerns do these logs indicate?
                2. **Attack Patterns**: Any signs of coordinated attacks, brute force, or unauthorized access?
                3. **Risk Level**: Assess the severity of findings
                4. **Immediate Actions**: What should be investigated or mitigated?
                5. **Recommendations**: Security measures to prevent future incidents

                Focus on genuine security threats and actionable insights.
                """

            elif query_context == "investigative_focused":
                prompt = f"""
                You are conducting a forensic analysis for: "{query}"

                Retrieved logs:
                {logs_text}

                Provide an INVESTIGATIVE analysis:
                1. **Timeline**: Sequence of events based on timestamps
                2. **Root Cause Analysis**: What likely caused these events?
                3. **Impact Assessment**: What systems/data may be affected?
                4. **Evidence**: Key indicators that support your conclusions
                5. **Next Steps**: What additional investigation is needed?

                Be thorough but stick to evidence-based conclusions.
                """

            else:  # general or audit_focused
                prompt = f"""
                You are analyzing audit logs for: "{query}"

                Retrieved logs:
                {logs_text}

                Provide a balanced analysis:
                1. **Summary**: Brief overview of findings
                2. **Key Activities**: Main actions and their frequency
                3. **Users and Systems**: Who did what and where
                4. **Observations**: Notable patterns or anomalies (mention only if significant)

                Keep it concise and focus on factual observations. Only highlight security concerns if clearly warranted.
                """

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600
            )

            analysis = response.choices[0].message.content.strip()

            # Add context note if results were filtered
            if len(filtered_results) < len(results) and filtered_results:
                analysis += f"\n\n*Note: Analysis focused on {len(filtered_results)} most relevant entries out of {len(results)} total results.*"

            return analysis

        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            return f"⚠️ AI analysis failed. Relevant logs ({len(analysis_results)} entries):\n\n{logs_text}"

    def get_command_statistics(self) -> Dict[str, int]:
        """Get statistics about commands in the dataset"""
        cmd_stats = defaultdict(int)
        for _, metadata in self.sentences_data:
            cmd = metadata.get("base_command", "")
            if cmd:
                cmd_stats[cmd] += 1
        return dict(cmd_stats)

    def get_user_statistics(self) -> Dict[str, int]:
        """Get statistics about users in the dataset"""
        user_stats = defaultdict(int)
        for _, metadata in self.sentences_data:
            user = metadata.get("user", "")
            if user:
                user_stats[user] += 1
        return dict(user_stats)


def main():
    """Main query function with comprehensive error handling"""
    try:
        print("🚀 Initializing AI-Powered Log Query System...")
        print("📋 Ready to search existing log database...")
        store = QueryOnlyLogStore()

        # Show dataset statistics
        print("\n📊 Dataset Statistics:")
        cmd_stats = store.get_command_statistics()
        user_stats = store.get_user_statistics()

        print(f"   Total log entries: {len(store.sentences_data)}")
        print(f"   Unique commands: {len(cmd_stats)}")
        print(f"   Unique users: {len(user_stats)}")

        if cmd_stats:
            top_commands = sorted(cmd_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top commands: {', '.join([f'{cmd}({count})' for cmd, count in top_commands])}")

        if user_stats:
            top_users = sorted(user_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top users: {', '.join([f'{user}({count})' for user, count in top_users])}")

        print("\n✅ Query system ready!")
        print("\n💡 Example queries for your log format:")
        print("  - 'list all rm commands'")
        print("  - 'show chmod operations by frank'")
        print("  - 'unauthorized login attempts'")
        print("  - 'failed authentication'")
        print("  - 'files deleted by grace'")
        print("  - 'cp commands on directories'")
        print("  - 'user alice file operations'")
        print("  - 'authentication failures'")
        print("  - 'chmod on build.sh'")

        while True:
            query = input("\n💭 Enter your search query (or 'quit' to exit): ")

            if query.lower().strip() in ["quit", "exit", "q"]:
                print("👋 Exiting. Goodbye!")
                break

            if query.lower().strip() in ["stats", "statistics"]:
                print("\n📊 Current Statistics:")
                print(f"Commands: {dict(list(cmd_stats.items())[:10])}")
                print(f"Users: {dict(list(user_stats.items())[:10])}")
                continue

            try:
                # Step 1: Search logs
                results = store.search(query, k=50)

                # Step 2: Display formatted results
                store.display_results(results, query)

                # Step 3: AI-powered analysis with query-aware filtering
                if results:
                    print("\n🤖 AI Analysis:")
                    print("=" * 80)
                    insight = store.analyze_results_with_ai(query, results)
                    print(insight)
                else:
                    print("\n💡 Search Tips:")
                    print("- Try broader keywords (e.g., 'delete' instead of 'remove files')")
                    print("- Use command names directly (rm, cp, mv, chmod, ls)")
                    print("- Include user names for user-specific searches")
                    print("- Try 'authentication' for login-related events")
                    print("- Use 'failed' or 'unauthorized' for security issues")

            except Exception as e:
                print(f"❌ Error during search: {e}")
                logger.error(traceback.format_exc())

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()


