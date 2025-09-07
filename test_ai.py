__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import shutil
import re
import json
import logging
from typing import List, Dict, Any, Set, Tuple, Optional, Union
from collections import defaultdict
import traceback
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
CHROMA_PATH = "chroma_db"
INPUT_FILE = "/home/opc/logai/output/nl_full1.log"
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
    "truncate": "empty file clear reduce size shrink zero"
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
    "network": ["network", "ssh", "curl", "wget", "download"],
    "system": ["system", "mount", "df", "du", "disk"]
}

class RobustAICommandStore:
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

    def reset_chroma(self):
        """Safely reset Chroma database"""
        try:
            if os.path.exists(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
            logger.info("✅ Reset Chroma DB")
        except Exception as e:
            logger.error(f"❌ Failed to reset Chroma DB: {e}")

    def _safe_ai_query(self, prompt: str, default_response: Dict[str, Any]) -> Dict[str, Any]:
        """Safely query AI with fallback"""
        if not self.groq_available or not self.groq_client:
            return default_response

        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
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

        # Extract user (simple pattern matching)
        user = ""
        user_patterns = [r"user:\s*([A-Za-z0-9._-]+)",   # matches user: dev02 | user:dev-02 | user: bob.admin
    r"\buser\s+([A-Za-z0-9._-]+)",  # matches "user dev02"
    r"\b([A-Za-z0-9._-]+)\s+executed",  # matches "dev02 executed"
    r"\b([A-Za-z0-9._-]+)\s+ran",       # matches "bob ran"
    r"\bby\s+([A-Za-z0-9._-]+)" ]

        for pattern in user_patterns:
            match = re.search(pattern, query_lower)
            if match:
                user = match.group(1)
                break

        # Extract keywords (filter common words)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = re.findall(r'\b\w+\b', query_lower)
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]

        # Extract time filter (simple patterns)
        time_filter = ""
        time_patterns = [r"\b(yesterday|today|last week|this week|last month)\b",
                        r"\b(\d{4}-\d{2}-\d{2})\b", r"\b(\d{1,2}:\d{2})\b"]
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
          "intent": "one of: delete, copy, move, view, search, create, execute, service, process, permission, network, system, or empty string",
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
                model="llama3-8b-8192",
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

    def parse_log_line(self, line: str) -> Tuple[str, Dict[str, Any]]:
        """Enhanced log line parsing with multiple patterns"""
        try:
            # Pattern 1: Standard syscall format
            patterns = [
                r"On (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}), the user (\w+) executed ([^,]+), syscall (\d+), and the result was (\w+)",
                r"User (\w+) executed ([^,]+) on (\d{4}-\d{2}-\d{2}) at (\d{2}:\d{2}:\d{2})",
                r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) - (\w+): (.+)",
                r"(\w+) ran (.+) at (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})"
            ]

            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()

                    if len(groups) >= 4:
                        # Extract components based on pattern
                        if "syscall" in line:
                            date, time, user, command, syscall, result = groups
                            metadata = {
                                "date": date,
                                "time": time,
                                "user": user,
                                "full_command": command.strip(),
                                "syscall": syscall,
                                "result": result
                            }
                        else:
                            # Adapt to other patterns
                            metadata = {
                                "date": groups[0] if len(groups) > 0 else "",
                                "time": groups[1] if len(groups) > 1 else "",
                                "user": groups[2] if len(groups) > 2 else "",
                                "full_command": groups[3] if len(groups) > 3 else "",
                                "syscall": "",
                                "result": ""
                            }

                        base_command = self._extract_base_command(metadata.get("full_command", ""))
                        metadata["base_command"] = base_command

                        return line.strip(), metadata

            # If no pattern matches, create basic metadata
            return line.strip(), {
                "base_command": self._extract_base_command(line),
                "full_command": line.strip(),
                "user": "",
                "date": "",
                "time": "",
                "syscall": "",
                "result": ""
            }

        except Exception as e:
            logger.warning(f"⚠️ Failed to parse line: {e}")
            return line.strip(), {"base_command": ""}

    def _extract_base_command(self, command: str) -> str:
        """Extract base command with better error handling"""
        try:
            if not command or not isinstance(command, str):
                return ""

            # Remove leading/trailing whitespace
            command = command.strip()
            if not command:
                return ""

            # Split by space and get first part
            parts = command.split()
            if not parts:
                return ""

            base_command = parts[0]

            # Remove path if present
            if '/' in base_command:
                base_command = base_command.split('/')[-1]

            # Remove file extensions
            if '.' in base_command:
                base_command = base_command.split('.')[0]

            return base_command.lower()

        except Exception as e:
            logger.warning(f"⚠️ Failed to extract base command from '{command}': {e}")
            return ""

    def _create_enhanced_content(self, sentence: str, metadata: Dict) -> str:
        """Create enhanced content for better searching"""
        try:
            parts = []

            # Add command information
            cmd = metadata.get("base_command", "").lower()
            if cmd and cmd in COMMAND_DESCRIPTIONS:
                parts.append(f"{cmd} {COMMAND_DESCRIPTIONS[cmd]}")

            # Add full command
            full_cmd = metadata.get("full_command", "")
            if full_cmd:
                parts.append(full_cmd.lower())

            # Add user information
            user = metadata.get("user", "")
            if user:
                parts.append(f"user {user.lower()}")

            # Add date/time context
            date = metadata.get("date", "")
            time = metadata.get("time", "")
            if date or time:
                parts.append(f"time {date} {time}")

            # Add result context
            result = metadata.get("result", "")
            if result:
                parts.append(f"result {result.lower()}")

            # Add original sentence
            parts.append(sentence.lower())

            return " ".join(filter(None, parts))

        except Exception as e:
            logger.warning(f"⚠️ Failed to create enhanced content: {e}")
            return sentence.lower()

    def load_sentences(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Load and parse sentences with comprehensive error handling"""
        try:
            if not os.path.exists(INPUT_FILE):
                raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")

            sentences = []
            seen = set()
            failed_lines = 0

            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line or line.startswith("---") or line.startswith("#"):
                            continue

                        sentence, metadata = self.parse_log_line(line)

                        # Skip duplicates
                        if sentence in seen:
                            continue

                        sentences.append((sentence, metadata))
                        seen.add(sentence)

                        # Build indices for fast lookup
                        idx = len(sentences) - 1

                        # Command index
                        cmd = metadata.get("base_command", "")
                        if cmd:
                            self.command_index[cmd].append(idx)

                        # User index
                        user = metadata.get("user", "")
                        if user:
                            self.user_index[user.lower()].append(idx)

                        # Keyword index
                        for word in sentence.lower().split():
                            if len(word) > 2:
                                self.keyword_index[word].append(idx)

                    except Exception as e:
                        failed_lines += 1
                        if failed_lines <= 10:  # Only log first 10 failures
                            logger.warning(f"⚠️ Failed to process line {line_num}: {e}")

            self.sentences_data = sentences

            logger.info(f"✅ Loaded {len(sentences)} sentences")
            if failed_lines > 0:
                logger.warning(f"⚠️ Failed to process {failed_lines} lines")

            return sentences

        except Exception as e:
            logger.error(f"❌ Failed to load sentences: {e}")
            raise

    def create_chroma(self, sentences: List[Tuple[str, Dict[str, Any]]]):
        """Create Chroma database with error handling"""
        try:
            if not sentences:
                raise ValueError("No sentences to process")

            docs = []
            failed_docs = 0

            for i, (txt, meta) in enumerate(sentences):
                try:
                    enhanced_content = self._create_enhanced_content(txt, meta)

                    # Ensure metadata is serializable
                    clean_meta = {}
                    for k, v in meta.items():
                        if isinstance(v, (str, int, float, bool, list, dict)) and v != "":
                            clean_meta[k] = v

                    clean_meta["original_content"] = txt
                    clean_meta["enhanced_content"] = enhanced_content
                    clean_meta["doc_id"] = i

                    docs.append(Document(page_content=enhanced_content, metadata=clean_meta))

                except Exception as e:
                    failed_docs += 1
                    logger.warning(f"⚠️ Failed to create document {i}: {e}")

            if not docs:
                raise ValueError("No valid documents created")

            # Create Chroma database in batches to avoid memory issues
            batch_size = 1000
            if len(docs) <= batch_size:
                self.db = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embedding_model,
                    persist_directory=CHROMA_PATH
                )
            else:
                # Process in batches
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i+batch_size]
                    if i == 0:
                        self.db = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embedding_model,
                            persist_directory=CHROMA_PATH
                        )
                    else:
                        self.db.add_documents(batch)

            logger.info(f"✅ Chroma DB created with {len(docs)} documents")
            if failed_docs > 0:
                logger.warning(f"⚠️ Failed to process {failed_docs} documents")

        except Exception as e:
            logger.error(f"❌ Failed to create Chroma DB: {e}")
            raise

    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Fallback keyword-based search"""
        try:
            query_words = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 2]

            # Score documents based on keyword matches
            doc_scores = defaultdict(float)

            for word in query_words:
                if word in self.keyword_index:
                    for doc_idx in self.keyword_index[word]:
                        doc_scores[doc_idx] += 1.0

                # Check command matches
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

            # Step 2: Vector search (primary method)
            candidates = []
            if self.db:
                try:
                    # Increase k for better reranking
                    search_k = min(k * 10, 150)  # Cap to prevent excessive results

                    # CORRECTED: Use the proper Chroma similarity_search method
                    semantic_results = self.db.similarity_search(query, k=search_k)

                    # semantic_results is a list of Document objects
                    for doc in semantic_results:
                        # Extract metadata from the Document object
                        meta = doc.metadata
                        candidates.append({
                            "content": meta.get("original_content", doc.page_content),
                            "command": meta.get("base_command", ""),
                            "user": meta.get("user", ""),
                            "full_command": meta.get("full_command", ""),
                            "date": meta.get("date", ""),
                            "time": meta.get("time", ""),
                            "result": meta.get("result", "")
                        })

                except Exception as e:
                    logger.warning(f"⚠️ Semantic search failed: {e}")
                    candidates = []

            # Step 3: Fallback to keyword search if vector search fails
            if not candidates:
                logger.info("🔄 Using keyword search fallback")
                candidates = self._keyword_search(query, k * 2)

            # Step 4: Filter based on structured query
            if structured and candidates:
                filtered_candidates = []

                for candidate in candidates:
                    include = True

                    # Filter by command if specified
                    if structured.get("commands"):
                        cmd_match = any(cmd in candidate.get("command", "").lower() or
                                      cmd in candidate.get("full_command", "").lower()
                                      for cmd in structured["commands"])
                        if not cmd_match:
                            include = False

                    # Filter by user if specified
                    if structured.get("user") and include:
                        user_match = structured["user"].lower() in candidate.get("user", "").lower()
                        if not user_match:
                            include = False

                    if include:
                        filtered_candidates.append(candidate)

                if filtered_candidates:
                    candidates = filtered_candidates

            # Step 5: AI reranking (with fallback to original order)
            final_results = self._safe_ai_rerank(query, candidates, k)

            # Step 6: Final fallback - return best candidates if all else fails
            if not final_results and candidates:
                final_results = candidates[:k]

            logger.info(f"✅ Found {len(final_results)} results")
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
        """Display search results with better formatting"""
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 60)

        if not results:
            print("❌ No results found")
            print("\nTips:")
            print("- Try using different keywords")
            print("- Check command names (rm, cp, mv, etc.)")
            print("- Include user names if searching by user")
            return

        for i, res in enumerate(results, 1):
            print(f"\n{i}. {res['content']}")

            # Show metadata
            info_parts = []
            if res.get("command"):
                info_parts.append(f"cmd: {res['command']}")
            if res.get("user"):
                info_parts.append(f"user: {res['user']}")
            if res.get("date") or res.get("time"):
                time_str = f"{res.get('date', '')} {res.get('time', '')}".strip()
                if time_str:
                    info_parts.append(f"time: {time_str}")
            if res.get("result"):
                info_parts.append(f"result: {res['result']}")

            if info_parts:
                print(f"   └─ {' | '.join(info_parts)}")

        print(f"\nShowing {len(results)} results")

    def analyze_results_with_ai(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Use AI model to generate insights from retrieved log results"""
        if not results:
            return "❌ No relevant logs found to analyze."

        # Format retrieved logs into context
        logs_text = "\n".join([f"- {r['content']} (user={r.get('user','')}, cmd={r.get('command','')}, time={r.get('date','')} {r.get('time','')})"
            for r in results
        ])

        # Fallback if Groq not available
        if not self.groq_available:
            return f"(AI disabled) Retrieved {len(results)} logs.\n\n{logs_text}"

        try:
            prompt = f"""
            You are an expert in Linux system logs.
            The user asked: "{query}"

            Here are the retrieved logs:
            {logs_text}

            Based on these logs, provide a clear and structured answer to the query.
            - If the query asks for a list, provide a bullet list or table.
            - If the query implies analysis (e.g., suspicious activity, root cause), explain clearly.
            - If nothing relevant is found, say so.
            """

            response = self.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600
        )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            return f"⚠️ Failed to analyze with AI. Showing raw logs:\n\n{logs_text}"


def main():
    """Main function with comprehensive error handling"""
    try:
        print("🚀 Initializing AI-Powered Command Search System...")
        store = RobustAICommandStore()

        print("🔄 Resetting database...")
        store.reset_chroma()

        print("📂 Loading sentences...")
        sentences = store.load_sentences()

        if not sentences:
            print("❌ No sentences loaded. Please check your input file.")
            return

        print("🏗️ Creating search database...")
        store.create_chroma(sentences)

        print("\n✅ System ready!")
        print("\nExample queries:")
        print("  - 'show me file deletions'")
        print("  - 'python scripts executed by alice'")
        print("  - 'failed rm commands'")
        print("  - 'who accessed /etc/passwd'")
        print("  - 'systemctl service restarts'")

        """while True:
            query = input("\n💭 Enter your search query (or 'quit' to exit): ")

            if query.lower().strip() in ["quit", "exit"]:
                print("👋 Exiting. Goodbye!")
                break

            try:
                results = store.search(query, k=10)
                store.display_results(results, query)

            except Exception as e:
                print(f"❌ Error during search: {e}")"""

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()

    while True:
        query = input("\n💭 Enter your search query (or 'quit' to exit): ")

        if query.lower().strip() in ["quit", "exit"]:
            print("👋 Exiting. Goodbye!")
            break

        try:
            # Step 1: search logs
            results = store.search(query, k=10)

            # Step 2: show raw logs
            store.display_results(results, query)

            # Step 3: mandatory AI insights
            insight = store.analyze_results_with_ai(query, results)
            print("\n🤖 AI Insights:")
            print("=" * 60)
            print(insight)

        except Exception as e:
            print(f"❌ Error during search: {e}")



if __name__ == "__main__":
    main()

