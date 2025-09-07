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
INPUT_FILE = "/home/opc/logai/output/nl_full2.log"
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
    "authentication": ["authentication", "login", "password", "failed", "auth"]
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
        """Enhanced log line parsing for new format"""
        try:
            # New patterns for the current log format
            patterns = [
                # Pattern 1: Standard command execution
                r"(Aug|Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul)\s+(\d{1,2})\s+(\d{2}:\d{2}:\d{2}):\s+User\s+(\w+)\s+executed\s+'([^']+)'\s+on\s+(file|directory)\s+'([^']+)'",
                
                # Pattern 2: Authentication events
                r"(Aug|Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul)\s+(\d{1,2})\s+(\d{2}:\d{2}:\d{2}):\s+Authentication\s+event\s+for\s+user\s+(\w+)\s+-\s+(.+)",
                
                # Pattern 3: Generic format fallback
                r"(Aug|Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul)\s+(\d{1,2})\s+(\d{2}:\d{2}:\d{2}):\s+(.+)",
                
                # Pattern 4: Original syscall format (for backward compatibility)
                r"On (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}), the user (\w+) executed ([^,]+), syscall (\d+), and the result was (\w+)"
            ]

            for i, pattern in enumerate(patterns):
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()
                    
                    if i == 0:  # Standard command execution
                        month, day, time, user, command, target_type, target_path = groups
                        metadata = {
                            "month": month,
                            "day": day,
                            "time": time,
                            "date": f"{month} {day}",  # Combined date
                            "user": user,
                            "command": command,
                            "target_type": target_type,
                            "target_path": target_path,
                            "full_command": f"{command} on {target_type} {target_path}",
                            "event_type": "command_execution"
                        }
                        
                    elif i == 1:  # Authentication events
                        month, day, time, user, auth_details = groups
                        metadata = {
                            "month": month,
                            "day": day,
                            "time": time,
                            "date": f"{month} {day}",
                            "user": user,
                            "command": "authentication",
                            "auth_details": auth_details,
                            "full_command": f"authentication: {auth_details}",
                            "event_type": "authentication"
                        }
                        
                    elif i == 2:  # Generic format
                        month, day, time, content = groups
                        # Try to extract user and command from content
                        user_match = re.search(r"User\s+(\w+)", content)
                        cmd_match = re.search(r"executed\s+'([^']+)'", content)
                        
                        metadata = {
                            "month": month,
                            "day": day,
                            "time": time,
                            "date": f"{month} {day}",
                            "user": user_match.group(1) if user_match else "",
                            "command": cmd_match.group(1) if cmd_match else "",
                            "full_command": content,
                            "event_type": "generic"
                        }
                        
                    else:  # Original syscall format
                        date, time, user, command, syscall, result = groups
                        metadata = {
                            "date": date,
                            "time": time,
                            "user": user,
                            "command": command.strip(),
                            "syscall": syscall,
                            "result": result,
                            "full_command": command.strip(),
                            "event_type": "syscall"
                        }

                    # Extract base command for all formats
                    base_command = self._extract_base_command(metadata.get("command", ""))
                    metadata["base_command"] = base_command

                    return line.strip(), metadata

            # If no pattern matches, create basic metadata
            return line.strip(), {
                "base_command": self._extract_base_command(line),
                "full_command": line.strip(),
                "user": "",
                "date": "",
                "time": "",
                "event_type": "unknown"
            }

        except Exception as e:
            logger.warning(f"⚠️ Failed to parse line: {e}")
            return line.strip(), {"base_command": "", "event_type": "error"}

    def _extract_base_command(self, command: str) -> str:
        """Extract base command with better error handling"""
        try:
            if not command or not isinstance(command, str):
                return ""

            # Remove leading/trailing whitespace and quotes
            command = command.strip().strip("'\"")
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
        """Create enhanced content for better searching with new format"""
        try:
            parts = []

            # Add command information with enhanced descriptions
            cmd = metadata.get("base_command", "").lower()
            if cmd and cmd in COMMAND_DESCRIPTIONS:
                parts.append(f"{cmd} {COMMAND_DESCRIPTIONS[cmd]}")
            
            # Add event type context
            event_type = metadata.get("event_type", "")
            if event_type:
                parts.append(f"event_type_{event_type}")

            # Add full command with target information
            full_cmd = metadata.get("full_command", "")
            if full_cmd:
                parts.append(full_cmd.lower())

            # Add target type and path information
            target_type = metadata.get("target_type", "")
            target_path = metadata.get("target_path", "")
            if target_type:
                parts.append(f"target_{target_type}")
            if target_path:
                # Extract meaningful parts from path
                path_parts = target_path.split('/')
                parts.extend([p.lower() for p in path_parts[-3:] if p])  # Last 3 path components

            # Add user information
            user = metadata.get("user", "")
            if user:
                parts.append(f"user_{user.lower()}")

            # Add date/time context
            date = metadata.get("date", "")
            time = metadata.get("time", "")
            if date:
                parts.append(f"date_{date.lower()}")
            if time:
                parts.append(f"time_{time}")

            # Add authentication details if present
            auth_details = metadata.get("auth_details", "")
            if auth_details:
                parts.append(f"auth_{auth_details.lower()}")

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

                        # Command index (enhanced for base command)
                        cmd = metadata.get("base_command", "")
                        if cmd:
                            self.command_index[cmd].append(idx)
                        
                        # Also index the original command
                        orig_cmd = metadata.get("command", "")
                        if orig_cmd and orig_cmd != cmd:
                            self.command_index[orig_cmd.lower()].append(idx)

                        # User index
                        user = metadata.get("user", "")
                        if user:
                            self.user_index[user.lower()].append(idx)

                        # Keyword index (enhanced)
                        enhanced_content = self._create_enhanced_content(sentence, metadata)
                        for word in enhanced_content.split():
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

            # Step 3: Fallback to keyword search if vector search fails
            if not candidates:
                logger.info("🔄 Using keyword search fallback")
                candidates = self._keyword_search(query, k * 2)

            # Step 4: Fallback to keyword search if vector search fails
            if not candidates:
                logger.info("Using keyword search fallback")
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
        """Enhanced AI analysis for new log format"""
        if not results:
            return "❌ No relevant logs found to analyze."

        # Format retrieved logs with enhanced context
        logs_text = "\n".join([
            f"- {r['content']}\n  └─ user={r.get('user','')}, cmd={r.get('command','')}, type={r.get('event_type','')}, time={r.get('date','')} {r.get('time','')}"
            for r in results
        ])

        # Fallback if Groq not available
        if not self.groq_available:
            return f"📋 Retrieved {len(results)} logs (AI analysis disabled):\n\n{logs_text}"

        try:
            prompt = f"""
            You are an expert Linux system administrator analyzing audit logs.
            User query: "{query}"

            Retrieved logs:
            {logs_text}

            Based on these logs, provide a comprehensive analysis:
            
            1. **Summary**: What do these logs show?
            2. **Key Findings**: 
               - Commands executed and their frequency
               - Users involved and their activities
               - File/directory operations
               - Any security-relevant events
            3. **Patterns**: Any notable patterns or anomalies?
            4. **Recommendations**: If applicable, any security or operational insights

            Format your response clearly with headers and bullet points where appropriate.
            If the query asks for a specific list (e.g., "list all rm commands"), provide that list first.
            """

            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            return f"⚠️ AI analysis failed. Raw logs ({len(results)} entries):\n\n{logs_text}"

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
    """Main function with comprehensive error handling"""
    try:
        print("🚀 Initializing AI-Powered Log Analysis System...")
        print("📋 Optimized for new log format: 'Aug DD HH:MM:SS: User X executed Y...'")
        store = RobustAICommandStore()

        print("🔄 Resetting database...")
        store.reset_chroma()

        print("📂 Loading and parsing log entries...")
        sentences = store.load_sentences()

        if not sentences:
            print("❌ No log entries loaded. Please check your input file.")
            return

        print("🏗️ Creating search database with enhanced indexing...")
        store.create_chroma(sentences)

        # Show dataset statistics
        print("\n📊 Dataset Statistics:")
        cmd_stats = store.get_command_statistics()
        user_stats = store.get_user_statistics()
        
        print(f"   Total log entries: {len(sentences)}")
        print(f"   Unique commands: {len(cmd_stats)}")
        print(f"   Unique users: {len(user_stats)}")
        
        if cmd_stats:
            top_commands = sorted(cmd_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top commands: {', '.join([f'{cmd}({count})' for cmd, count in top_commands])}")
        
        if user_stats:
            top_users = sorted(user_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top users: {', '.join([f'{user}({count})' for user, count in top_users])}")

        print("\n✅ System ready!")
        print("\n💡 Example queries for your log format:")
        print("  - 'list all rm commands'")
        print("  - 'show chmod operations by frank'")
        print("  - 'authentication failures'")
        print("  - 'files deleted by grace'") 
        print("  - 'cp commands on directories'")
        print("  - 'user alice file operations'")
        print("  - 'failed login attempts'")
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
                results = store.search(query, k=15)

                # Step 2: Display formatted results
                store.display_results(results, query)

                # Step 3: AI-powered analysis
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

            except Exception as e:
                print(f"❌ Error during search: {e}")
                logger.error(traceback.format_exc())

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
