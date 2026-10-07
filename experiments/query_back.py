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
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import calendar
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
    "authentication": ["authentication", "login", "password", "failed", "auth"]
}

class QueryOnlyLogStore:
    def __init__(self):
        self.db = None
        self.embedding_model = None
        self.sentences_data = []
        self.command_index = defaultdict(list)
        self.user_index = defaultdict(list)
        self.keyword_index = defaultdict(list)
        self.temporal_index = {}  # Maps doc_idx -> parsed datetime
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

    def _parse_timestamp(self, date_str: str, time_str: str) -> Optional[datetime]:
        """Intelligently parse date and time strings into datetime objects"""
        try:
            # Handle various date formats
            if date_str:
                # Try direct parsing first
                try:
                    parsed_date = date_parser.parse(date_str, fuzzy=True)
                except:
                    # Handle month-day format like "Aug 15"
                    month_day_patterns = [
                        r'(\w{3})\s+(\d{1,2})',  # Aug 15
                        r'(\w{3,9})\s+(\d{1,2})',  # August 15
                        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # 8/15/24 or 08-15-2024
                        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # 2024/08/15
                    ]
                    
                    parsed_date = None
                    for pattern in month_day_patterns:
                        match = re.search(pattern, date_str)
                        if match:
                            try:
                                if len(match.groups()) == 2:  # Month Day format
                                    month_name, day = match.groups()
                                    # Get current year as default
                                    year = datetime.now().year
                                    if month_name.isalpha():
                                        month_num = list(calendar.month_abbr).index(month_name[:3].title())
                                        parsed_date = datetime(year, month_num, int(day))
                                    break
                                elif len(match.groups()) == 3:  # Full date format
                                    if match.group(1).isdigit() and len(match.group(1)) == 4:
                                        # YYYY/MM/DD format
                                        year, month, day = match.groups()
                                        parsed_date = datetime(int(year), int(month), int(day))
                                    else:
                                        # MM/DD/YY format
                                        month, day, year = match.groups()
                                        if len(year) == 2:
                                            year = 2000 + int(year)
                                        parsed_date = datetime(int(year), int(month), int(day))
                                    break
                            except:
                                continue
                    
                    if not parsed_date:
                        return None
            else:
                parsed_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # Parse time if available
            if time_str:
                try:
                    time_part = date_parser.parse(time_str, fuzzy=True).time()
                    parsed_date = parsed_date.replace(
                        hour=time_part.hour,
                        minute=time_part.minute,
                        second=time_part.second,
                        microsecond=time_part.microsecond
                    )
                except:
                    pass

            return parsed_date

        except Exception as e:
            logger.debug(f"Failed to parse timestamp: {date_str} {time_str} - {e}")
            return None

    def _extract_temporal_queries(self, query: str) -> List[Dict[str, Any]]:
        """Extract all possible temporal references from query using intelligent parsing"""
        temporal_refs = []
        
        # Comprehensive temporal patterns (no hardcoding specific formats)
        patterns = [
            # Relative time expressions
            r'\b(yesterday|today|tomorrow)\b',
            r'\b(last|this|next)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b',
            r'\b(before|after|since|until)\s+(\S+(?:\s+\S+)*?)\b',
            
            # Absolute dates - flexible patterns
            r'\b(\w{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})?\b',  # August 15, 2024 or Aug 15
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',  # 8/15/24, 08-15-2024
            r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',    # 2024/08/15, 2024-08-15
            r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(\w{3,9})\s+(\d{4})\b',  # 15th August 2024
            
            # Time patterns
            r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm|AM|PM)?\b',
            r'\b(\d{1,2})\s*(am|pm|AM|PM)\b',
            
            # Time ranges
            r'\bbetween\s+(.+?)\s+and\s+(.+?)\b',
            r'\bfrom\s+(.+?)\s+to\s+(.+?)\b',
            
            # ISO-like formats (flexible)
            r'\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b',
        ]
        
        query_lower = query.lower()
        now = datetime.now()
        
        for pattern in patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                try:
                    match_text = match.group(0)
                    groups = match.groups()
                    
                    # Handle different pattern types
                    if 'yesterday' in match_text:
                        target_time = now - timedelta(days=1)
                        temporal_refs.append({
                            'type': 'absolute',
                            'start_time': target_time.replace(hour=0, minute=0, second=0),
                            'end_time': target_time.replace(hour=23, minute=59, second=59),
                            'original': match_text
                        })
                    
                    elif 'today' in match_text:
                        temporal_refs.append({
                            'type': 'absolute',
                            'start_time': now.replace(hour=0, minute=0, second=0),
                            'end_time': now.replace(hour=23, minute=59, second=59),
                            'original': match_text
                        })
                    
                    elif 'last week' in match_text:
                        week_start = now - timedelta(days=now.weekday() + 7)
                        week_end = week_start + timedelta(days=6)
                        temporal_refs.append({
                            'type': 'range',
                            'start_time': week_start.replace(hour=0, minute=0, second=0),
                            'end_time': week_end.replace(hour=23, minute=59, second=59),
                            'original': match_text
                        })
                    
                    elif 'between' in match_text or 'from' in match_text:
                        # Parse range expressions
                        parts = groups
                        if len(parts) >= 2:
                            try:
                                start_parsed = date_parser.parse(parts[0], fuzzy=True)
                                end_parsed = date_parser.parse(parts[1], fuzzy=True)
                                temporal_refs.append({
                                    'type': 'range',
                                    'start_time': start_parsed,
                                    'end_time': end_parsed,
                                    'original': match_text
                                })
                            except:
                                pass
                    
                    else:
                        # Try to parse as absolute date/time
                        try:
                            parsed_dt = date_parser.parse(match_text, fuzzy=True)
                            temporal_refs.append({
                                'type': 'absolute',
                                'start_time': parsed_dt,
                                'end_time': parsed_dt + timedelta(hours=23, minutes=59, seconds=59),
                                'original': match_text
                            })
                        except:
                            # Try manual parsing for complex formats
                            manual_parsed = self._manual_date_parse(match_text, groups)
                            if manual_parsed:
                                temporal_refs.append(manual_parsed)
                
                except Exception as e:
                    logger.debug(f"Failed to parse temporal reference: {match_text} - {e}")
                    continue
        
        return temporal_refs

    def _manual_date_parse(self, match_text: str, groups: tuple) -> Optional[Dict[str, Any]]:
        """Manual parsing for complex date formats"""
        try:
            if len(groups) >= 2:
                # Month Day format (Aug 15)
                if groups[0] and groups[1] and groups[0].isalpha():
                    month_name = groups[0]
                    day = int(groups[1])
                    year = int(groups[2]) if len(groups) > 2 and groups[2] else datetime.now().year
                    
                    # Convert month name to number
                    month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
                    month_abbr = month_name.lower()[:3]
                    if month_abbr in month_names:
                        month_num = month_names.index(month_abbr) + 1
                        target_dt = datetime(year, month_num, day)
                        return {
                            'type': 'absolute',
                            'start_time': target_dt.replace(hour=0, minute=0, second=0),
                            'end_time': target_dt.replace(hour=23, minute=59, second=59),
                            'original': match_text
                        }
                
                # Time format (14:30)
                elif ':' in match_text:
                    hour = int(groups[0])
                    minute = int(groups[1])
                    # Use today's date with specified time
                    today = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                    return {
                        'type': 'time',
                        'start_time': today,
                        'end_time': today + timedelta(minutes=59),
                        'original': match_text
                    }
        except:
            pass
        return None

    def _load_chroma_database(self):
        """Load existing Chroma database and build enhanced indices including temporal"""
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

                # Build temporal index - parse timestamps
                date_str = metadata.get("date", "")
                time_str = metadata.get("time", "")
                parsed_dt = self._parse_timestamp(date_str, time_str)
                if parsed_dt:
                    self.temporal_index[i] = parsed_dt

            logger.info(f"✅ Built temporal index with {len(self.temporal_index)} entries")
            logger.info("✅ All indices built successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load Chroma database: {e}")
            raise

    def _temporal_filter(self, doc_indices: List[int], temporal_queries: List[Dict[str, Any]]) -> List[int]:
        """Filter document indices based on temporal queries"""
        if not temporal_queries or not self.temporal_index:
            return doc_indices
        
        filtered_indices = []
        
        for doc_idx in doc_indices:
            if doc_idx not in self.temporal_index:
                continue
                
            doc_timestamp = self.temporal_index[doc_idx]
            match_found = False
            
            for temp_query in temporal_queries:
                start_time = temp_query.get('start_time')
                end_time = temp_query.get('end_time')
                
                if start_time and end_time:
                    if start_time <= doc_timestamp <= end_time:
                        match_found = True
                        break
                elif start_time:  # Only start time specified
                    if doc_timestamp >= start_time:
                        match_found = True
                        break
            
            if match_found:
                filtered_indices.append(doc_idx)
        
        return filtered_indices

    def _temporal_score_boost(self, results: List[Dict[str, Any]], temporal_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Boost scores for temporally relevant results"""
        if not temporal_queries:
            return results
        
        for result in results:
            # Try to parse timestamp from result metadata
            date_str = result.get("date", "")
            time_str = result.get("time", "")
            result_dt = self._parse_timestamp(date_str, time_str)
            
            if result_dt:
                for temp_query in temporal_queries:
                    start_time = temp_query.get('start_time')
                    end_time = temp_query.get('end_time')
                    
                    if start_time and end_time:
                        if start_time <= result_dt <= end_time:
                            # Exact temporal match - significant boost
                            current_score = result.get('score', 1.0)
                            result['score'] = current_score * 2.0
                            result['temporal_match'] = True
                            break
                    elif start_time:
                        # Calculate temporal distance for gradual scoring
                        time_diff = abs((result_dt - start_time).total_seconds())
                        if time_diff < 3600:  # Within 1 hour
                            current_score = result.get('score', 1.0)
                            result['score'] = current_score * 1.8
                            result['temporal_match'] = True
                        elif time_diff < 86400:  # Within 1 day
                            current_score = result.get('score', 1.0)
                            result['score'] = current_score * 1.5
                            result['temporal_match'] = True
        
        return results

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

        # Extract user (enhanced patterns)
        user = ""
        user_patterns = [
            r"user:\s*([A-Za-z0-9._-]+)",
            r"\buser\s+([A-Za-z0-9._-]+)",
            r"\b([A-Za-z0-9._-]+)\s+executed",
            r"\b([A-Za-z0-9._-]+)\s+ran",
            r"\bby\s+([A-Za-z0-9._-]+)",
            r"User\s+([A-Za-z0-9._-]+)\s+executed"
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

        return {
            "intent": intent,
            "commands": commands,
            "user": user,
            "keywords": keywords,
            "time_filter": ""
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
          "time_filter": "any temporal reference mentioned, otherwise empty string"
        }}

        Be precise and only include commands that are actually relevant.
        """

        result = self._safe_ai_query(prompt, default_response)
        logger.info(f"🤖 Query analysis: {result}")
        return result

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced search with intelligent temporal filtering and reranking"""
        if not query or not query.strip():
            return []

        query = query.strip()
        logger.info(f"🔍 Searching for: '{query}'")

        try:
            # Step 1: Extract temporal references from query
            temporal_queries = self._extract_temporal_queries(query)
            if temporal_queries:
                logger.info(f"⏰ Found temporal queries: {[tq['original'] for tq in temporal_queries]}")

            # Step 2: AI-powered query restructuring
            structured = self.ai_query_restructure(query)

            # Step 3: Get initial candidates using semantic search
            candidates = []
            if self.db:
                try:
                    logger.info("Using enhanced semantic search")
                    # Increase search space for better temporal filtering
                    search_k = min(k * 20, 500) if temporal_queries else min(k * 10, 150)
                    
                    semantic_results = self.db.similarity_search(query, k=search_k)
                    seen_content = set()

                    for doc in semantic_results:
                        meta = doc.metadata
                        content = meta.get("original_content", doc.page_content)

                        if content not in seen_content:
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
                                "auth_details": meta.get("auth_details", ""),
                                "score": 1.0  # Base semantic score
                            })

                except Exception as e:
                    logger.warning(f"⚠️ Semantic search failed: {e}")
                    candidates = []

            # Step 4: Apply temporal filtering if temporal queries exist
            if temporal_queries and candidates:
                logger.info(f"⏰ Applying temporal filtering to {len(candidates)} candidates")
                
                # Convert candidates back to indices for temporal filtering
                doc_indices = []
                candidate_map = {}
                
                for i, candidate in enumerate(candidates):
                    # Find matching document index in sentences_data
                    for doc_idx, (sentence, metadata) in enumerate(self.sentences_data):
                        if sentence == candidate["content"]:
                            doc_indices.append(doc_idx)
                            candidate_map[doc_idx] = i
                            break
                
                # Apply temporal filter
                filtered_indices = self._temporal_filter(doc_indices, temporal_queries)
                
                # Reconstruct candidates from filtered indices
                if filtered_indices:
                    temporal_candidates = []
                    for doc_idx in filtered_indices:
                        if doc_idx in candidate_map:
                            candidate_idx = candidate_map[doc_idx]
                            temporal_candidates.append(candidates[candidate_idx])
                    
                    candidates = temporal_candidates
                    logger.info(f"⏰ Temporal filter retained {len(candidates)} results")
                else:
                    logger.info("⏰ No temporal matches found, keeping original results")

            # Step 5: Apply traditional filtering (commands, users, etc.)
            if structured and candidates:
                filtered_candidates = []

                for candidate in candidates:
                    include = True

                    # Command filtering
                    if structured.get("commands"):
                        candidate_cmd = candidate.get("command", "").lower()
                        cmd_match = any(cmd.lower() == candidate_cmd for cmd in structured["commands"])
                        if not cmd_match:
                            full_cmd = candidate.get("full_command", "").lower()
                            cmd_in_full = any(cmd.lower() in full_cmd for cmd in structured["commands"])
                            if not cmd_in_full:
                                include = False

                    # User filtering
                    if structured.get("user") and include:
                        user_match = structured["user"].lower() in candidate.get("user", "").lower()
                        if not user_match:
                            include = False

                    # Intent-based filtering
                    if structured.get("intent") and include:
                        intent = structured["intent"]
                        if intent in INTENT_KEYWORDS:
                            intent_keywords = INTENT_KEYWORDS[intent]
                            content_lower = candidate.get("content", "").lower()
                            intent_match = any(kw in content_lower for kw in intent_keywords)
                            if not intent_match:
                                cmd = candidate.get("command", "").lower()
                                cmd_match = any(kw == cmd for kw in intent_keywords)
                                if not cmd_match:
                                    include = False

                    if include:
                        filtered_candidates.append(candidate)

                if filtered_candidates:
                    candidates = filtered_candidates

            # Step 6: Apply temporal score boosting
            if temporal_queries:
                candidates = self._temporal_score_boost(candidates, temporal_queries)

            # Step 7: Sort by score (temporal matches get highest scores)
            candidates.sort(key=lambda x: x.get('score', 1.0), reverse=True)

            # Step 8: AI reranking with temporal context
            final_results = self._safe_ai_rerank_with_temporal(query, candidates, temporal_queries, k)

            # Step 9: Final fallback
            if not final_results and candidates:
                final_results = candidates[:k]

            logger.info(f"Found {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            logger.error(traceback.format_exc())
            return []

    def _safe_ai_rerank_with_temporal(self, query: str, candidates: List[Dict[str, Any]], 
                                    temporal_queries: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """AI reranking with temporal context awareness"""
        if not self.groq_available or not candidates:
            return candidates[:k]

        try:
            max_candidates = min(len(candidates), 15)
            limited_candidates = candidates[:max_candidates]

            # Include temporal context in the prompt
            temporal_context = ""
            if temporal_queries:
                temporal_refs = [tq['original'] for tq in temporal_queries]
                temporal_context = f"\nTemporal references in query: {', '.join(temporal_refs)}"

            docs_text = "\n".join([
                f"{i+1}. {c['content'][:150]}... [Date: {c.get('date', 'N/A')} {c.get('time', 'N/A')}] [Score: {c.get('score', 1.0):.2f}] [Temporal: {'✓' if c.get('temporal_match') else '✗'}]"
                for i, c in enumerate(limited_candidates)
            ])

            prompt = f"""
            Query: "{query}"{temporal_context}

            Rank these log entries by relevance, considering both semantic relevance and temporal accuracy:
            {docs_text}

            Prioritize entries that match temporal references in the query.
            Return ONLY a valid JSON array of integers (1-indexed).
            Return exactly {min(k, max_candidates)} indices.
            Example: [1, 3, 2]
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

    def display_results(self, results: List[Dict[str, Any]], query: str = ""):
        """Enhanced display with temporal match indicators"""
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 80)

        if not results:
            print("❌ No results found")
            print("\n💡 Tips:")
            print("- Try using different keywords")
            print("- Check command names (rm, cp, mv, chmod, ls, etc.)")
            print("- Include user names if searching by user")
            print("- Try temporal references: 'yesterday', 'Aug 15', '14:30', 'last week'")
            print("- Use specific file/directory names")
            return

        for i, res in enumerate(results, 1):
            # Add temporal match indicator
            temporal_indicator = "⏰" if res.get('temporal_match') else ""
            score_indicator = f"[{res.get('score', 1.0):.2f}]" if res.get('score', 1.0) != 1.0 else ""
            
            print(f"\n{i}. {temporal_indicator} {res['content']} {score_indicator}")

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

            if info_parts:
                print(f"   └─ {' | '.join(info_parts)}")

            # Show target path if available
            target_path = res.get("target_path", "")
            if target_path and len(target_path) > 0:
                path_display = f"...{target_path[-57:]}" if len(target_path) > 60 else target_path
                print(f"   └─ path: {path_display}")

        temporal_matches = sum(1 for r in results if r.get('temporal_match'))
        if temporal_matches > 0:
            print(f"\n📊 Showing {len(results)} results ({temporal_matches} temporal matches)")
        else:
            print(f"\n📊 Showing {len(results)} results")

    def analyze_results_with_ai(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Enhanced AI analysis with temporal insights"""
        if not results:
            return "❌ No relevant logs found to analyze."

        # Check for temporal patterns
        temporal_results = [r for r in results if r.get('temporal_match')]
        
        logs_text = "\n".join([
            f"- {r['content']}\n  └─ user={r.get('user','')}, cmd={r.get('command','')}, time={r.get('date','')} {r.get('time','')}, temporal_match={'✓' if r.get('temporal_match') else '✗'}"
            for r in results[:10]  # Limit for token efficiency
        ])

        if not self.groq_available:
            temporal_summary = f" ({len(temporal_results)} temporal matches)" if temporal_results else ""
            return f"📋 Retrieved {len(results)} logs{temporal_summary}:\n\n{logs_text}"

        try:
            temporal_context = ""
            if temporal_results:
                temporal_context = f"\n\nIMPORTANT: {len(temporal_results)} of these logs match temporal criteria from the query."

            prompt = f"""
            You are an expert Linux system administrator analyzing audit logs with temporal awareness.
            User query: "{query}"

            Retrieved logs:
            {logs_text}{temporal_context}

            Provide a comprehensive analysis:

            1. **Summary**: What do these logs reveal?
            2. **Temporal Analysis**: If temporal matches exist, highlight time-based patterns
            3. **Key Findings**:
               - Commands and their frequency
               - Users and their activities  
               - File/directory operations
               - Security-relevant events
            4. **Patterns**: Notable patterns or anomalies
            5. **Recommendations**: Security or operational insights

            Prioritize temporal matches if they exist. Format clearly with headers and bullets.
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

    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get temporal statistics from the dataset"""
        if not self.temporal_index:
            return {"error": "No temporal index available"}
        
        timestamps = list(self.temporal_index.values())
        if not timestamps:
            return {"error": "No valid timestamps found"}
        
        timestamps.sort()
        
        return {
            "total_entries": len(timestamps),
            "earliest": timestamps[0].strftime("%Y-%m-%d %H:%M:%S"),
            "latest": timestamps[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "date_range_days": (timestamps[-1] - timestamps[0]).days,
            "unique_dates": len(set(dt.date() for dt in timestamps)),
            "entries_per_day": len(timestamps) / max(1, (timestamps[-1] - timestamps[0]).days + 1)
        }

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
    """Main function with enhanced temporal query support"""
    try:
        print("🚀 Initializing AI-Powered Temporal Log Query System...")
        print("⏰ Enhanced with intelligent date/time search capabilities...")
        store = TemporalLogStore()

        # Show dataset statistics including temporal info
        print("\n📊 Dataset Statistics:")
        cmd_stats = store.get_command_statistics()
        user_stats = store.get_user_statistics()
        temporal_stats = store.get_temporal_statistics()

        print(f"   Total log entries: {len(store.sentences_data)}")
        print(f"   Unique commands: {len(cmd_stats)}")
        print(f"   Unique users: {len(user_stats)}")
        
        if "error" not in temporal_stats:
            print(f"   Temporal coverage: {temporal_stats['date_range_days']} days")
            print(f"   Date range: {temporal_stats['earliest']} to {temporal_stats['latest']}")
            print(f"   Entries with timestamps: {temporal_stats['total_entries']}")
        
        if cmd_stats:
            top_commands = sorted(cmd_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top commands: {', '.join([f'{cmd}({count})' for cmd, count in top_commands])}")

        print("\n✅ Temporal query system ready!")
        print("\n💡 Example queries with temporal support:")
        print("  🕒 Temporal: 'rm commands yesterday', 'files deleted on Aug 15', 'authentication failures last week'")
        print("  👤 User-based: 'show chmod operations by frank', 'alice file operations today'")
        print("  🔧 Command-specific: 'list all cp commands', 'grep operations between 14:00 and 15:00'")
        print("  🔍 Complex: 'failed login attempts on Aug 15', 'chmod on build.sh yesterday'")

        while True:
            query = input("\n💭 Enter your search query (or 'quit' to exit): ")

            if query.lower().strip() in ["quit", "exit", "q"]:
                print("👋 Exiting. Goodbye!")
                break

            if query.lower().strip() in ["stats", "statistics"]:
                print("\n📊 Current Statistics:")
                print(f"Commands: {dict(list(cmd_stats.items())[:10])}")
                print(f"Users: {dict(list(user_stats.items())[:10])}")
                if "error" not in temporal_stats:
                    print(f"Temporal: {temporal_stats}")
                continue

            try:
                # Search with enhanced temporal support
                results = store.search(query, k=15)

                # Display results with temporal indicators
                store.display_results(results, query)

                # AI analysis with temporal insights
                if results:
                    print("\n🤖 AI Analysis:")
                    print("=" * 80)
                    insight = store.analyze_results_with_ai(query, results)
                    print(insight)
                else:
                    print("\n💡 Search Tips:")
                    print("⏰ Try temporal references: 'yesterday', 'Aug 15', '14:30', 'last week', 'today'")
                    print("🔧 Use command names: rm, cp, mv, chmod, ls, grep, ssh")
                    print("👤 Include usernames: 'by alice', 'user frank', 'grace executed'")
                    print("🔍 Combine terms: 'chmod operations yesterday', 'failed auth today'")

            except Exception as e:
                print(f"❌ Error during search: {e}")
                logger.error(traceback.format_exc())

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
