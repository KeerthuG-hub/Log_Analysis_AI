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

class LogIngestStore:
    def __init__(self):
        self.embedding_model = None
        self.sentences_data = []
        self.command_index = defaultdict(list)
        self.user_index = defaultdict(list)
        self.keyword_index = defaultdict(list)
        
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

    def reset_chroma(self):
        """Safely reset Chroma database"""
        try:
            if os.path.exists(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
            logger.info("✅ Reset Chroma DB")
        except Exception as e:
            logger.error(f"❌ Failed to reset Chroma DB: {e}")

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
                db = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embedding_model,
                    persist_directory=CHROMA_PATH
                )
            else:
                # Process in batches
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i+batch_size]
                    if i == 0:
                        db = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embedding_model,
                            persist_directory=CHROMA_PATH
                        )
                    else:
                        db.add_documents(batch)

            logger.info(f"✅ Chroma DB created with {len(docs)} documents")
            if failed_docs > 0:
                logger.warning(f"⚠️ Failed to process {failed_docs} documents")

        except Exception as e:
            logger.error(f"❌ Failed to create Chroma DB: {e}")
            raise

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
    """Main ingestion function with comprehensive error handling"""
    try:
        print("🚀 Initializing Log Ingestion System...")
        print("📋 Optimized for new log format: 'Aug DD HH:MM:SS: User X executed Y...'")
        store = LogIngestStore()

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

        print("\n✅ Ingestion complete! Database ready for querying.")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
