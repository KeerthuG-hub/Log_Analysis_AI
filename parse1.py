import re
import json
from collections import defaultdict
from datetime import datetime
import os

# Input files
AUDIT_LOG = "/home/opc/logai/enterprise_mnc_audit_sim/logs/aggregate/all_audit_logs.log"
AUTH_LOG = "/home/opc/logai/enterprise_mnc_audit_sim/logs/aggregate/all_auth_logs.log"

# Output directory
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# UID mapping
uid_to_user = {
    2001: "alice", 2002: "bob", 2003: "carol", 2004: "dave", 2005: "eve",
    2006: "frank", 2007: "grace", 2008: "heidi", 2009: "john", 2010: "sarah"
}

# Regex patterns
syscall_re = re.compile(
    r'(?P<timestamp>\w+\s+\d+\s[\d:]+).*audit: type=SYSCALL.*pid=(?P<pid>\d+).*uid=(?P<uid>\d+).*comm="(?P<comm>[^"]+)".*exe="(?P<exe>[^"]+)"'
)
path_re = re.compile(
    r'(?P<timestamp>\w+\s+\d+\s[\d:]+).*audit: type=PATH.*item=\d+ name="(?P<name>[^"]+)".*'
)
auth_re = re.compile(
    r'(?P<timestamp>\w+\s+\d+\s[\d:]+).*sshd\[\d+\]: (?P<event>.*)for (?P<user>\w+) from [\d.]+'
)

# --- Parse audit logs ---
pid_to_syscall = {}
audit_events = []

with open(AUDIT_LOG) as f:
    for line in f:
        sc_match = syscall_re.match(line)
        if sc_match:
            data = sc_match.groupdict()
            pid_to_syscall[data["pid"]] = data
            continue
        path_match = path_re.match(line)
        if path_match:
            data = path_match.groupdict()
            # correlate with nearest syscall by timestamp before path
            path_time = datetime.strptime(data["timestamp"], "%b %d %H:%M:%S")
            closest_sc = None
            min_diff = None
            for sc in pid_to_syscall.values():
                sc_time = datetime.strptime(sc["timestamp"], "%b %d %H:%M:%S")
                diff = (path_time - sc_time).total_seconds()
                if diff >= 0 and (min_diff is None or diff < min_diff):
                    min_diff = diff
                    closest_sc = sc
            if closest_sc:
                user = uid_to_user.get(int(closest_sc["uid"]), f"uid_{closest_sc['uid']}")
                audit_events.append({
                    "timestamp": data["timestamp"],
                    "user": user,
                    "command": closest_sc["comm"],
                    "exe": closest_sc["exe"],
                    "file": data["name"]
                })

# --- Parse auth logs ---
auth_events = []
with open(AUTH_LOG) as f:
    for line in f:
        match = auth_re.match(line)
        if match:
            data = match.groupdict()
            auth_events.append({
                "timestamp": data["timestamp"],
                "user": data["user"],
                "event": data["event"].strip()
            })

# --- Combine all events ---
all_events = sorted(audit_events + auth_events, key=lambda x: x["timestamp"])

# --- JSON outputs ---
json_by_day = defaultdict(list)
json_by_user = defaultdict(list)
json_all = []

for e in all_events:
    day = e["timestamp"].split()[1]
    if "command" in e:
        record = {"user": e["user"], "command": e["command"], "exe": e["exe"], "file": e["file"]}
    else:
        record = {"user": e["user"], "auth_event": e["event"]}
    json_by_day[day].append(record)
    json_by_user[e["user"]].append({**record, "timestamp": e["timestamp"]})
    json_all.append({**record, "timestamp": e["timestamp"]})

with open(os.path.join(OUTPUT_DIR, "events_by_day.json"), "w") as f:
    json.dump(json_by_day, f, indent=4)
with open(os.path.join(OUTPUT_DIR, "events_by_user.json"), "w") as f:
    json.dump(json_by_user, f, indent=4)
with open(os.path.join(OUTPUT_DIR, "events_all.json"), "w") as f:
    json.dump(json_all, f, indent=4)

# --- Natural language outputs ---
nl_by_day = defaultdict(list)
nl_by_user = defaultdict(list)
nl_all = []
nl_auth_only = []

for e in all_events:
    if "command" in e:
        nl_text = f"{e['timestamp']}: User {e['user']} executed '{e['command']}' ({e['exe']}) on file '{e['file']}'"
    else:
        nl_text = f"{e['timestamp']}: Authentication event for user {e['user']} - {e['event']}"
        nl_auth_only.append(nl_text)
    nl_by_day[e["timestamp"].split()[1]].append(nl_text)
    nl_by_user[e["user"]].append(nl_text)
    nl_all.append(nl_text)

# Write NL files
with open(os.path.join(OUTPUT_DIR, "nl_by_day.log"), "w") as f:
    for day, logs in nl_by_day.items():
        f.write(f"--- Day {day} ---\n")
        f.write("\n".join(logs) + "\n")

with open(os.path.join(OUTPUT_DIR, "nl_by_user.log"), "w") as f:
    for user, logs in nl_by_user.items():
        f.write(f"--- User {user} ---\n")
        f.write("\n".join(logs) + "\n")

with open(os.path.join(OUTPUT_DIR, "events_nl.log"), "w") as f:
    f.write("\n".join(nl_all))

with open(os.path.join(OUTPUT_DIR, "auth_nl.log"), "w") as f:
    f.write("\n".join(nl_auth_only))

print(f"Processing complete! All outputs are in {OUTPUT_DIR}")

