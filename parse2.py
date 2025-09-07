import re
import json
from collections import defaultdict
from datetime import datetime

# File paths
AUDIT_LOG = "/home/opc/logai/enterprise_mnc_audit_sim/logs/aggregate/all_audit_logs.log"
AUTH_LOG = "/home/opc/logai/enterprise_mnc_audit_sim/logs/aggregate/all_auth_logs.log"

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
    r'(?P<timestamp>\w+\s+\d+\s[\d:]+).*audit: type=PATH.*item=\d+ name="(?P<name>[^"]+)".*mode=(?P<mode>\d+).*'
)
auth_re = re.compile(
    r'(?P<timestamp>\w+\s+\d+\s[\d:]+).*sshd\[\d+\]: (?P<event>.*for (?P<user>\w+) from [\d.]+).*'
)

# Function to detect file vs directory from mode
def mode_to_type(mode_str):
    mode = int(mode_str, 8)
    if (mode & 0o170000) == 0o040000:
        return "directory"
    elif (mode & 0o170000) == 0o100000:
        return "file"
    else:
        return "unknown"

# Step 1: Parse audit logs with PID correlation
audit_events = []
pid_to_last_syscall = {}

with open(AUDIT_LOG) as f:
    for line in f:
        sc_match = syscall_re.match(line)
        if sc_match:
            data = sc_match.groupdict()
            pid_to_last_syscall[data["pid"]] = data
            continue

        path_match = path_re.match(line)
        if path_match:
            path_data = path_match.groupdict()
            # correlate with nearest syscall by timestamp before this path
            path_time = datetime.strptime(path_data["timestamp"], "%b %d %H:%M:%S")
            closest_sc = None
            min_delta = None
            for sc in pid_to_last_syscall.values():
                sc_time = datetime.strptime(sc["timestamp"], "%b %d %H:%M:%S")
                delta = (path_time - sc_time).total_seconds()
                if delta >= 0 and (min_delta is None or delta < min_delta):
                    closest_sc = sc
                    min_delta = delta
            if closest_sc:
                user = uid_to_user.get(int(closest_sc["uid"]), f"uid_{closest_sc['uid']}")
                entry_type = mode_to_type(path_data["mode"])
                audit_events.append({
                    "timestamp": path_data["timestamp"],
                    "user": user,
                    "command": closest_sc["comm"],
                    "exe": closest_sc["exe"],
                    "file": path_data["name"],
                    "type": entry_type
                })

# Step 2: Parse auth logs
auth_events = []
with open(AUTH_LOG) as f:
    for line in f:
        match = auth_re.match(line)
        if match:
            data = match.groupdict()
            auth_events.append({
                "timestamp": data["timestamp"],
                "user": data["user"],
                "auth_event": data["event"]
            })

# Step 3: Combine and sort all events
all_events = sorted(audit_events + auth_events, key=lambda x: x["timestamp"])

# Step 4: JSON outputs
json_by_day = defaultdict(list)
json_by_user = defaultdict(list)
json_full = []

for e in all_events:
    day = e["timestamp"].split()[1]  # Day number
    if "command" in e:
        json_obj = {
            "user": e["user"], "command": e["command"], "exe": e["exe"],
            "file": e["file"], "type": e["type"]
        }
        json_by_day[day].append(json_obj)
        json_by_user[e["user"]].append({**json_obj, "timestamp": e["timestamp"]})
        json_full.append({**json_obj, "timestamp": e["timestamp"]})
    else:
        json_obj = {"user": e["user"], "auth_event": e["auth_event"]}
        json_by_day[day].append(json_obj)
        json_by_user[e["user"]].append({**json_obj, "timestamp": e["timestamp"]})
        json_full.append({**json_obj, "timestamp": e["timestamp"]})

# Write JSON files
OUT_DIR = "output_dir"
import os
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "json_day.json"), "w") as f:
    json.dump(json_by_day, f, indent=4)

with open(os.path.join(OUT_DIR, "json_user.json"), "w") as f:
    json.dump(json_by_user, f, indent=4)

with open(os.path.join(OUT_DIR, "json_full.json"), "w") as f:
    json.dump(json_full, f, indent=4)

# Step 5: NL outputs
nl_by_day = defaultdict(list)
nl_by_user = defaultdict(list)
nl_full = []

for e in all_events:
    if "command" in e:
        nl_entry = f"{e['timestamp']}: User {e['user']} executed '{e['command']}' ({e['exe']}) on {e['type']} '{e['file']}'"
    else:
        nl_entry = f"{e['timestamp']}: Authentication event for user {e['user']} - {e['auth_event']}"

    day = e["timestamp"].split()[1]
    nl_by_day[day].append(nl_entry)
    nl_by_user[e["user"]].append(nl_entry)
    nl_full.append(nl_entry)

with open(os.path.join(OUT_DIR, "nl_day.log"), "w") as f:
    for lines in nl_by_day.values():
        f.write("\n".join(lines) + "\n")

with open(os.path.join(OUT_DIR, "nl_user.log"), "w") as f:
    for lines in nl_by_user.values():
        f.write("\n".join(lines) + "\n")

with open(os.path.join(OUT_DIR, "nl_full.log"), "w") as f:
    f.write("\n".join(nl_full))

# Separate NL for auth logs only
auth_nl = [f"{e['timestamp']}: Authentication event for user {e['user']} - {e['auth_event']}" for e in auth_events]
with open(os.path.join(OUT_DIR, "nl_auth.log"), "w") as f:
    f.write("\n".join(auth_nl))

print("✅ All JSON and NL logs generated in", OUT_DIR)

