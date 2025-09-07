from collections import Counter

file_path = "/home/opc/logai/output/nl_full1.log"
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = [line.rstrip("\n").strip() for line in f if line.strip()]

# Count occurrences
line_counts = Counter(lines)

# Keep only duplicates
duplicates = {line: count for line, count in line_counts.items() if count > 1}

# Sort by count descending
for line, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
    print(f"{count} | {line}")

