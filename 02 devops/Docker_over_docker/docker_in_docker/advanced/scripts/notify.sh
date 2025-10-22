#!/bin/bash

# Notification Script for Docker-in-Docker project
# Sends notifications via email and Slack

set -e

# Load environment variables
SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-}
EMAIL_SMTP_SERVER=${EMAIL_SMTP_SERVER:-}
EMAIL_SMTP_PORT=${EMAIL_SMTP_PORT:-587}
EMAIL_USERNAME=${EMAIL_USERNAME:-}
EMAIL_PASSWORD=${EMAIL_PASSWORD:-}
EMAIL_FROM=${EMAIL_FROM:-admin@example.com}
EMAIL_TO=${EMAIL_TO:-admin@example.com}

MESSAGE=$1
SUBJECT=${2:-"Docker-in-Docker Notification"}
LEVEL=${3:-info}

# Colors for console output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Slack notification function
send_slack_notification() {
    local message="$1"
    local level="$2"

    if [ -z "$SLACK_WEBHOOK_URL" ]; then
        echo "Slack webhook URL not configured"
        return 1
    fi

    local color
    case $level in
        error) color="danger" ;;
        warning) color="warning" ;;
        success) color="good" ;;
        *) color="#808080" ;;
    esac

    local payload=$(cat <<EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "Docker-in-Docker $level",
            "text": "$message",
            "footer": "Docker-in-Docker Monitoring",
            "ts": $(date +%s)
        }
    ]
}
EOF
    )

    if curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" >/dev/null; then
        echo -e "${GREEN}✅ Slack notification sent${NC}"
    else
        echo -e "${RED}❌ Failed to send Slack notification${NC}"
    fi
}

# Email notification function
send_email_notification() {
    local subject="$1"
    local message="$2"

    if [ -z "$EMAIL_SMTP_SERVER" ] || [ -z "$EMAIL_USERNAME" ]; then
        echo "Email configuration not complete"
        return 1
    fi

    # Create email content
    local email_content=$(cat <<EOF
From: $EMAIL_FROM
To: $EMAIL_TO
Subject: $subject
Content-Type: text/plain; charset=UTF-8

$message

--
Docker-in-Docker Monitoring System
Generated at: $(date)
EOF
    )

    # Send email using curl (for SMTP)
    if echo "$email_content" | curl -s --url "smtp://$EMAIL_SMTP_SERVER:$EMAIL_SMTP_PORT" \
        --mail-from "$EMAIL_FROM" \
        --mail-rcpt "$EMAIL_TO" \
        --user "$EMAIL_USERNAME:$EMAIL_PASSWORD" \
        --insecure \
        --upload-file - >/dev/null; then
        echo -e "${GREEN}✅ Email notification sent${NC}"
    else
        echo -e "${RED}❌ Failed to send email notification${NC}"
    fi
}

# Main notification logic
echo "📢 Sending notifications..."

# Send Slack notification
send_slack_notification "$MESSAGE" "$LEVEL"

# Send email notification
send_email_notification "$SUBJECT" "$MESSAGE"

echo -e "${GREEN}📢 Notification process completed${NC}"