import json
import csv
import os
from datetime import datetime
from pylint.lint import Run
from pylint.reporters.text import TextReporter
from io import StringIO

MINIMAL_PYLINT_SCORE = 8.5

def run_pylint_on_content(file_content):
    """
    Runs pylint on a given file content and returns the score.
    """
    # Create a temporary file to run pylint on
    with open("temp_pylint_file.py", "w") as f:
        f.write(file_content)

    reporter_output = StringIO()
    reporter = TextReporter(reporter_output)
    
    Run(['temp_pylint_file.py'], reporter=reporter, exit=False)
    
    output = reporter_output.getvalue()
    print(f"Pylint output: {output}")  # Re-enabled for debugging
    score_line = [line for line in output.split('\n') if 'Your code has been rated at' in line]
    
    os.remove("temp_pylint_file.py")

    if score_line:
        try:
            score = float(score_line[0].split(' ')[-2].split('/')[0])
            return score
        except (ValueError, IndexError):
            return 0.0
    return 0.0

def analyze_pull_request(pr_data, details, client):
    """
    Analyzes a pull request data and returns a summary.
    """
    pylint_scores = []
    for file in details.get('files', []):
        if file['filename'].endswith('.py'):
            content = client.get_file_content(file['raw_url'])
            if content:
                score = run_pylint_on_content(content)
                pylint_scores.append(score)

    pylint_score = sum(pylint_scores) / len(pylint_scores) if pylint_scores else 0.0
    
    has_at_least_one_approval = details["approved_count"] > 0
    CAN_APPROVE_PR = (
        not details["has_changes_requested"] and
        has_at_least_one_approval and
        pylint_score > MINIMAL_PYLINT_SCORE
    )

    analysis_summary = {
        "pr_number": pr_data["number"],
        "title": pr_data["title"],
        "author": pr_data["user"]["login"],
        "created_at": pr_data["created_at"],
        "updated_at": pr_data["updated_at"],
        "mergeable": pr_data["mergeable"],
        "additions": pr_data["additions"],
        "deletions": pr_data["deletions"],
        "changed_files": pr_data["changed_files"],
        "url": pr_data["html_url"],
        "approved_count": details["approved_count"],
        "has_changes_requested": details["has_changes_requested"],
        "pylint_score": pylint_score,
        "has_at_least_one_approval": has_at_least_one_approval,
        "CAN_APPROVE_PR": CAN_APPROVE_PR,
    }
    return analysis_summary

def save_repo_report(repo_owner, repo_name, all_pr_summaries):
    """
    Saves the analysis report for the entire repo to a JSON file.
    """
    report_dir = f'/report/{repo_owner}/{repo_name}'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    report_path = os.path.join(report_dir, "report.json")
    
    with open(report_path, "w") as f:
        json.dump(all_pr_summaries, f, indent=4)
    print(f"Repository report saved to {report_path}")