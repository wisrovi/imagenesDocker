import os
import sys

MINIMAL_SCORE = 8.5

# File paths used in the analysis
PR_FILES_REPORT_CSV = "/report/pr_files_report.csv"
PYLINT_SCORES_CSV = "/report/pr_files_report_pylint_scores.csv"
PYLINT_RESULTS_CSV = "/report/pr_files_report_pylint.csv"

def load_config():
    """
    Loads configuration from environment variables.
    """
    repo_owner = os.environ.get("REPO_OWNER")
    repo_name = os.environ.get("REPO_NAME")

    if not repo_owner or not repo_name:
        print("Error: REPO_OWNER and REPO_NAME environment variables are required.", file=sys.stderr)
        sys.exit(1)

    return {
        "repo_owner": repo_owner,
        "repo_name": repo_name
    }