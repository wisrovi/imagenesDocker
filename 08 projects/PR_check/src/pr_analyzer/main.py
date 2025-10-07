import os
from .github_client import GitHubClient
from .analysis import analyze_pull_request, save_repo_report
from .config import load_config

def main():
    """
    Main function to run the PR analysis for all open PRs.
    """
    config = load_config()
    repo_owner = config["repo_owner"]
    repo_name = config["repo_name"]

    client = GitHubClient(owner=repo_owner, repo=repo_name)
    open_prs = client.get_open_pull_requests()

    all_pr_summaries = []
    if open_prs:
        print(f"Found {len(open_prs)} open pull requests.")
        for pr_summary in open_prs:
            pr_number = pr_summary["number"]
            print(f"Analyzing PR #{pr_number}...\n") # Added newline for better readability
            pr_data = client.get_pull_request(pr_number)
            if pr_data:
                details = client.get_pr_details(pr_number)
                analysis_summary = analyze_pull_request(pr_data, details, client)
                all_pr_summaries.append(analysis_summary)
    else:
        print("No open pull requests found.")

    if all_pr_summaries:
        save_repo_report(repo_owner, repo_name, all_pr_summaries)
        
    print("PR analysis complete.")

if __name__ == "__main__":
    main()
