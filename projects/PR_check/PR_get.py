import os
import sys
from typing import Dict, List, Any

import pandas as pd
import requests
from dotenv import load_dotenv

class GitHubPRAnalyzer:
    """
    A class to interact with the GitHub API to analyze Pull Requests.

    This class handles authentication, fetching PRs, and retrieving detailed
    information about their reviews and modified files.
    """
    def __init__(self, owner: str, repo: str):
        """
        Initializes the GitHubPRAnalyzer with repository information and API token.

        Args:
            owner: The GitHub repository owner's username or organization name.
            repo: The name of the GitHub repository.
        """
        self.owner = owner
        self.repo = repo
        self.github_token = self._get_github_token()
        self.headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.base_url = f"https://api.github.com/repos/{self.owner}/{self.repo}"

    def _get_github_token(self) -> str:
        """
        Loads the GitHub token from environment variables.

        Returns:
            The GitHub Personal Access Token.

        Raises:
            SystemExit: If the GITHUB_PAT environment variable is not set.
        """
        load_dotenv('secrets.env')
        token = os.environ.get("GITHUB_PAT")
        if not token:
            print("Error: GITHUB_PAT environment variable is missing. Cannot connect to the GitHub API.")
            sys.exit(1)
        return token

    def get_open_pull_requests(self) -> List[Dict[str, Any]]:
        """
        Fetches all open Pull Requests from the repository.

        Returns:
            A list of dictionaries, where each dictionary represents an open PR.
            Returns an empty list if no open PRs are found.
        
        Raises:
            requests.exceptions.RequestException: If the API call fails.
        """
        print(f"Listing open PRs for {self.owner}/{self.repo}...")
        api_url = f"{self.base_url}/pulls?state=open"
        
        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()
            pull_requests = response.json()
            if not pull_requests:
                print("No open Pull Requests found in this repository.")
            return pull_requests
        except requests.exceptions.RequestException as e:
            print(f"Error contacting the GitHub API: {e}")
            sys.exit(1)
            
    def get_pr_details(self, pr_number: int) -> Dict[str, Any]:
        """
        Gets detailed information for a specific Pull Request, including reviews and files.

        Args:
            pr_number: The number of the Pull Request.

        Returns:
            A dictionary with the PR details, including approval count, changes requested status,
            and a list of modified files.

        Raises:
            requests.exceptions.RequestException: If any of the nested API calls fail.
        """
        try:
            # Get PR review details
            reviews_url = f"{self.base_url}/pulls/{pr_number}/reviews"
            reviews_response = requests.get(reviews_url, headers=self.headers)
            reviews_response.raise_for_status()
            reviews = reviews_response.json()

            approved_count = 0
            has_changes_requested = False
            for review in reviews:
                state = review.get("state")
                if state == "APPROVED":
                    approved_count += 1
                elif state == "CHANGES_REQUESTED":
                    has_changes_requested = True

            # Get PR file details
            files_url = f"{self.base_url}/pulls/{pr_number}/files"
            files_response = requests.get(files_url, headers=self.headers)
            files_response.raise_for_status()
            files = files_response.json()

            return {
                "approved_count": approved_count,
                "has_changes_requested": 'Sí' if has_changes_requested else 'No',
                "files": files
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching details for PR #{pr_number}: {e}", file=sys.stderr)
            return {
                "approved_count": 0,
                "has_changes_requested": 'Error',
                "files": []
            }

def process_and_save_report(pr_data: List[Dict[str, Any]]):
    """
    Processes the raw PR data, creates a pandas DataFrame, and saves it to a CSV file.

    Args:
        pr_data: A list of dictionaries containing all collected PR information.
    """
    if not pr_data:
        print("No modified file data found in any open PR.")
        return
        
    df = pd.DataFrame(pr_data)
    output_csv_file = "/report/pr_files_report.csv"
    df.to_csv(output_csv_file, index=False)
    print(f"\nComplete report successfully generated at '{output_csv_file}'!")

def main():
    """
    Main function to orchestrate the PR analysis workflow.
    """
    REPO_OWNER = "cimacorporate"
    REPO_NAME = "001-AREPO"

    analyzer = GitHubPRAnalyzer(REPO_OWNER, REPO_NAME)
    pull_requests = analyzer.get_open_pull_requests()
    all_pr_data = []

    if pull_requests:
        print(f"\n--- Processing Open Pull Requests in {REPO_OWNER}/{REPO_NAME} ---")
        for pr in pull_requests:
            pr_number = pr.get("number")
            pr_title = pr.get("title")
            pr_url = pr.get("html_url")
            source_branch = pr['head']['ref']
            target_branch = pr['base']['ref']
            requested_reviewers_logins = [r.get('login') for r in pr.get('requested_reviewers', [])]

            details = analyzer.get_pr_details(pr_number)
            approved_count = details["approved_count"]
            has_changes_requested = details["has_changes_requested"]
            files_data = details["files"]

            # Determine overall review status for console output
            review_status = "No reviews"
            if requested_reviewers_logins:
                review_status = f"Requested: {', '.join(requested_reviewers_logins)}"
            if approved_count > 0:
                review_status = f"Approved ({approved_count} approvals)"
            if has_changes_requested == 'Sí':
                review_status = "Changes Requested"

            print(f"\n- PR #{pr_number}: {pr_title}")
            print(f"  From branch: '{source_branch}' To branch: '{target_branch}'")
            print(f"  URL: {pr_url}")
            print(f"  Requested Reviewers: {', '.join(requested_reviewers_logins) if requested_reviewers_logins else 'None'}")
            print(f"  Approvals: {approved_count}")
            print(f"  Changes Requested: {has_changes_requested}")
            print(f"  Overall Review Status: {review_status}")

            if files_data:
                print(f"  Modified files in PR #{pr_number}:")
                for file_info in files_data:
                    filename = file_info.get("filename")
                    status = file_info.get("status")
                    additions = file_info.get("additions", 0)
                    deletions = file_info.get("deletions", 0)
                    changes = file_info.get("changes", 0)
                    
                    all_pr_data.append({
                        "pr_number": pr_number,
                        "pr_title": pr_title,
                        "pr_url": pr_url,
                        "source_branch": source_branch,
                        "target_branch": target_branch,
                        "requested_reviewers": ', '.join(requested_reviewers_logins) if requested_reviewers_logins else 'None',
                        "review_status": review_status,
                        "approved_count": approved_count,
                        "changes_requested": has_changes_requested,
                        "filename": filename,
                        "status": status,
                        "additions": additions,
                        "deletions": deletions,
                        "changes": changes
                    })
                    print(f"    - {filename} ({status}) - {additions} additions, {deletions} deletions")
            else:
                print(f"  No modified files found for PR #{pr_number}.")

            print("-" * 30)

    process_and_save_report(all_pr_data)

if __name__ == "__main__":
    main()
