import os
import sys
from typing import Dict, List, Any

import requests
from dotenv import load_dotenv

class GitHubClient:
    """
    A client to interact with the GitHub API.
    """
    def __init__(self, owner: str, repo: str):
        """
        Initializes the GitHubClient.

        Args:
            owner: The repository owner.
            repo: The repository name.
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
        Loads the GitHub token from the 'secrets.env' file.
        """
        load_dotenv('/config/secrets.env')
        token = os.environ.get("GITHUB_PAT")
        if not token:
            print("Error: GITHUB_PAT environment variable is missing.", file=sys.stderr)
            sys.exit(1)
        return token

    def get_open_pull_requests(self) -> List[Dict[str, Any]]:
        """
        Fetches all open Pull Requests from the repository.
        """
        print(f"Fetching open PRs for {self.owner}/{self.repo}...")
        api_url = f"{self.base_url}/pulls?state=open"
        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching open PRs: {e}", file=sys.stderr)
            sys.exit(1)

    def get_pull_request(self, pr_number: int) -> Dict[str, Any]:
        """
        Fetches a single pull request from a repository.
        """
        url = f"{self.base_url}/pulls/{pr_number}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching PR #{pr_number}: {e}", file=sys.stderr)
            return None

    def get_pr_details(self, pr_number: int) -> Dict[str, Any]:
        """
        Gets detailed information for a specific Pull Request.
        """
        try:
            reviews_url = f"{self.base_url}/pulls/{pr_number}/reviews"
            reviews_response = requests.get(reviews_url, headers=self.headers)
            reviews_response.raise_for_status()
            reviews = reviews_response.json()

            approved_count = sum(1 for r in reviews if r.get("state") == "APPROVED")
            has_changes_requested = any(r.get("state") == "CHANGES_REQUESTED" for r in reviews)

            files_url = f"{self.base_url}/pulls/{pr_number}/files"
            files_response = requests.get(files_url, headers=self.headers)
            files_response.raise_for_status()
            files = files_response.json()

            return {
                "approved_count": approved_count,
                "has_changes_requested": has_changes_requested,
                "files": files
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching details for PR #{pr_number}: {e}", file=sys.stderr)
            return {"approved_count": 0, "has_changes_requested": False, "files": []}

    def get_file_content(self, url: str) -> str:
        """
        Fetches the content of a file from a raw URL.
        """
        print(f"Fetching file content from: {url}") # Added for debugging
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching file content from {url}: {e}", file=sys.stderr)
            return None