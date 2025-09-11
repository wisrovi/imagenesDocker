import os
import sys
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv


def get_github_token() -> str:
    """
    Loads the GitHub token from environment variables.

    Returns:
        The GitHub Personal Access Token.

    Raises:
        SystemExit: If the GITHUB_PAT environment variable is not set.
    """
    load_dotenv("secrets.env")
    token = os.environ.get("GITHUB_PAT")
    if not token:
        print("Error: Falta GITHUB_PAT. No se puede conectar a la API de GitHub.")
        sys.exit(1)
    return token


def list_pr_branches(owner: str, repo: str, token: str):
    """
    Lists the branches for all open Pull Requests in a repository.

    Args:
        owner: The GitHub repository owner's username or organization.
        repo: The name of the GitHub repository.
        token: The GitHub Personal Access Token for authentication.
    """
    print(f"Listando ramas de PRs abiertos para {owner}/{repo}...")

    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=open"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        pull_requests = response.json()

        pendientes = []

        if not pull_requests:
            print("No hay Pull Requests abiertos en este repositorio.")
            return pendientes

        # print(f"\n--- Ramas de los Pull Requests Abiertos en {owner}/{repo} ---")

        for pr in pull_requests:
            pr_number = pr.get("number")
            pr_title = pr.get("title")
            source_branch = pr["head"]["ref"]
            target_branch = pr["base"]["ref"]

            pendientes.append(
                {
                    "pr_number": pr_number,
                    "pr_title": pr_title,
                    "source_branch": source_branch,
                    "target_branch": target_branch,
                }
            )

        return pendientes

    except requests.exceptions.RequestException as e:
        print(f"Error al contactar la API de GitHub: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    REPO_OWNER = "cimacorporate"
    REPO_NAME = "001-AREPO"
    GITHUB_TOKEN = get_github_token()
    pendientes = list_pr_branches(REPO_OWNER, REPO_NAME, GITHUB_TOKEN)

    for pendiente in pendientes:
        pr_number = pendiente["pr_number"]
        pr_title = pendiente["pr_title"]
        source_branch = pendiente["source_branch"]
        target_branch = pendiente["target_branch"]

        print(f"\n- PR #{pr_number}: {pr_title}")
        print(f"  Rama de origen: '{source_branch}'")
        print(f"  Rama de destino: '{target_branch}'")
