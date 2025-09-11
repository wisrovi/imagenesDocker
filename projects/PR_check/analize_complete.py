import argparse
import subprocess
import sys

import pandas as pd
from get_branches_opened import list_pr_branches, get_github_token
from PR_get import GitHubPRAnalyzer, process_and_save_report
from python_pylint import analyze_pull_request_status, PylintAnalysis, MINIMAL_SCORE


GITHUB_TOKEN = get_github_token()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lista las ramas de los Pull Requests abiertos en un repositorio de GitHub."
    )
    parser.add_argument(
        "--repo_owner",
        type=str,
        # required=True,
        default="cimacorporate",
        help="El nombre de usuario u organización del propietario del repositorio.",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        # required=True,
        default="001-AREPO",
        help="El nombre del repositorio de GitHub.",
    )
    args = parser.parse_args()

    REPO_OWNER = args.repo_owner
    REPO_NAME = args.repo_name

    pendientes = list_pr_branches(REPO_OWNER, REPO_NAME, GITHUB_TOKEN)

    command = ["git", "clone", f"git@github.com:{REPO_OWNER}/{REPO_NAME}.git", "/app"]

    try:
        report_result = subprocess.run(
            command,
            cwd="/app",
        )
    except:
        print("Fail clone")
        sys.exit(0)
        
    aproved_pr = {}

    for pendiente in pendientes:
        pr_number = pendiente["pr_number"]
        pr_title = pendiente["pr_title"]
        source_branch = pendiente["source_branch"]
        target_branch = pendiente["target_branch"]

        # print(f"\n- PR #{pr_number}: {pr_title}")
        # print(f"  Rama de origen: '{source_branch}'")
        # print(f"  Rama de destino: '{target_branch}'")

        branch = ["git", "checkout", f"{source_branch}"] # , "--depth", "1", "/app"
        
        
        try:
            report_result = subprocess.run(
                branch,
                cwd="/app",
            )
        except:
            print("Fail checkout")
            sys.exit(0)
            
        analyzer = GitHubPRAnalyzer(REPO_OWNER, REPO_NAME)
        pull_requests = analyzer.get_open_pull_requests()
        all_pr_data = []

        if pull_requests:
            # print(f"\n--- Processing Open Pull Requests in {REPO_OWNER}/{REPO_NAME} ---")
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

                # print(f"\n- PR #{pr_number}: {pr_title}")
                # print(f"  From branch: '{source_branch}' To branch: '{target_branch}'")
                # print(f"  URL: {pr_url}")
                # print(f"  Requested Reviewers: {', '.join(requested_reviewers_logins) if requested_reviewers_logins else 'None'}")
                # print(f"  Approvals: {approved_count}")
                # print(f"  Changes Requested: {has_changes_requested}")
                # print(f"  Overall Review Status: {review_status}")

                if files_data:
                    # print(f"  Modified files in PR #{pr_number}:")
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
                        # print(f"    - {filename} ({status}) - {additions} additions, {deletions} deletions")
                else:
                    # print(f"  No modified files found for PR #{pr_number}.")
                    pass

                # print("-" * 30)

            process_and_save_report(all_pr_data)

            
        PR_FILES_REPORT_CSV = "/report/pr_files_report.csv"
        PYLINT_SCORES_CSV = "/report/pr_files_report_pylint_scores.csv"
        PYLINT_RESULTS_CSV = "/report/pr_files_report_pylint.csv"    
        df = pd.read_csv(PR_FILES_REPORT_CSV)
        if df.empty:
            # print("The PR report CSV is empty. No files to analyze.")
            sys.exit(0)
            
        has_reviewer_approval, has_pending_changes = analyze_pull_request_status(df)
        python_files_to_check = df['filename'].dropna().unique()
        python_files_to_check = [f for f in python_files_to_check if isinstance(f, str) and f.lower().endswith(".py")]
        if not python_files_to_check:
            # print("No Python files found in the PR to check with Pylint.")
            sys.exit(0)
            
        pylint_analyzer = PylintAnalysis()
        all_results = []
        pylint_scores = {}
        total_score = 0

        for file_path in python_files_to_check:
            # print(f"Analyzing {file_path}...")
            results, score = pylint_analyzer.get_pylint_score_and_report(file_path)
            all_results.extend(results)
            pylint_scores[file_path] = score
            total_score += score

        # 5. Create and save new DataFrames
        df_scores = pd.DataFrame(list(pylint_scores.items()), columns=['filename', 'pylint_score'])
        df_merged = pd.merge(df, df_scores, on='filename', how='left')
        df_merged['pylint_score'] = df_merged['pylint_score'].fillna('N/A')

        df_merged.to_csv(PYLINT_SCORES_CSV, index=False)
        # print(f"\nReport with Pylint scores saved to '{PYLINT_SCORES_CSV}'.")
        
        df_raw_results = pd.DataFrame(all_results)
        if not df_raw_results.empty:
            df_raw_results.to_csv(PYLINT_RESULTS_CSV, index=False)
            # print(f"Raw Pylint report saved to '{PYLINT_RESULTS_CSV}'.")

        # 6. Print the final summary
        medium_score = total_score / len(python_files_to_check)
        meets_minimal_score = medium_score >= MINIMAL_SCORE
        can_be_approved = meets_minimal_score and has_reviewer_approval and not has_pending_changes
        
        # print("\n--- Pull Request Status Summary ---")
        # print(f"Number of Python files analyzed: {len(python_files_to_check)}")
        # print(f"Average Pylint Score: {medium_score:.2f}")
        # print(f"Meets Minimal Score ({MINIMAL_SCORE}): {meets_minimal_score}")
        # print(f"Has Reviewer Approval: {has_reviewer_approval}")
        # print(f"Has Pending Changes: {has_pending_changes}")
        # print(f"Pull Request is Approvable: {can_be_approved}")
        
        aproved_pr[pr_number] ={
            "can_be_approved": bool(can_be_approved),
            "has_pending_changes": bool(has_pending_changes),
            "has_reviewer_approval": bool(has_reviewer_approval),
            "medium_score": f"{medium_score}/{MINIMAL_SCORE}",
        }
        
    print(aproved_pr)
            


# example:
# python analize_complete.py --repo_owner cimacorporate --repo_name 001-AREPO
