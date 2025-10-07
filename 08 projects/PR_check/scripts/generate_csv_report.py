import os
import json
import csv

def generate_csv_report():
    """
    Reads all JSON reports from the /report directory and generates a global CSV report.
    """
    all_pr_summaries = []
    report_base_dir = '/report'

    for root, dirs, files in os.walk(report_base_dir):
        for file in files:
            if file == 'report.json':
                json_path = os.path.join(root, file)
                with open(json_path, 'r') as f:
                    repo_prs = json.load(f)
                    all_pr_summaries.extend(repo_prs)

    csv_path = os.path.join(report_base_dir, "report.csv")

    if not all_pr_summaries:
        with open(csv_path, "w", newline="") as f:
            f.write("")
        print("No PR summaries found to generate CSV. CSV report cleared.")
        return

    # Ensure all dictionaries have the same keys for CSV header
    # This is a simple approach, might need more robust handling for complex/missing fields
    fieldnames = set()
    for pr_summary in all_pr_summaries:
        fieldnames.update(pr_summary.keys())
    
    # Sort fieldnames for consistent CSV output
    fieldnames = sorted(list(fieldnames))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_pr_summaries)
    
    print(f"Global CSV report generated at {csv_path}")

if __name__ == "__main__":
    generate_csv_report()
