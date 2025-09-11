# import json
# import subprocess

# import pandas as pd

# from config import MINIMAL_SCORE


# def get_pylint_score(file_path):
#     """
#     Ejecuta Pylint en un archivo dado y devuelve su puntaje.
#     """

#     file_path_to_check = f"/app/{file_path}"
#     command_list = ['pylint', '--output-format=json', file_path_to_check]

#     try:
#         result = subprocess.run(
#             command_list,  # Pasa la lista directamente
#             capture_output=True,
#             text=True,
#             check=True,  # Esto lanzará una excepción si pylint falla
#             shell=False,  # No es necesario usar shell=True si pasas una lista
#             cwd="/app"
#         )
#         problems = result.stdout
#     except subprocess.CalledProcessError as e:
#         problems = e.stdout
#     except FileNotFoundError:
#         print("Error: 'pylint' no se encontró. Asegúrate de que esté instalado y en tu PATH.")

#     output_lines = problems.strip().split('\n')
#     output_lines = " ".join(output_lines)
#     data = json.loads(output_lines)

#     command_list_score = ['pylint', '--reports=y', '--output-format=text', file_path_to_check]
#     try:
#         result_score = subprocess.run(
#             command_list_score,
#             capture_output=True,
#             text=True,
#             check=False,
#             shell=False,
#             cwd="/app"
#         )
#         score_output = result_score.stdout
#         # Buscar la línea que contiene el puntaje
#         for line in score_output.splitlines():
#             if 'Your code has been rated at' in line:
#                 score = line.split(': ')[-1].split('/')[0].strip()
#                 break
#     except subprocess.CalledProcessError as e:
#         print(f"Error obteniendo el puntaje de Pylint para {file_path}: {e}")
#         score = 'Error'

#     return data, float(score)


# # Lista de tus archivos
# tiene_alguna_aprobacion_de_revisor = True
# tiene_pendiente_cambios = False

# CSV_FILENAME = "/app/pr_files_report.csv"
# df = pd.read_csv(CSV_FILENAME)

# approved_count = df['approved_count'].dropna().unique()
# if approved_count == 0:
#     tiene_alguna_aprobacion_de_revisor = False
    
# changes_requested = df['changes_requested'].dropna().unique()
# if changes_requested != "No":
#     tiene_pendiente_cambios = True


# python_files_to_check = df['filename'].dropna().unique()
# python_files_to_check = [f for f in python_files_to_check if isinstance(f, str) and f.lower().endswith(".py")]



# medium_score = 0
# pylint_scores = {}

# all_results = []

# # Obtiene y muestra el puntaje para cada archivo
# for file_path in python_files_to_check:
#     pylint_results, score = get_pylint_score(file_path)

#     for pylint_reesult in pylint_results:
#         all_results.append(pylint_reesult)

#     pylint_scores[file_path] = score
#     medium_score += score

# df_scores = pd.DataFrame(list(pylint_scores.items()), columns=['filename', 'pylint_score'])
# df_merged = pd.merge(df, df_scores, on='filename', how='left')
# df_merged['pylint_score'] = df_merged['pylint_score'].fillna('N/A')

# final_csv_filename = "/app/pr_files_report_pylint_scores.csv"
# df_merged.to_csv(final_csv_filename, index=False)

# file_csv_results_pylint = "/app/pr_files_report_pylint.csv"
# df = pd.DataFrame(all_results)
# df.to_csv(file_csv_results_pylint)

# medium_score = medium_score / len(python_files_to_check)
# print(f"Medium score of PR is {medium_score}")
# tiene_minimo_score = medium_score>=MINIMAL_SCORE

# print(f"Cumple el minimo score: {tiene_minimo_score}")
# print(f"tiene_alguna_aprobacion_de_revisor: {tiene_alguna_aprobacion_de_revisor}")
# print(f"tiene_pendiente_cambios: {tiene_pendiente_cambios}")

# print(f"El PR se puede aprobar: {tiene_minimo_score and tiene_alguna_aprobacion_de_revisor and not tiene_pendiente_cambios}")
























import json
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

# The MINIMAL_SCORE constant would typically be imported from a config file.
# We'll define it here for this example.
MINIMAL_SCORE = 8.0  # Example minimal score

class PylintAnalysis:
    """
    A class to encapsulate the Pylint analysis logic.
    
    This class handles the execution of Pylint on a given file, parsing the
    output, and extracting both the raw report and the final score.
    """
    def __init__(self):
        """Initializes the PylintAnalysis instance."""
        pass

    def get_pylint_score_and_report(self, file_path: str) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes Pylint on a file and returns the raw JSON report and the score.

        This method makes two subprocess calls: one to get the JSON report
        and another to get the final score text. This is a common pattern as
        Pylint's JSON output does not always include the final score.

        Args:
            file_path: The path to the Python file to analyze.

        Returns:
            A tuple containing:
                - A list of dictionaries representing the raw Pylint report.
                - The final Pylint score as a float.

        Raises:
            subprocess.CalledProcessError: If Pylint fails to run.
            FileNotFoundError: If the 'pylint' command is not found.
            json.JSONDecodeError: If the Pylint JSON output is malformed.
        """
        file_path_to_check = f"/report/{file_path}"
        pylint_report = []
        pylint_score = 0.0

        try:
            # First call to get the JSON output for the report
            report_command = ['pylint', '--output-format=json', file_path_to_check]
            report_result = subprocess.run(
                report_command,
                capture_output=True,
                text=True,
                # check=True,
                cwd="/app"
            )
            report_output = report_result.stdout
            pylint_report = json.loads(report_output)

            # Second call to get the score from the text report
            score_command = ['pylint', '--reports=y', '--output-format=text', file_path_to_check]
            score_result = subprocess.run(
                score_command,
                capture_output=True,
                text=True,
                check=False,
                cwd="/app"
            )
            score_output = score_result.stdout
            for line in score_output.splitlines():
                if 'Your code has been rated at' in line:
                    score_str = line.split(': ')[-1].split('/')[0].strip()
                    pylint_score = float(score_str)
                    break

        except subprocess.CalledProcessError as e:
            print(f"Pylint analysis failed for {file_path}. Error: {e.stderr}", file=sys.stderr)
            # Try to get partial report from stdout if an error occurred
            try:
                pylint_report = json.loads(e.stdout)
            except json.JSONDecodeError:
                pylint_report = []
            return pylint_report, 0.0
        except FileNotFoundError:
            print("Error: 'pylint' command not found. Please ensure it is installed.", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing Pylint JSON output for {file_path}: {e}", file=sys.stderr)
            return [], 0.0

        return pylint_report, pylint_score

def analyze_pull_request_status(df: pd.DataFrame) -> Tuple[bool, bool]:
    """
    Analyzes the DataFrame to determine PR approval and changes requested status.

    Args:
        df: The DataFrame containing PR file information.

    Returns:
        A tuple containing:
            - A boolean indicating if the PR has any reviewer approvals.
            - A boolean indicating if the PR has any requested changes.
    """
    has_reviewer_approval = (df['approved_count'] > 0).any()
    has_pending_changes = (df['changes_requested'] == 'Sí').any()
    return has_reviewer_approval, has_pending_changes

def main():
    """
    Main function to run the Pylint analysis workflow.

    This function orchestrates the entire process: reading the PR report CSV,
    running Pylint on relevant files, generating new CSVs with scores and
    raw results, and printing a final summary.
    """
    PR_FILES_REPORT_CSV = "/report/pr_files_report.csv"
    PYLINT_SCORES_CSV = "/report/pr_files_report_pylint_scores.csv"
    PYLINT_RESULTS_CSV = "/report/pr_files_report_pylint.csv"

    # 1. Read the PR report CSV
    try:
        df = pd.read_csv(PR_FILES_REPORT_CSV)
    except FileNotFoundError:
        print(f"Error: The file '{PR_FILES_REPORT_CSV}' was not found.", file=sys.stderr)
        sys.exit(1)
    
    if df.empty:
        print("The PR report CSV is empty. No files to analyze.")
        sys.exit(0)

    # 2. Analyze PR status for approvals and changes
    has_reviewer_approval, has_pending_changes = analyze_pull_request_status(df)
    
    # 3. Filter for Python files
    python_files_to_check = df['filename'].dropna().unique()
    python_files_to_check = [f for f in python_files_to_check if isinstance(f, str) and f.lower().endswith(".py")]

    if not python_files_to_check:
        print("No Python files found in the PR to check with Pylint.")
        sys.exit(0)

    # 4. Run Pylint on each Python file
    pylint_analyzer = PylintAnalysis()
    all_results = []
    pylint_scores = {}
    total_score = 0

    for file_path in python_files_to_check:
        print(f"Analyzing {file_path}...")
        results, score = pylint_analyzer.get_pylint_score_and_report(file_path)
        all_results.extend(results)
        pylint_scores[file_path] = score
        total_score += score

    # 5. Create and save new DataFrames
    df_scores = pd.DataFrame(list(pylint_scores.items()), columns=['filename', 'pylint_score'])
    df_merged = pd.merge(df, df_scores, on='filename', how='left')
    df_merged['pylint_score'] = df_merged['pylint_score'].fillna('N/A')

    df_merged.to_csv(PYLINT_SCORES_CSV, index=False)
    print(f"\nReport with Pylint scores saved to '{PYLINT_SCORES_CSV}'.")
    
    df_raw_results = pd.DataFrame(all_results)
    if not df_raw_results.empty:
        df_raw_results.to_csv(PYLINT_RESULTS_CSV, index=False)
        print(f"Raw Pylint report saved to '{PYLINT_RESULTS_CSV}'.")

    # 6. Print the final summary
    medium_score = total_score / len(python_files_to_check)
    meets_minimal_score = medium_score >= MINIMAL_SCORE
    can_be_approved = meets_minimal_score and has_reviewer_approval and not has_pending_changes
    
    print("\n--- Pull Request Status Summary ---")
    print(f"Number of Python files analyzed: {len(python_files_to_check)}")
    print(f"Average Pylint Score: {medium_score:.2f}")
    print(f"Meets Minimal Score ({MINIMAL_SCORE}): {meets_minimal_score}")
    print(f"Has Reviewer Approval: {has_reviewer_approval}")
    print(f"Has Pending Changes: {has_pending_changes}")
    print(f"Pull Request is Approvable: {can_be_approved}")

if __name__ == "__main__":
    main()