import datetime
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from filelock import FileLock

# Define the directory for results, ensuring it exists.
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Define the schema based on CLASSIFICATION_RUNS.md, adapted for the multi-agent architecture.
EXCEL_COLUMNS = [
    "timestamp",
    "run_id",
    "document_id",
    "document_name",
    "supervisor_model",
    "agent_a_model",
    "agent_b_model",
    "supervisor_temp",
    "agent_a_temp",
    "agent_b_temp",
    "prompt_version",
    "classification",
    "confidence",
    "consensus_reached",
    "rounds_used",
    "estimated_cost",
    "risk_cost",
    "risk_flag",
    "input_tokens",
    "output_tokens",
    "ground_truth",  # For manual entry later
]


def log_run_to_excel(run_data: Dict[str, Any]):
    """
    Appends a single classification run to a daily Excel file.

    This function is thread-safe and process-safe, preventing data corruption
    from concurrent writes.

    Args:
        run_data: A dictionary containing the data for a single run,
                  matching the EXCEL_COLUMNS schema.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    excel_path = RESULTS_DIR / f"results_{today}.xlsx"
    lock_path = RESULTS_DIR / f"results_{today}.xlsx.lock"

    lock = FileLock(lock_path, timeout=15)

    with lock:
        try:
            # Load existing data if file exists
            df = pd.read_excel(excel_path)
        except FileNotFoundError:
            # Create a new DataFrame if file doesn't exist
            df = pd.DataFrame(columns=EXCEL_COLUMNS)

        # Flatten model_parameters from the input dict
        flat_run_data = run_data.copy()
        if 'model_parameters' in flat_run_data and isinstance(flat_run_data['model_parameters'], dict):
            params = flat_run_data.pop('model_parameters')
            flat_run_data.update(params)

        # Prepare the new row, ensuring it only contains defined columns
        new_row_filtered = {k: flat_run_data.get(k) for k in EXCEL_COLUMNS}

        new_row_df = pd.DataFrame([new_row_filtered])

        # Append the new row
        df = pd.concat([df, new_row_df], ignore_index=True)

        # Write back to the Excel file using the openpyxl engine
        df.to_excel(excel_path, index=False, engine="openpyxl")