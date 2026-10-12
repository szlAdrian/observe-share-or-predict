#!/usr/bin/env python3
"""
Metrics Summarizer Calculator
Reads episode_result_log.csv and calculates summarized_metrics.
Excludes: episode_id, instruction, run_id.
Skips rows that don't match the header column count.
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple
import io
import sys


def compute_metrics_summary(csv_file_path: str) -> Dict[str, float]:
    """
    Read CSV file and compute and the summarized metrics for numeric columns.
    
    Args:
        csv_file_path: Path to the episode_result_log.csv file
        
    Returns:
        Dictionary mapping column names to their average values
    """
    csv_path = Path(csv_file_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    # Columns to exclude
    exclude_columns = {"episode_id", "instruction", "run_id"}
    
    headers = []
    columns_to_average = []  # indices of columns to average
    column_sums = {}  # column_name -> sum of values
    column_counts = {}  # column_name -> count of non-empty values
    skipped_rows = 0
    processed_rows = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        
        # Read header row
        header_row = next(reader)
        headers = header_row
        expected_column_count = len(headers)
        
        # Identify columns to average
        for idx, col_name in enumerate(headers):
            #print(f"Processing column: {col_name} (index {idx})")
            # Skip if in exclude list or contains "accuracy"
            if col_name not in exclude_columns and "accuracy" not in col_name:
                columns_to_average.append(idx)
                column_sums[col_name] = 0.0
                column_counts[col_name] = 0
        
        # Print excluded and included columns
        excluded_cols = exclude_columns | {col for col in headers if "accuracy" in col}
        included_cols = [headers[idx] for idx in columns_to_average]
        print(f"Excluded columns from averaging: {sorted(excluded_cols)}")
        print(f"Included columns for averaging: {sorted(included_cols)}")
        print()
        
        # Process data rows
        for row in reader:
            # Skip rows with mismatched column count
            if len(row) != expected_column_count:
                skipped_rows += 1
                continue
            
            processed_rows += 1
            
            # Process each column to average
            for col_idx in columns_to_average:
                col_name = headers[col_idx]
                value_str = row[col_idx].strip()
                
                # Skip empty values
                if value_str == "":
                    continue
                
                try:
                    value = float(value_str)
                    column_sums[col_name] += value
                    column_counts[col_name] += 1
                except ValueError:
                    # Skip non-numeric values
                    pass
    
    # Calculate averages for the non-accuracy columns
    summarized_metrics = {}
    for col_name in column_sums.keys():
        if column_counts[col_name] > 0:
            summarized_metrics[col_name] = column_sums[col_name] / column_counts[col_name]
        else:
            summarized_metrics[col_name] = 0.0
    
    agent_ids = [0,1]
    
    accuracy_nominator_denominator_name_triplets  = [
        ("oaa_pred_action_accuracy_", "oaa_correct_action_pred_count_", "oaa_total_pred_count_"),
        ("oaa_pred_action_accuracy_curr_next_", "oaa_correct_action_pred_count_curr_next_", "oaa_total_pred_count_"),
        ("oaa_pred_principal_accuracy_", "oaa_correct_principal_pred_count_", "oaa_total_pred_count_"),
        ("oaa_pred_principal_accuracy_curr_next_", "oaa_correct_principal_pred_count_curr_next_", "oaa_total_pred_count_"),
        ("oaa_pred_action_principal_accuracy_", "oaa_correct_action_principal_pred_count_", "oaa_total_pred_count_"),
        ("oaa_pred_action_principal_accuracy_curr_next_", "oaa_correct_action_principal_pred_count_curr_next_", "oaa_total_pred_count_"),
        ("oaa_pred_all_args_accuracy_", "oaa_correct_all_args_pred_count_", "oaa_total_pred_count_"),
        ("oaa_pred_all_args_accuracy_curr_next_", "oaa_correct_all_args_pred_count_curr_next_", "oaa_total_pred_count_"),
        ("oaa_pred_action_all_args_accuracy_", "oaa_correct_action_all_args_pred_count_", "oaa_total_pred_count_"),
        ("oaa_pred_action_all_args_accuracy_curr_next_", "oaa_correct_action_all_args_pred_count_curr_next_", "oaa_total_pred_count_"),
        ("oaa_pred_match_success_execution_accuracy_", "oaa_pred_match_success_execution_count_", "oaa_correct_action_all_args_pred_not_done_count_"),
        ("oaa_pred_match_success_execution_accuracy_curr_next_", "oaa_pred_match_success_execution_count_curr_next_", "oaa_correct_action_all_args_pred_not_done_count_curr_next_"),
        ("oaa_correct_principal_if_action_pred_accuracy_", "oaa_correct_action_principal_pred_count_", "oaa_correct_action_pred_count_"),
        ("oaa_correct_all_args_if_action_pred_accuracy_", "oaa_correct_action_all_args_pred_count_", "oaa_correct_action_pred_count_"),
        ("oaa_correct_principal_if_action_pred_accuracy_curr_next_", "oaa_correct_action_principal_pred_count_curr_next_", "oaa_correct_action_pred_count_curr_next_"),
        ("oaa_correct_all_args_if_action_pred_accuracy_curr_next_", "oaa_correct_action_all_args_pred_count_curr_next_", "oaa_correct_action_pred_count_curr_next_"),
    ]
    
    # Calculate accuracies if applicable
    for idx, (accuracy_col_name,count_col_name,total_count_col_name) in enumerate(accuracy_nominator_denominator_name_triplets):
        for agent_id in agent_ids:
            accuracy_col_name_agent_id = accuracy_col_name + str(agent_id)
            count_col_name_agent_id = count_col_name + str(agent_id)
            total_count_col_name_agent_id = total_count_col_name + str(agent_id)
            if accuracy_col_name_agent_id in headers \
            and count_col_name_agent_id in column_sums \
            and total_count_col_name_agent_id in column_sums:
                #print(f"Calculating {accuracy_col_name_agent_id} using {count_col_name_agent_id} and {total_count_col_name_agent_id}")
                summarized_metrics[accuracy_col_name_agent_id] = column_sums[count_col_name_agent_id] / column_sums[total_count_col_name_agent_id] if column_sums[total_count_col_name_agent_id] > 0 else None
                
    accuracy_nominator_denominator_name_quadruplets  = [
        ("oaa_pred_unmatch_success_execution_accuracy_", "oaa_pred_unmatch_success_execution_count_", "oa_not_done_pred_count_", "oaa_correct_action_all_args_pred_not_done_count_"),
        ("oaa_pred_unmatch_success_execution_accuracy_curr_next_", "oaa_pred_unmatch_success_execution_count_curr_next_", "oa_not_done_pred_count_", "oaa_correct_action_all_args_pred_not_done_count_curr_next_"),
    ]
    
    # Calculate oaa_pred_unmatch_success_execution_accuracy separately since its denominator is different
    for idx, (accuracy_col_name,count_col_name,not_done_pred_count_col_name,correct_count_col_name) in enumerate(accuracy_nominator_denominator_name_quadruplets):
        for agent_id in agent_ids:
            accuracy_col_name_agent_id = accuracy_col_name + str(agent_id)
            count_col_name_agent_id = count_col_name + str(agent_id)
            not_done_pred_count_col_name_agent_id = not_done_pred_count_col_name + str(agent_id)
            correct_count_col_name_agent_id = correct_count_col_name + str(agent_id)
            if accuracy_col_name_agent_id in headers \
            and count_col_name_agent_id in column_sums \
            and not_done_pred_count_col_name_agent_id in column_sums \
            and correct_count_col_name_agent_id in column_sums:
                #print(f"Calculating {accuracy_col_name_agent_id} using {count_col_name_agent_id}, {not_done_pred_count_col_name_agent_id} and {correct_count_col_name_agent_id}")
                summarized_metrics[accuracy_col_name_agent_id] = column_sums[count_col_name_agent_id] / (column_sums[not_done_pred_count_col_name_agent_id] - column_sums[correct_count_col_name_agent_id]) if (column_sums[not_done_pred_count_col_name_agent_id] - column_sums[correct_count_col_name_agent_id]) > 0 else None                 

    # Print summary
    print(f"Metrics Summary Report")
    print(f"=" * 80)
    print(f"File: {csv_file_path}")
    print(f"Total episodes processed: {processed_rows}")
    print(f"Episodes skipped (column mismatch): {skipped_rows}")
    print(f"Columns aggregated: {len(summarized_metrics)}")
    print(f"\n{'Metric Name':<60} {'Aggregated Value':>15}")
    print("-" * 80)
    
    # Sort by column name for readability
    for col_name in sorted(summarized_metrics.keys()):
        avg_value = summarized_metrics[col_name]
        print(f"{col_name} {avg_value}")
        #print(f"{col_name:<60} {avg_value:>15.10f}")
    
    print("=" * 80)
    
    return summarized_metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_metrics_summary.py <experiment_folder>")
        sys.exit(1)

    exp_folder = sys.argv[1]
    results_folder = Path(exp_folder) / "results"
    csv_file = results_folder / "episode_result_log.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    # Capture the output from compute_metrics_summary
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    summarized_metrics = compute_metrics_summary(str(csv_file))
    sys.stdout = old_stdout

    # Save the captured output to a txt file in results folder
    output_txt_path = results_folder / "end_result_metrics_summary_report.txt"
    with open(output_txt_path, 'w') as f:
        f.write(captured_output.getvalue())

    print(f"Report saved to: {output_txt_path}")

    # Save summarized_metrics to CSV in results folder
    output_csv_path = results_folder / "end_result_metrics.csv"
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric Name", "Aggregated Value"])
        for col_name in sorted(summarized_metrics.keys()):
            writer.writerow([col_name, summarized_metrics[col_name]])

    print(f"summarized_metrics saved to: {output_csv_path}")
