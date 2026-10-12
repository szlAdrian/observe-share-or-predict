#!/usr/bin/env python
"""
Script to evaluate the OANA predictions

Usage:
    python oana_pred_eval.py <relative_path_to_experiment_folder>
    
"""

import json
import re
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any

# Actions whose first argument is not an object instance id and should be left untouched.
ACTIONS_WITHOUT_OBJECT_ID = {"Explore", "Wait", "Done", "Close", "Navigate", "Open"}


def strip_object_instance_id(action: List[Any]) -> List[Any]:
    """
    Strip the trailing instance suffix (e.g. "_1") from the first (principal)
    argument of an action, e.g. "candle_holder_1, on, table_10, None, None"
    becomes "candle_holder, on, table_10, None, None".

    Actions in ACTIONS_WITHOUT_OBJECT_ID are returned unchanged, since their
    first argument is not an object instance id (e.g. a room name).

    :param action: A [action_name, action_args, ...] prediction entry
    :return: The action with its principal argument's instance id stripped
    """
    if not action or action[0] in ACTIONS_WITHOUT_OBJECT_ID:
        return action

    action_name, action_args, *rest = action
    if not action_args:
        return action

    parts = action_args.split(',', 1)
    principal_arg = re.sub(r'_\d+$', '', parts[0].strip())
    new_args = principal_arg if len(parts) == 1 else f"{principal_arg},{parts[1]}"

    return [action_name, new_args, *rest]


def strip_all_instance_ids(action: List[Any]) -> List[Any]:
    """
    Strip trailing instance suffixes (e.g. "_1", "_10") from every argument of
    an action, regardless of action type, e.g.
    "candle_holder_1, on, table_10, None, None"
    becomes "candle_holder, on, table, None, None".

    :param action: A [action_name, action_args, ...] prediction entry
    :return: The action with instance ids stripped from all arguments
    """
    if not action:
        return action

    action_name, action_args, *rest = action
    if not action_args:
        return action

    stripped_args = ', '.join(
        re.sub(r'_\d+$', '', arg.strip()) for arg in action_args.split(',')
    )
    return [action_name, stripped_args, *rest]


def get_first_argument(action_args: str) -> str:
    """
    Returns the first argument of the action arguments string.
        
    :param action_args: The input string containing arguments (comma-separated)
    :return: The first argument (substring before the first comma, or the whole string if no comma exists)
        Returns None if input is None
    """
    if action_args is None:
        return None
    
    if ',' in action_args:
        return action_args.split(',')[0]
    else:
        return action_args


def compare_predictions_for_agent(preds_and_labels: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    """
    Compare predictions with actual action classes for a specific agent.
    
    Args:
        preds_and_labels: The "oana_preds_and_labels" data from the log
        agent_id: Agent ID ("0" or "1")
    
    Returns:
        Dictionary containing comparison results
    """
    if agent_id not in preds_and_labels:
        return None
    
    agent_data = preds_and_labels[agent_id]
    pred = [strip_all_instance_ids(p) for p in agent_data.get('pred', [])]
    #pred = [strip_object_instance_id(p) for p in agent_data.get('pred', [])]  # Strip instance IDs from principal arguments
    oa_current_action_obj_class = [strip_all_instance_ids(a) for a in agent_data.get('oa_current_action_obj_class', [])]
    oa_next_action_obj_class = [strip_all_instance_ids(a) for a in agent_data.get('oa_next_action_obj_class', [])]
    print('------------------------------------')
    print(f"Agent {agent_id} predictions (stripped): {pred}")
    print(f"Agent {agent_id} current action classes (stripped): {oa_current_action_obj_class}")
    print(f"Agent {agent_id} next action classes (stripped): {oa_next_action_obj_class}")
    print('------------------------------------')
    
    # Compare principal action parameters in pred with oa_current_action_obj_class and oa_next_action_obj_class
    current_next_matches_all_count = 0
    next_matches_all_count = 0
    current_next_matches_action_types_count = 0
    next_matches_action_types_count = 0
    current_next_matches_principal_count = 0
    next_matches_principal_count = 0
    current_next_matches_action_types_principal_count = 0
    next_matches_action_types_principal_count = 0
    current_next_matches_args_only_count = 0
    next_matches_args_only_count = 0
    for pred_oaa, current_oaa, next_oaa in zip(pred, oa_current_action_obj_class, oa_next_action_obj_class):
        # Compare action and all arguments of the action
        if pred_oaa == current_oaa or pred_oaa == next_oaa:
            current_next_matches_all_count += 1
        if pred_oaa == next_oaa:
            next_matches_all_count += 1
        
        # Compare only action types 
        if pred_oaa[0] == current_oaa[0] or pred_oaa[0] == next_oaa[0]:
            current_next_matches_action_types_count += 1
        if pred_oaa[0] == next_oaa[0]:
            next_matches_action_types_count += 1
        
        # Compare only principal argument of the action
        if get_first_argument(pred_oaa[1]) == get_first_argument(current_oaa[1]) or get_first_argument(pred_oaa[1]) == get_first_argument(next_oaa[1]):
            current_next_matches_principal_count += 1
        if get_first_argument(pred_oaa[1]) == get_first_argument(next_oaa[1]):
            next_matches_principal_count += 1
            
        # Compare action types and principal argument of the action
        if ((pred_oaa[0] == current_oaa[0]) and (get_first_argument(pred_oaa[1]) == get_first_argument(current_oaa[1]))) or ((pred_oaa[0] == next_oaa[0]) and (get_first_argument(pred_oaa[1]) == get_first_argument(next_oaa[1]))):
            current_next_matches_action_types_principal_count += 1
        if (pred_oaa[0] == next_oaa[0]) and (get_first_argument(pred_oaa[1]) == get_first_argument(next_oaa[1])):
            next_matches_action_types_principal_count += 1
            
        # Compare all arguments of the action ignoring the action type
        if pred_oaa[1] == current_oaa[1] or pred_oaa[1] == next_oaa[1]:
            current_next_matches_args_only_count += 1
        if pred_oaa[1] == next_oaa[1]:
            next_matches_args_only_count += 1
    
    return {
        'agent_id': agent_id,
        'total_predictions': len(pred),
        'current_next_matches_all_count': current_next_matches_all_count,
        'next_matches_all_count': next_matches_all_count,
        'current_next_matches_action_types_count': current_next_matches_action_types_count,
        'next_matches_action_types_count': next_matches_action_types_count,
        'current_next_matches_principal_count': current_next_matches_principal_count,
        'next_matches_principal_count': next_matches_principal_count,
        'current_next_matches_action_types_principal_count': current_next_matches_action_types_principal_count,
        'next_matches_action_types_principal_count': next_matches_action_types_principal_count,
        'current_next_matches_args_only_count': current_next_matches_args_only_count,
        'next_matches_args_only_count': next_matches_args_only_count,
    }


def process_logs(experiment_folder: str) -> None:
    """
    Read and compare planner logs from experiment folder.
    
    Args:
        experiment_folder: Path to the experiment folder (relative or absolute)
    """
    experiment_path = Path(experiment_folder)
    logs_dir = experiment_path / "results" / "val.json.gz" / "planner-last-step-log"
    
    if not logs_dir.exists():
        print(f"Error: Logs directory not found at: {logs_dir}")
        sys.exit(1)
    
    log_files = sorted(logs_dir.glob("planner-last-step-log-*.json"))
    
    if not log_files:
        print(f"Warning: No log files found in {logs_dir}")
        return
    
    print(f"Processing {len(log_files)} log file(s)...\n")
    
    invalid_count = 0
    
    results = {
        '0': {
            'total_predictions': 0,
            'current_next_matches_all_count': 0,
            'next_matches_all_count': 0,
            'current_next_matches_action_types_count': 0,
            'next_matches_action_types_count': 0,
            'current_next_matches_principal_count': 0,
            'next_matches_principal_count': 0,
            'current_next_matches_action_types_principal_count': 0,
            'next_matches_action_types_principal_count': 0,
            'current_next_matches_args_only_count': 0,
            'next_matches_args_only_count': 0,
            }, 
        '1': {
            'total_predictions': 0,
            'current_next_matches_all_count': 0,
            'next_matches_all_count': 0,
            'current_next_matches_action_types_count': 0,
            'next_matches_action_types_count': 0,
            'current_next_matches_principal_count': 0,
            'next_matches_principal_count': 0,
            'current_next_matches_action_types_principal_count': 0,
            'next_matches_action_types_principal_count': 0,
            'current_next_matches_args_only_count': 0,
            'next_matches_args_only_count': 0,
        }
    }
    
    for log_file in log_files[:]:
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            if 'oaa_preds_and_labels' not in log_data:
                invalid_count += 1
                continue
            
            preds_and_labels = log_data['oaa_preds_and_labels']
            
            print(f"{'='*70}")
            print(f"File: {log_file.name}")
            print(f"{'='*70}")
            
            for agent_id in ['0', '1']:
                comparison = compare_predictions_for_agent(preds_and_labels, agent_id)
                if not comparison:
                    continue
                results[agent_id]['total_predictions'] += comparison['total_predictions']
                results[agent_id]['current_next_matches_all_count'] += comparison['current_next_matches_all_count']
                results[agent_id]['next_matches_all_count'] += comparison['next_matches_all_count']
                results[agent_id]['current_next_matches_action_types_count'] += comparison['current_next_matches_action_types_count']
                results[agent_id]['next_matches_action_types_count'] += comparison['next_matches_action_types_count']
                results[agent_id]['current_next_matches_principal_count'] += comparison['current_next_matches_principal_count']
                results[agent_id]['next_matches_principal_count']   += comparison['next_matches_principal_count']
                results[agent_id]['current_next_matches_action_types_principal_count'] += comparison['current_next_matches_action_types_principal_count']
                results[agent_id]['next_matches_action_types_principal_count'] += comparison['next_matches_action_types_principal_count']
                results[agent_id]['current_next_matches_args_only_count'] += comparison['current_next_matches_args_only_count']
                results[agent_id]['next_matches_args_only_count'] += comparison['next_matches_args_only_count']

                print(f"\nAgent {agent_id}:")
                print(f"Total Predictions: {comparison['total_predictions']}")
                print(f"Matches with Current/Next (Action + All Args): {comparison['current_next_matches_all_count']} ({comparison['current_next_matches_all_count']/comparison['total_predictions']:.2%})")
                print(f"Matches with Next (Action + All Args): {comparison['next_matches_all_count']} ({comparison['next_matches_all_count']/comparison['total_predictions']:.2%})")
                print(f"Matches with Current/Next (Action Type Only): {comparison['current_next_matches_action_types_count']} ({comparison['current_next_matches_action_types_count']/comparison['total_predictions']:.2%})")
                print(f"Matches with Next (Action Type Only): {comparison['next_matches_action_types_count']} ({comparison['next_matches_action_types_count']/comparison['total_predictions']:.2%})")
                print(f"Matches with Current/Next (Principal Arg Only): {comparison['current_next_matches_principal_count']} ({comparison['current_next_matches_principal_count']/comparison['total_predictions']:.2%})")
                print(f"Matches with Next (Principal Arg Only): {comparison['next_matches_principal_count']} ({comparison['next_matches_principal_count']/comparison['total_predictions']:.2%})")
                print(f"Matches with Current/Next (Action Type + Principal Arg): {comparison['current_next_matches_action_types_principal_count']} ({comparison['current_next_matches_action_types_principal_count']/comparison['total_predictions']:.2%})")
                print(f"Matches with Next (Action Type + Principal Arg): {comparison['next_matches_action_types_principal_count']} ({comparison['next_matches_action_types_principal_count']/comparison['total_predictions']:.2%})")
                print(f"Matches with Current/Next (Args Only): {comparison['current_next_matches_args_only_count']} ({comparison['current_next_matches_args_only_count']/comparison['total_predictions']:.2%})")
                print(f"Matches with Next (Args Only): {comparison['next_matches_args_only_count']} ({comparison['next_matches_args_only_count']/comparison['total_predictions']:.2%})")


            print()
        
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON in {log_file.name}: {e}")
        except Exception as e:
            print(f"Error: Failed to process {log_file.name}: {e}")

    print(f"\n{'='*70}")
    print("Summary of Results:")
    for agent_id in ['0', '1']:
        total_predictions = results[agent_id]['total_predictions']
        if total_predictions == 0:
            continue
        print(f"\nAgent {agent_id}:")
        print(f"Total Predictions: {total_predictions}")
        print(f"Matches with Current/Next (Action + All Args): {results[agent_id]['current_next_matches_all_count']} ({results[agent_id]['current_next_matches_all_count']/total_predictions:.2%})")
        print(f"Matches with Next (Action + All Args): {results[agent_id]['next_matches_all_count']} ({results[agent_id]['next_matches_all_count']/total_predictions:.2%})")
        print(f"Matches with Current/Next (Action Type Only): {results[agent_id]['current_next_matches_action_types_count']} ({results[agent_id]['current_next_matches_action_types_count']/total_predictions:.2%})")
        print(f"Matches with Next (Action Type Only): {results[agent_id]['next_matches_action_types_count']} ({results[agent_id]['next_matches_action_types_count']/total_predictions:.2%})")
        print(f"Matches with Current/Next (Principal Arg Only): {results[agent_id]['current_next_matches_principal_count']} ({results[agent_id]['current_next_matches_principal_count']/total_predictions:.2%})")
        print(f"Matches with Next (Principal Arg Only): {results[agent_id]['next_matches_principal_count']} ({results[agent_id]['next_matches_principal_count']/total_predictions:.2%})")
        print(f"Matches with Current/Next (Action Type + Principal Arg): {results[agent_id]['current_next_matches_action_types_principal_count']} ({results[agent_id]['current_next_matches_action_types_principal_count']/total_predictions:.2%})")
        print(f"Matches with Next (Action Type + Principal Arg): {results[agent_id]['next_matches_action_types_principal_count']} ({results[agent_id]['next_matches_action_types_principal_count']/total_predictions:.2%})")
        print(f"Matches with Current/Next (Args Only): {results[agent_id]['current_next_matches_args_only_count']} ({results[agent_id]['current_next_matches_args_only_count']/total_predictions:.2%})")
        print(f"Matches with Next (Args Only): {results[agent_id]['next_matches_args_only_count']} ({results[agent_id]['next_matches_args_only_count']/total_predictions:.2%})")
    print(f"\nFinished processing logs. Invalid log files: {invalid_count}/{len(log_files)}")
    
    # Calculate accuracies and prepare output data
    results_with_accuracies = {}
    for agent_id in ['0', '1']:
        total_predictions = results[agent_id]['total_predictions']
        agent_results = results[agent_id].copy()
        
        if total_predictions > 0:
            agent_results['current_next_matches_all_accuracy'] = results[agent_id]['current_next_matches_all_count'] / total_predictions
            agent_results['next_matches_all_accuracy'] = results[agent_id]['next_matches_all_count'] / total_predictions
            agent_results['current_next_matches_action_types_accuracy'] = results[agent_id]['current_next_matches_action_types_count'] / total_predictions
            agent_results['next_matches_action_types_accuracy'] = results[agent_id]['next_matches_action_types_count'] / total_predictions
            agent_results['current_next_matches_principal_accuracy'] = results[agent_id]['current_next_matches_principal_count'] / total_predictions
            agent_results['next_matches_principal_accuracy'] = results[agent_id]['next_matches_principal_count'] / total_predictions
            agent_results['current_next_matches_action_types_principal_accuracy'] = results[agent_id]['current_next_matches_action_types_principal_count'] / total_predictions
            agent_results['next_matches_action_types_principal_accuracy'] = results[agent_id]['next_matches_action_types_principal_count'] / total_predictions
            agent_results['current_next_matches_args_only_accuracy'] = results[agent_id]['current_next_matches_args_only_count'] / total_predictions
            agent_results['next_matches_args_only_accuracy'] = results[agent_id]['next_matches_args_only_count'] / total_predictions
        
        results_with_accuracies[agent_id] = agent_results
    
    # Save results to JSON file
    output_path = experiment_path / "results" / "oana_pred_eval.json"
    output_data = {
        'total_log_files': len(log_files),
        'invalid_log_files': invalid_count,
        'valid_log_files': len(log_files) - invalid_count,
        'results': results_with_accuracies
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {output_path}")
    
    # Save results to CSV file
    csv_output_path = experiment_path / "results" / "oana_pred_eval.csv"
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        headers = ['metric', 'agent_0_count', 'agent_0_accuracy', 'agent_1_count', 'agent_1_accuracy']
        writer.writerow(headers)
        
        # Write metrics rows
        metrics = [
            ('total_predictions', 'total_predictions'),
            ('current_next_matches_all', 'current_next_matches_all_count'),
            ('next_matches_all', 'next_matches_all_count'),
            ('current_next_matches_action_types', 'current_next_matches_action_types_count'),
            ('next_matches_action_types', 'next_matches_action_types_count'),
            ('current_next_matches_principal', 'current_next_matches_principal_count'),
            ('next_matches_principal', 'next_matches_principal_count'),
            ('current_next_matches_action_types_principal', 'current_next_matches_action_types_principal_count'),
            ('next_matches_action_types_principal', 'next_matches_action_types_principal_count'),
            ('current_next_matches_args_only', 'current_next_matches_args_only_count'),
            ('next_matches_args_only', 'next_matches_args_only_count'),
        ]
        
        for metric_name, key in metrics:
            agent_0_count = results_with_accuracies['0'].get(key, 0)
            agent_0_accuracy = results_with_accuracies['0'].get(key.replace('_count', '_accuracy'), '') if key != 'total_predictions' else ''
            agent_1_count = results_with_accuracies['1'].get(key, 0)
            agent_1_accuracy = results_with_accuracies['1'].get(key.replace('_count', '_accuracy'), '') if key != 'total_predictions' else ''
            
            writer.writerow([metric_name, agent_0_count, agent_0_accuracy, agent_1_count, agent_1_accuracy])
    
    print(f"Results saved to: {csv_output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python oana_pred_eval.py <relative_path_to_experiment_folder>")
        sys.exit(1)
    
    experiment_folder = sys.argv[1]
    process_logs(experiment_folder)
