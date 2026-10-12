#!/usr/bin/env python3
"""
Script to analyze planner logs and count successful/failed action executions per agent.

This script reads planner logs from:
    <experiment_folder>/results/val.json.gz/planner-log/<logs>

And counts:
    - Successful action executions (for each agent)
    - Failed action executions (for each agent)
    - Total replanning count (for each agent)
"""

import json
import os
import sys
import csv
from pathlib import Path
from collections import defaultdict


def analyze_planner_logs(experiment_folder):
    """
    Analyze all planner logs in the experiment folder.
    
    Args:
        experiment_folder: Path to the experiment folder (can be relative or absolute)
    
    Returns:
        dict: Statistics for each agent
    """
    # Convert to absolute path
    exp_path = Path(experiment_folder).resolve()
    logs_dir = exp_path / "results" / "val.json.gz" / "planner-log"
    
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        sys.exit(1)
    
    # Find all JSON log files
    log_files = sorted(logs_dir.glob("*.json"))
    
    if not log_files:
        print(f"No log files found in: {logs_dir}")
        sys.exit(1)
    
    print(f"Found {len(log_files)} log files")
    
    # Statistics per agent
    agent_stats = {
        "0": {
            "successful_executions": 0,
            "failed_executions": 0,
            "total_replanning_count": 0,
            "replanning_count_after_other_agent_is_done": 0,
            "episodes_processed": 0,
        },
        "1": {
            "successful_executions": 0,
            "failed_executions": 0,
            "total_replanning_count": 0,
            "replanning_count_after_other_agent_is_done": 0,
            "episodes_processed": 0,
        }
    }
    
    # Process each log file
    for log_file in log_files:
        try:
            print(f"Processing log file: {log_file.name}")
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            if "steps" not in data:
                print(f"Warning: No 'steps' key in {log_file.name}")
                continue
            
            agent_stats["0"]["episodes_processed"] += 1
            agent_stats["1"]["episodes_processed"] += 1
            
            replanning_count_0 = data["steps"][-1]["replanning_count"]["0"]
            replanning_count_1 = data["steps"][-1]["replanning_count"]["1"]
            agent_stats["0"]["total_replanning_count"] += replanning_count_0
            agent_stats["1"]["total_replanning_count"] += replanning_count_1
            
            replanning_count_when_oa_is_done_0 = 0
            replanning_count_when_oa_is_done_1 = 0

            # Track when each agent finishes to count the other agent's replanning steps after that point
            agent_0_done = False
            agent_1_done = False

            # Process each step
            for step in data["steps"]:
                is_done = step.get("is_done", {})
                responses = step.get("responses", {})
                replanning_count = step.get("replanning_count", {})
                
                # Check if agent 0 just finished (transition from False to True)
                if not agent_0_done and is_done.get("0"):
                    agent_0_done = True
                    replanning_count_when_oa_is_done_1 = replanning_count.get("1", 0)

                # Check if agent 1 just finished (transition from False to True)
                if not agent_1_done and is_done.get("1"):
                    agent_1_done = True
                    replanning_count_when_oa_is_done_0 = replanning_count.get("0", 0)
                    
                for agent_id in ["0", "1"]:
                    response = responses.get(agent_id, "")
                    
                    # Classify as success or failure based on response content
                    if response:
                        response_lower = response.lower()
                        if "success" in response_lower:
                            agent_stats[agent_id]["successful_executions"] += 1
                        else:
                            agent_stats[agent_id]["failed_executions"] += 1

            agent_stats["0"]["replanning_count_after_other_agent_is_done"] += replanning_count_0 - replanning_count_when_oa_is_done_0
            agent_stats["1"]["replanning_count_after_other_agent_is_done"] += replanning_count_1 - replanning_count_when_oa_is_done_1

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON in {log_file.name}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {log_file.name}: {e}")
    
    return agent_stats


def print_results(agent_stats):
    """Print the analysis results in a formatted table."""
    print("\n" + "="*80)
    print("PLANNER LOG ANALYSIS RESULTS")
    print("="*80)
    
    for agent_id in ["0", "1"]:
        stats = agent_stats[agent_id]
        total_executions = stats["successful_executions"] + stats["failed_executions"]
        success_rate = (
            100 * stats["successful_executions"] / total_executions
            if total_executions > 0 else 0
        )
        
        print(f"\nAgent {agent_id}:")
        print(f"  Episodes processed:        {stats['episodes_processed']}")
        print(f"  Total action executions:   {total_executions}")
        print(f"  Successful executions:     {stats['successful_executions']}")
        print(f"  Failed executions:         {stats['failed_executions']}")
        print(f"  Success rate:              {success_rate:.1f}%")
        print(f"  Total replanning steps:    {stats['total_replanning_count']}")
        print(f"  Replanning steps after other agent is done: {stats['replanning_count_after_other_agent_is_done']}")
        print(f"  Extra steps rate after other agent is done: {stats['replanning_count_after_other_agent_is_done']/stats['total_replanning_count']}" if stats['total_replanning_count'] > 0 else "N/A")
    print("\n" + "="*80)
    
    # Summary comparison
    print("\nCOMPARISON:")
    print(f"Agent 0 success rate: {100 * agent_stats['0']['successful_executions'] / max(1, agent_stats['0']['successful_executions'] + agent_stats['0']['failed_executions']):.1f}%")
    print(f"Agent 1 success rate: {100 * agent_stats['1']['successful_executions'] / max(1, agent_stats['1']['successful_executions'] + agent_stats['1']['failed_executions']):.1f}%")
    print("="*80 + "\n")


def save_results_to_csv(agent_stats, output_file):
    """Save the analysis results to a CSV file."""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'agent_id',
            'episodes_processed',
            'total_action_executions',
            'successful_executions',
            'failed_executions',
            'success_rate_%',
            'total_replanning_steps',
            'replanning_steps_after_other_agent_done',
            'extra_steps_rate_after_other_agent_done_%'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for agent_id in ["0", "1"]:
            stats = agent_stats[agent_id]
            total_executions = stats["successful_executions"] + stats["failed_executions"]
            success_rate = (
                100 * stats["successful_executions"] / total_executions
                if total_executions > 0 else 0
            )
            extra_steps_rate = (
                100 * stats["replanning_count_after_other_agent_is_done"] / stats['total_replanning_count']
                if stats['total_replanning_count'] > 0 else 0
            )
            
            writer.writerow({
                'agent_id': agent_id,
                'episodes_processed': stats['episodes_processed'],
                'total_action_executions': total_executions,
                'successful_executions': stats['successful_executions'],
                'failed_executions': stats['failed_executions'],
                'success_rate_%': f"{success_rate:.2f}",
                'total_replanning_steps': stats['total_replanning_count'],
                'replanning_steps_after_other_agent_done': stats['replanning_count_after_other_agent_is_done'],
                'extra_steps_rate_after_other_agent_done_%': f"{extra_steps_rate:.2f}"
            })


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_planner_logs.py <experiment_folder_path> [output_csv_file]")
        print("\nExample:")
        print("  python analyze_planner_logs.py ./outputs/my_experiment")
        print("  python analyze_planner_logs.py ./outputs/my_experiment results.csv")
        sys.exit(1)
    
    experiment_folder = sys.argv[1]
    
    # Set default output CSV path in experiment folder's results directory
    exp_path = Path(experiment_folder).resolve()
    default_csv = exp_path / "results" / "successfull_rate_and_extra_replans.csv"
    
    # Optional: output CSV file (can override the default)
    output_csv = sys.argv[2] if len(sys.argv) > 2 else str(default_csv)
    
    print(f"Analyzing planner logs in: {experiment_folder}")
    
    agent_stats = analyze_planner_logs(experiment_folder)
    print_results(agent_stats)
    
    # Save results to CSV
    save_results_to_csv(agent_stats, output_csv)
    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()
