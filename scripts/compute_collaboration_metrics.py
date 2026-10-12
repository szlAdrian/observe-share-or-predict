import argparse
import json
import os
from typing import Dict, Any, List, Tuple, Union

import numpy as np
from scipy import stats

def compute_statistics(
    data: List[Union[float, int]]
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Calculate the mean, standard deviation, and 95% confidence interval for a given list of numbers.

    Args:
        data (list): A list of numbers.

    Returns:
        tuple: A tuple containing the mean, standard deviation, and 95% confidence interval.
    """
    data = np.array(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    std_err = std_dev / np.sqrt(len(data))
    if std_err == 0.0:
        confidence_interval = (mean, mean)
    else:
        confidence_interval = stats.t.interval(
            0.95, len(data) - 1, loc=mean, scale=std_err
        )

    return mean, std_err, confidence_interval

def compute_exploration_efficiency(data:dict, filename:str) -> Dict[str, int]:
    """Return exploration efficiency for both agent 0 and agent 1.

    For each agent, return the number of exploration steps that occurred
    before the first successful Pick or picking part of Rearrange action. If not found,
    return -1 for that agent.
    """
    efficiency = {"agent_0": -1, "agent_1": -1}

    for step in data.get('steps', []):
        # Note:
        # If an agent is standing after picking up an object 
        # without (action failed) response in case of the Rearrange 
        # action, it means that the agent successfully picked up the object
        
        # Check agent 0
        if efficiency['agent_0'] == -1:
            if (("0" in step.get("high_level_actions", {})) and \
                step["high_level_actions"]["0"][0] == "Pick" and \
                ("0" in step.get("responses", {})) and \
                step["responses"]["0"] == "Successful execution!" \
                ) or \
                (("0" in step.get("high_level_actions", {})) and \
                step["high_level_actions"]["0"][0] == "Rearrange" and \
                ("0" in step.get("responses", {})) and \
                step["responses"]["0"] == "" and \
                ("0" in step.get("agent_states", {})) and \
                ("Standing" in step["agent_states"]["0"])):
                efficiency['agent_0'] = step.get("sim_step_count", -1)

        # Check agent 1
        if efficiency['agent_1'] == -1:
            if (("1" in step.get("high_level_actions", {})) and \
                step["high_level_actions"]["1"][0] == "Pick" and \
                ("1" in step.get("responses", {})) and \
                step["responses"]["1"] == "Successful execution!" \
                ) or \
                (("1" in step.get("high_level_actions", {})) and \
                step["high_level_actions"]["1"][0] == "Rearrange" and \
                ("1" in step.get("responses", {})) and \
                step["responses"]["1"] == "" and \
                ("1" in step.get("agent_states", {})) and \
                ("Standing" in step["agent_states"]["1"])):
                efficiency['agent_1'] = step.get("sim_step_count", -1)

        if efficiency['agent_0'] != -1 and efficiency['agent_1'] != -1:
            break

    if efficiency['agent_0'] == -1:
        print(
            f"No successfull Pick in: {filename} for agent 0. "
            f"Agent_0: {efficiency['agent_0']}, Agent_1: {efficiency['agent_1']}. "
            f"Total steps in the episode: {len(data.get('steps', []))}"
        )
    if efficiency['agent_1'] == -1:
        print(
            f"No successfull Pick in: {filename} for agent 1. "
            f"Agent_0: {efficiency['agent_0']}, Agent_1: {efficiency['agent_1']}. "
            f"Total steps in the episode: {len(data.get('steps', []))}"
        )
        
    return efficiency

def compute_task_distribution(data:dict) -> Tuple[int, int, int]:
    tasks_agent_0, tasks_agent_1, all_tasks, old_task_percent_complete = 0, 0, 0, 0
    for step in data['steps']:
        if "responses" not in step:
            continue

        # agent event occured if there is a non-empty response for that agent.
        agent_1_event = ("1" in step["responses"]) and (step["responses"]["1"] != "")
        agent_0_event = ("0" in step["responses"]) and (step["responses"]["0"] != "")

        if (
            agent_1_event
            and step["stats"]["task_percent_complete"] > old_task_percent_complete
        ):
            tasks_agent_1 += 1
            all_tasks += 1
            old_task_percent_complete = step["stats"]["task_percent_complete"]
        elif (
            agent_0_event
            and step["stats"]["task_percent_complete"] > old_task_percent_complete
        ):
            tasks_agent_0 += 1
            all_tasks += 1
            old_task_percent_complete = step["stats"]["task_percent_complete"] 
        elif agent_1_event and (step["high_level_actions"]["1"][0] == "Place" or step["high_level_actions"]["1"][0] == "Rearrange"):
            all_tasks += 1
        elif agent_0_event and (step["high_level_actions"]["0"][0] == "Place" or step["high_level_actions"]["0"][0] == "Rearrange"):
            all_tasks += 1

    #print(f'tasks_agent_0: {tasks_agent_0}, tasks_agent_1: {tasks_agent_1}, all_tasks: {all_tasks}')
    return tasks_agent_0, tasks_agent_1, all_tasks

def main(output_dir: str,dataset: str, recompute: bool = False) -> None:
    """Search under outputs/habitat_llm/<output_dir>/results/<dataset>.json.gz/planner-log. Load all
    planner-log-episode_*.json(.gz) files into a list of dicts and print a
    small summary.
    """
    metrics_file = os.path.join("outputs", "habitat_llm", output_dir, "results", "collaboration_metrics.json")
    # check if data has already been processed
    if os.path.exists(metrics_file) and not recompute:
        with open(metrics_file) as f:
            metrics = json.load(f)

        if ("ratio_agent_0" in metrics) and ("exploration_efficiency_agent_0" in metrics) and ("ratio_extraneous_actions" in metrics):
            print("Collaboration metrics already computed. Skipping...")
            return
    
    metrics_for_all_episodes: Dict[str, List[Any]] = {
        'exploration_efficiency_agent_0': [],
        'exploration_efficiency_agent_1': [],
        'exploration_efficiency_both_agents': [],
        'ratio_agent_0': [],
        'ratio_agent_1': [],
        'tasks_agent_0': [],
        'tasks_agent_1': [],
        'tasks_all': [],
        'ratio_extraneous_actions': []
    }

    # Walk the outputs/results folder
    planner_log_dir = os.path.join("outputs", "habitat_llm", output_dir, "results", f"{dataset}.json.gz", 'planner-log')
    if not os.path.isdir(planner_log_dir):
        print(f"No results directory at {planner_log_dir}")
        return

    print(f'Computing the collaboration metrics for {len(os.listdir(planner_log_dir))} episodes...')
    skipped_files = 0
    # collect planner-log files
    for filename in os.listdir(planner_log_dir):
        data = {}
        if not filename.startswith("planner-log-episode_"):
            continue
        full_filename = os.path.join(planner_log_dir, filename)
        try:
            with open(full_filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {full_filename}: {e}")
        
        exploration_efficiency = compute_exploration_efficiency(data, filename)
        if exploration_efficiency['agent_0'] >= 0:
            metrics_for_all_episodes['exploration_efficiency_agent_0'].append(exploration_efficiency['agent_0'])
        if exploration_efficiency['agent_1'] >= 0:
            metrics_for_all_episodes['exploration_efficiency_agent_1'].append(exploration_efficiency['agent_1'])

        if min(exploration_efficiency['agent_0'], exploration_efficiency['agent_1']) < 0:
            if max(exploration_efficiency['agent_0'], exploration_efficiency['agent_1']) > 0:
                metrics_for_all_episodes['exploration_efficiency_both_agents'].append(max(exploration_efficiency['agent_0'], exploration_efficiency['agent_1']))
        elif min(exploration_efficiency['agent_0'], exploration_efficiency['agent_1']) > 0:
            metrics_for_all_episodes['exploration_efficiency_both_agents'].append(min(exploration_efficiency['agent_0'], exploration_efficiency['agent_1']))
                
        try:
            if data["steps"][-1]["stats"]["task_percent_complete"] < 0.99:
                continue
        except (KeyError, IndexError):
            print(f"Skipping {full_filename}: Incomplete or malformed data")
            skipped_files += 1
            continue
        
        tasks_agent_0, tasks_agent_1, all_tasks = compute_task_distribution(data)
        
        metrics_for_all_episodes['tasks_agent_0'].append(tasks_agent_0)
        metrics_for_all_episodes['tasks_agent_1'].append(tasks_agent_1)
        metrics_for_all_episodes['tasks_all'].append(all_tasks)
        
        tasks_agents = tasks_agent_0 + tasks_agent_1
        if tasks_agents != 0:
            ratio_0 = tasks_agent_0 / tasks_agents
            ratio_1 = tasks_agent_1 / tasks_agents
            metrics_for_all_episodes['ratio_agent_0'].append(ratio_0)
            metrics_for_all_episodes['ratio_agent_1'].append(ratio_1)
        
        if all_tasks != 0:    
            ratio_extra = (all_tasks - tasks_agents) / all_tasks
            metrics_for_all_episodes['ratio_extraneous_actions'].append(ratio_extra)

    stats_dict = {}
    for key, values in metrics_for_all_episodes.items():
        if len(values) > 0:
            mean, std_err, ci = compute_statistics(values)
            stats_dict[f"{key}_mean"] = mean
            stats_dict[f"{key}_std_err"] = std_err
            stats_dict[f"{key}_ci_lower"] = ci[0]
            stats_dict[f"{key}_ci_upper"] = ci[1]
        else:
            stats_dict[f"{key}_mean"] = 0.0
            stats_dict[f"{key}_std_err"] = 0.0
            stats_dict[f"{key}_ci_lower"] = 0.0
            stats_dict[f"{key}_ci_upper"] = 0.0
        stats_dict[f"{key}_len"] = len(values)

    # Compute the sum-averaged extraneous actions
    total_tasks_all = sum(metrics_for_all_episodes["tasks_all"])
    total_tasks_agents = sum(metrics_for_all_episodes["tasks_agent_0"]) + sum(metrics_for_all_episodes["tasks_agent_1"])
    ratio_extraneous_actions_sum_averaged = (total_tasks_all - total_tasks_agents) / total_tasks_all if total_tasks_all > 0 else 0

    metrics = {
        **stats_dict,
        "ratio_extraneous_actions_sum_averaged": ratio_extraneous_actions_sum_averaged,
    }

    print(f"Computed collaboration metrics: {metrics}")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved collaboration metrics to {metrics_file}")
    print(f"Skipped {skipped_files} files due to incomplete or malformed data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute offloading/explore/extraneous metrics from planner-log JSONs."
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Root folder containing planner-log files")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument(
        "--recompute",
        action=argparse.BooleanOptionalAction,
        help="recompute the collaboration metrics",
    )
    args = parser.parse_args()

    main(args.output_dir, args.dataset, bool(args.recompute))