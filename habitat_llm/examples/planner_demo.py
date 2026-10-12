#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import sys
import time
import os
import traceback
import json
import shutil
from omegaconf import OmegaConf


# append the path of the
# parent directory
sys.path.append("..")

import hydra
import numpy as np
from typing import Any, Dict, Optional, Set

from torch import multiprocessing as mp

from habitat_llm.agent.env.evaluation.evaluation_functions import (
    aggregate_measures,
    aggregate_measures_oaa,
    aggregate_action_measures
)

from habitat_llm.utils import cprint, setup_config, fix_config


from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.evaluation import (
    CentralizedEvaluationRunner,
    DecentralizedEvaluationRunner,
    EvaluationRunner,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
# from habitat_baselines.utils.info_dict import extract_scalars_from_info

# from habitat_baselines.utils.info_dict for extract_scalars_from_info
# These metrics are not scalars and cannot be easily reported
# (unless using videos)
NON_SCALAR_METRICS = {"top_down_map", "collisions.is_collision"}

# modified from: habitat_baselines.utils.info_dict
def extract_scalars_from_info(
    info: Dict[str, Any], ignore_keys: Optional[Set[str]] = None
) -> Dict[str, float]:
    r"""From an environment info dictionary, returns a flattened
    dictionary of string to floats by filtering all non-scalar
    metrics.

        Args:
            info: A gym.Env  info dict
            ignore_keys: The list of info key names to exclude in the result.

        Returns:
            dictionary of scalar values
    """
    if ignore_keys is None:
        ignore_keys = set()
    ignore_keys.update(NON_SCALAR_METRICS)
    result = {}
    for k, v in info.items():
        if not isinstance(k, str) or k in ignore_keys:
            continue

        if isinstance(v, dict):
            result.update(
                {
                    k + "." + subk: subv
                    for subk, subv in extract_scalars_from_info(v).items()
                    if isinstance(subk, str)
                    and k + "." + subk not in ignore_keys
                }
            )
        # Things that are scalar-like will have an np.size of 1.
        # Strings also have an np.size of 1, so explicitly ban those
        elif np.size(v) == 1 and not isinstance(v, str):
            if v is None:
                result[k] = v
            else:
                result[k] = float(v)

    return result

def get_output_file(config, env_interface):
    dataset_file = env_interface.conf.habitat.dataset.data_path.split("/")[-1]
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    output_file = os.path.join(
        config.paths.results_dir,
        dataset_file,
        "stats",
        f"{episode_id}.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file


# Function to write data to the CSV file
def write_to_csv(file_name, result_dict):
    # Sort the dictionary by keys
    # Needed to ensure sanity in multi-process operation
    result_dict = dict(sorted(result_dict.items()))
    with open(file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

        # Check if the file is empty (to write headers)
        file.seek(0, 2)
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(result_dict)


def save_exception_message(config, env_interface):
    output_file = get_output_file(config, env_interface)
    exc_string = traceback.format_exc()
    failure_dict = {"success": False, "info": str(exc_string)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


def save_success_message(config, env_interface, info):
    output_file = get_output_file(config, env_interface)
    failure_dict = {"success": True, "stats": json.dumps(info)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


# Write the config file into the results folders
def write_config(config):
    dataset_file = config.habitat.dataset.data_path.split("/")[-1]
    output_file = os.path.join(config.paths.results_dir, dataset_file)
    os.makedirs(output_file, exist_ok=True)
    with open(f"{output_file}/config.yaml", "w+") as f:
        f.write(OmegaConf.to_yaml(config))

    # Copy over the RLM config
    planner_configs = []
    suffixes = []
    if "planner" in config.evaluation:
        # Centralized
        if "plan_config" in config.evaluation.planner is not None:
            planner_configs = [config.evaluation.planner.plan_config]
            suffixes = [""]
    else:
        for agent_name in config.evaluation.agents:
            suffixes.append(f"_{agent_name}")
            planner_configs.append(
                config.evaluation.agents[agent_name].planner.plan_config
            )

    for plan_config, suffix_rlm in zip(planner_configs, suffixes):
        if "llm" in plan_config and "serverdir" in plan_config.llm:
            yaml_rlm_path = plan_config.llm.serverdir
            if len(yaml_rlm_path) > 0:
                yaml_rlm_file = f"{yaml_rlm_path}/config.yaml"
                if os.path.isfile(yaml_rlm_file):
                    shutil.copy(
                        yaml_rlm_file, f"{output_file}/config_rlm{suffix_rlm}.yaml"
                    )


# Method to load agent planner from the config
@hydra.main(config_path="../conf")
def run_eval(config):
    fix_config(config)
    # Setup a seed
    # seed = 48212516
    seed = 47668090
    t0 = time.time()
    # Setup config
    config = setup_config(config, seed)
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    # print(f'dataset has {len(dataset.episodes)} episodes.')
    # print(f'dataset: {dataset}')
    # print(f'dataset episodes: {type(dataset.episodes)}')
    # print(f'dataset episodes: {dataset.episodes[0]}')
    # asd
    #episode_subset = [dataset.episodes[x] for x in [3,5,7,8,12,21,25,29,33]]
    # episode_subset = (dataset.episodes)[:2]
    # dataset = CollaborationDatasetV0(
    #     config=config.habitat.dataset, episodes=episode_subset
    # )

    write_config(config)
    if config.get("resume", False):
        dataset_file = config.habitat.dataset.data_path.split("/")[-1]
        # stats_dir = os.path.join(config.paths.results_dir, dataset_file, "stats")
        plan_log_dir = os.path.join(
            config.paths.results_dir, dataset_file, "planner-log"
        )

        # Find incomplete episodes
        incomplete_episodes = []
        for episode in dataset.episodes:
            episode_id = episode.episode_id
            # stats_file = os.path.join(stats_dir, f"{episode_id}.json")
            planlog_file = os.path.join(
                plan_log_dir, f"planner-log-episode_{episode_id}_0.json"
            )
            if not os.path.exists(planlog_file):
                incomplete_episodes.append(episode)
        print(
            f"Resuming with {len(incomplete_episodes)} incomplete episodes: {[e.episode_id for e in incomplete_episodes]}"
        )
        # Update dataset with only incomplete episodes
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=incomplete_episodes
        )

    # filter episodes by mod for running on multiple nodes
    if config.get("episode_mod_filter", None) is not None:
        rem, mod = config.episode_mod_filter
        episode_subset = [x for x in dataset.episodes if int(x.episode_id) % mod == rem]
        print(f"Mod filter: {rem}, {mod}")
        print(f"Episodes: {[e.episode_id for e in episode_subset]}")
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )

    num_episodes = len(dataset.episodes)
    if config.num_proc == 1:
        if config.get("episode_indices", None) is not None:
            if config.get("resume", False):
                raise ValueError("episode_indices and resume cannot be used together")
            episode_subset = [dataset.episodes[x] for x in config.episode_indices]
            dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )
        run_planner(config, dataset)
    else:
        # Process episodes in parallel
        mp_ctx = mp.get_context("forkserver")
        proc_infos = []
        config.num_proc = min(config.num_proc, num_episodes)
        ochunk_size = num_episodes // config.num_proc
        # Prepare chunked datasets
        chunked_datasets = []
        # TODO: we may want to chunk by scene
        start = 0
        for i in range(config.num_proc):
            chunk_size = ochunk_size
            if i < (num_episodes % config.num_proc):
                chunk_size += 1
            end = min(start + chunk_size, num_episodes)
            indices = slice(start, end)
            chunked_datasets.append(indices)
            start += chunk_size

        for episode_index_chunk in chunked_datasets:
            episode_subset = dataset.episodes[episode_index_chunk]
            new_dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )

            parent_conn, child_conn = mp_ctx.Pipe()
            proc_args = (config, new_dataset, child_conn)
            p = mp_ctx.Process(target=run_planner, args=proc_args)
            p.start()
            proc_infos.append((parent_conn, p))
            print("START PROCESS")

        # Get back info
        all_stats_episodes: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
        }
        all_action_metrics: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
            }
        all_action_obj_in_graph_metrics: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
            }
        all_action_curr_next_metrics: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
            }

        for conn, proc in proc_infos:
            if config.evaluation.get("predict_other_agent_actions", False):
                stats_episodes, action_metrics, action_obj_in_graph_metrics, action_curr_next_metrics = conn.recv()
            else:
                stats_episodes = conn.recv()
            for run_id, stats_run in stats_episodes.items():
                all_stats_episodes[str(run_id)].update(stats_run)
                if config.evaluation.get("predict_other_agent_actions", False):
                    all_action_metrics[str(run_id)].update(action_metrics[str(run_id)])
                    all_action_obj_in_graph_metrics[str(run_id)].update(action_obj_in_graph_metrics[str(run_id)])
                    all_action_curr_next_metrics[str(run_id)].update(action_curr_next_metrics[str(run_id)])
            proc.join()

        if config.evaluation.get("predict_other_agent_actions", False):
            agg_measures = aggregate_measures_oaa
            
            if config.evaluation.get("predict_other_agent_actions", False) and all_action_metrics != {}:    
                all_aggregated_action_metrics = aggregate_action_measures(
                    {run_id: aggregate_action_measures(v) for run_id, v in all_action_metrics.items()}
                )
            
                file_path_all_action_metrics_json = config.paths.end_action_result_file_path
                
                # Make directory if it doesn't exists already
                os.makedirs(os.path.dirname(file_path_all_action_metrics_json), exist_ok=True)
                        
                with open(file_path_all_action_metrics_json, "w+") as file:
                    file.write(json.dumps(all_aggregated_action_metrics))
                    
            if config.evaluation.get("predict_other_agent_actions", False) and all_action_obj_in_graph_metrics != {}:    
                all_aggregated_action_obj_in_graph_metrics = aggregate_action_measures(
                    {run_id: aggregate_action_measures(v, is_obj_in_graph = True) for run_id, v in all_action_obj_in_graph_metrics.items()}, is_obj_in_graph = True
                )
            
                file_path_all_action_obj_in_graph_metrics_json = config.paths.end_action_obj_in_graph_result_file_path
                
                # Make directory if it doesn't exists already
                os.makedirs(os.path.dirname(file_path_all_action_obj_in_graph_metrics_json), exist_ok=True)
                        
                with open(file_path_all_action_obj_in_graph_metrics_json, "w+") as file:
                    file.write(json.dumps(all_aggregated_action_obj_in_graph_metrics))
            
            if config.evaluation.get("predict_other_agent_actions", False) and all_action_curr_next_metrics != {}:    
                all_aggregated_action_curr_next_metrics = aggregate_action_measures(
                    {run_id: aggregate_action_measures(v) for run_id, v in all_action_curr_next_metrics.items()}
                )
        
                file_path_all_action_curr_next_metrics_json = config.paths.end_action_curr_next_result_file_path
                
                # Make directory if it doesn't exists already
                os.makedirs(os.path.dirname(file_path_all_action_curr_next_metrics_json), exist_ok=True)
                        
                with open(file_path_all_action_curr_next_metrics_json, "w+") as file:
                    file.write(json.dumps(all_aggregated_action_curr_next_metrics))
                    
        else:
            agg_measures = aggregate_measures
            
        all_metrics = agg_measures(
            {run_id: agg_measures(v) for run_id, v in all_stats_episodes.items()}
        )
        cprint("\n---------------------------------", "blue")
        cprint("Metrics Across All Runs:", "blue")
        for k, v in all_metrics.items():
            if v is not None and isinstance(v, (int, float)):
                cprint(f"{k}: {v:.3f}", "blue")
            else:
                cprint(f"{k}: {v}", "blue")
        cprint("\n---------------------------------", "blue")
        # Write aggregated results across experiment
        write_to_csv(config.paths.end_result_file_path, all_metrics)

    e_t = time.time() - t0
    print(f"Time elapsed since start of experiment: {e_t} seconds.")


def run_planner(config, dataset: CollaborationDatasetV0 = None, conn=None):
    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return

    # Setup interface with the simulator if the planner depends on it
    if config.env == "habitat":
        # Remove sensors if we are not saving video

        # TODO: have a flag for this, or some check
        keep_rgb = False
        if "use_rgb" in config.evaluation:
            keep_rgb = config.evaluation.use_rgb
        if not config.evaluation.save_video and not keep_rgb:
            remove_visual_sensors(config)

        # TODO: Can we move this inside the EnvironmentInterface?
        # We register the dynamic habitat sensors
        register_sensors(config)
        # We register custom actions
        register_actions(config)
        # We register custom measures
        register_measures(config)

        # Initialize the environment interface for the agent
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        try:
            env_interface.initialize_perception_and_world_graph()
        except Exception:
            print("Error initializing the environment")
            if config.evaluation.log_data:
                save_exception_message(config, env_interface)
    else:
        env_interface = None

    # Instantiate the agent planner
    eval_runner: EvaluationRunner = None
    if config.evaluation.type == "centralized":
        eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)
    elif config.evaluation.type == "decentralized":
        eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)
    else:
        cprint(
            "Invalid planner type. Please select between 'centralized' or 'decentralized'. Exiting",
            "red",
        )
        return

    # Print the planner
    cprint(f"Successfully constructed the '{config.evaluation.type}' planner!", "green")
    print(eval_runner)

    # Declare observability mode
    cprint(
        f"Partial observability is set to: '{config.world_model.partial_obs}'", "green"
    )

    # Print the agent list
    print("\nAgent List:")
    print(eval_runner.agent_list)

    # Print the agent description
    print("\nAgent Description:")
    print(eval_runner.agent_descriptions)

    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    # cprint(f"LLM model: {config.planner.llm.llm._target_}", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    cprint("---------------------------------------\n", "blue")

    os.makedirs(config.paths.results_dir, exist_ok=True)
    
    if config.evaluation.get("predict_other_agent_actions", False):
        agg_measures = aggregate_measures_oaa
        action_metrics = {str(i): {} for i in range(config.num_runs_per_episode)}
        action_metrics_obj_in_graph = {str(i): {} for i in range(config.num_runs_per_episode)}
        action_metrics_curr_next = {str(i): {} for i in range(config.num_runs_per_episode)}
    else:
        agg_measures = aggregate_measures

    # Run the planner
    if config.mode == "cli":
        instruction = "Go to the bed" if not config.instruction else config.instruction

        cprint(f'\nExecuting instruction: "{instruction}"', "blue")
        try:
            info = eval_runner.run_instruction(instruction)
        except Exception as e:
            print("An error occurred:", e)

    else:
        stats_episodes: Dict[str, Dict] = {
            str(i): {} for i in range(config.num_runs_per_episode)
        }

        num_episodes = len(env_interface.env.episodes)
        for run_id in range(config.num_runs_per_episode):
            for _ in range(num_episodes):
                # Get episode id
                episode_id = env_interface.env.env.env._env.current_episode.episode_id

                # Get instruction
                instruction = env_interface.env.env.env._env.current_episode.instruction
                print("\n\nEpisode", episode_id)

                try:
                    info = eval_runner.run_instruction(
                        output_name=f"episode_{episode_id}_{run_id}"
                    )

                    info_episode = {
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "instruction": instruction,
                    }
                    stats_keys = {
                        "task_percent_complete",
                        "task_state_success",
                        "sim_step_count",
                        "replanning_count",
                        "runtime",
                    }

                    # add replanning counts to stats_keys as scalars if replanning_count is a dict
                    if "replanning_count" in info and isinstance(
                        info["replanning_count"], dict
                    ):
                        for agent_id, replan_count in info["replanning_count"].items():
                            stats_keys.add(f"replanning_count_{agent_id}")
                            info[f"replanning_count_{agent_id}"] = replan_count

                    # add reasoning_about_other_agent_count to stats_keys as scalars if reasoning_about_other_agent_count is a dict
                    if "reasoning_about_other_agent_count" in info and isinstance(
                        info["reasoning_about_other_agent_count"], dict
                    ):
                        for agent_id, reasoning_about_other_agent_count in info["reasoning_about_other_agent_count"].items():
                            stats_keys.add(f"reasoning_about_other_agent_count_{agent_id}")
                            info[f"reasoning_about_other_agent_count_{agent_id}"] = reasoning_about_other_agent_count
                    
                    # add total oaa pred count to stats_keys as scalars if oaa_total_pred_count is a dict
                    if "oaa_total_pred_count" in info and isinstance(
                        info["oaa_total_pred_count"], dict
                    ):
                        for agent_id, oaa_total_pred_count in info["oaa_total_pred_count"].items():
                            stats_keys.add(f"oaa_total_pred_count_{agent_id}")
                            info[f"oaa_total_pred_count_{agent_id}"] = oaa_total_pred_count

                    # add total oaa pred count to stats_keys as scalars if oaa_total_pred_count is a dict
                    if "oa_not_done_pred_count" in info and isinstance(
                        info["oa_not_done_pred_count"], dict
                    ):
                        for agent_id, oa_not_done_pred_count in info["oa_not_done_pred_count"].items():
                            stats_keys.add(f"oa_not_done_pred_count_{agent_id}")
                            info[f"oa_not_done_pred_count_{agent_id}"] = oa_not_done_pred_count
                    
                    # add correct oaa action pred count to stats_keys as scalars if oaa_correct_action_pred_count is a dict
                    if "oaa_correct_action_pred_count" in info and isinstance(
                        info["oaa_correct_action_pred_count"], dict
                    ):
                        for agent_id, oaa_correct_action_pred_count in info["oaa_correct_action_pred_count"].items():
                            stats_keys.add(f"oaa_correct_action_pred_count_{agent_id}")
                            info[f"oaa_correct_action_pred_count_{agent_id}"] = oaa_correct_action_pred_count

                    # add oaa action pred accuracy to stats_keys as scalars if oaa_pred_action_accuracy is a dict
                    if "oaa_pred_action_accuracy" in info and isinstance(
                        info["oaa_pred_action_accuracy"], dict
                    ):
                        for agent_id, oaa_pred_action_accuracy in info["oaa_pred_action_accuracy"].items():
                            stats_keys.add(f"oaa_pred_action_accuracy_{agent_id}")
                            info[f"oaa_pred_action_accuracy_{agent_id}"] = oaa_pred_action_accuracy

                    # add correct oaa principal pred count to stats_keys as scalars if oaa_correct_principal_pred_count is a dict
                    if "oaa_correct_principal_pred_count" in info and isinstance(
                        info["oaa_correct_principal_pred_count"], dict
                    ):
                        for agent_id, oaa_correct_principal_pred_count in info["oaa_correct_principal_pred_count"].items():
                            stats_keys.add(f"oaa_correct_principal_pred_count_{agent_id}")
                            info[f"oaa_correct_principal_pred_count_{agent_id}"] = oaa_correct_principal_pred_count

                    # add oaa principal pred accuracy to stats_keys as scalars if oaa_pred_principal_accuracy is a dict
                    if "oaa_pred_principal_accuracy" in info and isinstance(
                        info["oaa_pred_principal_accuracy"], dict
                    ):
                        for agent_id, oaa_pred_principal_accuracy in info["oaa_pred_principal_accuracy"].items():
                            stats_keys.add(f"oaa_pred_principal_accuracy_{agent_id}")
                            info[f"oaa_pred_principal_accuracy_{agent_id}"] = oaa_pred_principal_accuracy

                    # add correct oaa action principal pred count to stats_keys as scalars if oaa_correct_action_principal_pred_count is a dict
                    if "oaa_correct_action_principal_pred_count" in info and isinstance(
                        info["oaa_correct_action_principal_pred_count"], dict
                    ):
                        for agent_id, oaa_correct_action_principal_pred_count in info["oaa_correct_action_principal_pred_count"].items():
                            stats_keys.add(f"oaa_correct_action_principal_pred_count_{agent_id}")
                            info[f"oaa_correct_action_principal_pred_count_{agent_id}"] = oaa_correct_action_principal_pred_count

                    # add oaa action principal pred accuracy to stats_keys as scalars if oaa_pred_action_principal_accuracy is a dict
                    if "oaa_pred_action_principal_accuracy" in info and isinstance(
                        info["oaa_pred_action_principal_accuracy"], dict
                    ):
                        for agent_id, oaa_pred_action_principal_accuracy in info["oaa_pred_action_principal_accuracy"].items():
                            stats_keys.add(f"oaa_pred_action_principal_accuracy_{agent_id}")
                            info[f"oaa_pred_action_principal_accuracy_{agent_id}"] = oaa_pred_action_principal_accuracy

                    # add correct oaa all args pred count to stats_keys as scalars if oaa_correct_all_args_pred_count is a dict
                    if "oaa_correct_all_args_pred_count" in info and isinstance(
                        info["oaa_correct_all_args_pred_count"], dict
                    ):
                        for agent_id, oaa_correct_all_args_pred_count in info["oaa_correct_all_args_pred_count"].items():
                            stats_keys.add(f"oaa_correct_all_args_pred_count_{agent_id}")
                            info[f"oaa_correct_all_args_pred_count_{agent_id}"] = oaa_correct_all_args_pred_count

                    # add oaa all args pred accuracy to stats_keys as scalars if oaa_pred_all_args_accuracy is a dict
                    if "oaa_pred_all_args_accuracy" in info and isinstance(
                        info["oaa_pred_all_args_accuracy"], dict
                    ):
                        for agent_id, oaa_pred_all_args_accuracy in info["oaa_pred_all_args_accuracy"].items():
                            stats_keys.add(f"oaa_pred_all_args_accuracy_{agent_id}")
                            info[f"oaa_pred_all_args_accuracy_{agent_id}"] = oaa_pred_all_args_accuracy

                    # add correct oaa action all args pred count to stats_keys as scalars if oaa_correct_action_all_args_pred_count is a dict
                    if "oaa_correct_action_all_args_pred_count" in info and isinstance(
                        info["oaa_correct_action_all_args_pred_count"], dict
                    ):
                        for agent_id, oaa_correct_action_all_args_pred_count in info["oaa_correct_action_all_args_pred_count"].items():
                            stats_keys.add(f"oaa_correct_action_all_args_pred_count_{agent_id}")
                            info[f"oaa_correct_action_all_args_pred_count_{agent_id}"] = oaa_correct_action_all_args_pred_count
                            
                    # add oaa action all args pred accuracy to stats_keys as scalars if oaa_pred_action_all_args_accuracy is a dict
                    if "oaa_pred_action_all_args_accuracy" in info and isinstance(
                        info["oaa_pred_action_all_args_accuracy"], dict
                    ):
                        for agent_id, oaa_pred_action_all_args_accuracy in info["oaa_pred_action_all_args_accuracy"].items():
                            stats_keys.add(f"oaa_pred_action_all_args_accuracy_{agent_id}")
                            info[f"oaa_pred_action_all_args_accuracy_{agent_id}"] = oaa_pred_action_all_args_accuracy                    
                    
                    # add oaa pred match success execution count to stats_keys as scalars if oaa_pred_match_success_execution_count is a dict
                    if "oaa_pred_match_success_execution_count" in info and isinstance(
                        info["oaa_pred_match_success_execution_count"], dict
                    ):
                        for agent_id, oaa_pred_match_success_execution_count in info["oaa_pred_match_success_execution_count"].items():
                            stats_keys.add(f"oaa_pred_match_success_execution_count_{agent_id}")
                            info[f"oaa_pred_match_success_execution_count_{agent_id}"] = oaa_pred_match_success_execution_count
                    
                    # add oaa pred match success execution accuracy to stats_keys as scalars if oaa_pred_match_success_execution_accuracy is a dict
                    if "oaa_pred_match_success_execution_accuracy" in info and isinstance(
                        info["oaa_pred_match_success_execution_accuracy"], dict
                    ):
                        for agent_id, oaa_pred_match_success_execution_accuracy in info["oaa_pred_match_success_execution_accuracy"].items():
                            stats_keys.add(f"oaa_pred_match_success_execution_accuracy_{agent_id}")
                            info[f"oaa_pred_match_success_execution_accuracy_{agent_id}"] = oaa_pred_match_success_execution_accuracy
                    
                    # add oaa pred unmatch success execution count to stats_keys as scalars if oaa_pred_unmatch_success_execution_count is a dict
                    if "oaa_pred_unmatch_success_execution_count" in info and isinstance(
                        info["oaa_pred_unmatch_success_execution_count"], dict
                    ):
                        for agent_id, oaa_pred_unmatch_success_execution_count in info["oaa_pred_unmatch_success_execution_count"].items():
                            stats_keys.add(f"oaa_pred_unmatch_success_execution_count_{agent_id}")
                            info[f"oaa_pred_unmatch_success_execution_count_{agent_id}"] = oaa_pred_unmatch_success_execution_count
                    
                    # add oaa pred unmatch success execution accuracy to stats_keys as scalars if oaa_pred_unmatch_success_execution_accuracy is a dict
                    if "oaa_pred_unmatch_success_execution_accuracy" in info and isinstance(
                        info["oaa_pred_unmatch_success_execution_accuracy"], dict
                    ):
                        for agent_id, oaa_pred_unmatch_success_execution_accuracy in info["oaa_pred_unmatch_success_execution_accuracy"].items():
                            stats_keys.add(f"oaa_pred_unmatch_success_execution_accuracy_{agent_id}")
                            info[f"oaa_pred_unmatch_success_execution_accuracy_{agent_id}"] = oaa_pred_unmatch_success_execution_accuracy
                    
                    if "oaa_correct_action_principal_if_action_pred_accuracy" in info and isinstance(
                        info["oaa_correct_action_principal_if_action_pred_accuracy"], dict):
                        for agent_id, oaa_correct_action_principal_if_action_pred_accuracy in info["oaa_correct_action_principal_if_action_pred_accuracy"].items():
                            stats_keys.add(f"oaa_correct_action_principal_if_action_pred_accuracy_{agent_id}")
                            info[f"oaa_correct_action_principal_if_action_pred_accuracy_{agent_id}"] = oaa_correct_action_principal_if_action_pred_accuracy
                    
                    if "oaa_correct_action_all_args_if_action_pred_accuracy" in info and isinstance(
                        info["oaa_correct_action_all_args_if_action_pred_accuracy"], dict):
                        for agent_id, oaa_correct_action_all_args_if_action_pred_accuracy in info["oaa_correct_action_all_args_if_action_pred_accuracy"].items():
                            stats_keys.add(f"oaa_correct_action_all_args_if_action_pred_accuracy_{agent_id}")
                            info[f"oaa_correct_action_all_args_if_action_pred_accuracy_{agent_id}"] = oaa_correct_action_all_args_if_action_pred_accuracy
                    
                    if "oaa_correct_action_all_args_pred_not_done_count" in info and isinstance(
                        info["oaa_correct_action_all_args_pred_not_done_count"], dict):
                        for agent_id, oaa_correct_action_all_args_pred_not_done_count in info["oaa_correct_action_all_args_pred_not_done_count"].items():
                            stats_keys.add(f"oaa_correct_action_all_args_pred_not_done_count_{agent_id}")
                            info[f"oaa_correct_action_all_args_pred_not_done_count_{agent_id}"] = oaa_correct_action_all_args_pred_not_done_count
                    
                    if config.evaluation.get("predict_other_agent_actions", False) and "per_action_metrics" in info and isinstance(
                        info["per_action_metrics"], dict):
                        action_metrics[str(run_id)][episode_id] = info["per_action_metrics"]
                              
                    # other replanned _obj_in_graph metrics
                    
                    if "oaa_total_pred_count_obj_in_graph" in info and isinstance(
                        info["oaa_total_pred_count_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_total_pred_count in info["oaa_total_pred_count_obj_in_graph"].items():
                            stats_keys.add(f"oaa_total_pred_count_obj_in_graph_{agent_id}")
                            info[f"oaa_total_pred_count_obj_in_graph_{agent_id}"] = oaa_total_pred_count

                    if "oaa_correct_action_pred_count_obj_in_graph" in info and isinstance(
                        info["oaa_correct_action_pred_count_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_correct_action_pred_count in info["oaa_correct_action_pred_count_obj_in_graph"].items():
                            stats_keys.add(f"oaa_correct_action_pred_count_obj_in_graph_{agent_id}")
                            info[f"oaa_correct_action_pred_count_obj_in_graph_{agent_id}"] = oaa_correct_action_pred_count

                    if "oaa_pred_action_accuracy_obj_in_graph" in info and isinstance(
                        info["oaa_pred_action_accuracy_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_pred_action_accuracy in info["oaa_pred_action_accuracy_obj_in_graph"].items():
                            stats_keys.add(f"oaa_pred_action_accuracy_obj_in_graph_{agent_id}")
                            info[f"oaa_pred_action_accuracy_obj_in_graph_{agent_id}"] = oaa_pred_action_accuracy

                    if "oaa_correct_principal_pred_count_obj_in_graph" in info and isinstance(
                        info["oaa_correct_principal_pred_count_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_correct_principal_pred_count in info["oaa_correct_principal_pred_count_obj_in_graph"].items():
                            stats_keys.add(f"oaa_correct_principal_pred_count_obj_in_graph_{agent_id}")
                            info[f"oaa_correct_principal_pred_count_obj_in_graph_{agent_id}"] = oaa_correct_principal_pred_count

                    if "oaa_pred_principal_accuracy_obj_in_graph" in info and isinstance(
                        info["oaa_pred_principal_accuracy_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_pred_principal_accuracy in info["oaa_pred_principal_accuracy_obj_in_graph"].items():
                            stats_keys.add(f"oaa_pred_principal_accuracy_obj_in_graph_{agent_id}")
                            info[f"oaa_pred_principal_accuracy_obj_in_graph_{agent_id}"] = oaa_pred_principal_accuracy

                    if "oaa_correct_action_principal_pred_count_obj_in_graph" in info and isinstance(
                        info["oaa_correct_action_principal_pred_count_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_correct_action_principal_pred_count in info["oaa_correct_action_principal_pred_count_obj_in_graph"].items():
                            stats_keys.add(f"oaa_correct_action_principal_pred_count_obj_in_graph_{agent_id}")
                            info[f"oaa_correct_action_principal_pred_count_obj_in_graph_{agent_id}"] = oaa_correct_action_principal_pred_count

                    if "oaa_pred_action_principal_accuracy_obj_in_graph" in info and isinstance(
                        info["oaa_pred_action_principal_accuracy_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_pred_action_principal_accuracy in info["oaa_pred_action_principal_accuracy_obj_in_graph"].items():
                            stats_keys.add(f"oaa_pred_action_principal_accuracy_obj_in_graph_{agent_id}")
                            info[f"oaa_pred_action_principal_accuracy_obj_in_graph_{agent_id}"] = oaa_pred_action_principal_accuracy

                    if "oaa_correct_all_args_pred_count_obj_in_graph" in info and isinstance(
                        info["oaa_correct_all_args_pred_count_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_correct_all_args_pred_count in info["oaa_correct_all_args_pred_count_obj_in_graph"].items():
                            stats_keys.add(f"oaa_correct_all_args_pred_count_obj_in_graph_{agent_id}")
                            info[f"oaa_correct_all_args_pred_count_obj_in_graph_{agent_id}"] = oaa_correct_all_args_pred_count

                    if "oaa_pred_all_args_accuracy_obj_in_graph" in info and isinstance(
                        info["oaa_pred_all_args_accuracy_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_pred_all_args_accuracy in info["oaa_pred_all_args_accuracy_obj_in_graph"].items():
                            stats_keys.add(f"oaa_pred_all_args_accuracy_obj_in_graph_{agent_id}")
                            info[f"oaa_pred_all_args_accuracy_obj_in_graph_{agent_id}"] = oaa_pred_all_args_accuracy

                    if "oaa_correct_action_all_args_pred_count_obj_in_graph" in info and isinstance(
                        info["oaa_correct_action_all_args_pred_count_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_correct_action_all_args_pred_count in info["oaa_correct_action_all_args_pred_count_obj_in_graph"].items():
                            stats_keys.add(f"oaa_correct_action_all_args_pred_count_obj_in_graph_{agent_id}")
                            info[f"oaa_correct_action_all_args_pred_count_obj_in_graph_{agent_id}"] = oaa_correct_action_all_args_pred_count
                            
                    if "oaa_pred_action_all_args_accuracy_obj_in_graph" in info and isinstance(
                        info["oaa_pred_action_all_args_accuracy_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_pred_action_all_args_accuracy in info["oaa_pred_action_all_args_accuracy_obj_in_graph"].items():
                            stats_keys.add(f"oaa_pred_action_all_args_accuracy_obj_in_graph_{agent_id}")
                            info[f"oaa_pred_action_all_args_accuracy_obj_in_graph_{agent_id}"] = oaa_pred_action_all_args_accuracy                    
                
                    if "oaa_pred_match_success_execution_count_obj_in_graph" in info and isinstance(
                        info["oaa_pred_match_success_execution_count_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_pred_match_success_execution_count in info["oaa_pred_match_success_execution_count_obj_in_graph"].items():
                            stats_keys.add(f"oaa_pred_match_success_execution_count_obj_in_graph_{agent_id}")
                            info[f"oaa_pred_match_success_execution_count_obj_in_graph_{agent_id}"] = oaa_pred_match_success_execution_count
                    
                    if "oaa_pred_match_success_execution_accuracy_obj_in_graph" in info and isinstance(
                        info["oaa_pred_match_success_execution_accuracy_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_pred_match_success_execution_accuracy in info["oaa_pred_match_success_execution_accuracy_obj_in_graph"].items():
                            stats_keys.add(f"oaa_pred_match_success_execution_accuracy_obj_in_graph_{agent_id}")
                            info[f"oaa_pred_match_success_execution_accuracy_obj_in_graph_{agent_id}"] = oaa_pred_match_success_execution_accuracy
                    
                    if "oaa_pred_unmatch_success_execution_count_obj_in_graph" in info and isinstance(
                        info["oaa_pred_unmatch_success_execution_count_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_pred_unmatch_success_execution_count in info["oaa_pred_unmatch_success_execution_count_obj_in_graph"].items():
                            stats_keys.add(f"oaa_pred_unmatch_success_execution_count_obj_in_graph_{agent_id}")
                            info[f"oaa_pred_unmatch_success_execution_count_obj_in_graph_{agent_id}"] = oaa_pred_unmatch_success_execution_count
                    
                    if "oaa_pred_unmatch_success_execution_accuracy_obj_in_graph" in info and isinstance(
                        info["oaa_pred_unmatch_success_execution_accuracy_obj_in_graph"], dict
                    ):
                        for agent_id, oaa_pred_unmatch_success_execution_accuracy in info["oaa_pred_unmatch_success_execution_accuracy_obj_in_graph"].items():
                            stats_keys.add(f"oaa_pred_unmatch_success_execution_accuracy_obj_in_graph_{agent_id}")
                            info[f"oaa_pred_unmatch_success_execution_accuracy_obj_in_graph_{agent_id}"] = oaa_pred_unmatch_success_execution_accuracy
                    
                    if "oaa_correct_action_principal_if_action_pred_accuracy_obj_in_graph" in info and isinstance(
                        info["oaa_correct_action_principal_if_action_pred_accuracy_obj_in_graph"], dict):
                        for agent_id, oaa_correct_action_principal_if_action_pred_accuracy in info["oaa_correct_action_principal_if_action_pred_accuracy_obj_in_graph"].items():
                            stats_keys.add(f"oaa_correct_action_principal_if_action_pred_accuracy_obj_in_graph_{agent_id}")
                            info[f"oaa_correct_action_principal_if_action_pred_accuracy_obj_in_graph_{agent_id}"] = oaa_correct_action_principal_if_action_pred_accuracy
                    
                    if "oaa_correct_action_all_args_if_action_pred_accuracy_obj_in_graph" in info and isinstance(
                        info["oaa_correct_action_all_args_if_action_pred_accuracy_obj_in_graph"], dict):
                        for agent_id, oaa_correct_action_all_args_if_action_pred_accuracy in info["oaa_correct_action_all_args_if_action_pred_accuracy_obj_in_graph"].items():
                            stats_keys.add(f"oaa_correct_action_all_args_if_action_pred_accuracy_obj_in_graph_{agent_id}")
                            info[f"oaa_correct_action_all_args_if_action_pred_accuracy_obj_in_graph_{agent_id}"] = oaa_correct_action_all_args_if_action_pred_accuracy
                    
                    if config.evaluation.get("predict_other_agent_actions", False) and "per_action_metrics_obj_in_graph" in info and isinstance(
                        info["per_action_metrics_obj_in_graph"], dict):
                        action_metrics_obj_in_graph[str(run_id)][episode_id] = info["per_action_metrics_obj_in_graph"]
                    
                    # _curr_next metrics
                    

                    if "oaa_correct_action_pred_count_curr_next" in info and isinstance(
                        info["oaa_correct_action_pred_count_curr_next"], dict
                    ):
                        for agent_id, oaa_correct_action_pred_count in info["oaa_correct_action_pred_count_curr_next"].items():
                            stats_keys.add(f"oaa_correct_action_pred_count_curr_next_{agent_id}")
                            info[f"oaa_correct_action_pred_count_curr_next_{agent_id}"] = oaa_correct_action_pred_count

                    if "oaa_pred_action_accuracy_curr_next" in info and isinstance(
                        info["oaa_pred_action_accuracy_curr_next"], dict
                    ):
                        for agent_id, oaa_pred_action_accuracy in info["oaa_pred_action_accuracy_curr_next"].items():
                            stats_keys.add(f"oaa_pred_action_accuracy_curr_next_{agent_id}")
                            info[f"oaa_pred_action_accuracy_curr_next_{agent_id}"] = oaa_pred_action_accuracy

                    if "oaa_correct_principal_pred_count_curr_next" in info and isinstance(
                        info["oaa_correct_principal_pred_count_curr_next"], dict
                    ):
                        for agent_id, oaa_correct_principal_pred_count in info["oaa_correct_principal_pred_count_curr_next"].items():
                            stats_keys.add(f"oaa_correct_principal_pred_count_curr_next_{agent_id}")
                            info[f"oaa_correct_principal_pred_count_curr_next_{agent_id}"] = oaa_correct_principal_pred_count

                    if "oaa_pred_principal_accuracy_curr_next" in info and isinstance(
                        info["oaa_pred_principal_accuracy_curr_next"], dict
                    ):
                        for agent_id, oaa_pred_principal_accuracy in info["oaa_pred_principal_accuracy_curr_next"].items():
                            stats_keys.add(f"oaa_pred_principal_accuracy_curr_next_{agent_id}")
                            info[f"oaa_pred_principal_accuracy_curr_next_{agent_id}"] = oaa_pred_principal_accuracy

                    if "oaa_correct_action_principal_pred_count_curr_next" in info and isinstance(
                        info["oaa_correct_action_principal_pred_count_curr_next"], dict
                    ):
                        for agent_id, oaa_correct_action_principal_pred_count in info["oaa_correct_action_principal_pred_count_curr_next"].items():
                            stats_keys.add(f"oaa_correct_action_principal_pred_count_curr_next_{agent_id}")
                            info[f"oaa_correct_action_principal_pred_count_curr_next_{agent_id}"] = oaa_correct_action_principal_pred_count

                    if "oaa_pred_action_principal_accuracy_curr_next" in info and isinstance(
                        info["oaa_pred_action_principal_accuracy_curr_next"], dict
                    ):
                        for agent_id, oaa_pred_action_principal_accuracy in info["oaa_pred_action_principal_accuracy_curr_next"].items():
                            stats_keys.add(f"oaa_pred_action_principal_accuracy_curr_next_{agent_id}")
                            info[f"oaa_pred_action_principal_accuracy_curr_next_{agent_id}"] = oaa_pred_action_principal_accuracy

                    if "oaa_correct_all_args_pred_count_curr_next" in info and isinstance(
                        info["oaa_correct_all_args_pred_count_curr_next"], dict
                    ):
                        for agent_id, oaa_correct_all_args_pred_count in info["oaa_correct_all_args_pred_count_curr_next"].items():
                            stats_keys.add(f"oaa_correct_all_args_pred_count_curr_next_{agent_id}")
                            info[f"oaa_correct_all_args_pred_count_curr_next_{agent_id}"] = oaa_correct_all_args_pred_count

                    if "oaa_pred_all_args_accuracy_curr_next" in info and isinstance(
                        info["oaa_pred_all_args_accuracy_curr_next"], dict
                    ):
                        for agent_id, oaa_pred_all_args_accuracy in info["oaa_pred_all_args_accuracy_curr_next"].items():
                            stats_keys.add(f"oaa_pred_all_args_accuracy_curr_next_{agent_id}")
                            info[f"oaa_pred_all_args_accuracy_curr_next_{agent_id}"] = oaa_pred_all_args_accuracy

                    if "oaa_correct_action_all_args_pred_count_curr_next" in info and isinstance(
                        info["oaa_correct_action_all_args_pred_count_curr_next"], dict
                    ):
                        for agent_id, oaa_correct_action_all_args_pred_count in info["oaa_correct_action_all_args_pred_count_curr_next"].items():
                            stats_keys.add(f"oaa_correct_action_all_args_pred_count_curr_next_{agent_id}")
                            info[f"oaa_correct_action_all_args_pred_count_curr_next_{agent_id}"] = oaa_correct_action_all_args_pred_count

                    if "oaa_pred_action_all_args_accuracy_curr_next" in info and isinstance(
                        info["oaa_pred_action_all_args_accuracy_curr_next"], dict
                    ):
                        for agent_id, oaa_pred_action_all_args_accuracy in info["oaa_pred_action_all_args_accuracy_curr_next"].items():
                            stats_keys.add(f"oaa_pred_action_all_args_accuracy_curr_next_{agent_id}")
                            info[f"oaa_pred_action_all_args_accuracy_curr_next_{agent_id}"] = oaa_pred_action_all_args_accuracy                    
                
                    if "oaa_pred_match_success_execution_count_curr_next" in info and isinstance(
                        info["oaa_pred_match_success_execution_count_curr_next"], dict
                    ):
                        for agent_id, oaa_pred_match_success_execution_count in info["oaa_pred_match_success_execution_count_curr_next"].items():
                            stats_keys.add(f"oaa_pred_match_success_execution_count_curr_next_{agent_id}")
                            info[f"oaa_pred_match_success_execution_count_curr_next_{agent_id}"] = oaa_pred_match_success_execution_count
                    
                    if "oaa_pred_match_success_execution_accuracy_curr_next" in info and isinstance(
                        info["oaa_pred_match_success_execution_accuracy_curr_next"], dict
                    ):
                        for agent_id, oaa_pred_match_success_execution_accuracy in info["oaa_pred_match_success_execution_accuracy_curr_next"].items():
                            stats_keys.add(f"oaa_pred_match_success_execution_accuracy_curr_next_{agent_id}")
                            info[f"oaa_pred_match_success_execution_accuracy_curr_next_{agent_id}"] = oaa_pred_match_success_execution_accuracy
                    
                    if "oaa_pred_unmatch_success_execution_count_curr_next" in info and isinstance(
                        info["oaa_pred_unmatch_success_execution_count_curr_next"], dict
                    ):
                        for agent_id, oaa_pred_unmatch_success_execution_count in info["oaa_pred_unmatch_success_execution_count_curr_next"].items():
                            stats_keys.add(f"oaa_pred_unmatch_success_execution_count_curr_next_{agent_id}")
                            info[f"oaa_pred_unmatch_success_execution_count_curr_next_{agent_id}"] = oaa_pred_unmatch_success_execution_count
                    
                    if "oaa_pred_unmatch_success_execution_accuracy_curr_next" in info and isinstance(
                        info["oaa_pred_unmatch_success_execution_accuracy_curr_next"], dict
                    ):
                        for agent_id, oaa_pred_unmatch_success_execution_accuracy in info["oaa_pred_unmatch_success_execution_accuracy_curr_next"].items():
                            stats_keys.add(f"oaa_pred_unmatch_success_execution_accuracy_curr_next_{agent_id}")
                            info[f"oaa_pred_unmatch_success_execution_accuracy_curr_next_{agent_id}"] = oaa_pred_unmatch_success_execution_accuracy
                    
                    if "oaa_correct_action_principal_if_action_pred_accuracy_curr_next" in info and isinstance(
                        info["oaa_correct_action_principal_if_action_pred_accuracy_curr_next"], dict):
                        for agent_id, oaa_correct_action_principal_if_action_pred_accuracy in info["oaa_correct_action_principal_if_action_pred_accuracy_curr_next"].items():
                            stats_keys.add(f"oaa_correct_action_principal_if_action_pred_accuracy_curr_next_{agent_id}")
                            info[f"oaa_correct_action_principal_if_action_pred_accuracy_curr_next_{agent_id}"] = oaa_correct_action_principal_if_action_pred_accuracy
                    
                    if "oaa_correct_action_all_args_if_action_pred_accuracy_curr_next" in info and isinstance(
                        info["oaa_correct_action_all_args_if_action_pred_accuracy_curr_next"], dict):
                        for agent_id, oaa_correct_action_all_args_if_action_pred_accuracy in info["oaa_correct_action_all_args_if_action_pred_accuracy_curr_next"].items():
                            stats_keys.add(f"oaa_correct_action_all_args_if_action_pred_accuracy_curr_next_{agent_id}")
                            info[f"oaa_correct_action_all_args_if_action_pred_accuracy_curr_next_{agent_id}"] = oaa_correct_action_all_args_if_action_pred_accuracy

                    if "oaa_correct_action_all_args_pred_not_done_count_curr_next" in info and isinstance(
                        info["oaa_correct_action_all_args_pred_not_done_count_curr_next"], dict):
                        for agent_id, oaa_correct_action_all_args_pred_not_done_count in info["oaa_correct_action_all_args_pred_not_done_count_curr_next"].items():
                            stats_keys.add(f"oaa_correct_action_all_args_pred_not_done_count_curr_next_{agent_id}")
                            info[f"oaa_correct_action_all_args_pred_not_done_count_curr_next_{agent_id}"] = oaa_correct_action_all_args_pred_not_done_count

                    if config.evaluation.get("predict_other_agent_actions", False) and "per_action_metrics_curr_next" in info and isinstance(
                        info["per_action_metrics_curr_next"], dict):
                        action_metrics_curr_next[str(run_id)][episode_id] = info["per_action_metrics_curr_next"]
                    
                    # if "per_action_metrics" in info and isinstance(
                    #     info["per_action_metrics"], dict):
                    #     for agent_id, agent_metrics in info["per_action_metrics"].items():
                    #         for action_name, action_metrics in agent_metrics.items():
                    #             for metric_name, metric_value in action_metrics.items():
                    #                 stats_keys.add(f"action_{action_name}_{metric_name}_{agent_id}")
                    #                 info[f"action_{action_name}_{metric_name}_{agent_id}"] = metric_value
                    
                    # print('**********************************')
                    # print(f'info:  {info}')
                    # print(f'stats_keys: {stats_keys}')
                    # print('**********************************')
                    stats_episode = extract_scalars_from_info(
                        info, ignore_keys=info.keys() - stats_keys
                    )
                    stats_episodes[str(run_id)][episode_id] = stats_episode

                    cprint("\n---------------------------------", "blue")
                    cprint(f"Metrics For Run {run_id} Episode {episode_id}:", "blue")
                    for k, v in stats_episodes[str(run_id)][episode_id].items():
                        if v is not None and isinstance(v, (int, float)):
                            cprint(f"{k}: {v:.3f}", "blue")
                        else:
                            cprint(f"{k}: {v}", "blue")
                    cprint("\n---------------------------------", "blue")
                    # Log results onto a CSV
                    epi_metrics = stats_episodes[str(run_id)][episode_id] | info_episode
                    if config.evaluation.log_data:
                        save_success_message(config, env_interface, stats_episode)
                    write_to_csv(config.paths.epi_result_file_path, epi_metrics)
                except Exception as e:
                    # print exception and trace
                    traceback.print_exc()
                    print("An error occurred while running the episode:", e)
                    print(f"Skipping evaluating episode: {episode_id}")
                    if config.evaluation.log_data:
                        save_exception_message(config, env_interface)

                try:
                    # Reset env_interface (moves onto the next episode in the dataset)
                    env_interface.reset_environment()
                except Exception as e:
                    # print exception and trace
                    traceback.print_exc()
                    print("An error occurred while resetting the env_interface:", e)
                    print("Skipping evaluating episode.")
                    if config.evaluation.log_data:
                        save_exception_message(config, env_interface)

                # Reset evaluation runner
                eval_runner.reset()

            # aggregate metrics across the current run.
            run_metrics = agg_measures(stats_episodes[str(run_id)])
            cprint("\n---------------------------------", "blue")
            cprint(f"Metrics For Run {run_id}:", "blue")
            for k, v in run_metrics.items():
                if v is not None and isinstance(v, (int, float)):
                    cprint(f"{k}: {v:.3f}", "blue")
                else:
                    cprint(f"{k}: {v}", "blue")
            cprint("\n---------------------------------", "blue")

            # Write aggregated results across run
            write_to_csv(config.paths.run_result_file_path, run_metrics)
            
            if config.evaluation.get("predict_other_agent_actions", False) and "per_action_metrics" in info and isinstance(
                info["per_action_metrics"], dict):
                
                run_action_metrics = aggregate_action_measures(action_metrics[str(run_id)])
                
                file_path_action_metrics_json = config.paths.run_action_result_file_path
                        
                # Make directory if it doesn't exists already
                os.makedirs(os.path.dirname(file_path_action_metrics_json), exist_ok=True)
                        
                with open(file_path_action_metrics_json, "a") as file:
                    file.write(json.dumps(run_action_metrics))
                    
            if config.evaluation.get("predict_other_agent_actions", False) and "per_action_metrics_obj_in_graph" in info and isinstance(
                info["per_action_metrics_obj_in_graph"], dict):
                
                run_action_obj_in_graph_metrics = aggregate_action_measures(action_metrics_obj_in_graph[str(run_id)], is_obj_in_graph = True)
                
                file_path_action_obj_in_graph_metrics_json = config.paths.run_action_obj_in_graph_result_file_path
                        
                # Make directory if it doesn't exists already
                os.makedirs(os.path.dirname(file_path_action_obj_in_graph_metrics_json), exist_ok=True)
                        
                with open(file_path_action_obj_in_graph_metrics_json, "a") as file:
                    file.write(json.dumps(run_action_obj_in_graph_metrics))
            
            if config.evaluation.get("predict_other_agent_actions", False) and "per_action_metrics_curr_next" in info and isinstance(
                info["per_action_metrics_curr_next"], dict):
                
                run_action_curr_next_metrics = aggregate_action_measures(action_metrics_curr_next[str(run_id)])
                
                file_path_action_curr_next_metrics_json = config.paths.run_action_curr_next_result_file_path
                        
                # Make directory if it doesn't exists already
                os.makedirs(os.path.dirname(file_path_action_curr_next_metrics_json), exist_ok=True)
                        
                with open(file_path_action_curr_next_metrics_json, "a") as file:
                    file.write(json.dumps(run_action_curr_next_metrics))

        # aggregate metrics across all runs.
        if conn is None:
            all_metrics = agg_measures(
                {run_id: agg_measures(v) for run_id, v in stats_episodes.items()}
            )
            cprint("\n---------------------------------", "blue")
            cprint("Metrics Across All Runs:", "blue")
            for k, v in all_metrics.items():
                if v is not None and isinstance(v, (int, float)):
                    cprint(f"{k}: {v:.3f}", "blue")
                else:
                    cprint(f"{k}: {v}", "blue")
            cprint("\n---------------------------------", "blue")
            # Write aggregated results across experiment
            write_to_csv(config.paths.end_result_file_path, all_metrics)
            
            if config.evaluation.get("predict_other_agent_actions", False) and action_metrics != {}:
                all_action_metrics = aggregate_action_measures(
                    {run_id: aggregate_action_measures(v) for run_id, v in action_metrics.items()}
                )
        
                file_path_all_action_metrics_json = config.paths.end_action_result_file_path
               
                # Make directory if it doesn't exists already
                os.makedirs(os.path.dirname(file_path_all_action_metrics_json), exist_ok=True)
                        
                with open(file_path_all_action_metrics_json, "w+") as file:
                    file.write(json.dumps(all_action_metrics))
                    
            if config.evaluation.get("predict_other_agent_actions", False) and action_metrics_obj_in_graph != {}:
                all_action_obj_in_graph_metrics = aggregate_action_measures(
                    {run_id: aggregate_action_measures(v, is_obj_in_graph = True) for run_id, v in action_metrics_obj_in_graph.items()}, is_obj_in_graph = True
                )
        
                file_path_all_action_obj_in_graph_metrics_json = config.paths.end_action_obj_in_graph_result_file_path
               
                # Make directory if it doesn't exists already
                os.makedirs(os.path.dirname(file_path_all_action_obj_in_graph_metrics_json), exist_ok=True)
                        
                with open(file_path_all_action_obj_in_graph_metrics_json, "w+") as file:
                    file.write(json.dumps(all_action_obj_in_graph_metrics))
            
            if config.evaluation.get("predict_other_agent_actions", False) and action_metrics_curr_next != {}:
                all_action_curr_next_metrics = aggregate_action_measures(
                    {run_id: aggregate_action_measures(v) for run_id, v in action_metrics_curr_next.items()}
                )
        
                file_path_all_action_curr_next_metrics_json = config.paths.end_action_curr_next_result_file_path
        
                # Make directory if it doesn't exists already
                os.makedirs(os.path.dirname(file_path_all_action_curr_next_metrics_json), exist_ok=True)
                        
                with open(file_path_all_action_curr_next_metrics_json, "w+") as file:
                    file.write(json.dumps(all_action_curr_next_metrics))
             
        else:
            if config.evaluation.get("predict_other_agent_actions", False):
                conn.send((stats_episodes, action_metrics, action_metrics_obj_in_graph, action_metrics_curr_next))
            else:
                conn.send(stats_episodes)

    env_interface.env.close()
    del env_interface

    if conn is not None:
        # Potentially we may want to send something

        conn.close()


if __name__ == "__main__":
    cprint(
        "\nStart of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    # Run planner
    run_eval()

    cprint(
        "\nEnd of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )
