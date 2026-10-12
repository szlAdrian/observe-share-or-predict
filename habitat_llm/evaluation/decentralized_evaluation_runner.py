#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
"""
This module contains the DecentralizedEvaluationRunner class, which implements a decentralized evaluation strategy
where multiple planners independently control different agents. It extends the base EvaluationRunner class
to provide functionality for running multi-agent evaluations with decentralized control.
"""

from typing import TYPE_CHECKING, Any, Dict, Tuple, List

from hydra.utils import instantiate

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from habitat_llm.world_model.world_graph import WorldGraph

from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.evaluation import EvaluationRunner
from habitat_llm.tools.motor_skills.motor_skill_tool import MotorSkillTool
from habitat_llm.world_model import Object


# Evaluation runner, will go over episodes, run planners and store necessary data
class DecentralizedEvaluationRunner(EvaluationRunner):
    """
    Evaluation runner that implements a decentralized control strategy where separate planners
    independently control different agents. This class handles episode execution, planner initialization,
    and action generation with each agent having its own dedicated planner.
    """

    def __init__(
        self,
        evaluation_runner_config_arg: "DictConfig",  # Hydra config type
        env_arg: EnvironmentInterface,
    ) -> None:
        # Call EvaluationRunner class constructor
        super().__init__(evaluation_runner_config_arg, env_arg)
        
        self.per_action_metrics = {0: {}, 1: {}}
        
        if self.evaluation_runner_config.get("predict_other_agent_actions", False):
            self.oaa_total_pred_count = {0: 0, 1: 0}
            self.oa_not_done_pred_count = {0: 0, 1: 0}
            self.oaa_correct_action_pred_count = {0: 0, 1: 0}
            self.oaa_pred_action_accuracy = {0: None, 1: None}
            self.oaa_correct_principal_pred_count = {0: 0, 1: 0}
            self.oaa_pred_principal_accuracy = {0: None, 1: None}
            self.oaa_correct_action_principal_pred_count = {0: 0, 1: 0}
            self.oaa_pred_action_principal_accuracy = {0: None, 1: None}
            self.oaa_correct_all_args_pred_count = {0: 0, 1: 0}
            self.oaa_pred_all_args_accuracy = {0: None, 1: None}
            self.oaa_correct_action_all_args_pred_count = {0: 0, 1: 0}
            self.oaa_pred_action_all_args_accuracy = {0: None, 1: None}
            self.last_predicted_oaa_matched = {0: None, 1: None}
            self.oaa_pred_match_success_execution_count = {0: 0, 1: 0}
            self.oaa_pred_match_success_execution_accuracy = {0: None, 1: None}
            self.oaa_pred_unmatch_success_execution_count = {0: 0, 1: 0}
            self.oaa_pred_unmatch_success_execution_accuracy = {0: None, 1: None}
            self.oaa_correct_action_all_args_pred_not_done_count = {0: 0, 1: 0}
            
            if self.evaluation_runner_config.get("pred_oaa_same_inference_mode", False):
                self.oaa_correct_action_pred_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_action_accuracy_curr_next = {0: None, 1: None}
                self.oaa_correct_principal_pred_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_principal_accuracy_curr_next = {0: None, 1: None}
                self.oaa_correct_action_principal_pred_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_action_principal_accuracy_curr_next = {0: None, 1: None}
                self.oaa_correct_all_args_pred_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_all_args_accuracy_curr_next = {0: None, 1: None}
                self.oaa_correct_action_all_args_pred_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_action_all_args_accuracy_curr_next = {0: None, 1: None}
                self.last_predicted_oaa_matched_curr_next = {0: None, 1: None}
                self.oaa_pred_match_success_execution_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_match_success_execution_accuracy_curr_next = {0: None, 1: None}
                self.oaa_pred_unmatch_success_execution_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_unmatch_success_execution_accuracy_curr_next = {0: None, 1: None}
                self.oaa_correct_action_all_args_pred_not_done_count_curr_next = {0: 0, 1: 0}
                self.per_action_metrics_curr_next = {0: {}, 1: {}}
            else:
                self.oaa_total_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_correct_action_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_action_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_correct_principal_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_principal_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_correct_action_principal_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_action_principal_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_correct_all_args_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_all_args_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_correct_action_all_args_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_action_all_args_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_pred_match_success_execution_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_match_success_execution_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_pred_unmatch_success_execution_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_unmatch_success_execution_accuracy_obj_in_graph = {0: None, 1: None}
                
                self.per_action_metrics_obj_in_graph = {0: {}, 1: {}}
                self.last_pred_calc_obj_in_graph = {0: None, 1: None}
                
            self.should_check_predicted_action_execution_success = {0: True, 1: True}
            
            self.oaa_preds_and_labels = {0: {'pred':[], 'oa_current_action':[], 'oa_next_action':[], 'oa_current_action_response':[], 'oa_next_action_response':[], 'len_pred': 0, 'len_oa_current_action': 0, 'len_oa_next_action': 0}, 1: {'pred':[], 'oa_current_action':[], 'oa_next_action':[], 'oa_current_action_response':[], 'oa_next_action_response':[], 'len_pred': 0, 'len_oa_current_action': 0, 'len_oa_next_action': 0}} # store the predicted and label actions for each prediction made, used for analysis after the evaluation

    def _initialize_planners(self) -> None:
        """
        Initialize separate planners for each agent based on the evaluation runner configuration.
        Sets up each planner with its corresponding agent and configures special planning modes
        if specified.
        """
        self.planner = {}

        # Set an agent to each planner
        for agent_conf in self.evaluation_runner_config.agents.values():
            planner_conf = agent_conf.planner
            planner = instantiate(planner_conf)
            planner = planner(env_interface=self.env_interface)
            planner.agents = [self.agents[agent_conf.uid]]
            if self.evaluation_runner_config.get("predict_other_agent_actions", False):
                other_agent_uid = 1 - agent_conf.uid
                planner.agents[0].other_agents_tool_descriptions = self.agents[other_agent_uid].tool_descriptions
                planner.agents[0].other_agents_tools = self.agents[other_agent_uid].tools
            self.planner[agent_conf.uid] = planner
            if (
                "planning_mode" in planner_conf.plan_config
                and planner_conf.plan_config.planning_mode == "st"
            ):
                for v in self.planner.values():
                    for agent in v.agents:
                        for tool in agent.tools.values():
                            if isinstance(tool, MotorSkillTool):
                                tool.error_mode = "st"

    # Method to print the object
    def __str__(self) -> str:
        """
        Return string with state of the evaluator
        """
        assert isinstance(self.planner, dict)
        planner_str = " ".join(
            [
                f"{planner_id}:{type(planner_val)}"
                for planner_id, planner_val in self.planner.items()
            ]
        )
        out = f"Decentralized Planner: {planner_str}\n"
        out += f"Number of Agents: {len(self.agents)}"
        return out
    
    def reset(self) -> None:
        """Reset metrics and stats to be ready for the next episode."""
        super().reset()
        self.per_action_metrics = {0: {}, 1: {}}
        
        if self.evaluation_runner_config.get("predict_other_agent_actions", False):
            self.oaa_total_pred_count = {0: 0, 1: 0}
            self.oa_not_done_pred_count = {0: 0, 1: 0}
            self.oaa_correct_action_pred_count = {0: 0, 1: 0}
            self.oaa_pred_action_accuracy = {0: None, 1: None}
            self.oaa_correct_principal_pred_count = {0: 0, 1: 0}
            self.oaa_pred_principal_accuracy = {0: None, 1: None}
            self.oaa_correct_action_principal_pred_count = {0: 0, 1: 0}
            self.oaa_pred_action_principal_accuracy = {0: None, 1: None}
            self.oaa_correct_all_args_pred_count = {0: 0, 1: 0}
            self.oaa_pred_all_args_accuracy = {0: None, 1: None}
            self.oaa_correct_action_all_args_pred_count = {0: 0, 1: 0}
            self.oaa_pred_action_all_args_accuracy = {0: None, 1: None}
            self.last_predicted_oaa_matched = {0: None, 1: None}
            self.oaa_pred_match_success_execution_count = {0: 0, 1: 0}
            self.oaa_pred_match_success_execution_accuracy = {0: None, 1: None}
            self.oaa_pred_unmatch_success_execution_count = {0: 0, 1: 0}
            self.oaa_pred_unmatch_success_execution_accuracy = {0: None, 1: None}
            self.oaa_correct_action_all_args_pred_not_done_count = {0: 0, 1: 0}
            
            if self.evaluation_runner_config.get("pred_oaa_same_inference_mode", False):
                self.oaa_correct_action_pred_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_action_accuracy_curr_next = {0: None, 1: None}
                self.oaa_correct_principal_pred_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_principal_accuracy_curr_next = {0: None, 1: None}
                self.oaa_correct_action_principal_pred_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_action_principal_accuracy_curr_next = {0: None, 1: None}
                self.oaa_correct_all_args_pred_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_all_args_accuracy_curr_next = {0: None, 1: None}
                self.oaa_correct_action_all_args_pred_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_action_all_args_accuracy_curr_next = {0: None, 1: None}
                self.last_predicted_oaa_matched_curr_next = {0: None, 1: None}
                self.oaa_pred_match_success_execution_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_match_success_execution_accuracy_curr_next = {0: None, 1: None}
                self.oaa_pred_unmatch_success_execution_count_curr_next = {0: 0, 1: 0}
                self.oaa_pred_unmatch_success_execution_accuracy_curr_next = {0: None, 1: None}
                self.oaa_correct_action_all_args_pred_not_done_count_curr_next = {0: 0, 1: 0}
                self.per_action_metrics_curr_next = {0: {}, 1: {}}
            else:
                self.oaa_total_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_correct_action_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_action_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_correct_principal_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_principal_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_correct_action_principal_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_action_principal_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_correct_all_args_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_all_args_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_correct_action_all_args_pred_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_action_all_args_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_pred_match_success_execution_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_match_success_execution_accuracy_obj_in_graph = {0: None, 1: None}
                self.oaa_pred_unmatch_success_execution_count_obj_in_graph = {0: 0, 1: 0}
                self.oaa_pred_unmatch_success_execution_accuracy_obj_in_graph = {0: None, 1: None}
                
                self.per_action_metrics_obj_in_graph = {0: {}, 1: {}}
                self.last_pred_calc_obj_in_graph = {0: None, 1: None}
                
            self.should_check_predicted_action_execution_success = {0: True, 1: True}

            self.oaa_preds_and_labels = {0: {'pred':[], 'oa_current_action':[], 'oa_next_action':[], 'oa_current_action_response':[], 'oa_next_action_response':[], 'len_pred': 0, 'len_oa_current_action': 0, 'len_oa_next_action': 0}, 1: {'pred':[], 'oa_current_action':[], 'oa_next_action':[], 'oa_current_action_response':[], 'oa_next_action_response':[], 'len_pred': 0, 'len_oa_current_action': 0, 'len_oa_next_action': 0}} # store the predicted and label actions for each prediction made, used for analysis after the evaluation
    def reset_planners(self) -> None:
        """
        Method to reset planner parameters.
        Usually called after finishing one episode.
        """
        assert isinstance(self.planner, dict)
        for planner in self.planner.values():
            planner.reset()
            
    def _get_first_argument(self, action_args: str) -> str:
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
    
    def _check_object_in_agent_graph(self, action: Tuple, agent_uid: int) -> Tuple[bool, str]:
        """
        Check if the first parameter of an action is an object and if it exists in the agent's world graph.
        
        :param action: The action tuple in format (action_name, action_args, ...)
        :param agent_uid: The UID of the agent to check against
        :return: Tuple (is_object_in_agent_graph, object_name) where:
                - is_object_in_agent_graph: True if the first arg is an Object and exists in agent's graph
                - object_name: The name of the object if it's an object, None otherwise
        """
        if not isinstance(action, tuple) or len(action) < 2:
            return False, None
        
        # Get the first argument
        action_args = action[1]
        first_arg = self._get_first_argument(action_args)
        
        if first_arg is None:
            return False, None
        
        # Check if this is an object in the full world graph
        try:
            node = self.env_interface.full_world_graph.get_node_from_name(first_arg)
            #print('<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            print(f'first_arg: {first_arg}')
            print(f'node: {node}')
            print(f'all objects from agent {agent_uid} graph:')
            for node_ in self.env_interface.world_graph[agent_uid].get_all_objects():
                print(node_)
            
            #print('<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            # Check if it's an Object type
            if not isinstance(node, Object):
                return False, None
        except ValueError:
            # Node doesn't exist in full world graph
            return False, None
        
        # Check if it exists in the agent's world graph
        if self.env_interface.world_graph[agent_uid].has_node(first_arg):
            return True, first_arg
        else:
            return False, first_arg
    
    def _compare_actions_calc_metrics_other_replanned(self, label_action: Any, pred_action: Any, agent_uid: int) -> Dict[str, bool]:
        """
        Compare two actions and/or its arguments for equality and calculate and update metrics. 
        Handles tuple-based actions.
        
        :param label_action: First action with its arguments to compare
        :param pred_action: Second action with its arguments to compare
        :param agent_uid: The UID of the agent whose prediction is being compared
        :return: Dict that contains boolean values indicating whether different levels of action/argument matches occurred
        """
        preds_match = {}
        
        # Handle tuple-based actions (most common format)
        if isinstance(label_action, tuple) and isinstance(pred_action, tuple):
            # Compare the action type and main arguments
            # Ignore None arguments in comparison
            
            # Compare arguments, ignoring None values
            if len(label_action) > 0 and len(pred_action) > 0:
                label_principal_arg = self._get_first_argument(label_action[1])
                pred_principal_arg = self._get_first_argument(pred_action[1])
                # print('-----------------------compare_actions-----------------------')
                # print(f'label_action: {label_action}, pred_action: {pred_action}')
                # print(f'label_action 1st arg: {label_principal_arg}, pred_action 1st arg: {pred_principal_arg}')
                # print('------------------------------------------------------------')
                if label_action[0] not in self.per_action_metrics[agent_uid]:
                    self.per_action_metrics[agent_uid][label_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_when_pred': 0, 'failure_count_when_pred': 0, 'success_count_all': 0, 'failure_count_all': 0}
                if label_action[0] not in self.per_action_metrics_obj_in_graph[agent_uid] and self.last_pred_calc_obj_in_graph[agent_uid]:
                    self.per_action_metrics_obj_in_graph[agent_uid][label_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_when_pred': 0, 'failure_count_when_pred': 0}
                if pred_action[0] not in self.per_action_metrics[agent_uid]:
                    self.per_action_metrics[agent_uid][pred_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_when_pred': 0, 'failure_count_when_pred': 0, 'success_count_all': 0, 'failure_count_all': 0}
                if pred_action[0] not in self.per_action_metrics_obj_in_graph[agent_uid] and self.last_pred_calc_obj_in_graph[agent_uid]:
                    self.per_action_metrics_obj_in_graph[agent_uid][pred_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_when_pred': 0, 'failure_count_when_pred': 0} 
                
                if label_action[0] == pred_action[0]: 
                    # Compare action type
                    self.oaa_correct_action_pred_count[agent_uid] += 1
                    preds_match['action_match'] = True
                    if self.last_pred_calc_obj_in_graph[agent_uid]:
                        self.oaa_correct_action_pred_count_obj_in_graph[agent_uid] += 1
                    
                    # Update the per-action metrics
                    self.per_action_metrics[agent_uid][label_action[0]]['tp'] += 1
                    if self.last_pred_calc_obj_in_graph[agent_uid]:
                        self.per_action_metrics_obj_in_graph[agent_uid][label_action[0]]['tp'] += 1
                    
                else: 
                    # Different action type
                    preds_match['action_match'] = False
                    
                    # Update the per-action metrics
                    self.per_action_metrics[agent_uid][pred_action[0]]['fp'] += 1
                    self.per_action_metrics[agent_uid][label_action[0]]['fn'] += 1
                    if self.last_pred_calc_obj_in_graph[agent_uid]:
                        self.per_action_metrics_obj_in_graph[agent_uid][pred_action[0]]['fp'] += 1
                        self.per_action_metrics_obj_in_graph[agent_uid][label_action[0]]['fn'] += 1
                
                self.oaa_pred_action_accuracy[agent_uid] = self.oaa_correct_action_pred_count[agent_uid]/self.oaa_total_pred_count[agent_uid]
                if self.last_pred_calc_obj_in_graph[agent_uid]:
                    self.oaa_pred_action_accuracy_obj_in_graph[agent_uid] = self.oaa_correct_action_pred_count_obj_in_graph[agent_uid]/self.oaa_total_pred_count_obj_in_graph[agent_uid]        
                
                if label_principal_arg == pred_principal_arg: 
                    # Compare principal argument
                    self.oaa_correct_principal_pred_count[agent_uid] += 1
                    preds_match['principal_match'] = True
                    if self.last_pred_calc_obj_in_graph[agent_uid]:
                        self.oaa_correct_principal_pred_count_obj_in_graph[agent_uid] += 1
                else: 
                    # Different principal argument
                    preds_match['principal_match'] = False 
                    
                self.oaa_pred_principal_accuracy[agent_uid] = self.oaa_correct_principal_pred_count[agent_uid]/self.oaa_total_pred_count[agent_uid]
                if self.last_pred_calc_obj_in_graph[agent_uid]:
                    self.oaa_pred_principal_accuracy_obj_in_graph[agent_uid] = self.oaa_correct_principal_pred_count_obj_in_graph[agent_uid]/self.oaa_total_pred_count_obj_in_graph[agent_uid]    
                
                if (label_action[0] == pred_action[0]) and (label_principal_arg == pred_principal_arg): 
                    # Compare action type and principal argument
                    self.oaa_correct_action_principal_pred_count[agent_uid] += 1
                    preds_match['action_principal_match'] = True
                    if self.last_pred_calc_obj_in_graph[agent_uid]:
                        self.oaa_correct_action_principal_pred_count_obj_in_graph[agent_uid] += 1
                else: 
                    # Different action type or principal argument
                    preds_match['action_principal_match'] = False
                    
                self.oaa_pred_action_principal_accuracy[agent_uid] = self.oaa_correct_action_principal_pred_count[agent_uid]/self.oaa_total_pred_count[agent_uid]
                if self.last_pred_calc_obj_in_graph[agent_uid]:
                    self.oaa_pred_action_principal_accuracy_obj_in_graph[agent_uid] = self.oaa_correct_action_principal_pred_count_obj_in_graph[agent_uid]/self.oaa_total_pred_count_obj_in_graph[agent_uid]
                    
                if label_action[1] == pred_action[1]:
                    # Compare all arguments
                    self.oaa_correct_all_args_pred_count[agent_uid] += 1
                    preds_match['all_args_match'] = True
                    if self.last_pred_calc_obj_in_graph[agent_uid]:
                        self.oaa_correct_all_args_pred_count_obj_in_graph[agent_uid] += 1
                else: 
                    # Different arguments
                    preds_match['all_args_match'] = False
                
                self.oaa_pred_all_args_accuracy[agent_uid] = self.oaa_correct_all_args_pred_count[agent_uid]/self.oaa_total_pred_count[agent_uid]
                if self.last_pred_calc_obj_in_graph[agent_uid]:
                    self.oaa_pred_all_args_accuracy_obj_in_graph[agent_uid] = self.oaa_correct_all_args_pred_count_obj_in_graph[agent_uid]/self.oaa_total_pred_count_obj_in_graph[agent_uid]
                
                if label_action[:2] == pred_action[:2]: 
                    # Compare action type and all arguments
                    self.oaa_correct_action_all_args_pred_count[agent_uid] += 1
                    preds_match['action_all_args_match'] = True
                    self.last_predicted_oaa_matched[agent_uid] = True
                    if self.last_pred_calc_obj_in_graph[agent_uid]:
                        self.oaa_correct_action_all_args_pred_count_obj_in_graph[agent_uid] += 1
                    
                else: 
                    # Different action type and/or arguments
                    preds_match['action_all_args_match'] = False
                    self.last_predicted_oaa_matched[agent_uid] = False
                    
                self.oaa_pred_action_all_args_accuracy[agent_uid] = self.oaa_correct_action_all_args_pred_count[agent_uid]/self.oaa_total_pred_count[agent_uid]
                if self.last_pred_calc_obj_in_graph[agent_uid]:
                    self.oaa_pred_action_all_args_accuracy_obj_in_graph[agent_uid] = self.oaa_correct_action_all_args_pred_count_obj_in_graph[agent_uid]/self.oaa_total_pred_count_obj_in_graph[agent_uid]
        
        # TODO: Handle other action formats if necessary
        
        return preds_match
    
    def _replace_object_arg_with_obj_class(self, action: Tuple, world_graph) -> Tuple:
        """
        Replace object arguments in the action with their object class (without "_<ID>").
        
        :param action: The action tuple (action_name, action_args, ...)
        :param world_graph: The WorldGraph to check for objects
        :return: The modified action tuple with object names replaced by classes
        """
        if not isinstance(action, tuple) or len(action) < 2:
            return action
        
        action_name = action[0]
        action_args = action[1]
        
        if not isinstance(action_args, str):
            return action
        
        args_list = [arg.strip() for arg in action_args.split(',')]
        new_args_list = []
        
        for arg in args_list:
            if arg:
                try:
                    node = world_graph.get_node_from_name(arg)
                    if isinstance(node, Object):
                        # Remove the "_<ID>" part to get the class
                        obj_class = node.name.rsplit('_', 1)[0] if '_' in node.name else node.name
                        new_args_list.append(obj_class)
                    else:
                        new_args_list.append(arg)
                except ValueError:
                    new_args_list.append(arg)
            else:
                new_args_list.append(arg)
        
        new_action_args = ', '.join(new_args_list)
        return (action_name, new_action_args) + action[2:]
    
    def _compare_actions_calc_metrics(self, world_graph: Dict[int, 'WorldGraph']) -> Dict[str, bool]:
        """
        Compare the actions and/or its arguments in self.oaa_preds_and_labels for equality and calculate and update metrics. 
        Handles tuple-based actions.
    
        :return: Dict that contains boolean values indicating whether different levels of action/argument matches occurred
        """
        preds_match = {0:{},1:{}}
        
        # loop through the agents
        for agent_uid, pred_label_actions_dict in self.oaa_preds_and_labels.items():
            other_agent_uid = 1 - agent_uid
            assert pred_label_actions_dict['len_pred'] == pred_label_actions_dict['len_oa_current_action'] == pred_label_actions_dict['len_oa_next_action']
            # loop through the action predictions and labels for the whole episode
            for i in range(pred_label_actions_dict['len_pred']):
                # Get prediction and label actions
                pred_action = pred_label_actions_dict['pred'][i]
                label_current_action = pred_label_actions_dict['oa_current_action'][i]
                label_next_action = pred_label_actions_dict['oa_next_action'][i]
                # print('--------------label_current_action, label_next_action--------------')
                # print(label_current_action)
                # print(label_next_action)
                # print('--------------pred_action--------------')
                # Creating object class versions of the current and next label actions if the config specifies to do so
                # trim down the '_<id>' from the object names in the action arguments to get the object class
                if self.evaluation_runner_config.get("pred_oaa_obj_class_mode", False):
                    if 'oa_current_action_obj_class' not in self.oaa_preds_and_labels[agent_uid]:
                        self.oaa_preds_and_labels[agent_uid]['oa_current_action_obj_class'] = []
                    if 'oa_next_action_obj_class' not in self.oaa_preds_and_labels[agent_uid]:
                        self.oaa_preds_and_labels[agent_uid]['oa_next_action_obj_class'] = []
                    label_current_action = self._replace_object_arg_with_obj_class(label_current_action, world_graph[other_agent_uid])
                    label_next_action = self._replace_object_arg_with_obj_class(label_next_action, world_graph[other_agent_uid])
                    self.oaa_preds_and_labels[agent_uid]['oa_current_action_obj_class'].append(label_current_action)
                    self.oaa_preds_and_labels[agent_uid]['oa_next_action_obj_class'].append(label_next_action)
                # print('--------------label_current_action, label_next_action--------------')
                # print(label_current_action)
                # print(label_next_action)
                # print('--------------pred_action--------------')
                # Handle tuple-based actions (most common format)
                if isinstance(label_current_action, tuple) and isinstance(label_next_action, tuple) and isinstance(pred_action, tuple):
                    # Compare the action type and main arguments
                    # Ignore None arguments in comparison
                    
                    # Compare arguments, ignoring None values
                    if len(label_current_action) > 0 and len(label_next_action) > 0 and len(pred_action) > 0:
                        label_curr_principal_arg = self._get_first_argument(label_current_action[1])
                        label_next_principal_arg = self._get_first_argument(label_next_action[1])
                        pred_principal_arg = self._get_first_argument(pred_action[1])
                        
                        # Initialize per_action_metrics for the given action type
                        if label_next_action[0] not in self.per_action_metrics[agent_uid]:
                            self.per_action_metrics[agent_uid][label_next_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_all': 0, 'failure_count_all': 0}
                        if label_next_action[0] not in self.per_action_metrics_curr_next[agent_uid]:
                            self.per_action_metrics_curr_next[agent_uid][label_next_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_all': 0, 'failure_count_all': 0}
                        if label_current_action[0] not in self.per_action_metrics_curr_next[agent_uid]:
                            self.per_action_metrics_curr_next[agent_uid][label_current_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_all': 0, 'failure_count_all': 0}
                        if pred_action[0] not in self.per_action_metrics[agent_uid]:
                            self.per_action_metrics[agent_uid][pred_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_all': 0, 'failure_count_all': 0}
                        if pred_action[0] not in self.per_action_metrics_curr_next[agent_uid]:
                            self.per_action_metrics_curr_next[agent_uid][pred_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_all': 0, 'failure_count_all': 0} 
                        
                        if 'action_match_next' not in preds_match[agent_uid]:
                            preds_match[agent_uid]['action_match_next'] =  []
                        if 'action_match_curr_next' not in preds_match[agent_uid]:
                            preds_match[agent_uid]['action_match_curr_next'] =  []
                        
                        if label_next_action[0] == pred_action[0]: 
                            # Compare action type
                            self.oaa_correct_action_pred_count[agent_uid] += 1
                            self.oaa_correct_action_pred_count_curr_next[agent_uid] += 1
                            preds_match[agent_uid]['action_match_next'].append(True)
                            preds_match[agent_uid]['action_match_curr_next'].append(True)
                        
                            # Update the per-action metrics
                            self.per_action_metrics[agent_uid][label_next_action[0]]['tp'] += 1
                            self.per_action_metrics_curr_next[agent_uid][label_next_action[0]]['tp'] += 1
                        else: 
                            # Different action type
                            preds_match[agent_uid]['action_match_next'].append(False)
                            
                            # Update the per-action metrics
                            self.per_action_metrics[agent_uid][pred_action[0]]['fp'] += 1
                            self.per_action_metrics[agent_uid][label_next_action[0]]['fn'] += 1
                            if label_current_action[0] == pred_action[0]:
                                # Compare action type
                                self.oaa_correct_action_pred_count_curr_next[agent_uid] += 1
                                preds_match[agent_uid]['action_match_curr_next'].append(True)
                                    
                                # Update the per-action metrics
                                self.per_action_metrics_curr_next[agent_uid][label_current_action[0]]['tp'] += 1
                            else:
                                # Different action type
                                preds_match[agent_uid]['action_match_curr_next'].append(False)
                                
                                # Update the per-action metrics
                                self.per_action_metrics_curr_next[agent_uid][pred_action[0]]['fp'] += 1
                                self.per_action_metrics_curr_next[agent_uid][label_current_action[0]]['fn'] += 1             
                        
                        if 'principal_match_next' not in preds_match[agent_uid]:
                            preds_match[agent_uid]['principal_match_next'] =  []
                        if 'principal_match_curr_next' not in preds_match[agent_uid]:
                            preds_match[agent_uid]['principal_match_curr_next'] =  []
                        
                        if label_next_principal_arg == pred_principal_arg: 
                            # Compare principal argument
                            self.oaa_correct_principal_pred_count[agent_uid] += 1
                            self.oaa_correct_principal_pred_count_curr_next[agent_uid] += 1
                            preds_match[agent_uid]['principal_match_next'].append(True)
                            preds_match[agent_uid]['principal_match_curr_next'].append(True)
                        else: 
                            # Different principal argument
                            preds_match[agent_uid]['principal_match_next'].append(False)
                            if label_curr_principal_arg == pred_principal_arg:
                                # Compare principal argument
                                self.oaa_correct_principal_pred_count_curr_next[agent_uid] += 1
                                preds_match[agent_uid]['principal_match_curr_next'].append(True)
                            else: 
                                # Different principal argument
                                preds_match[agent_uid]['principal_match_curr_next'].append(False) 
                        
                        if 'action_principal_match' not in preds_match[agent_uid]:
                            preds_match[agent_uid]['action_principal_match'] =  []
                        if 'action_principal_match_curr_next' not in preds_match[agent_uid]:
                            preds_match[agent_uid]['action_principal_match_curr_next'] =  []
                        
                        if (label_next_action[0] == pred_action[0]) and (label_next_principal_arg == pred_principal_arg): 
                            # Compare action type and principal argument
                            self.oaa_correct_action_principal_pred_count[agent_uid] += 1
                            self.oaa_correct_action_principal_pred_count_curr_next[agent_uid] += 1
                            preds_match[agent_uid]['action_principal_match'].append(True)
                            preds_match[agent_uid]['action_principal_match_curr_next'].append(True)
                        else: 
                            # Different action type or principal argument
                            preds_match[agent_uid]['action_principal_match'].append(False)
                            if (label_current_action[0] == pred_action[0]) and (label_curr_principal_arg == pred_principal_arg): 
                                # Compare action type and principal argument
                                self.oaa_correct_action_principal_pred_count_curr_next[agent_uid] += 1
                                preds_match[agent_uid]['action_principal_match_curr_next'].append(True)
                            else:
                                # Different action type or principal argument
                                preds_match[agent_uid]['action_principal_match_curr_next'].append(False)
                            
                        if 'all_args_match' not in preds_match[agent_uid]:
                            preds_match[agent_uid]['all_args_match'] =  []
                        if 'all_args_match_curr_next' not in preds_match[agent_uid]:
                            preds_match[agent_uid]['all_args_match_curr_next'] =  []
                            
                        if label_next_action[1] == pred_action[1]:
                            # Compare all arguments
                            self.oaa_correct_all_args_pred_count[agent_uid] += 1
                            self.oaa_correct_all_args_pred_count_curr_next[agent_uid] += 1
                            preds_match[agent_uid]['all_args_match'].append(True)
                            preds_match[agent_uid]['all_args_match_curr_next'].append(True)
                        else: 
                            # Different arguments
                            preds_match[agent_uid]['all_args_match'].append(False)
                            if label_current_action[1] == pred_action[1]:
                                # Compare all arguments
                                self.oaa_correct_all_args_pred_count_curr_next[agent_uid] += 1
                                preds_match[agent_uid]['all_args_match_curr_next'].append(True)
                            else: 
                                # Different arguments
                                preds_match[agent_uid]['all_args_match_curr_next'].append(False)
                        
                        if 'action_all_args_match' not in preds_match[agent_uid]:
                            preds_match[agent_uid]['action_all_args_match'] =  []
                        if 'action_all_args_match_curr_next' not in preds_match[agent_uid]:
                            preds_match[agent_uid]['action_all_args_match_curr_next'] =  []
                        
                        if label_next_action[:2] == pred_action[:2]: 
                            # Compare action type and all arguments
                            self.oaa_correct_action_all_args_pred_count[agent_uid] += 1
                            self.oaa_correct_action_all_args_pred_count_curr_next[agent_uid] += 1
                            preds_match[agent_uid]['action_all_args_match'].append(True)
                            preds_match[agent_uid]['action_all_args_match_curr_next'].append(True)
                            
                            if label_next_action[0] != "Done": 
                                self.oaa_correct_action_all_args_pred_not_done_count[agent_uid] += 1
                                self.oaa_correct_action_all_args_pred_not_done_count_curr_next[agent_uid] += 1
                            
                            # Calculate pred match success execution metrics for next oaa
                            # Response for Done action is None
                            if label_next_action[0] != "Done" \
                                and pred_label_actions_dict['oa_next_action_response'][i] is not None \
                                and "successful execution!" in pred_label_actions_dict['oa_next_action_response'][i].lower():
                                self.oaa_pred_match_success_execution_count[agent_uid] += 1
                                self.oaa_pred_match_success_execution_count_curr_next[agent_uid] += 1
                        else: 
                            # Different action type and/or arguments
                            preds_match[agent_uid]['action_all_args_match'].append(False)
                            
                            # Calculate pred match success execution metrics for next oaa
                            if label_next_action[0] != "Done" \
                                and pred_label_actions_dict['oa_next_action_response'][i] is not None \
                                and "successful execution!" in pred_label_actions_dict['oa_next_action_response'][i].lower():
                                self.oaa_pred_unmatch_success_execution_count[agent_uid] += 1
                            
                            if label_current_action[:2] == pred_action[:2]: 
                                # Compare action type and all arguments
                                self.oaa_correct_action_all_args_pred_count_curr_next[agent_uid] += 1
                                preds_match[agent_uid]['action_all_args_match_curr_next'].append(True)
                                
                                if label_current_action[0] != "Done": 
                                    self.oaa_correct_action_all_args_pred_not_done_count_curr_next[agent_uid] += 1
                                
                                # Calculate pred match success execution metrics for current oaa
                                if label_current_action[0] != "Done" \
                                    and pred_label_actions_dict['oa_current_action_response'][i] is not None \
                                    and "successful execution!" in pred_label_actions_dict['oa_current_action_response'][i].lower():
                                    self.oaa_pred_match_success_execution_count_curr_next[agent_uid] += 1
                            else: 
                                # Different action type and/or arguments
                                preds_match[agent_uid]['action_all_args_match_curr_next'].append(False)
                                # Calculate pred match success execution metrics for current oaa
                                if label_current_action[0] != "Done" \
                                    and pred_label_actions_dict['oa_current_action_response'][i] is not None \
                                    and "successful execution!" in pred_label_actions_dict['oa_current_action_response'][i].lower():
                                    self.oaa_pred_unmatch_success_execution_count_curr_next[agent_uid] += 1
                
                # Handle other action formats if necessary
            
            # Calculate the accuracies from the counts
            self.oaa_pred_action_accuracy[agent_uid] = self.oaa_correct_action_pred_count[agent_uid]/self.oaa_total_pred_count[agent_uid] if self.oaa_total_pred_count[agent_uid] > 0 else None
            self.oaa_pred_action_accuracy_curr_next[agent_uid] = self.oaa_correct_action_pred_count_curr_next[agent_uid]/self.oaa_total_pred_count[agent_uid] if self.oaa_total_pred_count[agent_uid] > 0 else None
            
            self.oaa_pred_principal_accuracy[agent_uid] = self.oaa_correct_principal_pred_count[agent_uid]/self.oaa_total_pred_count[agent_uid] if self.oaa_total_pred_count[agent_uid] > 0 else None
            self.oaa_pred_principal_accuracy_curr_next[agent_uid] = self.oaa_correct_principal_pred_count_curr_next[agent_uid]/self.oaa_total_pred_count[agent_uid] if self.oaa_total_pred_count[agent_uid] > 0 else None
        
            self.oaa_pred_action_principal_accuracy[agent_uid] = self.oaa_correct_action_principal_pred_count[agent_uid]/self.oaa_total_pred_count[agent_uid] if self.oaa_total_pred_count[agent_uid] > 0 else None
            self.oaa_pred_action_principal_accuracy_curr_next[agent_uid] = self.oaa_correct_action_principal_pred_count_curr_next[agent_uid]/self.oaa_total_pred_count[agent_uid] if self.oaa_total_pred_count[agent_uid] > 0 else None
                        
            self.oaa_pred_all_args_accuracy[agent_uid] = self.oaa_correct_all_args_pred_count[agent_uid]/self.oaa_total_pred_count[agent_uid] if self.oaa_total_pred_count[agent_uid] > 0 else None
            self.oaa_pred_all_args_accuracy_curr_next[agent_uid] = self.oaa_correct_all_args_pred_count_curr_next[agent_uid]/self.oaa_total_pred_count[agent_uid] if self.oaa_total_pred_count[agent_uid] > 0 else None
        
            self.oaa_pred_action_all_args_accuracy[agent_uid] = self.oaa_correct_action_all_args_pred_count[agent_uid]/self.oaa_total_pred_count[agent_uid] if self.oaa_total_pred_count[agent_uid] > 0 else None
            self.oaa_pred_action_all_args_accuracy_curr_next[agent_uid] = self.oaa_correct_action_all_args_pred_count_curr_next[agent_uid]/self.oaa_total_pred_count[agent_uid] if self.oaa_total_pred_count[agent_uid] > 0 else None
        
            self.oaa_pred_match_success_execution_accuracy[agent_uid] = self.oaa_pred_match_success_execution_count[agent_uid]/self.oaa_correct_action_all_args_pred_not_done_count[agent_uid] if self.oaa_correct_action_all_args_pred_not_done_count[agent_uid] > 0 else None
            self.oaa_pred_match_success_execution_accuracy_curr_next[agent_uid] = self.oaa_pred_match_success_execution_count_curr_next[agent_uid]/self.oaa_correct_action_all_args_pred_not_done_count_curr_next[agent_uid] if self.oaa_correct_action_all_args_pred_not_done_count_curr_next[agent_uid] > 0 else None
            
            self.oaa_pred_unmatch_success_execution_accuracy[agent_uid] = self.oaa_pred_unmatch_success_execution_count[agent_uid]/(self.oa_not_done_pred_count[agent_uid] - self.oaa_correct_action_all_args_pred_not_done_count[agent_uid]) if self.oa_not_done_pred_count[agent_uid] - self.oaa_correct_action_all_args_pred_not_done_count[agent_uid] > 0 else None
            self.oaa_pred_unmatch_success_execution_accuracy_curr_next[agent_uid] = self.oaa_pred_unmatch_success_execution_count_curr_next[agent_uid]/(self.oa_not_done_pred_count[agent_uid] - self.oaa_correct_action_all_args_pred_not_done_count_curr_next[agent_uid]) if self.oa_not_done_pred_count[agent_uid] - self.oaa_correct_action_all_args_pred_not_done_count_curr_next[agent_uid] > 0 else None

        return preds_match
        
    def _calc_match_success_execution_metrics(self, response: str, other_agent_uid: int) -> None:
        """
        Calculate and update metrics related to the match between predicted action and label action, and the success of the execution of the label action.
        
        :param response: The response from the environment after executing the label action, used to determine if the execution was successful
        :param other_agent_uid: The UID of the other agent who predicted the action 
        """
        # print(f'<<<<<<<<<<<<<<<<<<<<<<<')
        # print(f'response: {response}, other_agent_uid: {other_agent_uid}, self.last_predicted_oaa_matched[other_agent_uid]: {self.last_predicted_oaa_matched[other_agent_uid]}')
        # print(f'<<<<<<<<<<<<<<<<<<<<<<<')
        # update all time metrics
        if self.last_predicted_oaa_matched[other_agent_uid]:
            # Predicted action matches the labeled action
            if "successful execution!" in response.lower():
                # successful execution of label the action
                self.oaa_pred_match_success_execution_count[other_agent_uid] += 1
                    
                # print(f'<<<<<<<<<<<<<<<<<<<<<<<')
                # print(f'other agent uid: {other_agent_uid}, \nself.oaa_pred_match_success_execution_count: {self.oaa_pred_match_success_execution_count[other_agent_uid]}, \nself.oaa_correct_action_all_args_pred_count: {self.oaa_correct_action_all_args_pred_count[other_agent_uid]}')
                # self.oaa_correct_action_all_args_pred_count[other_agent_uid] cannot be 0 if the program execution comes here
                self.oaa_pred_match_success_execution_accuracy[other_agent_uid] = self.oaa_pred_match_success_execution_count[other_agent_uid]/self.oaa_correct_action_all_args_pred_count[other_agent_uid]
                # print(f'self.oaa_pred_match_success_execution_accuracy: {self.oaa_pred_match_success_execution_accuracy[other_agent_uid]}')
                # print(f'<<<<<<<<<<<<<<<<<<<<<<<')
        else:
            # Predicted action doesn't match the labeled action
            if "successful execution!" in response.lower():
                # successful execution of the label action
                self.oaa_pred_unmatch_success_execution_count[other_agent_uid] += 1
                    
                # print(f'<<<<<<<<<<<<<<<<<<<<<<<')
                # print(f'other agent uid: {other_agent_uid}, \nself.oaa_pred_unmatch_success_execution_count: {self.oaa_pred_unmatch_success_execution_count[other_agent_uid]}, \nself.oaa_total_pred_count: {self.oaa_total_pred_count[other_agent_uid]}, \nself.oaa_correct_action_all_args_pred_count: {self.oaa_correct_action_all_args_pred_count[other_agent_uid]}')
                # If self.last_predicted_oaa_matched[other_agent_uid] is False, then self.oaa_total_pred_count[other_agent_uid] > self.oaa_correct_action_all_args_pred_count[other_agent_uid], so the denominator cannot be 0 here
                self.oaa_pred_unmatch_success_execution_accuracy[other_agent_uid] = self.oaa_pred_unmatch_success_execution_count[other_agent_uid]/(self.oaa_total_pred_count[other_agent_uid] - self.oaa_correct_action_all_args_pred_count[other_agent_uid])
                # print(f'self.oaa_pred_unmatch_success_execution_accuracy: {self.oaa_pred_unmatch_success_execution_accuracy[other_agent_uid]}')
                # print(f'<<<<<<<<<<<<<<<<<<<<<<<')
                
        # update obj_in_graph metrics
        if self.last_pred_calc_obj_in_graph[other_agent_uid]:
            if self.last_predicted_oaa_matched[other_agent_uid]:
                # Predicted action matches the labeled action
                if "successful execution!" in response.lower():
                    # successful execution of label the action
                    self.oaa_pred_match_success_execution_count_obj_in_graph[other_agent_uid] += 1
                        
                    # print(f'<<<<<<<<<<<<<<<<<<<<<<<')
                    # print(f'other agent uid: {other_agent_uid}, \nself.oaa_pred_match_success_execution_count_obj_in_graph: {self.oaa_pred_match_success_execution_count_obj_in_graph[other_agent_uid]}, \nself.oaa_correct_action_all_args_pred_count_obj_in_graph: {self.oaa_correct_action_all_args_pred_count_obj_in_graph[other_agent_uid]}')
                    # self.oaa_correct_action_all_args_pred_count_obj_in_graph[other_agent_uid] cannot be 0 if the program execution comes here
                    self.oaa_pred_match_success_execution_accuracy_obj_in_graph[other_agent_uid] = self.oaa_pred_match_success_execution_count_obj_in_graph[other_agent_uid]/self.oaa_correct_action_all_args_pred_count_obj_in_graph[other_agent_uid]
                    # print(f'self.oaa_pred_match_success_execution_accuracy_obj_in_graph: {self.oaa_pred_match_success_execution_accuracy_obj_in_graph[other_agent_uid]}')
                    # print(f'<<<<<<<<<<<<<<<<<<<<<<<')
            else:
                # Predicted action doesn't match the labeled action
                if "successful execution!" in response.lower():
                    # successful execution of the label action
                    self.oaa_pred_unmatch_success_execution_count_obj_in_graph[other_agent_uid] += 1
                        
                    # print(f'<<<<<<<<<<<<<<<<<<<<<<<')
                    # print(f'other agent uid: {other_agent_uid}, \nself.oaa_pred_unmatch_success_execution_count_obj_in_graph: {self.oaa_pred_unmatch_success_execution_count_obj_in_graph[other_agent_uid]}, \nself.oaa_total_pred_count_obj_in_graph: {self.oaa_total_pred_count_obj_in_graph[other_agent_uid]}, \nself.oaa_correct_action_all_args_pred_count_obj_in_graph: {self.oaa_correct_action_all_args_pred_count_obj_in_graph[other_agent_uid]}')
                    # If self.last_predicted_oaa_matched[other_agent_uid] is False, then self.oaa_total_pred_count_obj_in_graph[other_agent_uid] > self.oaa_correct_action_all_args_pred_count_obj_in_graph[other_agent_uid], so the denominator cannot be 0 here
                    self.oaa_pred_unmatch_success_execution_accuracy_obj_in_graph[other_agent_uid] = self.oaa_pred_unmatch_success_execution_count_obj_in_graph[other_agent_uid]/(self.oaa_total_pred_count_obj_in_graph[other_agent_uid] - self.oaa_correct_action_all_args_pred_count_obj_in_graph[other_agent_uid])
                    # print(f'self.oaa_pred_unmatch_success_execution_accuracy_obj_in_graph: {self.oaa_pred_unmatch_success_execution_accuracy_obj_in_graph[other_agent_uid]}')
                    # print(f'<<<<<<<<<<<<<<<<<<<<<<<')
    
    def _calc_success_failure_action_execution_other_replanned(self, high_level_action: Any, response: str, agent_uid: int, when_pred = False) -> None:
        """
        Calculate and update the success or failure count of the execution of a high-level action, only when action prediction happend.
        
        :param high_level_action: 
        :param response: The response from the environment after executing the label action, used to determine if the execution was successful
        :param agent_uid: The UID of the agent whose action is being evaluated
        :param when_pred: Whether to count only, when prediction is made by the other agent or all the time, used to determine which metrics to update
        """
        other_agent_uid = 1 - agent_uid
        if high_level_action[0] not in self.per_action_metrics[agent_uid]:
            if self.evaluation_runner_config.get("predict_other_agent_actions", False):
                self.per_action_metrics[agent_uid][high_level_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_when_pred': 0, 'failure_count_when_pred': 0, 'success_count_all': 0, 'failure_count_all': 0}
            else:
                self.per_action_metrics[agent_uid][high_level_action[0]] = {'success_count_all': 0, 'failure_count_all': 0}
        
        if (high_level_action[0] not in self.per_action_metrics_obj_in_graph[agent_uid]
        and self.evaluation_runner_config.get("predict_other_agent_actions", False)
        and self.last_pred_calc_obj_in_graph[other_agent_uid]
        and when_pred):
            self.per_action_metrics_obj_in_graph[agent_uid][high_level_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_when_pred': 0, 'failure_count_when_pred': 0}
        
        # only calling the function, if there is a response for the executed action
        if "successful execution!" in response.lower() and self.evaluation_runner_config.get("predict_other_agent_actions", False):
            self.per_action_metrics[agent_uid][high_level_action[0]]['success_count_all'] += 1
            if when_pred:
                self.per_action_metrics[agent_uid][high_level_action[0]]['success_count_when_pred'] += 1
                if self.last_pred_calc_obj_in_graph[other_agent_uid]:
                    self.per_action_metrics_obj_in_graph[agent_uid][high_level_action[0]]['success_count_when_pred'] += 1
        elif "successful execution!" not in response.lower() and self.evaluation_runner_config.get("predict_other_agent_actions", False):
            self.per_action_metrics[agent_uid][high_level_action[0]]['failure_count_all'] += 1
            if when_pred:
                self.per_action_metrics[agent_uid][high_level_action[0]]['failure_count_when_pred'] += 1
                if self.last_pred_calc_obj_in_graph[other_agent_uid]:
                    self.per_action_metrics_obj_in_graph[agent_uid][high_level_action[0]]['failure_count_when_pred'] += 1
            
    def _calc_success_failure_action_execution(self, high_level_action: Any, response: str, agent_uid: int) -> None:
        """
        Calculate and update the success or failure count of the execution of a high-level action.
        
        :param high_level_action: 
        :param response: The response from the environment after executing the label action, used to determine if the execution was successful
        :param agent_uid: The UID of the agent whose action is being evaluated
        """
        if high_level_action[0] not in self.per_action_metrics[agent_uid]:
            self.per_action_metrics[agent_uid][high_level_action[0]] = {'tp': 0, 'fp': 0, 'fn': 0, 'success_count_all': 0, 'failure_count_all': 0}
        
        # only calling the function, if there is a response for the executed action
        if "successful execution!" in response.lower():
            self.per_action_metrics[agent_uid][high_level_action[0]]['success_count_all'] += 1
        elif "successful execution!" not in response.lower():
            self.per_action_metrics[agent_uid][high_level_action[0]]['failure_count_all'] += 1

    def _fill_oaa_preds_and_labels(self, planner_info: Dict[str, Any]) -> None:
        """
        Fill self.oaa_preds_and_labels with the predicted high-level actions and the corresponding label high-level actions for both agents, when replanning happens.
        
        :param planner_info: The dictionary containing the planning info from all planners, which includes the current high-level actions, whether replanning happened, and the predicted high-level action by the other agent for each agent
        """
        # Loop through all available planners
        for planner in self.planner.values():
            agent_uid = planner.agents[0].uid
            other_agent_uid = 1 - agent_uid 
            # Filling the other agent's list of oa_next_action with 
            # the current high-level action of this agent
            if planner_info["replanned"][agent_uid]:
                list_size_diff = len(self.oaa_preds_and_labels[other_agent_uid]["pred"]) - len(self.oaa_preds_and_labels[other_agent_uid]["oa_next_action"])
                if len(self.oaa_preds_and_labels[other_agent_uid]["pred"]) > 0 and list_size_diff > 0:
                    self.oaa_preds_and_labels[other_agent_uid]["oa_next_action"].extend([planner_info["high_level_actions"][agent_uid] for _ in range(list_size_diff)])
                    self.oaa_preds_and_labels[other_agent_uid]["len_oa_next_action"] += list_size_diff
                    # may calculate action prediction metrics for the other agent here
            # Filling the other agent's list of oa_next_action_response 
            # and oa_current_action_response with the current response of this agent
            if planner_info.get("responses", False) and planner_info["responses"].get(agent_uid, False) and planner_info["responses"][agent_uid] != "":
                list_size_diff = len(self.oaa_preds_and_labels[other_agent_uid]["oa_current_action_response"]) - len(self.oaa_preds_and_labels[other_agent_uid]["oa_next_action_response"])
                if len(self.oaa_preds_and_labels[other_agent_uid]["oa_current_action_response"]) > 0 and list_size_diff > 0:
                    self.oaa_preds_and_labels[other_agent_uid]["oa_next_action_response"].extend([planner_info["responses"][agent_uid] for _ in range(list_size_diff)])
                list_size_diff = len(self.oaa_preds_and_labels[other_agent_uid]["pred"]) - len(self.oaa_preds_and_labels[other_agent_uid]["oa_current_action_response"])
                if len(self.oaa_preds_and_labels[other_agent_uid]["pred"]) > 0 and list_size_diff > 0:
                    self.oaa_preds_and_labels[other_agent_uid]["oa_current_action_response"].extend([planner_info["responses"][agent_uid] for _ in range(list_size_diff)])
            # Filling the other agent's list of oa_next_action_response 
            # and oa_current_action_response with None as response, if this agent is Done
            if planner_info['is_done'][agent_uid]:
                list_size_diff = len(self.oaa_preds_and_labels[other_agent_uid]["pred"]) - len(self.oaa_preds_and_labels[other_agent_uid]["oa_next_action_response"])
                if len(self.oaa_preds_and_labels[other_agent_uid]["pred"]) > 0 and list_size_diff > 0:
                    self.oaa_preds_and_labels[other_agent_uid]["oa_next_action_response"].extend([None for _ in range(list_size_diff)])
                list_size_diff = len(self.oaa_preds_and_labels[other_agent_uid]["pred"]) - len(self.oaa_preds_and_labels[other_agent_uid]["oa_current_action_response"])
                if len(self.oaa_preds_and_labels[other_agent_uid]["pred"]) > 0 and list_size_diff > 0:
                    self.oaa_preds_and_labels[other_agent_uid]["oa_current_action_response"].extend([None for _ in range(list_size_diff)])
                
        
        # Loop through all available planners
        for planner in self.planner.values():
            agent_uid = planner.agents[0].uid
            other_agent_uid = 1 - agent_uid 
            # Filling this agent's oaa_preds_and_labels lists when this agent is replanned
            if planner_info["replanned"][agent_uid]:
                self.oaa_preds_and_labels[agent_uid]["pred"].append(planner_info["other_agents_predicted_high_level_actions"][agent_uid])
                self.oaa_preds_and_labels[agent_uid]["len_pred"] += 1
                if planner_info['is_done'][other_agent_uid]:
                    self.oaa_preds_and_labels[agent_uid]["oa_current_action"].append(("Done", None, None))
                    self.oaa_preds_and_labels[agent_uid]["oa_next_action"].append(("Done", None, None))
                    self.oaa_preds_and_labels[agent_uid]["len_oa_current_action"] += 1
                    self.oaa_preds_and_labels[agent_uid]["len_oa_next_action"] += 1
                    self.oaa_preds_and_labels[agent_uid]["oa_current_action_response"].append(None)
                    self.oaa_preds_and_labels[agent_uid]["oa_next_action_response"].append(None)
                    # may calculate action prediction metrics for this agent here
                else:
                    self.oa_not_done_pred_count[agent_uid] += 1
                    self.oaa_preds_and_labels[agent_uid]["oa_current_action"].append(planner_info["high_level_actions"][other_agent_uid])
                    self.oaa_preds_and_labels[agent_uid]["len_oa_current_action"] += 1
    
    def get_low_level_actions(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Given a set of observations, gets a vector of low level actions,
        an info dictionary and a boolean indicating that the run should end.

        :param instruction: Natural language instruction describing the task to execute
        :param observations: Dictionary of observations from the environment
        :param world_graph: Dictionary mapping agent IDs to their world graph states

        :return: Tuple containing:
            - Dictionary mapping agent IDs to their low level actions
            - Dictionary with planner info about high level actions
            - Boolean indicating whether all planners are done
        """
        # Declare container to store planned low level actions
        # from all planners
        low_level_actions: Dict[int, Any] = {}

        # Declare container to store planning info from all planners
        planner_info: Dict[str, Any] = {}

        # Marks the end of all planners
        all_planners_are_done = True

        assert isinstance(self.planner, dict)
        # Loop through all available planners
        for planner in self.planner.values():
            # Get next action for this planner
            (
                this_planner_low_level_actions,
                this_planner_info,
                this_planner_is_done,
            ) = planner.get_next_action(instruction, observations, world_graph)
            agent_uid = planner.agents[0].uid
            other_agent_uid = 1 - agent_uid
            if self.evaluation_runner_config.get("predict_other_agent_actions", False) and not self.evaluation_runner_config.get("pred_oaa_same_inference_mode", False):
                # if the agent is replanned and the other is not done, 
                # then the other should predict an action
                if this_planner_info["replanned"][agent_uid] and (not self.planner[other_agent_uid].is_done):
                    self.oaa_total_pred_count[other_agent_uid] += 1
                    this_planner_info["predicted_high_level_action_by_the_other_agent"] = self.planner[other_agent_uid].predict_other_agent_high_level_action(instruction, observations, world_graph)
                    print('-----------------------------')
                    print(f'Agent {agent_uid} replanned: {this_planner_info["high_level_actions"]}')
                    is_obj_in_graph, obj_name = self._check_object_in_agent_graph(this_planner_info["high_level_actions"][agent_uid], other_agent_uid)
                    print(f'is_obj_in_graph, obj_name: {(is_obj_in_graph, obj_name)}')
                    calc_obj_in_graph = not (obj_name is not None and not is_obj_in_graph) # if obj_name is not None but is_obj_in_graph is False, it means the object exists in the full graph but not in the other agent's graph
                    if calc_obj_in_graph:
                        self.oaa_total_pred_count_obj_in_graph[other_agent_uid] += 1
                    self.last_pred_calc_obj_in_graph[other_agent_uid] = calc_obj_in_graph
                    preds_match = self._compare_actions_calc_metrics_other_replanned(this_planner_info["high_level_actions"][agent_uid], this_planner_info["predicted_high_level_action_by_the_other_agent"][other_agent_uid], other_agent_uid)
                    this_planner_info["oaa_pred_match"] = {other_agent_uid: preds_match}
                    print(f'Predicted high-level action for agent {agent_uid} by agent {other_agent_uid}: {this_planner_info["predicted_high_level_action_by_the_other_agent"]}')
                    print('-----------------------------')
                    
                # match_success_execution_metrics will be only calculated,
                # if the ground truth agent is finished the action execution 
                # with a response from the environment and there is any prediction 
                # left from the predictor agent
                if this_planner_info.get("responses", False) and this_planner_info["responses"].get(agent_uid, False) and this_planner_info["responses"][agent_uid] != "" and self.should_check_predicted_action_execution_success[other_agent_uid]:
                    self._calc_match_success_execution_metrics(this_planner_info["responses"][agent_uid], other_agent_uid)
                    self._calc_success_failure_action_execution_other_replanned(this_planner_info["high_level_actions"][agent_uid], this_planner_info["responses"][agent_uid], agent_uid, when_pred = True)
                    
                    # if the predictor agent is done, then no predictions will come from it
                    # so don't callculate anymore these metrics
                    if self.planner[other_agent_uid].is_done:
                        self.should_check_predicted_action_execution_success[other_agent_uid] = False  
            elif self.evaluation_runner_config.get("predict_other_agent_actions", False) and self.evaluation_runner_config.get("pred_oaa_same_inference_mode", False):
                if this_planner_info["replanned"][agent_uid]:
                    self.oaa_total_pred_count[agent_uid] += 1
                if this_planner_info.get("responses", False) and this_planner_info["responses"].get(agent_uid, False) and this_planner_info["responses"][agent_uid] != "":
                    self._calc_success_failure_action_execution(this_planner_info["high_level_actions"][agent_uid], this_planner_info["responses"][agent_uid], agent_uid)
            
            # calculate success/failure ratio of action execution 
            if not self.evaluation_runner_config.get("pred_oaa_same_inference_mode", False) and self.evaluation_runner_config.get("predict_other_agent_actions", False) and this_planner_info.get("responses", False) and this_planner_info["responses"].get(agent_uid, False) and this_planner_info["responses"][agent_uid] != "":
                self._calc_success_failure_action_execution_other_replanned(this_planner_info["high_level_actions"][agent_uid], this_planner_info["responses"][agent_uid], agent_uid)
            
            # Update the output dictionary with planned low level actions
            low_level_actions.update(this_planner_low_level_actions)
            # print('-----------------------------')
            # print(f'this_planner_info["replan_required"]: {this_planner_info["replan_required"]}') # planner_info["replan_required"]: {1: True}
            # print('-----------------------------')
            # Merges this_planner_info from all planners
            for key, val in this_planner_info.items():
                if type(val) == dict:
                    if key not in planner_info:
                        planner_info[key] = {}
                    planner_info[key].update(val)
                elif type(val) == str:
                    if key not in planner_info:
                        planner_info[key] = ""
                    planner_info[key] += val
                else:
                    raise ValueError(
                        "Logging entity can only be a dictionary or string!"
                    )

            all_planners_are_done = this_planner_is_done and all_planners_are_done
        # print(f'-------------------')
        # print(f'planner_info: {planner_info}')
        # print(f'-------------------')
        if self.evaluation_runner_config.get("predict_other_agent_actions", False) and self.evaluation_runner_config.get("pred_oaa_same_inference_mode", False):
            self._fill_oaa_preds_and_labels(planner_info)
            # Add oaa_preds_and_labels to planner_info only if all_planners_are_done
            # to avoid huge planner logs
            
            # if all planners are done, then calculate action prediction metrics
            # at the very end of the episode
            if all_planners_are_done:
                planner_info["oaa_preds_and_labels"] = self.oaa_preds_and_labels
                preds_match = self._compare_actions_calc_metrics(world_graph)
                planner_info['preds_match'] = preds_match
                
                planner_info["oaa_correct_action_pred_count_curr_next"] = self.oaa_correct_action_pred_count_curr_next
                planner_info["oaa_pred_action_accuracy_curr_next"] = self.oaa_pred_action_accuracy_curr_next
                planner_info["oaa_correct_principal_pred_count_curr_next"] = self.oaa_correct_principal_pred_count_curr_next
                planner_info["oaa_pred_principal_accuracy_curr_next"] = self.oaa_pred_principal_accuracy_curr_next
                planner_info["oaa_correct_action_principal_pred_count_curr_next"] = self.oaa_correct_action_principal_pred_count_curr_next
                planner_info["oaa_pred_action_principal_accuracy_curr_next"] = self.oaa_pred_action_principal_accuracy_curr_next
                planner_info["oaa_correct_all_args_pred_count_curr_next"] = self.oaa_correct_all_args_pred_count_curr_next
                planner_info["oaa_pred_all_args_accuracy_curr_next"] = self.oaa_pred_all_args_accuracy_curr_next
                planner_info["oaa_correct_action_all_args_pred_count_curr_next"] = self.oaa_correct_action_all_args_pred_count_curr_next
                planner_info["oaa_pred_action_all_args_accuracy_curr_next"] = self.oaa_pred_action_all_args_accuracy_curr_next
                planner_info["oaa_pred_match_success_execution_count_curr_next"] = self.oaa_pred_match_success_execution_count_curr_next
                planner_info["oaa_pred_match_success_execution_accuracy_curr_next"] = self.oaa_pred_match_success_execution_accuracy_curr_next
                planner_info["oaa_pred_unmatch_success_execution_count_curr_next"] = self.oaa_pred_unmatch_success_execution_count_curr_next
                planner_info["oaa_pred_unmatch_success_execution_accuracy_curr_next"] = self.oaa_pred_unmatch_success_execution_accuracy_curr_next
                planner_info["oaa_correct_action_principal_if_action_pred_accuracy_curr_next"] = {0:self.oaa_correct_action_principal_pred_count_curr_next[0]/self.oaa_correct_action_pred_count_curr_next[0] if self.oaa_correct_action_pred_count_curr_next[0] > 0 else None,
                                                                1:self.oaa_correct_action_principal_pred_count_curr_next[1]/self.oaa_correct_action_pred_count_curr_next[1] if self.oaa_correct_action_pred_count_curr_next[1] > 0 else None}
                planner_info["oaa_correct_action_all_args_if_action_pred_accuracy_curr_next"] = {0:self.oaa_correct_action_all_args_pred_count_curr_next[0]/self.oaa_correct_action_pred_count_curr_next[0] if self.oaa_correct_action_pred_count_curr_next[0] > 0 else None,
                                                                1:self.oaa_correct_action_all_args_pred_count_curr_next[1]/self.oaa_correct_action_pred_count_curr_next[1] if self.oaa_correct_action_pred_count_curr_next[1] > 0 else None}
                planner_info["oaa_correct_action_all_args_pred_not_done_count_curr_next"] = self.oaa_correct_action_all_args_pred_not_done_count_curr_next
                planner_info["per_action_metrics_curr_next"] = self.per_action_metrics_curr_next
            
        
        if self.evaluation_runner_config.get("predict_other_agent_actions", False):
            if all_planners_are_done:
                planner_info["oaa_total_pred_count"] = self.oaa_total_pred_count
                planner_info["oa_not_done_pred_count"] = self.oa_not_done_pred_count
                planner_info["oaa_correct_action_pred_count"] = self.oaa_correct_action_pred_count
                planner_info["oaa_pred_action_accuracy"] = self.oaa_pred_action_accuracy
                planner_info["oaa_correct_principal_pred_count"] = self.oaa_correct_principal_pred_count
                planner_info["oaa_pred_principal_accuracy"] = self.oaa_pred_principal_accuracy
                planner_info["oaa_correct_action_principal_pred_count"] = self.oaa_correct_action_principal_pred_count
                planner_info["oaa_pred_action_principal_accuracy"] = self.oaa_pred_action_principal_accuracy
                planner_info["oaa_correct_all_args_pred_count"] = self.oaa_correct_all_args_pred_count
                planner_info["oaa_pred_all_args_accuracy"] = self.oaa_pred_all_args_accuracy
                planner_info["oaa_correct_action_all_args_pred_count"] = self.oaa_correct_action_all_args_pred_count
                planner_info["oaa_pred_action_all_args_accuracy"] = self.oaa_pred_action_all_args_accuracy
                planner_info["oaa_pred_match_success_execution_count"] = self.oaa_pred_match_success_execution_count
                planner_info["oaa_pred_match_success_execution_accuracy"] = self.oaa_pred_match_success_execution_accuracy
                planner_info["oaa_pred_unmatch_success_execution_count"] = self.oaa_pred_unmatch_success_execution_count
                planner_info["oaa_pred_unmatch_success_execution_accuracy"] = self.oaa_pred_unmatch_success_execution_accuracy
                planner_info["oaa_correct_action_principal_if_action_pred_accuracy"] = {0:self.oaa_correct_action_principal_pred_count[0]/self.oaa_correct_action_pred_count[0] if self.oaa_correct_action_pred_count[0] > 0 else None,
                                                                1:self.oaa_correct_action_principal_pred_count[1]/self.oaa_correct_action_pred_count[1] if self.oaa_correct_action_pred_count[1] > 0 else None}
                planner_info["oaa_correct_action_all_args_if_action_pred_accuracy"] = {0:self.oaa_correct_action_all_args_pred_count[0]/self.oaa_correct_action_pred_count[0] if self.oaa_correct_action_pred_count[0] > 0 else None,
                                                                1:self.oaa_correct_action_all_args_pred_count[1]/self.oaa_correct_action_pred_count[1] if self.oaa_correct_action_pred_count[1] > 0 else None}
                planner_info["oaa_correct_action_all_args_pred_not_done_count"] = self.oaa_correct_action_all_args_pred_not_done_count
                planner_info["per_action_metrics"] = self.per_action_metrics
            
            if not self.evaluation_runner_config.get("pred_oaa_same_inference_mode", False):
            
                planner_info["oaa_total_pred_count_obj_in_graph"] = self.oaa_total_pred_count_obj_in_graph
                planner_info["oaa_correct_action_pred_count_obj_in_graph"] = self.oaa_correct_action_pred_count_obj_in_graph
                planner_info["oaa_pred_action_accuracy_obj_in_graph"] = self.oaa_pred_action_accuracy_obj_in_graph
                planner_info["oaa_correct_principal_pred_count_obj_in_graph"] = self.oaa_correct_principal_pred_count_obj_in_graph
                planner_info["oaa_pred_principal_accuracy_obj_in_graph"] = self.oaa_pred_principal_accuracy_obj_in_graph
                planner_info["oaa_correct_action_principal_pred_count_obj_in_graph"] = self.oaa_correct_action_principal_pred_count_obj_in_graph
                planner_info["oaa_pred_action_principal_accuracy_obj_in_graph"] = self.oaa_pred_action_principal_accuracy_obj_in_graph
                planner_info["oaa_correct_all_args_pred_count_obj_in_graph"] = self.oaa_correct_all_args_pred_count_obj_in_graph
                planner_info["oaa_pred_all_args_accuracy_obj_in_graph"] = self.oaa_pred_all_args_accuracy_obj_in_graph
                planner_info["oaa_correct_action_all_args_pred_count_obj_in_graph"] = self.oaa_correct_action_all_args_pred_count_obj_in_graph
                planner_info["oaa_pred_action_all_args_accuracy_obj_in_graph"] = self.oaa_pred_action_all_args_accuracy_obj_in_graph
                planner_info["oaa_pred_match_success_execution_count_obj_in_graph"] = self.oaa_pred_match_success_execution_count_obj_in_graph
                planner_info["oaa_pred_match_success_execution_accuracy_obj_in_graph"] = self.oaa_pred_match_success_execution_accuracy_obj_in_graph
                planner_info["oaa_pred_unmatch_success_execution_count_obj_in_graph"] = self.oaa_pred_unmatch_success_execution_count_obj_in_graph
                planner_info["oaa_pred_unmatch_success_execution_accuracy_obj_in_graph"] = self.oaa_pred_unmatch_success_execution_accuracy_obj_in_graph
                planner_info["oaa_correct_action_principal_if_action_pred_accuracy_obj_in_graph"] = {0:self.oaa_correct_action_principal_pred_count_obj_in_graph[0]/self.oaa_correct_action_pred_count_obj_in_graph[0] if self.oaa_correct_action_pred_count_obj_in_graph[0] > 0 else None,
                                                                1:self.oaa_correct_action_principal_pred_count_obj_in_graph[1]/self.oaa_correct_action_pred_count_obj_in_graph[1] if self.oaa_correct_action_pred_count_obj_in_graph[1] > 0 else None}
                planner_info["oaa_correct_action_all_args_if_action_pred_accuracy_obj_in_graph"] = {0:self.oaa_correct_action_all_args_pred_count_obj_in_graph[0]/self.oaa_correct_action_pred_count_obj_in_graph[0] if self.oaa_correct_action_pred_count_obj_in_graph[0] > 0 else None,
                                                                1:self.oaa_correct_action_all_args_pred_count_obj_in_graph[1]/self.oaa_correct_action_pred_count_obj_in_graph[1] if self.oaa_correct_action_pred_count_obj_in_graph[1] > 0 else None}
                planner_info["per_action_metrics_obj_in_graph"] = self.per_action_metrics_obj_in_graph
        
        # print('-----------------------------')
        # print(f'planner_info: {planner_info}') # planner_info: {'replanned': {0: True, 1: True}, 'replan_required': {0: True, 1: True},... 'high_level_actions': {0: ('Explore', 'bedroom_1', None), 1: ('Explore', 'bedroom_1', None)},
        # print('-----------------------------')
        return low_level_actions, planner_info, all_planners_are_done
