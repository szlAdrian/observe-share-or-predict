# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import time
from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

from build.lib.habitat_llm.utils import world_graph
from habitat_llm.llm.instruct.utils import get_objects_descr
from habitat_llm.planner import LLMPlanner
from habitat_llm.utils.grammar import (
    FREE_TEXT,
)
from habitat_llm.world_model.entity import Object, SpotRobot, Human

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph


class ZeroShotReactPlanner(LLMPlanner):
    """
    This class builds the prompt for the, zero shot llm react planner format.
    """

    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ) -> None:
        """
        Initialize the ZeroShotReactPlanner.

        :param plan_config: The planner configuration.
        :param env_interface: The environment interface.
        """
        super().__init__(plan_config, env_interface)
        self.skip_actions_num: int = 0 # Number of other agent's actions to skip when constructing the action history
        self.is_reasoning_about_other_agent: bool = False
        self.reasoning_about_other_agent_count: int = 0 # Number of times the planner is reasoning about the other agent, used for logging and analysis purposes

    def reset(self):
        """
        Reset the planner state.
        """
        self.skip_actions_num: int = 0
        self.is_reasoning_about_other_agent: bool = False
        self.reasoning_about_other_agent_count: int = 0
        super().reset()
        
    def build_response_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts all valid responses based on a world graph.

        :param world_graph: The world graph.
        :return: The response grammar.
        """
        delimiter = "\\n"
        tool_rules = self.build_tool_grammar(world_graph)

        # rule for the PARTNR baseline response
        root_rule = (
            f'root ::= {FREE_TEXT} "{delimiter}" tool_call "{delimiter}Assigned!"'
        )

        return "\n".join([root_rule, tool_rules])

    def build_action_only_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts only a tool call (no free-text reasoning
        prefix), optionally preceded by whitespace/newlines.

        Used for "think-then-constrain" generation, where the reasoning
        (e.g. a <think>...</think> block) is generated unconstrained and
        only the action call afterwards is CFG-constrained.

        :param world_graph: The world graph.
        :return: The response grammar.
        """
    
        delimiter = "\\n"
        tool_rules = self.build_tool_grammar(world_graph)
        root_rule = f'root ::= tool_call "{delimiter}Assigned!"'
        return "\n".join([root_rule, tool_rules])

    # modified from habitat_llm/tools/perception/find_agent_action_tool.py
    def _get_state_history(self, include_only_new_actions: bool) -> str:
        """Method to get state history of the other agent"""

        # Set other agent id - assumes there are only two agents named 0 and 1
        assert len(self.agents) == 1
        other_agent_id = 1 - self.agents[0].uid

        # Query the state history from the env_interface
        if self.env_interface.conf.get("observe_oaa_same_room",False):
            # use same room observations
            history_elements = self.env_interface.agent_observed_state_history[other_agent_id]
        else:
            # use all observations
            history_elements = self.env_interface.agent_state_history[other_agent_id]
            
        if len(history_elements) == 0:
            return "No action observed yet"

        states = [el.state for el in history_elements]

        # Construct the state history
        if include_only_new_actions:
            result = ", ".join(states[self.skip_actions_num:])
            self.skip_actions_num = len(states)
            if result=="":
                return states[-1] 
            return result
        
        return ", ".join(states)
    
    def _get_action_history(self, include_only_new_actions: bool) -> str:
        """Method to get the action history of the other agent"""

        # Set other agent id - assumes there are only two agents named 0 and 1
        assert len(self.agents) == 1
        other_agent_id = 1 - self.agents[0].uid

        # Query the state history from the env_interface
        if self.env_interface.conf.get("observe_oaa_same_room",False):
            # use same room observations
            history_elements = self.env_interface.agent_observed_action_history[other_agent_id]
        else:
            # use all observations
            history_elements = self.env_interface.agent_action_history[other_agent_id]
        
        if len(history_elements) == 0:
            return "No action observed yet"

        actions = [f"{el.action[0]}[{el.action[1]}]" for el in history_elements]

        # Construct the state history
        if include_only_new_actions:
            result = ", ".join(actions[self.skip_actions_num:])
            self.skip_actions_num = len(actions)
            if result=="":
                return actions[-1]
            return result
        
        return ", ".join(actions)
    
    def _add_responses_to_prompt(self, responses: Dict[int, str]) -> str:
        """
        Add skill responses to the prompt optionally including object descriptions (depending on the config).

        :param responses: A dictionary of agent responses.
        :return: The updated print string.
        """
        if self.planner_config.objects_response:
            assert len(self.agents) == 1
            agent = self.agents[0]
            result = ""
            world_graph = self.env_interface.world_graph[agent.uid]
            if responses[agent.uid] != "":
                objects = get_objects_descr(
                    world_graph,
                    agent.uid,
                    include_room_name=True,
                    add_state_info=self.planner_config.objects_response_include_states,
                )
                # conditions when the human agent shouldn't observe the robot's actions 
                # (not partnr's partial observability settings)
                should_observe_oaa = not (self.env_interface.partial_obs and self.env_interface.conf.agent_asymmetry and not self.env_interface.conf.get('robot_asymmetry',False) and self.env_interface.conf.get('update_other_agent_world_graph_by_agent_action',True))
                # human agent can only observe oaa, if it's not the partnr's partial observability settings
                # because in partnr's partial observability settings the robot agent can observe the human, 
                # but the human cannot observe the robot
                if self.planner_config.get("other_agents_action",False) and ((agent.uid == 0) or should_observe_oaa):
                    if self.planner_config.include_only_new_actions:
                        response_format = (
                            "{user_tag}Result: {result}\nThe other agent's actions since the previous observation: {other_agents_actions}\nObjects: {objects}{eot_tag}"
                        )
                    else:    
                        response_format = (
                            "{user_tag}Result: {result}\nAll of the other agent's actions until now: {other_agents_actions}\nObjects: {objects}{eot_tag}"
                        )

                    if self.planner_config.action_description_type=='states':
                        # Extract state history from environment
                        other_agents_actions = self._get_state_history(self.planner_config.include_only_new_actions)
                    elif self.planner_config.action_description_type=='actions':
                        # Extract action history from environment
                        other_agents_actions = self._get_action_history(self.planner_config.include_only_new_actions)    
                    elif self.planner_config.action_description_type=='faat':
                        # Use FindAgentActionTool to get the action history
                        if self.planner_config.include_only_new_actions:
                            other_agents_actions = self._get_state_history(self.planner_config.include_only_new_actions)
                            if other_agents_actions != "None":
                                low_level_action, other_agents_actions = agent.process_high_level_action(
                                    "FindAgentActionTool", None, None, faat_wait=False, faat_other_agents_states=other_agents_actions
                                )
                        else:
                            low_level_action, other_agents_actions = agent.process_high_level_action(
                                "FindAgentActionTool", None, None, faat_wait=False
                            )
                        if other_agents_actions=="":
                            other_agents_actions="None"
                    else:
                        raise ValueError(f"Unknown action_description_type: {self.planner_config.action_description_type}")
                                        
                    result = response_format.format(
                        result=responses[agent.uid],
                        objects=objects,
                        other_agents_actions=other_agents_actions,
                        user_tag=self.planner_config.llm.user_tag,
                        eot_tag=self.planner_config.llm.eot_tag,
                    )
                else:
                    response_format = (
                        "{user_tag}Result: {result}\nObjects: {objects}{eot_tag}"
                    )
                
                    result = response_format.format(
                        result=responses[agent.uid],
                        objects=objects,
                        user_tag=self.planner_config.llm.user_tag,
                        eot_tag=self.planner_config.llm.eot_tag,
                    )
                self.curr_prompt += result + self.planner_config.llm.assistant_tag
                print(result + self.planner_config.llm.assistant_tag, end="")
                self.trace += result + self.planner_config.llm.assistant_tag
        else:
            result = super()._add_responses_to_prompt(responses)
        return result
    
    def replan(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ):
        """
        Replan a high level action using the LLM/VLM.

        If `think_then_constrain` is enabled, generation is unconstrained
        until the model emits `think_close_tag` (e.g. "</think>"), and only
        the action call afterwards is CFG-constrained to a valid tool call.
        This lets reasoning models (e.g. DeepSeek-R1-Distill) freely emit a
        <think>...</think> block before selecting an action.
        """
        if not self.planner_config.get("think_then_constrain", False):
            return super().replan(instruction, observations, world_graph)

        think_open = self.planner_config.get("think_open_tag", "<think>\n")
        think_close = self.planner_config.get("think_close_tag", "</think>")

        llm_response = self.llm.generate(
            self.curr_prompt + think_open,
            self.stopword,
            generation_args={
                "grammar_definition": self.build_action_only_grammar(
                    world_graph[self._agents[0].uid]
                ),
                "think_gate_close_tag": think_close,
                "think_gate_max_tokens": self.planner_config.get(
                    "think_gate_max_tokens", None
                ),
            },
        )
        llm_response = self.format_response(llm_response, self.end_expression)
        llm_response = think_open + llm_response

        return {"llm_response": llm_response}

    def parse_thought(self, input_string: str) -> str:
        """
        Extract thought from the LLM response, handling <think>...</think>
        blocks produced when `think_then_constrain` is enabled.

        :param input_string: The input string to parse.
        :return: The extracted thought.
        """
        if "</think>" in input_string:
            return input_string.split("</think>")[0].lstrip("<think>").strip()
        return super().parse_thought(input_string)

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
        verbose: bool = False,
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Get the next low-level action to execute.

        :param instruction: The instruction for the task.
        :param observations: The current observations.
        :param world_graph: The world graph for each agent.
        :param verbose: Whether to print verbose output. Defaults to False.
        :return: A tuple containing:
                 - The low-level actions for each agent
                 - Planner information
                 - Whether the planner is done
        """
        planner_info: Dict[str, Union[Any, str]] = {}
        # Early return if planner is already done
        if self.is_done:
            planner_info = {
                "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                "traces": {agent.uid: self.trace for agent in self.agents},
                "replanning_count": {
                    agent.uid: self.replanning_count for agent in self.agents
                },
                "replanned": {agent.uid: False for agent in self.agents},
                "replan_required": {
                    agent.uid: self.replan_required for agent in self.agents
                },
                'reasoning_about_other_agent_count': {agent.uid: self.reasoning_about_other_agent_count for agent in self.agents},
                'agent_states': {agent.uid: 'Standing' for agent in self.agents},
                "is_done": {agent.uid: self.is_done for agent in self.agents},
            }
            return {}, planner_info, self.is_done

        if self.curr_prompt == "":
            # Prepare prompts
            self.curr_prompt, self.params = self.prepare_prompt(
                instruction, world_graph[self._agents[0].uid], observations=observations
            )
            self.curr_obj_states = get_objects_descr(
                world_graph[self._agents[0].uid],
                self._agents[0].uid,
                include_room_name=True,
                add_state_info=self.planner_config.objects_response_include_states,
                centralized=self.planner_config.centralized,
            )

        if self.trace == "":
            self.trace += f"Task: {instruction}\nThought: "

        print_str = ""
        self.is_done = False

        if self.replan_required:
            for attr_name, attr_value in world_graph.items():
                #print(f"\n--------------[DEBUG] World Graph of Agent {attr_name}------------\n")
                self.trace += f"\n--------------[DEBUG] World Graph of Agent {attr_name}------------\n"
                for k,v in attr_value.graph.items():
                    if isinstance(k,Object) or isinstance(k,SpotRobot) or isinstance(k,Human):
                        #print(k, ": ", v)
                        #print()
                        self.trace += f"{k} : {v}\n"
                #print(f"-------------------------------------")
                self.trace += f"-------------------------------------\n"

            planner_info["replanned"] = {agent.uid: True for agent in self.agents}
            if verbose:
                # calculate the total time of response generation
                start_time = time.time()

            response_info = self.replan(instruction, observations, world_graph)
            #print(f'response_info: {response_info}')
            llm_response = response_info["llm_response"]
            #print(f'llm_response: {llm_response}')
            # parse thought from the response
            thought = self.parse_thought(llm_response)
            #print(f'thought: {thought}')
            
            if verbose:
                total_time = time.time() - start_time
                print(
                    f"Time taken for LLM response generation: {total_time}; replanning_count: {self.replanning_count}"
                )

            # Update prompt with the first response
            print_str += f"""----------agent {(self.agents[0]).uid}----------\n{llm_response}\n{self.stopword}\n----------agent {(self.agents[0]).uid}----------\n"""
            prompt_addition = (
                f"""{llm_response}\n{self.stopword}{self.planner_config.llm.eot_tag}"""
            )
            self.curr_prompt += prompt_addition
            self.trace += prompt_addition

            # Check if the planner should stop
            # Stop if the replanning count exceed a certain threshold
            # or end expression is found in llm response
            # This is helpful to break infinite planning loop.
            self.is_done = (self.check_if_agent_done(llm_response)) or (
                self.replanning_count == self.planner_config.replanning_threshold
            )
            # Increment the llm call counter on every replan
            # doesn't get incremented before comparison as first "replan" is technically
            # the first required plan
            self.replanning_count += 1
            
            self.is_reasoning_about_other_agent = self.check_if_reasoning_about_other_agent(llm_response)
            if self.is_reasoning_about_other_agent:
                self.reasoning_about_other_agent_count += 1
            
            # Early return if stop is required
            if self.is_done:
                planner_info = {
                    "print": print_str,
                    # "print_no_tags": print_str_no_tags,
                    "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                    "traces": {agent.uid: self.trace for agent in self.agents},
                    "replanning_count": {
                        agent.uid: self.replanning_count for agent in self.agents
                    },
                    "replan_required": {
                        agent.uid: self.replan_required for agent in self.agents
                    },
                    "replanned": {agent.uid: True for agent in self.agents},
                    "is_done": {agent.uid: self.is_done for agent in self.agents},
                    "thought": {agent.uid: thought for agent in self.agents},
                    "high_level_actions": {
                        agent.uid: ("Done", None, None) for agent in self.agents
                    },
                    'agent_states': {agent.uid: 'Standing' for agent in self.agents},
                    'is_reasoning_about_other_agent': {agent.uid: self.is_reasoning_about_other_agent for agent in self.agents},
                    'reasoning_about_other_agent_count': {agent.uid: self.reasoning_about_other_agent_count for agent in self.agents},
                }
                return {}, planner_info, self.is_done

            # Parse high level action directives from llm response
            high_level_actions = self.actions_parser(
                self.agents, llm_response, self.params
            )

            print(f"\n\n[DEBUG] Now Executing: {high_level_actions}\n\n")

            # Get low level actions and/or responses
            low_level_actions, responses = self.process_high_level_actions(
                high_level_actions, observations
            )

            # Store last executed high level action
            self.last_high_level_actions = high_level_actions
        else:
            planner_info["replanned"] = {agent.uid: False for agent in self.agents}
            # Set thought to None
            thought = None

            # Get low level actions and/or responses using last high level actions
            low_level_actions, responses = self.process_high_level_actions(
                self.last_high_level_actions, observations
            )

        # Log if replanning was done or not before overwriting the value
        planner_info["replan_required"] = {
            agent.uid: self.replan_required for agent in self.agents
        }

        # Check if replanning is required
        # Replanning is required when any of the actions being executed
        # have a response indicating success or failure (and the reason)
        self.replan_required = any(responses.values())
        print_str += self._add_responses_to_prompt(responses)

        # Update planner info
        planner_info["responses"] = responses
        planner_info["thought"] = {agent.uid: thought for agent in self.agents}
        planner_info["is_done"] = {agent.uid: self.is_done for agent in self.agents}
        planner_info["print"] = print_str
        # planner_info["print_no_tags"] = print_str_no_tags
        planner_info["high_level_actions"] = self.last_high_level_actions
        planner_info["prompts"] = {agent.uid: self.curr_prompt for agent in self.agents}
        planner_info["traces"] = {agent.uid: self.trace for agent in self.agents}
        planner_info["replanning_count"] = {
            agent.uid: self.replanning_count for agent in self.agents
        }
        planner_info['reasoning_about_other_agent_count'] = {agent.uid: self.reasoning_about_other_agent_count for agent in self.agents}
        planner_info["agent_states"] = self.get_last_agent_states()
        planner_info["agent_positions"] = self.get_last_agent_positions()
        planner_info["agent_collisions"] = self.get_agent_collisions()
        return low_level_actions, planner_info, self.is_done
    
    def check_if_reasoning_about_other_agent(self, llm_response: str) -> bool:
        """
        Check if the agent is about the other agent in the LLM response.

        :param llm_response: The LLM response to check.
        :return: True if the agent is reasoning about other, False otherwise.
        """
        return (llm_response is not None) and ("other agent" in llm_response.lower())
