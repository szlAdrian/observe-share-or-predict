# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

# TODO: 
# Implement and add to config if the Knowledge Summary agent should see only the last observation or the full history; if the planner should see only the last Knowledge Summary or the full history
# Delete the first line of the observation for the Knowledge summarizer? (Search Object:) Result: .... 
import re
import time

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from habitat_llm.llm.instruct.utils import get_objects_descr, get_world_descr
from habitat_llm.planner import ZeroShotReactPlanner
from habitat_llm.utils.grammar import FREE_TEXT

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph


class ZeroShotKnowledgeReactPlanner(ZeroShotReactPlanner):
    """
    This class builds the prompt for the, zero shot llm knowledge react planner format.
    Before every new action taking there is a knowledge inference about the objects and the other agent's state and intent?
    This knowledge is then also used to build the prompt for the react planner.
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
        
        self.curr_prompt_no_knowledge_summary: str = ""
        
        # Initialie container to store the prompt for the knowledge summary generation
        self.curr_knowledge_summary_prompt: str = ""
        
        # Initialize the knowledge summary prompt
        self.knowledge_summary_prompt = self.instruct.knowledge_summary_prompt
        
    def reset(self):
        """
        Reset the planner state.
        """
        self.curr_knowledge_summary_prompt: str = ""
        self.curr_prompt_no_knowledge_summary: str = ""
        super().reset()
    
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
                response_format_with_knowledge_summary = (
                    "{user_tag}Result: {result}\nObjects: {objects}\n{knowledge_summary}{eot_tag}"
                )
                
                response_format_knowledge_summary_prompt = (
                    "Result: {result}\nObjects: {objects}"
                )
                
                objects = get_objects_descr(
                    world_graph,
                    agent.uid,
                    include_room_name=True,
                    add_state_info=self.planner_config.objects_response_include_states,
                )
                
                if self.planner_config.get("only_last_reasoning_considered",False):
                    self.curr_prompt = self.curr_prompt_no_knowledge_summary
                    response_format_no_knowledge_summary_prompt = (
                        "{user_tag}Result: {result}\nObjects: {objects}{eot_tag}"
                    )
                    result_no_knowledge_summary_prompt = response_format_no_knowledge_summary_prompt.format(
                        result=responses[agent.uid],
                        objects=objects,
                        user_tag=self.planner_config.llm.user_tag,
                        eot_tag=self.planner_config.llm.eot_tag,
                    )
                    self.curr_prompt_no_knowledge_summary += result_no_knowledge_summary_prompt + self.planner_config.llm.assistant_tag
                
                result_knowledge_summary_prompt = response_format_knowledge_summary_prompt.format(
                    result=responses[agent.uid],
                    objects=objects,
                )
                
                self.curr_knowledge_summary_prompt += result_knowledge_summary_prompt
                
                # get the knowledge summary after every new observation
                knowledge_summary_response_info = self._get_knowledge_summary()
                llm_knowledge_summary_response = knowledge_summary_response_info["llm_response"]
                
                result = response_format_with_knowledge_summary.format(
                    result=responses[agent.uid],
                    objects=objects,
                    knowledge_summary=llm_knowledge_summary_response,
                    user_tag=self.planner_config.llm.user_tag,
                    eot_tag=self.planner_config.llm.eot_tag,
                )
                
                self.curr_prompt += result + self.planner_config.llm.assistant_tag
                print(result + self.planner_config.llm.assistant_tag, end="")
                self.trace += result + self.planner_config.llm.assistant_tag
        else:
            result = super()._add_responses_to_prompt(responses)
        return result
    
    def parse_action(self, input_string: str) -> str:
        """
        Extract action from the LLM response.

        :param input_string: The input string to parse.
        :return: The extracted action.
        """
        # Define the patterns for Agent actions
        pattern = r"\n|Final Thought"

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        if match:
            # Extract the text before the pattern
            return input_string[match.start()+1:].strip()
        else:
            # If no pattern is found, return the whole string
            return ""
    
    def prepare_knowledge_summary_prompt(
        self, input_instruction: str, world_graph: "WorldGraph", **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare the prompt for the LLM for knowledge summary.

        :param input_instruction: The input instruction.
        :param world_graph: The world graph.
        :return: The prepared prompt and parameters.
        """
        params = {
            "input": input_instruction,
            "tool_list": self.tool_list,
            "world_graph": world_graph,
            "id": self.agents[0].uid,
        }

        if "{agent_descriptions}" in self.knowledge_summary_prompt:
            params["agent_descriptions"] = self.agent_descriptions
        if "{system_tag}" in self.knowledge_summary_prompt:
            params["system_tag"] = self.planner_config.llm.system_tag
        if "{user_tag}" in self.knowledge_summary_prompt:
            params["user_tag"] = self.planner_config.llm.user_tag
        if "{assistant_tag}" in self.knowledge_summary_prompt:
            params["assistant_tag"] = self.planner_config.llm.assistant_tag
        if "{eot_tag}" in self.knowledge_summary_prompt:
            params["eot_tag"] = self.planner_config.llm.eot_tag
        if "{agent_role_description}" in self.knowledge_summary_prompt:
            # only support agent role description when planning for a single agent
            assert len(self.agents) == 1
            if str(self.agents[0].uid) == "1":
                agent_role_description = '\nYou are playing the role of the task giver. This means if the instruction says something like "You should move the object and I will wash it", then the other agent should be moving the object, and you should washing the it.\n'
            else:
                agent_role_description = '\nYou are playing the role of the task receiver. This means if the instruction says something like "You should move the object and I will wash it", then you should move the object and the other agent should wash it.\n'
            params["agent_role_description"] = agent_role_description
        if "{world_description}" in self.knowledge_summary_prompt:
            # only designed for the decentralized setting
            assert len(self.agents) == 1
            world_description = get_world_descr(
                world_graph,
                agent_uid=self.agents[0].uid,
                add_state_info=self.planner_config.objects_response_include_states,
                include_room_name=True,
                centralized=self.planner_config.centralized,
            )
            params["world_description"] = world_description

        if "should_format" in kwargs and not kwargs["should_format"]:
            # In some cases a subclass may want to fill the extra arguments here, so we don't format
            # because those arguments would be missing.
            output_prompt = ""
        else:
            output_prompt = self.knowledge_summary_prompt.format(**params)
        return output_prompt, params
    
    def _get_knowledge_summary(self):
        """
        Get the knowledge summary using the LLM/VLM
        """
        prompt = self.curr_knowledge_summary_prompt + self.planner_config.llm.eot_tag + self.planner_config.llm.assistant_tag  
        # Generate response
        llm_response = self.llm.generate(prompt, self.stopword)
 
        # Format the response
        # This removes extra text followed by end expression when needed.
        llm_response = self.format_response(llm_response, self.end_expression)

        info = {"llm_response": llm_response}
        
        return info
    
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
                "is_done": {agent.uid: self.is_done for agent in self.agents},
            }
            return {}, planner_info, self.is_done

        if self.curr_prompt == "":
            # Prepare prompts
            self.curr_prompt, self.params = self.prepare_prompt(
                instruction, world_graph[self._agents[0].uid], observations=observations
            )
            if self.planner_config.get("only_last_reasoning_considered",False):
                self.curr_prompt_no_knowledge_summary = self.curr_prompt
            
            self.curr_obj_states = get_objects_descr(
                world_graph[self._agents[0].uid],
                self._agents[0].uid,
                include_room_name=True,
                add_state_info=self.planner_config.objects_response_include_states,
                centralized=self.planner_config.centralized,
            )
        
        if self.curr_knowledge_summary_prompt == "":
            # Prepare knowledge summary prompts
            self.curr_knowledge_summary_prompt, self.knowledge_summary_params = self.prepare_knowledge_summary_prompt(
                instruction, world_graph[self._agents[0].uid], observations=observations
            )

        if self.trace == "":
            self.trace += f"Task: {instruction}\nThought: "

        print_str = ""
        self.is_done = False

        if self.replan_required:
            planner_info["replanned"] = {agent.uid: True for agent in self.agents}
            if verbose:
                # calculate the total time of response generation
                start_time = time.time()
            
            response_info = self.replan(instruction, observations, world_graph)
            llm_response = response_info["llm_response"]
            # parse thought from the response
            thought = self.parse_thought(llm_response)

            if verbose:
                total_time = time.time() - start_time
                print(
                    f"Time taken for LLM response generation: {total_time}; replanning_count: {self.replanning_count}"
                )

            # Update prompt with the first response
            print_str += f"""{llm_response}\n{self.stopword}\n"""
            prompt_addition = (
                f"""{llm_response}\n{self.stopword}{self.planner_config.llm.eot_tag}"""
            )
            self.curr_prompt += prompt_addition
            if self.planner_config.get("only_last_reasoning_considered",False):
                self.curr_prompt_no_knowledge_summary += prompt_addition
            self.trace += prompt_addition
            
            # Parse the action from the LLM response
            action = self.parse_action(llm_response)
            self.curr_knowledge_summary_prompt += f"\n\nSelected action: {action}\n"
            
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
        planner_info["agent_states"] = self.get_last_agent_states()
        planner_info["agent_positions"] = self.get_last_agent_positions()
        planner_info["agent_collisions"] = self.get_agent_collisions()

        return low_level_actions, planner_info, self.is_done
