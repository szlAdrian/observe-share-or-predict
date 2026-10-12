# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import time
from hydra.utils import instantiate
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from habitat_llm.llm.instruct.utils import get_objects_descr, get_world_descr
from habitat_llm.planner import ZeroShotReactPlanner
from habitat_llm.utils.grammar import (
    FREE_TEXT,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph

from habitat_llm.world_model.entity import Object, SpotRobot, Human


class ZeroShotReactPredictOAAOtherReplannedPlanner(ZeroShotReactPlanner):
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
        self.oaactions_parser = instantiate(self.instruct.oaactions_parser)

        # Initialie container to store the prompt for predicting other agent's high-level action
        self.curr_predict_other_agent_high_level_action_prompt: str = ""
        
        # Initialize the other agent action prediction prompt
        self.predict_other_agent_high_level_action_prompt = self.instruct.predict_other_agent_action_prompt
        
    def reset(self):
        """
        Reset the planner state.
        """
        self.curr_predict_other_agent_high_level_action_prompt: str = ""
        super().reset()
        
    def build_response_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts all valid responses based on a world graph.

        :param world_graph: The world graph.
        :return: The response grammar.
        """
        delimiter = "\\n"
        tool_rules = self.build_tool_grammar(world_graph)

        if self.planner_config.get("response_grammar_type",False) == "partner-current-next-self":
            # rule for the predicing current and next action of the partner along with self next action
            # for zero_shot_predict_oaa_prompt
            root_rule = (
                f'root ::= "The other agent\'s current action: " tool_call "{delimiter}The other agent\'s next action: " tool_call "{delimiter}" {FREE_TEXT} "{delimiter}" tool_call "{delimiter}Assigned!"'
            )
        elif self.planner_config.get("response_grammar_type",False) == "partner-next-self":
            # rule for the predicing current and next action of the partner along with self next action
            # for zero_shot_predict_oaa_prompt
            root_rule = (
                f'root ::= "The other agent\'s next action: " tool_call "{delimiter}" {FREE_TEXT} "{delimiter}" tool_call "{delimiter}Assigned!"'
            )
        else:
            # rule for the PARTNR baseline response
            root_rule = (
                f'root ::= {FREE_TEXT} "{delimiter}" tool_call "{delimiter}Assigned!"'
            )

        return "\n".join([root_rule, tool_rules])
    
    # modified from habitat_llm/tools/perception/find_agent_action_tool.py
    def _get_state_history_for_oaa(self, include_only_new_actions: bool) -> str:
        """Method to get state history of the other agent.
        
        from _get_state_history method, but removed the increment 
        of self.skip_actions_num. It shouldn't be updated when 
        predicting the other agent's action.
        """

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
            if result=="":
                return states[-1] 
            return result
        
        return ", ".join(states)
    
    def _get_action_history_for_oaa(self, include_only_new_actions: bool) -> str:
        """Method to get the action history of the other agent
        
        from _get_action_history method, but removed the increment 
        of self.skip_actions_num. It shouldn't be updated when 
        predicting the other agent's action.
        """

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
                self.curr_predict_other_agent_high_level_action_prompt += result + self.planner_config.llm.assistant_tag
                #print(result + self.planner_config.llm.assistant_tag, end="")
                #print(f"\n\n[DEBUG] Current curr_predict_other_agent_high_level_action_prompt: {self.curr_predict_other_agent_high_level_action_prompt}\n\n")
                self.trace += result + self.planner_config.llm.assistant_tag
        else:
            result = super()._add_responses_to_prompt(responses)
        return result
    
    def _generate_current_observation_for_predict_oaa_prompt(self) -> str:
        """
        Generate current observations for the predict other agent high level action prompt optionally including object descriptions (depending on the config).
        From _add_responses_to_prompt method.
        
        :return: The observation string.
        """
        assert self.planner_config.objects_response
        if self.planner_config.objects_response:
            assert len(self.agents) == 1
            agent = self.agents[0]
            result = ""
            world_graph = self.env_interface.world_graph[agent.uid]
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
                        "{user_tag}Result: {result}\nThe other agent's actions since the previous observation: {other_agents_actions}\nYour observations:\nObjects: {objects}{eot_tag}"
                    )
                else:    
                    response_format = (
                        "{user_tag}Result: {result}\nAll of the other agent's actions until now: {other_agents_actions}\nYour observations:\nObjects: {objects}{eot_tag}"
                    )

                if self.planner_config.action_description_type=='states':
                    # Extract state history from environment
                    other_agents_actions = self._get_state_history_for_oaa(self.planner_config.include_only_new_actions)
                elif self.planner_config.action_description_type=='actions':
                    # Extract action history from environment
                    other_agents_actions = self._get_action_history_for_oaa(self.planner_config.include_only_new_actions)    
                elif self.planner_config.action_description_type=='faat':
                    # Use FindAgentActionTool to get the action history
                    if self.planner_config.include_only_new_actions:
                        other_agents_actions = self._get_state_history_for_oaa(self.planner_config.include_only_new_actions)
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
                    result="Your last action is still under execution.",
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
                    result="Your last action is still under execution.",
                    objects=objects,
                    user_tag=self.planner_config.llm.user_tag,
                    eot_tag=self.planner_config.llm.eot_tag,
                )
            
        return result
    
    def prepare_predict_other_agent_high_level_action_prompt(
        self, input_instruction: str, world_graph: "WorldGraph", **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare the prompt for the LLM for predicting the other agent's high-level action.
        From the LLMPlanner.prepare_prompt method.
        
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
        
        # We modify the prompt if we want to use RAG and the prompt has not
        # been modified
        if "{rag_examples}" in self.predict_other_agent_high_level_action_prompt:
            if self.rag is not None:
                _, index = self.rag.retrieve_top_k_given_query(
                    input_instruction, top_k=1, agent_id=self._agents[0].uid
                )
                index = index[0]

                example_str = (
                    f"{self.planner_config.llm.user_tag}Below are some example solutions from different settings:\nExample 1:\n"
                    + self.rag.data_dict[index]["trace"]
                    + "\n"
                )
                params["rag_examples"] = example_str
            else:
                params["rag_examples"] = ""
        if "{tool_descriptions}" in self.predict_other_agent_high_level_action_prompt:
            params["tool_descriptions"] = self.agents[0].other_agents_tool_descriptions
        if "{agent_descriptions}" in self.predict_other_agent_high_level_action_prompt:
            params["agent_descriptions"] = self.agent_descriptions
        if "{tool_list}" in self.predict_other_agent_high_level_action_prompt:
            params["tool_list"] = self.tool_list
        if "{system_tag}" in self.predict_other_agent_high_level_action_prompt:
            params["system_tag"] = self.planner_config.llm.system_tag
        if "{user_tag}" in self.predict_other_agent_high_level_action_prompt:
            params["user_tag"] = self.planner_config.llm.user_tag
        if "{assistant_tag}" in self.predict_other_agent_high_level_action_prompt:
            params["assistant_tag"] = self.planner_config.llm.assistant_tag
        if "{eot_tag}" in self.predict_other_agent_high_level_action_prompt:
            params["eot_tag"] = self.planner_config.llm.eot_tag
        if "{agent_role_description}" in self.predict_other_agent_high_level_action_prompt:
            # only support agent role description when planning for a single agent
            assert len(self.agents) == 1
            if str(self.agents[0].uid) == "1":
                agent_role_description = '\nYou are playing the role of the task giver. This means if the instruction says something like "You should move the object and I will wash it", then the other agent should be moving the object, and you should washing the it.\n'
            else:
                agent_role_description = '\nYou are playing the role of the task receiver. This means if the instruction says something like "You should move the object and I will wash it", then you should move the object and the other agent should wash it.\n'
            params["agent_role_description"] = agent_role_description
        if "{world_description}" in self.predict_other_agent_high_level_action_prompt:
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
            output_prompt = self.predict_other_agent_high_level_action_prompt.format(**params)
        # print('------------------prepare_predict_other_agent_high_level_action_prompt-----------------')
        # print(f'self.agents[0].uid: {self.agents[0].uid}')
        # print(f"\n\n[DEBUG] output_prompt: {output_prompt}\n\n")
        # print('---------------------------------------------------------')
        return output_prompt, params
    
    def generate_prediction(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
        prompt: str,
    ):
        """
        Generate the prediction of the other agent's high level action using the LLM/VLM
        From the LLMPlanner.replan method.
        """
        # Generate response
        if self.planner_config.get("constrained_generation", False):
            llm_response = self.llm.generate(
                prompt,
                self.stopword,
                generation_args={
                    "grammar_definition": self.build_response_grammar(
                        world_graph[self._agents[0].uid]
                    )
                },
            )
        else:
            llm_response = self.llm.generate(prompt, self.stopword)

        # Format the response
        # This removes extra text followed by end expression when needed.
        llm_response = self.format_response(llm_response, self.end_expression)

        info = {"llm_response": llm_response}
        return info
    
    def predict_other_agent_high_level_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
        verbose: bool = False,
    ) -> Dict[int, str]:
        """
        Predict the other agent's high-level action.

        :param instruction: The instruction for the task.
        :param observations: The current observations.
        :param world_graph: The world graph for each agent.
        :param verbose: Whether to print verbose output. Defaults to False.
        :return: The predicted high-level action of the other agent.
        """
        
        if self.curr_predict_other_agent_high_level_action_prompt == "":
            # Prepare other agent action prediction prompt
            self.curr_predict_other_agent_high_level_action_prompt, params_ = self.prepare_predict_other_agent_high_level_action_prompt(
                instruction,
                world_graph[self._agents[0].uid],
                observations=observations,
            )
        
        result = self._generate_current_observation_for_predict_oaa_prompt()
        predict_prompt = self.curr_predict_other_agent_high_level_action_prompt + result + self.planner_config.llm.assistant_tag
        # print('----------------predict prompt for other agent high-level action-----------------')
        # print(f"\n\n[DEBUG] predict_prompt: {predict_prompt}\n\n")
        # print('---------------------------------------------------------')

        response_info = self.generate_prediction(instruction, observations, world_graph, prompt=predict_prompt)
        # print('---------------------------------------------------------')
        # print(f'response_info oaa: {response_info}')
        # print('---------------------------------------------------------')
        llm_response = response_info["llm_response"]
        
        # Parse high level action directives from llm response
        predicted_other_agent_high_level_action = self.actions_parser(
            self.agents, llm_response, self.params
        )
        # print('---------------------------------------------------------')
        # print(f'predicted_other_agent_high_level_action: {predicted_other_agent_high_level_action}')
        # print('---------------------------------------------------------')
            
        trace_addition = (
            f"""\n-----------------current observation & predicting the other agent's high-level action-----------------\n{result}{self.planner_config.llm.assistant_tag}\n{llm_response}\n{self.planner_config.llm.eot_tag}\nPredicted other agent's high-level action: {predicted_other_agent_high_level_action}\n----------------------------------"""
        )
        self.trace += trace_addition
        
        return predicted_other_agent_high_level_action
    
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
        if self.curr_predict_other_agent_high_level_action_prompt == "":
            # Prepare other agent action prediction prompt
            self.curr_predict_other_agent_high_level_action_prompt, params_ = self.prepare_predict_other_agent_high_level_action_prompt(
                instruction,
                world_graph[self._agents[0].uid],
                observations=observations,
            )
        #print(f"\n\n[DEBUG] Current curr_predict_other_agent_high_level_action_prompt: {self.curr_predict_other_agent_high_level_action_prompt}\n\n")
        #print(f"params_: {params_}\n\n")
        # asd
        if self.trace == "":
            self.trace += f"Task: {instruction}\nThought: "

        print_str = ""
        self.is_done = False
        '''
        if self.counter % 10 == 0:
            self.prev_robot_graph = world_graph[0].graph
            self.prev_human_graph = world_graph[1].graph
            for attr_name, attr_value in world_graph.items():
                    print(f"\n--------------[DEBUG] World Graph of Agent {attr_name}------------\n")
                    #self.trace += f"\n--------------[DEBUG] World Graph of Agent {attr_name}------------\n"
                    for k,v in attr_value.graph.items():
                        if isinstance(k,Object) or isinstance(k,SpotRobot) or isinstance(k,Human):
                            print(k, ": ", v)
                            print()
                            #self.trace += f"{k} : {v}\n"
                    print(f"-------------------------------------")
                    #self.trace += f"-------------------------------------\n"
        
        if self.counter > 0 and (self.prev_robot_graph != world_graph[0].graph or self.prev_human_graph != world_graph[1].graph):
            self.prev_robot_graph = world_graph[0].graph
            self.prev_human_graph = world_graph[1].graph
            for attr_name, attr_value in world_graph.items():
                    print(f"\n--------------[DEBUG] World Graph of Agent {attr_name}------------\n")
                    #self.trace += f"\n--------------[DEBUG] World Graph of Agent {attr_name}------------\n"
                    for k,v in attr_value.graph.items():
                        if isinstance(k,Object) or isinstance(k,SpotRobot) or isinstance(k,Human):
                            print(k, ": ", v)
                            print()
                            #self.trace += f"{k} : {v}\n"
                    print(f"-------------------------------------")
                    #self.trace += f"-------------------------------------\n"
        self.counter += 1
        '''
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
            self.curr_predict_other_agent_high_level_action_prompt += prompt_addition
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
                }
                return {}, planner_info, self.is_done

            # Parse high level action directives from llm response
            high_level_actions = self.actions_parser(
                self.agents, llm_response, self.params
            )
            # Parse other agent's high level action directives from llm response
            # oa_high_level_actions = self.oaactions_parser(
            #     self.agents, llm_response, self.params
            # )
            # print(f"\n\n[DEBUG] Other Agent's Predicted Action: {oa_high_level_actions}\n\n")

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
