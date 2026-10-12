# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import time
from typing import TYPE_CHECKING, Any, Dict, Tuple, Union, List
from hydra.utils import instantiate

from build.lib.habitat_llm.planner.zero_shot_react_planner import ZeroShotReactPlanner
from habitat_llm.llm.instruct.utils import get_objects_descr, get_world_descr
from habitat_llm.planner import ZeroShotReactPlanner
from habitat_llm.utils.grammar import (
    FREE_TEXT,
    FURNITURE,
    NAV_TARGET,
    NAV_TARGET_OBJECT_CLASS,
    OBJECT,
    OBJECT_OR_FURNITURE,
    OBJECT_CLASS,
    OBJECT_CLASS_OR_FURNITURE,
    ROOM,
    SPATIAL_CONSTRAINT,
    SPATIAL_RELATION, 
)

from habitat_llm.world_model.entity import Object, SpotRobot, Human

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph


class ZeroShotReactPredictOAAPlanner(ZeroShotReactPlanner):
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
        self.skip_actions_num: int = 0 # Number of other agent's actions to skip when constructing the action history

    def reset(self):
        """
        Reset the planner state.
        """
        self.skip_actions_num: int = 0
        super().reset()
    
    def build_other_agent_tool_call_grammar(self, world_graph: "WorldGraph") -> str:
        """
        This method builds a grammar that accepts all valid tool calls based a world graph
        The grammar is specified in the EBNF grammar description format
        see https://github.com/epfl-dlab/transformers-CFG for details and examples

        :param world_graph: The world graph.
        """
        tool_grammar = {}
        objects = world_graph.get_all_objects()
        
        for agent in self.agents:
            for tool_name, tool in agent.other_agents_tools.items():
                if tool_name not in tool_grammar:
                    # skip tools that require objects when there are no objects
                    if OBJECT in tool.argument_types and len(objects) == 0:
                        continue
                    tool_grammar[tool_name] = tool.grammar()
        tool_grammar["Done"] = '"Done[]"'
        grammar_str = "other_agent_tool_call ::= " + " | ".join(tool_grammar.keys()) + "\n"
        for tool_name, tool_grammar_str in tool_grammar.items():
            # "Done" is already defined by build_tool_grammar()'s output, which
            # this grammar fragment is always concatenated with in
            # build_response_grammar(). Re-emitting it here would define the
            # "Done" nonterminal twice, which xgrammar's EBNF parser rejects
            # ("Invalid grammar specification") even though transformers_cfg
            # (used by the HF fallback) tolerates it.
            if tool_name == "Done":
                continue
            grammar_str += f"{tool_name} ::= {tool_grammar_str}\n"

        
        return grammar_str
    
    def build_tool_grammar_oaa_obj_class(self, world_graph: "WorldGraph") -> str:
        """
        This method builds a grammar that accepts all valid tool calls based a world graph
        The grammar is specified in the EBNF grammar description format
        see https://github.com/epfl-dlab/transformers-CFG for details and examples

        :param world_graph: The world graph.
        """
        tool_grammar = {}
        tool_grammar_other_agent = {}
        objects = world_graph.get_all_objects()
        # trim down the '_<id>' from the object names to get the object class
        object_classes = set([obj.name.rsplit('_', 1)[0] if '_' in obj.name else obj.name for obj in self.env_interface.full_world_graph.get_all_objects()])
        # print('----------------full_world_graph_object_classes----------------')
        # print(object_classes)
        # print('----------------full_world_graph_object_classes----------------')
        rules = []
        # we cannot include rules which have objects when there are no objects
        if len(objects) != 0:
            object_expansion = " | ".join(
                (f'"{x.name}"' for x in world_graph.get_all_objects())
            )
            object_rule = f"{OBJECT} ::= " + object_expansion
            nav_target_rule = f"{NAV_TARGET} ::= ({FURNITURE} | {ROOM} | {OBJECT})"
            object_of_furniture_rule = (
                f"{OBJECT_OR_FURNITURE} ::= ({FURNITURE} | {OBJECT})"
            )
            rules.append(nav_target_rule)
            rules.append(object_rule)
            rules.append(object_of_furniture_rule)
        else:
            object_of_furniture_rule = f"{OBJECT_OR_FURNITURE} ::= {FURNITURE}"
            nav_target_rule = f"{NAV_TARGET} ::= ({FURNITURE} | {ROOM})"
            rules.append(nav_target_rule)
            rules.append(object_of_furniture_rule)
        
        # Add object class rules everytime, so the agent can predict object classes 
        # even when there are no objects in the current world graph, which can be 
        # helpful for generalization and also for the cases when the agent cannot 
        # observe the objects in the environment 
        # Designed to use for predicting the other agent's action
        object_class_expansion = " | ".join(
                (f'"{x}"' for x in object_classes)
            )
        object_class_rule = f"{OBJECT_CLASS} ::= " + object_class_expansion
        nav_target_object_class_rule = f"{NAV_TARGET_OBJECT_CLASS} ::= ({FURNITURE} | {ROOM} | {OBJECT_CLASS})"
        object_class_or_furniture_rule = (
            f"{OBJECT_CLASS_OR_FURNITURE} ::= ({FURNITURE} | {OBJECT_CLASS})"
        )
        rules.append(nav_target_object_class_rule)
        rules.append(object_class_rule)
        rules.append(object_class_or_furniture_rule)
            
        # tool call grammar for the action call of this agent    
        for agent in self.agents:
            for tool_name, tool in agent.tools.items():
                if tool_name not in tool_grammar:
                    # skip tools that require objects when there are no objects
                    if OBJECT in tool.argument_types and len(objects) == 0:
                        continue
                    tool_grammar[tool_name] = tool.grammar()
        tool_grammar["Done"] = '"Done[]"'
        grammar_str = "tool_call ::= " + " | ".join(tool_grammar.keys()) + "\n"
        for tool_name, tool_grammar_str in tool_grammar.items():
            grammar_str += f"{tool_name} ::= {tool_grammar_str}\n"
            
        # tool call grammar for the predicted action call of the other agent
        for agent in self.agents:
            for tool_name, tool in agent.other_agents_tools.items():
                if tool_name not in tool_grammar_other_agent:
                    # skip tools that require objects when there are no objects
                    # if OBJECT in tool.argument_types and len(objects) == 0:
                    #     continue
                    tool_name = tool_name + "_obj_class"
                    tool_grammar_other_agent[tool_name] = tool.grammar_obj_class()
        tool_grammar_other_agent["Done"] = '"Done[]"'
        grammar_str += "other_agent_tool_call ::= " + " | ".join(tool_grammar_other_agent.keys()) + "\n"
        for tool_name, tool_grammar_str in tool_grammar_other_agent.items():
            # "Done" was already emitted above for tool_grammar/tool_call --
            # skip it here to avoid defining the "Done" nonterminal twice in
            # the same grammar (xgrammar rejects duplicate rule definitions).
            if tool_name == "Done":
                continue
            grammar_str += f"{tool_name} ::= {tool_grammar_str}\n"

        # build rules for each of the argument types
        furniture_rule = f"{FURNITURE} ::= " + " | ".join(
            (f'"{x.name}"' for x in world_graph.get_all_furnitures())
        )
        room_rule = f"{ROOM} ::= " + " | ".join(
            (f'"{x.name}"' for x in world_graph.get_all_rooms())
        )
        spatial_constraint_rule = f'{SPATIAL_CONSTRAINT} ::= "next_to"'
        spatial_relation_rule = f'{SPATIAL_RELATION} ::= "on" | "within"'
        free_text_rule = f"{FREE_TEXT} ::= [ \"'.:,!a-zA-Z_0-9]*"
        white_space_rule = "WS ::= [ ]*"
        rules.append(furniture_rule)
        rules.append(room_rule)
        rules.append(spatial_constraint_rule)
        rules.append(spatial_relation_rule)
        rules.append(free_text_rule)
        rules.append(white_space_rule)
        grammar_str += "\n".join(rules)
       
        return grammar_str
    
    def build_response_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts all valid responses based on a world graph.

        :param world_graph: The world graph.
        :return: The response grammar.
        """
        delimiter = "\\n"
        if self.planner_config.get('response_grammar_type', False) == "predict_oaa_obj_class":
            tool_rules = self.build_tool_grammar_oaa_obj_class(world_graph)
            
            # rule for the PARTNR baseline response
            root_rule = (
                f'root ::= {FREE_TEXT} "{delimiter}The other agent\'s predicted next action: " other_agent_tool_call "{delimiter}My next action: " tool_call "{delimiter}Assigned!"'
            )
            # root_rule = (
            #     f'root ::= {FREE_TEXT} "{delimiter}" other_agent_tool_call "{delimiter}" tool_call "{delimiter}Assigned!"'
            # )
            # print('--------------------------------build_response_grammar--------------------------------')
            # print("\n".join([root_rule, tool_rules]))
            # print('--------------------------------build_response_grammar--------------------------------')
            # asd
            return "\n".join([root_rule, tool_rules])
        else:
            tool_rules = self.build_tool_grammar(world_graph)
            other_agent_tool_call_rule = self.build_other_agent_tool_call_grammar(world_graph)
            # print('--------------------------------build_response_grammar--------------------------------')
            # print(f'other_agent_tool_call_rule: {other_agent_tool_call_rule}')
            # print('--------------------------------')
            
            # rule for the PARTNR baseline response
            root_rule = (
                f'root ::= {FREE_TEXT} "{delimiter}The other agent\'s predicted next action: " other_agent_tool_call "{delimiter}My next action: " tool_call "{delimiter}Assigned!"'
            )
            # root_rule = (
            #     f'root ::= {FREE_TEXT} "{delimiter}" other_agent_tool_call "{delimiter}" tool_call "{delimiter}Assigned!"'
            # )
            # print("\n".join([root_rule, other_agent_tool_call_rule, tool_rules]))
            # print('--------------------------------build_response_grammar--------------------------------')
            return "\n".join([root_rule, other_agent_tool_call_rule, tool_rules])
    
    def prepare_prompt(
        self, input_instruction: str, world_graph: "WorldGraph", **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare the prompt for the LLM.

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
        if "{rag_examples}" in self.prompt:
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
        if "{tool_descriptions}" in self.prompt:
            params["tool_descriptions"] = self.agents[0].tool_descriptions
        if "{your_tool_descriptions}" in self.prompt:
            params["your_tool_descriptions"] = self.agents[0].tool_descriptions
        if "{other_agent_tool_descriptions}" in self.prompt:
            params["other_agent_tool_descriptions"] = self.agents[0].other_agents_tool_descriptions
        if "{agent_descriptions}" in self.prompt:
            params["agent_descriptions"] = self.agent_descriptions
        if "{tool_list}" in self.prompt:
            params["tool_list"] = self.tool_list
        if "{system_tag}" in self.prompt:
            params["system_tag"] = self.planner_config.llm.system_tag
        if "{user_tag}" in self.prompt:
            params["user_tag"] = self.planner_config.llm.user_tag
        if "{assistant_tag}" in self.prompt:
            params["assistant_tag"] = self.planner_config.llm.assistant_tag
        if "{eot_tag}" in self.prompt:
            params["eot_tag"] = self.planner_config.llm.eot_tag
        if "{agent_role_description}" in self.prompt:
            # only support agent role description when planning for a single agent
            assert len(self.agents) == 1
            if str(self.agents[0].uid) == "1":
                agent_role_description = '\nYou are playing the role of the task giver. This means if the instruction says something like "You should move the object and I will wash it", then the other agent should be moving the object, and you should washing the it.\n'
            else:
                agent_role_description = '\nYou are playing the role of the task receiver. This means if the instruction says something like "You should move the object and I will wash it", then you should move the object and the other agent should wash it.\n'
            params["agent_role_description"] = agent_role_description
        if "{world_description}" in self.prompt:
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
            output_prompt = self.prompt.format(**params)
        # print('--------------------------------prepare_prompt--------------------------------')
        # print(output_prompt)
        # print('--------------------------------prepare_prompt--------------------------------')
        return output_prompt, params
    
    def format_response(
        self, response: str, end_expression: Union[str, List[str]]
    ) -> str:
        """
        Format the LLM response by trimming it up to the last appearance of end_expression.

        :param response: The LLM response to format.
        :param end_expression: The end expression(s) to look for.
        :return: The formatted response.
        """
        response = response.rstrip("\n")
        if type(end_expression) == str:
            index = response.rfind(end_expression)
            target_end_expression = end_expression
        else:
            index = -1
            target_end_expression = ""
            for _end_expression in end_expression:
                _index = response.rfind(_end_expression)
                if _index > index:
                    index = _index
                    target_end_expression = _end_expression
        return (
            response[: index + len(target_end_expression)] if index != -1 else response
        )
    
    def replan(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ):
        """
        Replan a high level action using the LLM/VLM
        """
        # Generate response
        if self.planner_config.get("constrained_generation", False):
            llm_response = self.llm.generate(
                self.curr_prompt,
                self.stopword,
                generation_args={
                    "grammar_definition": self.build_response_grammar(
                        world_graph[self._agents[0].uid]
                    )
                },
            )
        else:
            llm_response = self.llm.generate(self.curr_prompt, self.stopword)

        # Format the response
        # This removes extra text followed by the last occurence of the end expression when needed.
        llm_response = self.format_response(llm_response, self.end_expression)
        
        info = {"llm_response": llm_response}

        return info
    
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
                            "{user_tag}Result of your action: {result}\nThe other agent's actions since the previous observation: {other_agents_actions}\nYour observations:\nObjects: {objects}{eot_tag}"
                        )
                    else:    
                        response_format = (
                            "{user_tag}Result of your action: {result}\nAll of the other agent's actions until now: {other_agents_actions}\nYour observations:\nObjects: {objects}{eot_tag}"
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

            if self.planner_config.get('include_oaa_pred_in_prompt', True):
                llm_response_prompt_addition = llm_response
            else:
                lines = llm_response.split('\n')
                if len(lines) >= 2:
                    llm_response_prompt_addition = "\n".join(lines[:-2] + [lines[-1]])
                else:
                    llm_response_prompt_addition = llm_response

            # Update prompt with the first response
            print_str += f"""----------agent {(self.agents[0]).uid}----------\n{llm_response_prompt_addition}\n{self.stopword}\n----------agent {(self.agents[0]).uid}----------\n"""
            prompt_addition = (
                f"""{llm_response_prompt_addition}\n{self.stopword}{self.planner_config.llm.eot_tag}"""
            )
            self.curr_prompt += prompt_addition
            self.trace += prompt_addition

            # Check if the planner should stop
            # Stop if the replanning count exceed a certain threshold
            # or end expression is found in llm response's last line,
            # which is expected to be the action call
            # This is helpful to break infinite planning loop.
            self.is_done = (self.check_if_agent_done(llm_response.split('\n')[-1])) or (
                self.replanning_count == self.planner_config.replanning_threshold
            )
            # Increment the llm call counter on every replan
            # doesn't get incremented before comparison as first "replan" is technically
            # the first required plan
            self.replanning_count += 1
            
            self.is_reasoning_about_other_agent = self.check_if_reasoning_about_other_agent(llm_response)
            if self.is_reasoning_about_other_agent:
                self.reasoning_about_other_agent_count += 1
            
            other_agents_predicted_high_level_actions = self.oaactions_parser(llm_response, self.agents[0].uid)
            if other_agents_predicted_high_level_actions[self.agents[0].uid][0] == "Done":
                other_agents_predicted_high_level_actions[self.agents[0].uid] = ("Done", None, None)
            self.last_other_agents_predicted_high_level_actions = other_agents_predicted_high_level_actions
            
            # print(f'\n\n llm_response:\n{llm_response}\n-------------\n\n')
            # print(f'\n\nParsed other agents predicted high level actions:\n{other_agents_predicted_high_level_actions}\n-------------\n\n')
            
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
                    'other_agents_predicted_high_level_actions': other_agents_predicted_high_level_actions,
                    'agent_states': {agent.uid: 'Standing' for agent in self.agents},
                    'is_reasoning_about_other_agent': {agent.uid: self.is_reasoning_about_other_agent for agent in self.agents},
                    'reasoning_about_other_agent_count': {agent.uid: self.reasoning_about_other_agent_count for agent in self.agents},
                }
                return {}, planner_info, self.is_done

            # Parse high level action directives from llm response
            high_level_actions = self.actions_parser(
                self.agents, llm_response, self.params
            )
            
            print(f"\n\n[DEBUG] Predicted other agent's action: {other_agents_predicted_high_level_actions}")
            print(f"[DEBUG] Now Executing: {high_level_actions}\n\n")

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
        planner_info["other_agents_predicted_high_level_actions"] = self.last_other_agents_predicted_high_level_actions
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

        return (llm_response is not None) and (("agent b" in llm_response.split('\n')[0].lower()) or ("other agent" in llm_response.split('\n')[0].lower()))
