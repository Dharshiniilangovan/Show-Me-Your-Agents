import os
import json
import uuid

from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from .agent_registry import AgentRegistry
from .context_store import ContextStore
from .workflow import Workflow


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(
    override=True
)


# ============================================================
# PROMPT FILE PATH
# ============================================================

PROMPT_PATH = (
    Path(__file__).parent
    / "prompt.txt"
)


# ============================================================
# ORCHESTRATOR AGENT
# ============================================================

class OrchestratorAgent:
    """
    Central Orchestrator Agent.

    Responsibilities:

    1. Understand the user's goal.
    2. Extract parameters.
    3. Determine required capabilities.
    4. Match capabilities to agents.
    5. Resolve agent dependencies.
    6. Determine required user inputs.
    7. Ask for missing information.
    8. Dynamically construct the workflow.
    9. Execute agents.
    10. Aggregate results.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.registry = (
            AgentRegistry()
        )

        self.context_store = (
            ContextStore()
        )

        self.client = Groq(
            api_key=os.getenv(
                "GROQ_API_KEY"
            )
        )

    # ========================================================
    # REGISTER AGENT
    # ========================================================

    def register_agent(
        self,
        agent_name,
        agent_function,
        capabilities=None,
        dependencies=None,
        required_user_inputs=None
    ):

        self.registry.register_agent(
            agent_name=agent_name,
            agent_function=agent_function,
            capabilities=capabilities,
            dependencies=dependencies,
            required_user_inputs=required_user_inputs
        )

    # ========================================================
    # LOAD SYSTEM PROMPT
    # ========================================================

    def load_prompt(self):

        with open(
            PROMPT_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    # ========================================================
    # UNDERSTAND USER QUERY
    # ========================================================

    def parse_query(
        self,
        query,
        conversation=None
    ):

        if conversation is None:

            conversation = {}

        # ----------------------------------------------------
        # GET CURRENTLY REGISTERED CAPABILITIES
        # ----------------------------------------------------

        available_capabilities = (
            self.registry
            .get_capabilities()
        )

        # ----------------------------------------------------
        # LOAD PROMPT FROM prompt.txt
        # ----------------------------------------------------

        prompt_template = (
            self.load_prompt()
        )

        # ----------------------------------------------------
        # INSERT CURRENT CAPABILITIES INTO PROMPT
        # ----------------------------------------------------

        system_prompt = (
            prompt_template.replace(
                "__AVAILABLE_CAPABILITIES__",
                json.dumps(
                    available_capabilities,
                    indent=2
                )
            )
        )

        # ----------------------------------------------------
        # CALL GROQ
        # ----------------------------------------------------

        try:

            response = (
                self.client
                .chat
                .completions
                .create(

                    model=
                        "openai/gpt-oss-20b",

                    messages=[
                        {
                            "role":
                                "system",

                            "content":
                                system_prompt
                        },

                        {
                            "role":
                                "system",

                            "content": (
                                "Existing conversation state:\n"
                                + json.dumps(
                                    conversation,
                                    indent=2
                                )
                            )
                        },

                        {
                            "role":
                                "user",

                            "content":
                                query
                        }
                    ],

                    response_format={
                        "type":
                            "json_object"
                    },

                    temperature=0
                )
            )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            parsed = (
                json.loads(
                    content
                )
            )

            # =================================================
            # NORMALIZE RESTAURANT ID
            # =================================================
            #
            # Groq determines that the value represents
            # a restaurant.
            #
            # Python only normalizes it to the dataset format.
            #
            # Examples:
            #
            # 1   -> R01
            # 2   -> R02
            # 10  -> R10
            # R1  -> R01
            # R01 -> R01
            # =================================================

            restaurant_id = (
                parsed.get(
                    "restaurant_id"
                )
            )

            if restaurant_id is not None:

                restaurant_id = (
                    str(
                        restaurant_id
                    )
                    .strip()
                )

                # ---------------------------------------------
                # Numeric restaurant ID
                # ---------------------------------------------

                if restaurant_id.isdigit():

                    parsed[
                        "restaurant_id"
                    ] = (
                        f"R{int(restaurant_id):02d}"
                    )

                # ---------------------------------------------
                # R-prefixed restaurant ID
                # ---------------------------------------------

                elif (
                    restaurant_id
                    .upper()
                    .startswith("R")

                    and

                    restaurant_id[1:]
                    .isdigit()
                ):

                    number = int(
                        restaurant_id[1:]
                    )

                    parsed[
                        "restaurant_id"
                    ] = (
                        f"R{number:02d}"
                    )

            # =================================================
            # NORMALIZE MENU ITEM ID
            # =================================================
            #
            # Examples:
            #
            # 1   -> M01
            # 2   -> M02
            # 12  -> M12
            # M1  -> M01
            # M12 -> M12
            # =================================================

            menu_item_id = (
                parsed.get(
                    "menu_item_id"
                )
            )

            if menu_item_id is not None:

                menu_item_id = (
                    str(
                        menu_item_id
                    )
                    .strip()
                )

                # ---------------------------------------------
                # Numeric menu item ID
                # ---------------------------------------------

                if menu_item_id.isdigit():

                    parsed[
                        "menu_item_id"
                    ] = (
                        f"M{int(menu_item_id):02d}"
                    )

                # ---------------------------------------------
                # M-prefixed menu item ID
                # ---------------------------------------------

                elif (
                    menu_item_id
                    .upper()
                    .startswith("M")

                    and

                    menu_item_id[1:]
                    .isdigit()
                ):

                    number = int(
                        menu_item_id[1:]
                    )

                    parsed[
                        "menu_item_id"
                    ] = (
                        f"M{number:02d}"
                    )

            # =================================================
            # RETURN PARSED QUERY
            # =================================================

            return {

                "original_query":
                    query,

                "goal":
                    parsed.get(
                        "goal"
                    ),

                "required_capabilities":
                    parsed.get(
                        "required_capabilities",
                        []
                    ),

                "restaurant_id":
                    parsed.get(
                        "restaurant_id"
                    ),

                "restaurant_name":
                    parsed.get(
                        "restaurant_name"
                    ),
                "restaurant_scope":
                      parsed.get(
                        "restaurant_scope"
                    ),
                "holiday_name":
                    parsed.get(
                        "holiday_name"
                    ),

                "special_event_name":
                    parsed.get(
                        "special_event_name"
                    ),

                "menu_item_id":
                    parsed.get(
                        "menu_item_id"
                    ),

                "menu_item_name":
                    parsed.get(
                        "menu_item_name"
                    ),

                "time_period":
                    parsed.get(
                        "time_period"
                    ),

                "forecast_horizon":
                    parsed.get(
                        "forecast_horizon"
                    ),

                "promotion_percentage":
                    parsed.get(
                        "promotion_percentage"
                    )
            }

        # ----------------------------------------------------
        # QUERY PARSING ERROR
        # ----------------------------------------------------

        except Exception as e:

            print(
                "[ORCHESTRATOR] "
                f"Query parsing failed: {e}"
            )

            return {

                "original_query":
                    query,

                "goal":
                    None,

                "required_capabilities":
                    [],

                "restaurant_id":
                    None,

                "restaurant_name":
                    None,

                "restaurant_scope":
                    None,

                "menu_item_id":
                    None,

                "menu_item_name":
                    None,

                "time_period":
                    None,
                "holiday_name":
                    None,

                "special_event_name":
                    None,

                "forecast_horizon":
                    None,

                "promotion_percentage":
                    None
            }

    # ========================================================
    # BUILD WORKFLOW
    # ========================================================

    def build_workflow(
        self,
        required_capabilities
    ):
        """
        Dynamically construct the workflow.

        The Orchestrator performs this step.
        """

        workflow = (
            Workflow()
        )

        visited_agents = set()

        resolving_agents = set()

        # ----------------------------------------------------
        # RESOLVE ONE CAPABILITY
        # ----------------------------------------------------

        def resolve_capability(
            capability
        ):

            agent_name = (
                self.registry
                .find_agent_by_capability(
                    capability
                )
            )

            # -----------------------------------------------
            # CAPABILITY NOT AVAILABLE
            # -----------------------------------------------

            if agent_name is None:

                raise ValueError(
                    "No registered agent can provide "
                    f"the capability '{capability}'."
                )

            # -----------------------------------------------
            # ALREADY RESOLVED
            # -----------------------------------------------

            if agent_name in visited_agents:

                return

            # -----------------------------------------------
            # CIRCULAR DEPENDENCY
            # -----------------------------------------------

            if agent_name in resolving_agents:

                raise ValueError(
                    "Circular agent dependency "
                    f"detected at '{agent_name}'."
                )

            resolving_agents.add(
                agent_name
            )

            # -----------------------------------------------
            # GET DEPENDENCIES
            # -----------------------------------------------

            dependencies = (
                self.registry
                .get_dependencies(
                    agent_name
                )
            )

            # -----------------------------------------------
            # DEPENDENCIES EXECUTE FIRST
            # -----------------------------------------------

            for dependency in (
                dependencies
            ):

                resolve_capability(
                    dependency
                )

            resolving_agents.remove(
                agent_name
            )

            # -----------------------------------------------
            # ADD AGENT TO WORKFLOW
            # -----------------------------------------------

            workflow.add_step(
                agent_name
            )

            visited_agents.add(
                agent_name
            )

        # ----------------------------------------------------
        # RESOLVE ALL FINAL CAPABILITIES
        # ----------------------------------------------------

        for capability in (
            required_capabilities
        ):

            resolve_capability(
                capability
            )

        return (
            workflow.get_steps()
        )

    # ========================================================
    # GET USER INPUT REQUIREMENTS
    # ========================================================

    def get_required_user_inputs(
        self,
        workflow
    ):
        """
        Collect all user inputs required by the agents
        participating in the workflow.
        """

        requirements = []

        for agent_name in (
            workflow
        ):

            agent_requirements = (
                self.registry
                .get_required_user_inputs(
                    agent_name
                )
            )

            for requirement in (
                agent_requirements
            ):

                if (
                    requirement
                    not in requirements
                ):

                    requirements.append(
                        requirement
                    )

        return requirements

    # ========================================================
    # FIND MISSING USER INPUTS
    # ========================================================

    def find_missing_user_inputs(
        self,
        required_inputs,
        conversation
    ):

        missing = []

        for requirement in (
            required_inputs
        ):

            # ------------------------------------------------
            # RESTAURANT
            # ------------------------------------------------

            if requirement == "restaurant":

              available = (

                conversation.get(
                    "restaurant_scope"
                ) == "all"

                or

                conversation.get(
                    "restaurant_id"
                )
                is not None

                or

                conversation.get(
                    "restaurant_name"
                )
                is not None
            )

              if not available:

                    missing.append(
                        "restaurant"
                    )

            # ------------------------------------------------
            # MENU ITEM
            # ------------------------------------------------

            elif requirement == "menu_item":

                available = (

                    conversation.get(
                        "menu_item_id"
                    )
                    is not None

                    or

                    conversation.get(
                        "menu_item_name"
                    )
                    is not None
                )

                if not available:

                    missing.append(
                        "menu_item"
                    )

            # ------------------------------------------------
            # NORMAL PARAMETER
            # ------------------------------------------------

            else:

                if (
                    conversation.get(
                        requirement
                    )
                    is None
                ):

                    missing.append(
                        requirement
                    )

        return missing

    # ========================================================
    # CREATE REQUEST
    # ========================================================

    def create_request(
        self,
        data
    ):

        return {

            "request_id":
                str(
                    uuid.uuid4()
                ),

            "timestamp":
                datetime.now()
                .isoformat(),

            "query":
                data.get(
                    "original_query"
                ),

            "goal":
                data.get(
                    "goal"
                ),

            "restaurant_id":
                data.get(
                    "restaurant_id"
                ),

            "restaurant_name":
                data.get(
                    "restaurant_name"
                ),
            "restaurant_scope":
                data.get(
                    "restaurant_scope"
               ),    
            "holiday_name":
                data.get(
                    "holiday_name"
                ),

            "special_event_name":
                data.get(
                    "special_event_name"
                ),   

            "menu_item_id":
                data.get(
                    "menu_item_id"
                ),

            "menu_item_name":
                data.get(
                    "menu_item_name"
                ),

            "time_period":
                data.get(
                    "time_period"
                ),

            "forecast_horizon":
                data.get(
                    "forecast_horizon"
                ),

            "promotion_percentage":
                data.get(
                    "promotion_percentage"
                )
        }

    # ========================================================
    # RUN ONE AGENT
    # ========================================================

    def run_agent(
        self,
        agent_name,
        state
    ):

        print(
            "\n[ORCHESTRATOR] "
            f"Running: {agent_name}"
        )

        # ----------------------------------------------------
        # CHECK REGISTRATION
        # ----------------------------------------------------

        if not (
            self.registry
            .is_registered(
                agent_name
            )
        ):

            error = (
                f"Agent '{agent_name}' "
                "is not registered."
            )

            state[
                "workflow_status"
            ][agent_name] = (
                "unavailable"
            )

            state[
                "errors"
            ].append(
                error
            )

            return state

        # ----------------------------------------------------
        # EXECUTE AGENT
        # ----------------------------------------------------

        try:

            state[
                "workflow_status"
            ][agent_name] = (
                "running"
            )

            agent_function = (
                self.registry
                .get_agent(
                    agent_name
                )
            )

            result = (
                agent_function(
                    state
                )
            )

            state[
                "agent_results"
            ][agent_name] = (
                result
            )

            state[
                "workflow_status"
            ][agent_name] = (
                "completed"
            )

            print(
                "[ORCHESTRATOR] "
                f"{agent_name} completed"
            )

        except Exception as e:

            state[
                "workflow_status"
            ][agent_name] = (
                "failed"
            )

            state[
                "errors"
            ].append({

                "agent":
                    agent_name,

                "error":
                    str(e)
            })

            print(
                "[ERROR] "
                f"{agent_name}: {e}"
            )

        return state

    # ========================================================
    # EXECUTE WORKFLOW
    # ========================================================

    def execute_workflow(
        self,
        state
    ):

        for agent_name in (
            state[
                "required_agents"
            ]
        ):

            state = (
                self.run_agent(
                    agent_name,
                    state
                )
            )

        # ----------------------------------------------------
        # FINAL WORKFLOW STATUS
        # ----------------------------------------------------

        if state[
            "errors"
        ]:

            state[
                "status"
            ] = (
                "completed_with_errors"
            )

        else:

            state[
                "status"
            ] = (
                "completed"
            )

        return state

    # ========================================================
    # PROCESS REQUEST
    # ========================================================

    def process_request(
        self,
        query
    ):

        # ----------------------------------------------------
        # STEP 1
        # GET EXISTING CONVERSATION
        # ----------------------------------------------------

        existing = (
            self.context_store
            .get_conversation()
        )

        # ----------------------------------------------------
        # STEP 2
        # UNDERSTAND CURRENT MESSAGE
        # ----------------------------------------------------

        parsed = (
            self.parse_query(
                query,
                existing
            )
        )

        # ----------------------------------------------------
        # STEP 3
        # UPDATE CONVERSATION
        # ----------------------------------------------------

        conversation = (
            self.context_store
            .update_conversation(
                parsed
            )
        )

        goal = (
            conversation.get(
                "goal"
            )
        )

        capabilities = (
            conversation.get(
                "required_capabilities",
                []
            )
        )

        # ----------------------------------------------------
        # STEP 4
        # GOAL UNKNOWN
        # ----------------------------------------------------

        if (
            goal is None
            or
            not capabilities
        ):

            return {

                "status":
                    "clarification_required",

                "message": (
                    "I'm not sure what analysis "
                    "you would like me to perform. "
                    "Please describe what you want "
                    "to know."
                )
            }

        # ----------------------------------------------------
        # STEP 5
        # BUILD WORKFLOW
        # ----------------------------------------------------

        try:

            workflow = (
                self.build_workflow(
                    capabilities
                )
            )

        except ValueError as e:

            return {

                "status":
                    "unsupported_request",

                "message":
                    str(e)
            }

        # ----------------------------------------------------
        # STEP 6
        # GET REQUIRED USER INPUTS
        # ----------------------------------------------------

        required_user_inputs = (
            self.get_required_user_inputs(
                workflow
            )
        )

        # ----------------------------------------------------
        # STEP 7
        # FIND MISSING INPUTS
        # ----------------------------------------------------

        missing = (
            self.find_missing_user_inputs(
                required_user_inputs,
                conversation
            )
        )

        # ----------------------------------------------------
        # STEP 8
        # ASK FOR MISSING INFORMATION
        # ----------------------------------------------------

        if missing:

            readable = {

                "restaurant":
                    (
                        "restaurant ID "
                        "or restaurant name"
                    ),

                "menu_item":
                    (
                        "menu item ID "
                        "or menu item name"
                    ),

                "forecast_horizon":
                    "forecast period",

                "promotion_percentage":
                    "promotion percentage"
            }

            missing_names = [

                readable.get(
                    parameter,
                    parameter
                )

                for parameter
                in missing
            ]

            return {

                "status":
                    "missing_parameters",

                "missing_parameters":
                    missing,

                "message": (
                    "Please provide the "
                    + " and ".join(
                        missing_names
                    )
                    + "."
                )
            }

        # ----------------------------------------------------
        # STEP 9
        # DISPLAY ORCHESTRATION DECISION
        # ----------------------------------------------------

        print(
            "\n[ORCHESTRATOR] "
            f"Goal: {goal}"
        )

        print(
            "[ORCHESTRATOR] "
            f"Required capabilities: "
            f"{capabilities}"
        )

        print(
            "[ORCHESTRATOR] "
            f"Dynamic workflow: "
            f"{workflow}"
        )

        # ----------------------------------------------------
        # STEP 10
        # CREATE REQUEST
        # ----------------------------------------------------

        request_data = {

            "original_query":
                query,

            **conversation
        }

        request = (
            self.create_request(
                request_data
            )
        )

        # ----------------------------------------------------
        # STEP 11
        # CREATE WORKFLOW STATE
        # ----------------------------------------------------

        state = (
            self.context_store
            .create_context(

                request[
                    "request_id"
                ],

                request,

                goal,

                capabilities,

                workflow
            )
        )

        # ----------------------------------------------------
        # STEP 12
        # EXECUTE WORKFLOW
        # ----------------------------------------------------

        state = (
            self.execute_workflow(
                state
            )
        )

        # ----------------------------------------------------
        # STEP 13
        # STORE WORKFLOW RESULT
        # ----------------------------------------------------

        self.context_store.update_context(

            request[
                "request_id"
            ],

            state
        )

        # ----------------------------------------------------
        # STEP 14
        # FINAL STRUCTURED RESPONSE
        # ----------------------------------------------------

        response = {

            "request_id":
                state[
                    "request_id"
                ],

            "goal":
                state[
                    "goal"
                ],

            "required_capabilities":
                state[
                    "required_capabilities"
                ],

            "workflow":
                state[
                    "required_agents"
                ],

            "workflow_status":
                state[
                    "workflow_status"
                ],

            "results":
                state[
                    "agent_results"
                ],

            "errors":
                state[
                    "errors"
                ],

            "status":
                state[
                    "status"
                ]
        }

        # ----------------------------------------------------
        # STEP 15
        # REQUEST COMPLETED
        # ----------------------------------------------------

        self.context_store.reset_conversation()

        return response