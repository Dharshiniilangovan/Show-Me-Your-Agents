class ContextStore:
    """
    Stores:
    1. Conversation memory
    2. Workflow execution state
    3. Shared knowledge/context for communication between agents
    """

    def __init__(self):

        # Stores workflow contexts using request_id
        self.context = {}

        # Stores conversational information between user turns
        self.conversation = (
            self._empty_conversation()
        )

    # ========================================================
    # CREATE EMPTY CONVERSATION
    # ========================================================

    def _empty_conversation(self):

        return {

            "goal": None,

            "required_capabilities": [],

            "restaurant_id": None,
            "restaurant_name": None,
            "restaurant_scope": None,

            "holiday_name": None,
            "special_event_name": None,

            "menu_item_id": None,
            "menu_item_name": None,

            "time_period": None,
            "forecast_horizon": None,

            "promotion_percentage": None,

            # Tells the orchestrator what parameter
            # it is currently waiting for.
            "awaiting_parameter": None
        }

    # ========================================================
    # GET CONVERSATION
    # ========================================================

    def get_conversation(self):

        return self.conversation.copy()

    # ========================================================
    # UPDATE CONVERSATION
    # ========================================================

    def update_conversation(
        self,
        new_data
    ):

        for key, value in new_data.items():

            # Ignore fields that are not part
            # of conversation memory
            if key not in self.conversation:
                continue

            # Do not overwrite existing information
            # with None
            if value is None:
                continue

            # Parameter-only follow-up responses may
            # return an empty capability list.
            #
            # Do not erase previously identified
            # capabilities in that situation.
            if (
                key == "required_capabilities"
                and value == []
            ):
                continue

            self.conversation[key] = value

        return self.conversation.copy()

    # ========================================================
    # SET PARAMETER WE ARE WAITING FOR
    # ========================================================

    def set_awaiting_parameter(
        self,
        parameter
    ):

        self.conversation[
            "awaiting_parameter"
        ] = parameter

    # ========================================================
    # CLEAR AWAITING PARAMETER
    # ========================================================

    def clear_awaiting_parameter(self):

        self.conversation[
            "awaiting_parameter"
        ] = None

    # ========================================================
    # RESET CONVERSATION
    # ========================================================

    def reset_conversation(self):

        self.conversation = (
            self._empty_conversation()
        )

    # ========================================================
    # CREATE WORKFLOW CONTEXT
    # ========================================================

    def create_context(
        self,
        request_id,
        request,
        goal,
        required_capabilities,
        required_agents
    ):

        state = {

            # ------------------------------------------------
            # REQUEST INFORMATION
            # ------------------------------------------------

            "request_id":
                request_id,

            "request":
                request,

            "goal":
                goal,

            # ------------------------------------------------
            # ORCHESTRATOR INFORMATION
            # ------------------------------------------------

            "required_capabilities":
                required_capabilities,

            "required_agents":
                required_agents,

            # ------------------------------------------------
            # WORKFLOW STATUS
            # ------------------------------------------------

            "workflow_status": {

                agent: "pending"

                for agent in required_agents
            },

            # ------------------------------------------------
            # AGENT RESULTS
            # ------------------------------------------------
            #
            # Stores the actual results returned during
            # this workflow.
            #
            # Example:
            #
            # agent_results["data_analyst"]
            # agent_results["context_seasonality"]
            # agent_results["customer_pattern"]
            #
            # ------------------------------------------------

            "agent_results": {},

            # =================================================
            # SHARED KNOWLEDGE / CONTEXT STORE
            # =================================================
            #
            # This is the common layer used by agents to
            # discover reusable outputs produced by other
            # agents.
            #
            # Agents should not need to hard-code each
            # other's file locations.
            #
            # =================================================

            "shared_context": {

                # ---------------------------------------------
                # DATA ANALYST OUTPUTS
                # ---------------------------------------------

                "data": {

                    # Full cleaned + processed dataset
                    "unified_demand_path": None,

                    # Context-enhanced dataset generated
                    # by Context / Seasonality
                    "context_features_path": None
                },

                # ---------------------------------------------
                # CONTEXT / SEASONALITY OUTPUTS
                # ---------------------------------------------

                "context": {

                    # Restaurant/SKU historical
                    # context signals
                    "context_signals_path": None,

                    # Machine-readable Context Agent contract
                    "agent_context_path": None
                },

                # ---------------------------------------------
                # CUSTOMER PATTERN OUTPUT
                # ---------------------------------------------

                "customer_pattern": None
            },

            # ------------------------------------------------
            # WORKFLOW ERRORS
            # ------------------------------------------------

            "errors": [],

            # ------------------------------------------------
            # OVERALL WORKFLOW STATUS
            # ------------------------------------------------

            "status": "processing"
        }

        # Save the state using request_id
        self.context[
            request_id
        ] = state

        return state

    # ========================================================
    # GET WORKFLOW CONTEXT
    # ========================================================

    def get_context(
        self,
        request_id
    ):

        return self.context.get(
            request_id
        )

    # ========================================================
    # UPDATE WORKFLOW CONTEXT
    # ========================================================

    def update_context(
        self,
        request_id,
        state
    ):

        self.context[
            request_id
        ] = state

        return state