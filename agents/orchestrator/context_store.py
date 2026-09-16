class ContextStore:
    """
    Stores conversation memory and workflow execution state.
    """

    def __init__(self):

        self.context = {}

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

            # Tells Groq what parameter the chatbot
            # is currently waiting for.
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

            if key not in self.conversation:
                continue

            if value is None:
                continue

            # Parameter-only follow-ups return [].
            # Do not erase previously stored capabilities.
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

            "request_id":
                request_id,

            "request":
                request,

            "goal":
                goal,

            "required_capabilities":
                required_capabilities,

            "required_agents":
                required_agents,

            "workflow_status": {

                agent: "pending"

                for agent in required_agents
            },

            "agent_results": {},

            "errors": [],

            "status": "processing"
        }

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