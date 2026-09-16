class AgentRegistry:
    """
    Registry of specialist agents.

    Each agent declares:

    1. Its Python implementation.
    2. Capabilities it provides.
    3. Capabilities it depends on.
    4. Information that must come from the user.
    """

    def __init__(self):
        self.agents = {}

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

        self.agents[agent_name] = {
            "function": agent_function,
            "capabilities": capabilities or [],
            "dependencies": dependencies or [],
            "required_user_inputs": required_user_inputs or []
        }

        print(
            f"[AGENT REGISTRY] Registered agent: {agent_name}"
        )

    # ========================================================
    # GET AGENT
    # ========================================================

    def get_agent(self, agent_name):

        agent = self.agents.get(agent_name)

        if agent is None:
            return None

        return agent["function"]

    # ========================================================
    # CHECK REGISTRATION
    # ========================================================

    def is_registered(self, agent_name):

        return agent_name in self.agents

    # ========================================================
    # FIND AGENT FOR CAPABILITY
    # ========================================================

    def find_agent_by_capability(self, capability):

        for agent_name, info in self.agents.items():

            if capability in info["capabilities"]:
                return agent_name

        return None

    # ========================================================
    # DEPENDENCIES
    # ========================================================

    def get_dependencies(self, agent_name):

        agent = self.agents.get(agent_name)

        if agent is None:
            return []

        return agent["dependencies"]

    # ========================================================
    # REQUIRED USER INPUTS
    # ========================================================

    def get_required_user_inputs(self, agent_name):

        agent = self.agents.get(agent_name)

        if agent is None:
            return []

        return agent["required_user_inputs"]

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def get_capabilities(self):

        result = {}

        for agent_name, info in self.agents.items():

            result[agent_name] = info["capabilities"]

        return result

    # ========================================================
    # ALL AGENTS
    # ========================================================

    def get_all_agents(self):

        return self.agents