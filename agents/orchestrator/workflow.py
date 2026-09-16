class Workflow:
    """
    Stores a workflow dynamically created by the
    Orchestrator Agent.

    This class does NOT decide which agents should run.
    """

    def __init__(self):
        self.steps = []

    def add_step(self, agent_name):
        """
        Add an agent once to the execution plan.
        """

        if agent_name not in self.steps:
            self.steps.append(agent_name)

    def get_steps(self):
        """
        Return the current execution plan.
        """

        return self.steps.copy()

    def clear(self):
        self.steps = []