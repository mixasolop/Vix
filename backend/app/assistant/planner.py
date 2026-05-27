from app.schemas.plans import Plan, PlanStep, PlanStepStatus


class Planner:
    def create_plan(self, user_message: str) -> Plan:
        goal = self._goal_from_message(user_message)
        return Plan(
            goal=goal,
            steps=[
                PlanStep(number=1, title="Understand request", status=PlanStepStatus.completed),
                PlanStep(number=2, title="Select tool", status=PlanStepStatus.completed),
                PlanStep(number=3, title="Execute", status=PlanStepStatus.completed),
            ],
        )

    @staticmethod
    def _goal_from_message(user_message: str) -> str:
        normalized = user_message.strip().lower()
        if "notepad" in normalized:
            return "Open Notepad"
        if normalized:
            return f"Handle request: {user_message.strip()}"
        return "Handle empty request"
