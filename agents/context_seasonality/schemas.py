from dataclasses import dataclass, asdict
@dataclass
class AgentRunSummary:
    rows: int
    start_date: str
    end_date: str
    restaurants: int
    menu_items: int
    missing_quantity: int
    output_files: list[str]
    def to_dict(self): return asdict(self)
