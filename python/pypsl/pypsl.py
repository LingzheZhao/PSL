from typing import Dict
from pypsl_lib import PlaneSweeper

class PSL:
    def __init__(self, configs: Dict) -> None:
        self.sweeper = PlaneSweeper(configs)
