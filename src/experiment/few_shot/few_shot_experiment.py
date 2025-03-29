from typing import Optional, Dict, List
from src.experiment.base_experiment import BaseExperiment

class FewShotExperiment(BaseExperiment):
    """Implements few-shot approach for MBPP experiment."""
    
    def process_task(self, task_id: int, data: List[Dict]) -> Optional[Dict]:
        """Process a single MBPP task using few-shot approach."""
        # Implementation will go here
        pass


class FewShotWithRepetitionExperiment(FewShotExperiment):
    """Implements few-shot with repetition approach for MBPP experiment."""
    # Implementation will go here
    pass 