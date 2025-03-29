from pathlib import Path
from src.cli.experiment_arguments import parse_experiment_arguments
from src.config.experiment_config import ExperimentConfig
from src.experiment.mbpp_experiment import MBPPExperiment


def main():
    """Main entry point with argument parsing."""
    args = parse_experiment_arguments()

    # Create config from command line args
    config = ExperimentConfig.from_args(args)

    # Initialize and run experiment
    experiment = MBPPExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
