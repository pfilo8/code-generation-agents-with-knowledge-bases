from src.cli.experiment_arguments import parse_experiment_arguments
from src.config.experiment_config import ExperimentConfig
from src.experiment.factory import ExperimentFactory


def main():
    """Main entry point with argument parsing."""
    args = parse_experiment_arguments()

    # Create config from command line args
    config = ExperimentConfig.from_args(args)

    # Create and run appropriate experiment using factory
    experiment = ExperimentFactory.create_experiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
