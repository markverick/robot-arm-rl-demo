import argparse
from importlib import import_module


SCENARIO_MODULES = {
    "figure8": "scenarios.figure8.train_ppo",
    "star5": "scenarios.star5.train_ppo",
}


def main():
    parser = argparse.ArgumentParser(description="Train PPO on a selected scenario")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_MODULES.keys()),
        default="figure8",
        help="Scenario name to train (default: figure8)",
    )
    args = parser.parse_args()

    module = import_module(SCENARIO_MODULES[args.scenario])
    module.main()


if __name__ == "__main__":
    main()
