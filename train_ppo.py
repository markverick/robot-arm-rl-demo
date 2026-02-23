import argparse
from importlib import import_module


SCENARIO_MODULES = {
    "circle": "scenarios.circle.train_ppo",
    "figure8": "scenarios.figure8.train_ppo",
    "star5": "scenarios.star5.train_ppo",
    "triangle": "scenarios.triangle.train_ppo",
}


def select_scenario_interactive() -> str:
    names = sorted(SCENARIO_MODULES.keys())
    print("Select scenario:")
    for idx, name in enumerate(names, start=1):
        print(f"  {idx}) {name}")

    default_name = "figure8" if "figure8" in names else names[0]
    prompt = f"Enter number (default: {default_name}): "

    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default_name
        if raw.isdigit():
            pick = int(raw)
            if 1 <= pick <= len(names):
                return names[pick - 1]
        if raw in SCENARIO_MODULES:
            return raw
        print("Invalid choice. Enter a number from the list or a scenario name.")


def main():
    parser = argparse.ArgumentParser(description="Train PPO on a selected scenario")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_MODULES.keys()),
        default=None,
        help="Scenario name to train (if omitted, an interactive selector is shown)",
    )
    args = parser.parse_args()

    scenario = args.scenario or select_scenario_interactive()
    module = import_module(SCENARIO_MODULES[scenario])
    module.main()


if __name__ == "__main__":
    main()
