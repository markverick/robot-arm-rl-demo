from pathlib import Path


def _list_model_candidates(prefix: str, model_dir: Path) -> list[Path]:
    return sorted(
        model_dir.glob(f"{prefix}*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def select_model_path(
    prefix: str,
    model_path: str | None = None,
    model_rank: int | None = None,
    interactive: bool = False,
    model_dir: str = "models",
) -> str:
    if model_path is not None:
        return model_path

    model_dir_path = Path(model_dir)
    candidates = _list_model_candidates(prefix=prefix, model_dir=model_dir_path)

    if not candidates:
        raise FileNotFoundError(
            f"No model found matching: {model_dir_path / (prefix + '*.zip')}"
        )

    if interactive:
        print("Select model:")
        for idx, candidate in enumerate(candidates, start=1):
            print(f"  {idx}) {candidate}")

        while True:
            raw = input("Enter number: ").strip()
            if raw.isdigit():
                pick = int(raw)
                if 1 <= pick <= len(candidates):
                    return str(candidates[pick - 1])
            print("Invalid choice. Enter a valid number from the list.")

    if model_rank is None:
        model_rank = 1

    if model_rank < 1:
        raise ValueError("model_rank must be >= 1")

    if model_rank > len(candidates):
        raise FileNotFoundError(
            f"Requested model_rank={model_rank}, but only {len(candidates)} model(s) found for prefix '{prefix}'"
        )

    return str(candidates[model_rank - 1])


def make_timestamped_model_stem(prefix: str, model_dir: str = "models") -> Path:
    from datetime import datetime

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return model_dir_path / f"{prefix}_{timestamp}"
