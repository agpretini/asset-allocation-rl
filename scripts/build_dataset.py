"""Build data/processed/monthly_dataset.parquet from raw CSV inputs."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Write the processed monthly dataset."""
    from data.dataset import write_monthly_dataset

    output_path = write_monthly_dataset()
    print(output_path)


if __name__ == "__main__":
    main()
