"""Entry point for automatic SAP2000 RC building model generation."""

from __future__ import annotations

from config import CONFIG
from model_generator import generate_models


def main() -> None:
    """Run the configured model generation workflow."""
    rows = generate_models(CONFIG)
    success_count = sum(row.uretim_durumu == "success" for row in rows)
    failed_count = len(rows) - success_count
    print(f"Generated {success_count} SAP2000 model(s); {failed_count} iteration(s) failed.")
    print(f"Metadata CSV: {CONFIG.output_dir / 'models_metadata.csv'}")


if __name__ == "__main__":
    main()
