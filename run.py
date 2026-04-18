"""
Entry point for the CMMS Monitoring System.
Prints a console monitoring report, then starts the FastAPI server on port 8000.

Usage:
    python run.py
"""

import sys
from pathlib import Path
from datetime import date

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from core.monitor import run_monitoring_report, print_monitoring_report

DATA_PATH = Path(__file__).parent / "data" / "sample_cmms_data.json"
REFERENCE_DATE = date(2024, 2, 1)


def main():
    # 1. Print console monitoring report
    print("\nGenerating CMMS monitoring report...")
    report = run_monitoring_report(str(DATA_PATH), reference_date=REFERENCE_DATE)
    print_monitoring_report(report)

    # 2. Start FastAPI server
    import uvicorn

    print("Starting CMMS Monitoring API on http://localhost:8000")
    print("Dashboard: http://localhost:8000/")
    print("API docs:  http://localhost:8000/docs")
    print("Press Ctrl+C to stop.\n")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
