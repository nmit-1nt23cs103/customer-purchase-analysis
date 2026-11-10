"""Run the whole project end to end:  python -m src.run_pipeline [--download] [--k 4]

preprocess -> EDA charts -> segmentation -> SQL analysis -> classification -> results summary
The summary is written to reports/results_summary.md and injected into README.md between the
RESULTS markers, so the README always shows numbers from your latest run.
"""
from __future__ import annotations

import argparse
import re

from . import classification, config as C, download_data, eda, preprocess, segmentation, sql_analysis
from .utils import step

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def update_readme(summary_for_readme: str) -> bool:
    readme = C.ROOT / "README.md"
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    block = f"{START}\n\n{summary_for_readme.strip()}\n\n{END}"
    readme.write_text(re.sub(f"{re.escape(START)}.*?{re.escape(END)}", lambda _: block, text, flags=re.S),
                      encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer purchase analysis pipeline")
    parser.add_argument("--download", action="store_true", help="download the dataset from Kaggle first")
    parser.add_argument("--k", type=int, default=None, help="force the number of customer segments")
    args = parser.parse_args()

    if args.download:
        with step("Download dataset"):
            download_data.main()
    with step("1/5 Clean data + load into SQLite"):
        prep = preprocess.run()
    with step("2/5 Exploratory charts"):
        eda.run()
    with step("3/5 Segmentation (K-Means)"):
        seg = segmentation.run(args.k)
    with step("4/5 SQL analysis"):
        sql = sql_analysis.run()
    with step("5/5 Classification"):
        clf = classification.run()

    sections = [prep["summary_md"], seg["summary_md"], sql["summary_md"], clf["summary_md"]]
    body = "\n".join(sections)
    eda_md = (
        "### Exploratory analysis\n\n"
        "![Distributions]({FIG}/eda_distributions.png)\n\n"
        "![Spend by category]({FIG}/eda_spend_by_category.png)\n\n"
        "![Spend by signup quarter]({FIG}/eda_spend_by_signup_quarter.png)\n\n"
        "![Correlation]({FIG}/eda_correlation.png)\n"
    )
    body = "\n".join([sections[0], eda_md, *sections[1:]])
    (C.REPORT_DIR / "results_summary.md").write_text(
        "# Results\n\n" + body.replace("{FIG}", "figures").replace("](sql/analysis.sql)", "](../sql/analysis.sql)"),
        encoding="utf-8")
    updated = update_readme(body.replace("{FIG}", "reports/figures"))
    print("\nDone. Results: reports/results_summary.md" + (" (README.md updated)" if updated else ""))


if __name__ == "__main__":
    main()
