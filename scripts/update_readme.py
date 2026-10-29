"""Copy results/SUMMARY.md into README.md between the RESULTS markers.

    python scripts/update_readme.py

Keeps every number in the README traceable to a run in results/.
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
summary = (root / "results" / "SUMMARY.md").read_text()
summary = summary.replace("](pr_", "](results/pr_")
readme = (root / "README.md").read_text()
new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
             "<!-- RESULTS:START -->\n" + summary + "\n<!-- RESULTS:END -->", readme, flags=re.S)
(root / "README.md").write_text(new)
print("README.md updated")
