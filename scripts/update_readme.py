"""Copy results/SUMMARY.md and results/SHIFT.md into README.md between their markers.

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
shift = root / "results" / "SHIFT.md"
if shift.exists():
    text = shift.read_text().replace("](shift_", "](results/shift_")
    text = re.sub(r"^# .*\n", "", text, count=1)          # the README already has the heading
    new = re.sub(r"<!-- SHIFT:START -->.*<!-- SHIFT:END -->",
                 lambda m: "<!-- SHIFT:START -->\n" + text + "\n<!-- SHIFT:END -->", new, flags=re.S)
(root / "README.md").write_text(new)
print("README.md updated")
