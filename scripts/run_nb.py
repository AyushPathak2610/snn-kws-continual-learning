"""Run every code cell of a notebook as one script. --real switches SMOKE_TEST off."""
import json, sys
from pathlib import Path

nb_path = Path(sys.argv[1])
src = "\n\n".join("".join(c["source"]) for c in json.loads(nb_path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code")
if "--real" in sys.argv:
    assert src.count("SMOKE_TEST = True ") == 1
    src = src.replace("SMOKE_TEST = True ", "SMOKE_TEST = False ")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
exec(compile(src, str(nb_path), "exec"), {"__name__": "__main__"})
