"""Run a command fully detached from the notebook cell, logging to a file.

Usage: python scripts/bg.py /content/gguf.log python scripts/run_gguf_eval.py
A plain `cmd &` in a Colab cell keeps the cell's terminal open, so the cell waits for
the job; here the child gets its own session and no handle on the terminal.
"""

import subprocess
import sys

log, cmd = sys.argv[1], sys.argv[2:]
with open(log, "a") as out:
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True)
print(f"started pid {p.pid}: {' '.join(cmd)} (log: {log})")
