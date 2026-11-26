"""Start `ollama serve` in the background (if not already up) and pull the given models.

Usage: python scripts/start_ollama.py qwen2.5:3b qwen2.5:7b
The server keeps running after this script exits (it is started in its own session).
"""

import subprocess
import sys
import time
import urllib.request

URL = "http://127.0.0.1:11434"


def up() -> bool:
    try:
        urllib.request.urlopen(URL, timeout=2)
        return True
    except Exception:
        return False


if not up():
    subprocess.Popen(["ollama", "serve"], stdout=open("/content/ollama.log", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        if up():
            break
        time.sleep(1)
    else:
        sys.exit("Ollama did not start; see /content/ollama.log")
print("Ollama is up")
for model in sys.argv[1:]:
    subprocess.run(["ollama", "pull", model], check=True, stdout=subprocess.DEVNULL)
    print("pulled", model)
