#!/bin/sh
# The interpreter the agent's workspace gets a `python` shim to.
#
# The observation is a 24x80 screen of text and, when the run gives it out, a
# tile-rendered PNG -- so an interpreter that cannot open a PNG or hold an array
# makes half of what the run writes unreadable, and /usr/bin/python3 here has
# neither numpy nor Pillow. The harness's own .env-venv is the wrong one to hand
# over: `from nle import nethack` from it reaches the glyph table, the object
# names and a second copy of the game the session could play offline. So: numpy,
# Pillow, and nothing else.
set -e
cd "$(dirname "$0")/.."
uv venv --python 3.12 .agent-venv
uv pip install --python .agent-venv/bin/python numpy pillow
.agent-venv/bin/python -c "import numpy, PIL"
for module in nle gymnasium; do
    if .agent-venv/bin/python -c "import $module" 2>/dev/null; then
        echo "make_agent_venv: the agent's interpreter can import $module — refusing" >&2
        exit 1
    fi
done
echo "agent interpreter ready: $(pwd)/.agent-venv/bin/python"
