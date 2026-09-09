#!/bin/sh
# The interpreter the harness runs under.
#
# NLE ships a compiled NetHack 3.6.6 and pins gymnasium; neither belongs in the
# parent project's dependency set, which resolves gym, torch and the rest of bai's
# ML stack against each other. So the harness gets its own interpreter, here, and
# run.py is launched with it: act.py inherits it through the workspace shim, which
# is how the daemon ends up holding a NetHack process.
#
# The version is pinned and the pin is checked rather than trusted: every number in
# notes/nle-harness-plan.md was measured against 1.3.0.
set -e
cd "$(dirname "$0")/.."
uv venv --python 3.12 .env-venv
uv pip install --python .env-venv/bin/python \
    "nle==1.3.0" "pydantic>=2" python-dotenv pytest pillow numpy
.env-venv/bin/python - <<'WARM'
import gymnasium as gym
import nle  # noqa: F401  registers the environments
from nle import nethack

env = gym.make("NetHackChallenge-v0")
obs, _ = env.reset(seed=0)
print(f"nle {nle.__version__}: {len(env.unwrapped.actions)} actions, "
      f"obs {sorted(obs)}")
print(f"tty {obs['tty_chars'].shape}, glyphs {obs['glyphs'].shape}, "
      f"{nethack.NLE_BL_SCORE=}")
env.close()
WARM
echo "harness interpreter ready: $(pwd)/.env-venv/bin/python"
