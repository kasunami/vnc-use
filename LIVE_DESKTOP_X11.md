# Live Ubuntu Desktop (X11) Setup

This project is designed to drive a VNC-accessible desktop. If you want it to control your **live** Ubuntu desktop
session under **X11**, the most direct approach is to expose your current `:0` display via `x11vnc` and connect
`vnc-use` to it.

## Prereqs

- Confirm you are on X11: `echo $XDG_SESSION_TYPE` → `x11`
- `x11vnc` installed (requires sudo on Ubuntu):
  - `sudo apt-get update && sudo apt-get install -y x11vnc`

## Start a localhost-only VNC server for your live session

1) Create a VNC password file (no sudo required):

- `mkdir -p ~/.vnc`
- `x11vnc -storepasswd` (writes `~/.vnc/passwd`)

2) Run `x11vnc` bound to localhost only:

- `x11vnc -display :0 -auth guess -rfbauth ~/.vnc/passwd -rfbport 5901 -localhost -forever -shared`

Notes:
- `-localhost` is important: it prevents LAN access to your live desktop.
- Keep this terminal open while the agent runs.

## Point vnc-use at your live desktop

1) Configure credentials (recommended: prompts for the password so it doesn’t land in your shell history):

- `uv run vnc-use-credentials set live-desktop --server localhost::5901`

2) Use a local/OpenAI-compatible vision model via Mesh Router:

- `export MODEL_PROVIDER=openai_compatible`
- `export OPENAI_BASE_URL=http://10.0.1.47:4010/v1`
- `export OPENAI_MODEL=<your_vision_model_name>`

If your router requires auth:
- `export OPENAI_API_KEY=<token>`

3) Run a task:

- `uv run vnc-use run --task "Open Firefox, go to …, fill the form, submit"`

Or via the MCP server:

- `uv run vnc-use-mcp` (defaults to `http://127.0.0.1:8000/mcp`)

## Safety levers worth keeping on

Even in “dangerous mode”, enabling HITL (`hitl_mode=True`) gives you a last-chance stop on obviously destructive
actions (deletes, irreversible submissions, etc.).

