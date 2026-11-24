# Command reference

Every `mnm` command and option, generated from the CLI itself. `mnm <command> --help` prints
the same at a terminal; [the guide](GUIDE.md) explains when to use each one.


## `mnm` itself

```
mnm [OPTIONS] COMMAND [ARGS]...
```

| Argument or option | What it does |
|---|---|
| `--version` | Print the version and the checkout's commit, and exit. |
| `--install-completion` | Install completion for the current shell. |
| `--show-completion` | Show completion for the current shell, to copy it or customize the installation. |

## Contents

- **Set up and check:** [`setup`](#mnm-setup), [`doctor`](#mnm-doctor), [`status`](#mnm-status), [`machine`](#mnm-machine), [`update`](#mnm-update), [`service`](#mnm-service), [`serve`](#mnm-serve), [`open`](#mnm-open)
- **Find and add models:** [`what`](#mnm-what), [`resolve`](#mnm-resolve), [`fit`](#mnm-fit), [`add`](#mnm-add), [`starter`](#mnm-starter), [`upgrade`](#mnm-upgrade)
- **Run models:** [`ls`](#mnm-ls), [`show`](#mnm-show), [`start`](#mnm-start), [`stop`](#mnm-stop), [`try`](#mnm-try), [`why`](#mnm-why), [`set`](#mnm-set), [`replan`](#mnm-replan), [`url`](#mnm-url), [`logs`](#mnm-logs), [`pin`](#mnm-pin), [`unpin`](#mnm-unpin), [`vram`](#mnm-vram), [`rm`](#mnm-rm), [`router`](#mnm-router), [`mcp`](#mnm-mcp)
- **Sets of models:** [`sets`](#mnm-sets)
- **Make things:** [`run`](#mnm-run), [`outputs`](#mnm-outputs), [`clean`](#mnm-clean), [`pipeline`](#mnm-pipeline), [`comfyui`](#mnm-comfyui), [`lora`](#mnm-lora)
- **The studio:** [`song`](#mnm-song), [`studio`](#mnm-studio), [`voices`](#mnm-voices)
- **Measure and tune:** [`calibration`](#mnm-calibration), [`tune`](#mnm-tune), [`nightly`](#mnm-nightly), [`engines`](#mnm-engines)
- **Workspace apps:** [`apps`](#mnm-apps)
- **Keep things safe:** [`backup`](#mnm-backup), [`restore`](#mnm-restore), [`disk`](#mnm-disk), [`report`](#mnm-report), [`notify`](#mnm-notify)

## Set up and check

Install the prerequisites, check the machine, and keep mnm itself up to date.

### `mnm setup`

Set this machine up, guided at a terminal: Docker, the NVIDIA runtime (and on native
Linux the driver), where models live, the router's network and key, tokens, the
background services and a first model. Re-running is safe: each step checks first.

```
mnm setup [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--dry-run` | Show what would be installed. |
| `--pull / --no-pull` | Also pull the llama.cpp engine image (~2 GB). Default: `on`. |
| `--yes / -y` | Unattended: every question takes its default. Without a terminal it is what lets setup install anything (else it prints the checks only). |
| `--install-driver` | Native Linux: install the NVIDIA driver without asking (it needs a reboot). |
| `--plain` | The checks table only, no guided sections. |

### `mnm doctor`

Check GPU, RAM, WSL networking, Docker, storage and token.

```
mnm doctor [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--deep / --no-deep` | Also run a GPU container smoke test (pulls ~100 MB) and measure the RAM's bandwidth (a few seconds, 2 GiB of RAM). |
| `--json` | Machine-readable output. |
| `--app [opencode\|librechat\|open-webui\|openhands\|goose\|odysseus\|hermes]` | Check the router from a workspace app's side instead: its model ids, a tool call, embeddings, MCP, the context floor and the key. |
| `--container STR` | With --app: the app's container, when mnm did not name it. |
| `--from [here\|docker\|lan]` | Check the router as a client sees it: from here, from a container here (docker), or at its LAN address (lan; on a client install, MNM_ROUTER_URL). |
| `--remote-script` | Print the same checks as a script for another machine (sh + curl, or --shell powershell); it reads MNM_API_KEY from its own environment. |
| `--shell [sh\|powershell]` | With --remote-script: sh or powershell. Default: `sh`. |

### `mnm status`

The box on one screen: health, the doctor's worst lines, the models by state, what
waits in the router, the last nightly and the web UI's address.

```
mnm status [OPTIONS]
```

### `mnm machine`

The box mnm runs on: its fingerprint, every box seen, what changed since plans were made.

```
mnm machine [OPTIONS] COMMAND [ARGS]...
```

#### `mnm machine show`

This box in full, what changed since the box before, and the instances planned on
another box.

```
mnm machine show [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--json` | The whole machine file as JSON (uuids included). |
| `--reveal` | Show the cards' uuids (private: never paste them anywhere). |

#### `mnm machine history`

Every box this state folder has seen, newest first, one line each.

```
mnm machine history [OPTIONS]
```

#### `mnm machine forget`

Drop a box that is not this one, with what was measured on it (its RAM bandwidth, its
expert rate and every reading taken on it). Instances planned on it stay, and still
count as planned elsewhere.

```
mnm machine forget [OPTIONS] MACHINE_ID
```

| Argument or option | What it does |
|---|---|
| `MACHINE_ID` | A box's id (`history`). |
| `--yes` | Do not ask first. |

#### `mnm machine adopt`

Plan every instance again for this box after its hardware changed: the RAM measured,
the quant free to change, every download listed with its size and asked once, tunes
queued. Never deletes an instance and never writes the secrets file.

```
mnm machine adopt [OPTIONS] [ONLY…]
```

| Argument or option | What it does |
|---|---|
| `ONLY` | Only these instances (`--only` too). (optional) |
| `--only STR` | Only this instance (repeatable). |
| `--yes` | Download what the plans need. |
| `--keep-quants` | Keep each instance's quant (no new files). |
| `--no-tune` | Queue no tunes. |
| `--dry-run` | Show the box and the plans; save nothing. |

### `mnm update`

Pull the latest code, install dependencies, restart the web UI and router services.

```
mnm update [OPTIONS]
```

### `mnm service`

Run the web UI as a background service (systemd --user).

```
mnm service [OPTIONS] COMMAND [ARGS]...
```

#### `mnm service install`

Install and start the web UI as a user service that survives logout and reboot.

```
mnm service install [OPTIONS]
```

#### `mnm service status`

Are the web UI and router services installed, enabled and running?

```
mnm service status [OPTIONS]
```

#### `mnm service logs`

The web UI and router services' recent log lines, interleaved (journalctl --user).

```
mnm service logs [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--lines / -n INT` | How many lines. Default: `100`. |

#### `mnm service uninstall`

Stop and remove the web UI service (the instances and models stay).

```
mnm service uninstall [OPTIONS]
```

### `mnm serve`

Start the web UI (instances, add model, doctor).

```
mnm serve [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--host STR` | Bind address. Default: `127.0.0.1`. |
| `--port INT` | Port (default MNM_PORT, 7800). |

### `mnm open`

Open the web UI in this machine's browser (under WSL, Windows' default browser).

```
mnm open [OPTIONS] [NAME]
```

| Argument or option | What it does |
|---|---|
| `NAME` | An instance whose page to open; the front page without one. (optional) |

## Find and add models

Look for models, see what fits before anything downloads, and add one.

### `mnm what`

Find models and grade each against this box with the planner: whether it fits, at
which quant, how fast, what it needs, and the line that adds it. Never downloads.

```
mnm what [OPTIONS] [WORDS…]
```

| Argument or option | What it does |
|---|---|
| `WORDS` | What to search for: `qwen3 coder`, `flux`. (optional) |
| `--type STR` | chat, coder, vision, pictures, video, memory (and voice, ear: the curated ones). |
| `--for STR` | No words: three of the curated picks for this use. |
| `--limit INT RANGE` | The most rows a source gives. Default: `6`. |
| `--vram FLOAT` | Override free VRAM in GiB (plan for another machine). |
| `--ram FLOAT` | Free system RAM in GiB, with --vram. Default: `14.0`. |
| `--gpu STR` | GPU name for the bandwidth table, with --vram. Default: `RTX 4070`. |

### `mnm resolve`

Show what a reference resolves to: modality, format, sizes, quants.

```
mnm resolve [OPTIONS] REFERENCE
```

| Argument or option | What it does |
|---|---|
| `REFERENCE` | HF link, org/name, org/name:QUANT, hf.co/… |
| `--json` | Machine-readable output. |

### `mnm fit`

Read the GGUF header remotely and show Quality / Balanced / Speed profiles.

```
mnm fit [OPTIONS] REFERENCE
```

| Argument or option | What it does |
|---|---|
| `REFERENCE` | A Hugging Face, Civitai or LM Studio link, org/name, or org/name/<file\|folder>. |
| `--version STR` | Which version of an image repo (a checkpoint file, a diffusers folder or a GGUF model; `mnm fit` lists them). |
| `--vram FLOAT` | Override free VRAM in GiB (plan for another machine). |
| `--ram FLOAT` | Free system RAM in GiB, with --vram. Default: `14.0`. |
| `--gpu STR` | GPU name for the bandwidth table, with --vram. Default: `RTX 4070`. |

### `mnm add`

Plan an instance for this machine, save its recipe, and download the files.

```
mnm add [OPTIONS] REFERENCE
```

| Argument or option | What it does |
|---|---|
| `REFERENCE` | A Hugging Face, Civitai or LM Studio link, org/name, or org/name/<file\|folder>. |
| `--name STR` | Instance name (default: from the repo). |
| `--profile [quality\|balanced\|speed\|custom]` | quality, balanced or speed (custom: what --quant or --ctx make). Default: `balanced`. |
| `--quant STR` | Explicit quant, e.g. Q5_K_M. |
| `--ctx INT` | Explicit context length. |
| `--kv [f16\|q8_0\|q4_0]` | KV cache type override. |
| `--version STR` | Which version of an image repo (a checkpoint file, a diffusers folder or a GGUF model; `mnm fit` lists them). |
| `--type [chat\|coding\|vision_chat\|embedding\|rerank\|image_gen\|image_edit\|video\|speech_to_text\|text_to_speech\|song\|stems\|voice_convert\|music_score\|unknown]` | Add a chat model as `coding` (or a coding model as `chat`). |
| `--for-agents` | The agent profile: type coding, context 32,768 or the model's own when smaller, tools on. |
| `--for STR` | The agent profile for one client's per-request floor: claude-code, codex, opencode, qwen-code, openhands, hermes, goose, aider, cline, zed, llama-vscode; implies --for-agents. |
| `--download / --no-download` | Download the files now. Default: `on`. |
| `--tune / --no-tune` | Queue a tune for when the card is idle (`mnm tune`). Default: `on`. |
| `--upscaler / --no-upscaler` | Image models: fetch the 18 MB ESRGAN upscaler too, for `/sdcpp/v1/upscale` and the engine page's upscale step. Default: `on`. |
| `--vram FLOAT` | Override free VRAM in GiB (plan for another machine). |
| `--ram FLOAT` | Free system RAM in GiB, with --vram. Default: `14.0`. |
| `--gpu STR` | GPU name for the bandwidth table, with --vram. Default: `RTX 4070`. |

### `mnm starter`

Add starter models that fit this card (each is `mnm add`, planned for this machine).

```
mnm starter [OPTIONS] [PICKS…]
```

| Argument or option | What it does |
|---|---|
| `PICKS` | Numbers or names from the list (chat, memory …), `all`, or none. (optional) |
| `--list` | Show the menu only. |

### `mnm upgrade`

Plan the newest family member of an instance's model with its settings, show what
changes, and add it beside the old one with --yes.

```
mnm upgrade [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | The instance to upgrade. |
| `--yes` | Add the new model beside the old one: it takes the name, the old one is kept as <name>-old (parked first, after its replies, when loaded). |
| `--replace` | Remove <name>-old and the files no other instance uses. |
| `--vram FLOAT` | Override free VRAM in GiB (plan for another machine). |
| `--ram FLOAT` | Free system RAM in GiB, with --vram. Default: `14.0`. |
| `--gpu STR` | GPU name for the bandwidth table, with --vram. Default: `RTX 4070`. |

## Run models

Start, stop, inspect, change and remove the models you added.

### `mnm ls`

List instances with their container and health state.

```
mnm ls [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--disk` | Each instance's files on disk and when it was last used. |

### `mnm show`

Show an instance's recipe, its settings with recommendations, and the exact command.

```
mnm show [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |

### `mnm start`

Start the container, wait until it is ready, and print the URLs to paste.

```
mnm start [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `--stop-others` | First park the running instances whose VRAM it needs (loaded again on request). |
| `--anyway` | Start beside the instances it will not fit with. |

### `mnm stop`

Stop and remove the container; the recipe and files stay. Every name is checked before
anything stops, and an instance that is not running is not an error. Stopped means off
(the router never loads it); --park means unloaded and loadable on request.

```
mnm stop [OPTIONS] [NAMES…]
```

| Argument or option | What it does |
|---|---|
| `NAMES` | — (optional) |
| `--all` | Stop every instance that has a container. |
| `--park` | Unload but keep it loadable: the router loads it at its next request (with --all, every instance, a stopped one too). |

### `mnm try`

Send a running instance the request its curl card shows, and print what came back.

Images are saved as <name>.png, clips as <name>.webm, speech as <name>.wav, in the
current folder. `--all` tries every saved instance in turn (a chat-like one's tool calls
too: the Tools column).

```
mnm try [OPTIONS] [NAME]
```

| Argument or option | What it does |
|---|---|
| `NAME` | — (optional) |
| `--all` | Every saved instance, largest first, each alone (the rest parked; a pinned one or a set in use is left as it is and not tried); one table, also saved as try-all-<time>.txt here. |
| `--audio FILE` | A clip to transcribe (speech-to-text instances). |
| `--speed` | Chat: 400 tokens, and llama.cpp's own tok/s. |
| `--tools` | Chat, coding, vision: the tool-call test an agent depends on (one call, the tool's result, two calls at once, streamed, with thinking on); the verdict shows on the instance's page. `--all` always runs it on chat-like instances. |
| `--measure` | Sample GPU memory every 0.25 s during the request. |
| `--agent` | Chat, coding: an agent's long prefix read cold, again warm, and (cache=on) after a park with its saved slot put back: what the prompt cache saves; kept as a reading. `--speed` is the decode rate. |
| `--prefix INT RANGE` | With --agent: its tokens (20,000, or what one request's context holds). |
| `--say STR` | Text-to-speech: the sentence to read (default: the card's). |
| `--image FILE` | A picture: an image model repaints it, an editor edits it, a video model animates it (image-to-video). |
| `--mask FILE` | With --image: white is redrawn, black kept (inpainting). |

### `mnm why`

Explain an instance's plan in words: why this quant, this context, this much on the
card, and what more room would change. Plans and saves nothing.

```
mnm why [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | The instance whose plan to explain. |
| `--more INT RANGE` | How many GiB more room to plan with for the last part. Default: `4`. |

### `mnm set`

Change settings; they persist and apply on the next start.

```
mnm set [OPTIONS] NAME ASSIGNMENTS…
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `ASSIGNMENTS` | key=value … (see `mnm show`). |

### `mnm replan`

Plan a saved instance again for this machine now (a new card, more RAM, a server, a
card freed): same name, port, type and profile; the settings the plan decides are its
again.

```
mnm replan [OPTIONS] [NAME]
```

| Argument or option | What it does |
|---|---|
| `NAME` | — (optional) |
| `--all` | Every instance, one after another. |
| `--vram FLOAT` | Override free VRAM in GiB (plan for another machine). |
| `--ram FLOAT` | Free system RAM in GiB, with --vram. Default: `14.0`. |
| `--gpu STR` | GPU name for the bandwidth table, with --vram. Default: `RTX 4070`. |
| `--any-quant` | Let the plan pick the quant too (may download). |
| `--yes` | Save without asking. |
| `--shrink` | Take the new plan even when it is smaller and the saved one still fits. |

### `mnm url`

Print the URLs (and config or commands) to paste into a harness or a workspace app.

```
mnm url [OPTIONS] [NAME]
```

| Argument or option | What it does |
|---|---|
| `NAME` | An instance; leave it out for a workspace app's whole file. (optional) |
| `--for [aider\|brave-leo\|claude-code\|cline\|codex\|continue\|curl\|frigate\|goose\|hermes\|home-assistant\|jetbrains\|karakeep\|librechat\|llama-vscode\|n8n\|obsidian-copilot\|odysseus\|open-webui\|openai-sdk\|opencode\|openhands\|qwen-code\|raycast\|zed]` | Who pastes it: a chat client, a coding agent, another app (GUIDE 13), or a workspace app (opencode, librechat, openhands, goose, odysseus: no name needed). Default: `odysseus`. |
| `--direct` | The engine's own address on 127.0.0.1, without the router. |
| `--write` | Write the app's file (its own place, or --to); merges into yours. |
| `--to PATH` | Where --write writes (default: the app's own). |
| `--force` | Replace a file mnm cannot merge into (a copy is kept). |
| `--from [here\|docker\|lan]` | Where the client sits: here (this machine, a Windows app), docker (a container here) or lan (another machine: the router's LAN address, the key required). Default: where that client usually sits. |

### `mnm logs`

Show the engine log, then every warning and error in it with what each one means.

```
mnm logs [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `--tail INT` | Lines from the end (0 = everything). Default: `400`. |
| `--review` | Only the warnings, errors and diagnosis. |

### `mnm pin`

Keep a model loaded: for good (`keep=always`), or `--for 2h` after each request.

```
mnm pin [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `--for STR` | Keep it loaded this long after its last request (2h, 30m) instead of for good. |

### `mnm unpin`

Let the router unload a model again: `keep=auto`, and its idle limit back to ten minutes.

```
mnm unpin [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |

### `mnm vram`

GPU memory in use, read every `--every` seconds for `--seconds`, and the highest.

```
mnm vram [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--seconds INT RANGE` | How long to read for. Default: `10`. |
| `--every FLOAT RANGE` | Seconds between readings. Default: `2.0`. |

### `mnm rm`

Stop the instance and delete its recipe (downloaded files are kept unless --files).

```
mnm rm [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `--files` | Also delete the downloads no other instance uses (frees the disk). |

### `mnm router`

Serve every instance on one URL (the systemd unit runs it); `mnm router manifest` prints what it tells apps.

```
mnm router [OPTIONS] COMMAND [ARGS]...
```

| Argument or option | What it does |
|---|---|
| `--host STR` | Where to listen: 0.0.0.0, local (this machine and its Docker containers) or addresses (default MNM_ROUTER_HOST, 0.0.0.0). |
| `--port INT` | Port (default MNM_ROUTER_PORT, 7801). |

#### `mnm router manifest`

What the router's `GET /__mnm/manifest` answers now: every instance with its desk
roles, paths, context, tools and vision, and the router's URLs from here, from containers
and from the LAN. Built here, so it answers with the router down too.

```
mnm router manifest [OPTIONS]
```

### `mnm mcp`

mnm's workflows and pipelines as MCP tools, for Odysseus (or any MCP client).

```
mnm mcp [OPTIONS] COMMAND [ARGS]...
```

#### `mnm mcp tools`

The tools the router offers now: one per workflow that runs here, the film, run_status.

```
mnm mcp tools [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--schema` | Each tool's arguments too. |

## Sets of models

Models with a role each, started together or taking turns.

### `mnm sets`

Sets: a chat model, a coder, an image model, a voice … started together, planned to fit together, one card per harness.

```
mnm sets [OPTIONS] COMMAND [ARGS]...
```

#### `mnm sets create`

Create a set from existing instances. Nothing starts.

```
mnm sets create [OPTIONS] NAME MEMBERS…
```

| Argument or option | What it does |
|---|---|
| `NAME` | The set's name. |
| `MEMBERS` | role=instance pairs: chat=qwen3-8b image=zimage … |
| `--swap` | The members take turns on the GPU: each starts at its first request, stopping the others when the card is short. |
| `--persistent` | Its members are never unloaded while they run: not for room, not when idle. |

#### `mnm sets edit`

Change a set's members or whether it is persistent (its instances stay; nothing starts).

```
mnm sets edit [OPTIONS] NAME [MEMBERS…]
```

| Argument or option | What it does |
|---|---|
| `NAME` | The set's name. |
| `MEMBERS` | role=instance pairs to add or replace: image=klein … (optional) |
| `--drop STR` | A role to take out (repeat for more). |
| `--persistent / --no-persistent` | Its members are never unloaded while they run (or are again). |

#### `mnm sets ls`

Every set with its members and whether they fit together right now.

```
mnm sets ls [OPTIONS]
```

#### `mnm sets show`

The members, each one's state and fit verdict, and the fit-together verdict.

```
mnm sets show [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |

#### `mnm sets default`

The desk: the set whose roles (coder, chat, memory …) are model ids an app's
config names, so the app survives a model swap. Without a name: which set it is.

```
mnm sets default [OPTIONS] [NAME]
```

| Argument or option | What it does |
|---|---|
| `NAME` | The set that becomes the desk. (optional) |
| `--clear` | No desk: no role is a model id. |

#### `mnm sets start`

Start every member that is not running, the largest first, and print their URLs.
When they do not fit, mnm's other running models are parked first, as the router would
park them (never a pinned one or a member of a set in use).

```
mnm sets start [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `--anyway` | Start even when the members will not fit together. |
| `--no-room` | Park nothing for the set: refuse when mnm's other running models hold its room. |

#### `mnm sets stop`

Stop every member that has a container; recipes and files stay.

```
mnm sets stop [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |

#### `mnm sets rm`

Remove the set; its instances stay.

```
mnm sets rm [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |

#### `mnm sets url`

Every member's URLs for one harness, under its role.

```
mnm sets url [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `--for [odysseus\|open-webui\|continue\|openai-sdk\|curl\|claude-code\|codex\|opencode\|qwen-code\|aider\|cline\|zed\|llama-vscode\|home-assistant\|karakeep\|frigate\|n8n\|jetbrains\|raycast\|obsidian-copilot\|brave-leo]` | Who pastes it: a chat client or a coding agent. Default: `odysseus`. |

## Make things

Runs and their outputs, pipelines, ComfyUI workflows and LoRAs.

### `mnm run`

Send an instance your prompt through the router and keep what comes back as a run
(the web's gallery, `mnm outputs`); `--repeat N` makes N runs of it, one after another
(a song's N are candidates under one run, to pick one from).

```
mnm run [OPTIONS] NAME [PROMPT]
```

| Argument or option | What it does |
|---|---|
| `NAME` | The instance. |
| `PROMPT` | What you want: a question, a picture, a clip, a line. (optional) |
| `--image FILE` | A picture to repaint, edit, animate or ask a vision model about. |
| `--mask FILE` |  |
| `--audio FILE` | A clip to transcribe, split into stems, convert or score. |
| `--lyrics FILE` | A song's lyrics, from a text file. |
| `--repeat INT RANGE` | Make it N times (1 to 20), one run each. Default: `1`. |

### `mnm outputs`

What runs made: list, show, delete, prune.

```
mnm outputs [OPTIONS] COMMAND [ARGS]...
```

#### `mnm outputs ls`

The runs, newest first, with their files and sizes.

```
mnm outputs ls [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--limit INT RANGE` | How many of the newest runs. Default: `30`. |

#### `mnm outputs show`

One run: what was asked, with which settings, and its files.

```
mnm outputs show [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |

#### `mnm outputs rm`

Delete runs: their records and their files.

```
mnm outputs rm [OPTIONS] RUN_IDS…
```

| Argument or option | What it does |
|---|---|
| `RUN_IDS` | Run ids (`mnm outputs ls`). |
| `--keep-files` | Forget the run, keep its folder. |

#### `mnm outputs prune`

Delete old runs (records and files), by age, by total size, or both.

```
mnm outputs prune [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--older-than STR` | An age: 12h, 30d, 2w. |
| `--max-gb FLOAT RANGE` | Delete the oldest runs until the rest fit. |
| `--yes` | Do not ask. |

### `mnm clean`

Empty what runs and engines left behind, one target at a time. Models, recipes, sets
and engine images are never touched (`mnm rm`, `uninstall.sh`).

```
mnm clean [OPTIONS] [TARGETS…]
```

| Argument or option | What it does |
|---|---|
| `TARGETS` | outputs, runs, logs, workflows, imports, comfy (none: show what each holds). (optional) |
| `--older-than STR` | Only what is older: 12h, 30d, 2w. |
| `--keep-files` | runs: forget the history, keep the files. |
| `--yes` | Do not ask. |
| `--auto` | What the nightly cleans: runs and logs older than `mnm nightly set clean=30d`'s age, without asking. |

### `mnm pipeline`

Pipelines: multi-step runs over your instances (Short film).

```
mnm pipeline [OPTIONS] COMMAND [ARGS]...
```

#### `mnm pipeline run`

Make a short film: a shot list, a still per shot, the lines, a clip per still, one film.
It stops for you after the shot list and after the stills unless --auto.

```
mnm pipeline run [OPTIONS] IDEA
```

| Argument or option | What it does |
|---|---|
| `IDEA` | The film's idea, in a sentence or two. |
| `--chat STR` | The chat instance that writes the shot list. |
| `--image STR` | The image instance that draws each still. |
| `--video STR` | The video instance that animates each still, or comfy:<workflow> for an imported ComfyUI workflow (a Wan 14B one, say). |
| `--voice STR` | A text-to-speech instance; none: a silent film. |
| `--shots INT RANGE` |  Default: `3`. |
| `--style STR` | The look every still shares. Default: `cinematic, soft light`. |
| `--auto` | Run straight through; no stop at the review points. |
| `--clip-size STR` | A comfy:<workflow>'s clip size, WxH (default 832x480); the stills are drawn at it. An instance's clips are its own size. |
| `--clip-frames INT RANGE` | A comfy:<workflow>'s frames per clip (default 33). |

#### `mnm pipeline go`

Past the review point the run waits at, and on.

```
mnm pipeline go [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |

#### `mnm pipeline resume`

Carry on a run that failed or was interrupted, from the first item not done.

```
mnm pipeline resume [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |

#### `mnm pipeline redo`

Make one item again (and the film); a still at the stills review stays there.

```
mnm pipeline redo [OPTIONS] RUN_ID STAGE [NUMBER]
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |
| `STAGE` | shotlist, stills, lines or clips. |
| `NUMBER` | Which one (1 = the first shot). (optional) |

#### `mnm pipeline ls`

The pipeline runs, newest first.

```
mnm pipeline ls [OPTIONS]
```

#### `mnm pipeline show`

One run's steps: each item, its state and its file.

```
mnm pipeline show [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |

### `mnm comfyui`

ComfyUI: share mnm's picture models with it (no subcommand), or run it as an optional engine for workflows (install, start, stop, status, logs).

```
mnm comfyui [OPTIONS] COMMAND [ARGS]...
```

| Argument or option | What it does |
|---|---|
| `--write PATH` | Your ComfyUI folder: add the block to its extra_model_paths.yaml (the file as it was before the first write is kept as .bak). |

#### `mnm comfyui install`

Build ComfyUI's image on this box (an Ubuntu base, ComfyUI at mnm's pinned release,
PyTorch). The image before it stays for `mnm comfyui rollback`.

```
mnm comfyui install [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--cpu` | PyTorch for the CPU (no NVIDIA card; slow). |
| `--ca FILE` | A proxy's CA certificate, for a build behind one. |
| `--yes` | Do not ask first. |

#### `mnm comfyui start`

Start ComfyUI (on its own network, no internet) and wait until it answers.

```
mnm comfyui start [OPTIONS]
```

#### `mnm comfyui stop`

Stop ComfyUI (its image, workflows and outputs stay).

```
mnm comfyui stop [OPTIONS]
```

#### `mnm comfyui status`

Installed or not, running or not, and what it holds on the card.

```
mnm comfyui status [OPTIONS]
```

#### `mnm comfyui logs`

ComfyUI's log.

```
mnm comfyui logs [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--tail INT RANGE` |  Default: `200`. |

#### `mnm comfyui rollback`

Use the image before the last install again (restart ComfyUI to apply it).

```
mnm comfyui rollback [OPTIONS]
```

#### `mnm comfyui uninstall`

Remove ComfyUI's container, images and network (workflows and outputs stay).

```
mnm comfyui uninstall [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--yes` |  |

#### `mnm comfyui models`

Each model file the workflow names: in place, on disk (an instance's), fetchable, or
unknown; `--fetch` downloads and links them.

```
mnm comfyui models [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | An imported workflow. |
| `--set STR` | FILE=LINK: where a file comes from (Hugging Face or Civitai). |
| `--find` | Look for files without a link in known repositories. |
| `--fetch` | Download what is missing. |
| `--yes` | Do not ask before downloading. |

#### `mnm comfyui run`

Run an imported workflow; ComfyUI starts if it is off, mnm's models make room, and the
outputs are kept as a run (`mnm outputs`). `--repeat N`: N runs, a new seed for each
after the first.

```
mnm comfyui run [OPTIONS] NAME [VALUES…]
```

| Argument or option | What it does |
|---|---|
| `NAME` | An imported workflow. |
| `VALUES` | key=value settings (`mnm comfyui workflow fields`). (optional) |
| `--file STR` | key=path: a picture, clip or sound from this folder. |
| `--repeat INT RANGE` | Make it N times (1 to 20), one run each. Default: `1`. |

#### `mnm comfyui again`

A workflow run's settings and uploads again, as a new run.

```
mnm comfyui again [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | A workflow run (`mnm outputs ls`). |
| `--new-seed` | Random seeds this time. |

#### `mnm comfyui packs`

Custom node packs: the vetted list, approvals, the install.

```
mnm comfyui packs [OPTIONS] COMMAND [ARGS]...
```

##### `mnm comfyui packs ls`

The vetted packs, the approved ones and what the running layer holds.

```
mnm comfyui packs ls [OPTIONS]
```

##### `mnm comfyui packs add`

Approve packs: a vetted one at once, any other after showing what the Registry (or, for
`github:owner/repo@tag`, the code and GitHub) says.

```
mnm comfyui packs add [OPTIONS] [PACKS…]
```

| Argument or option | What it does |
|---|---|
| `PACKS` | Registry ids, or github:owner/repo@<tag or commit>. (optional) |
| `--version STR` | A version other than the vetted one. |
| `--for STR` | Every pack an imported workflow uses. |
| `--yes` | Do not ask first. |

##### `mnm comfyui packs rm`

Withdraw an approval (`install` then builds the layer without it).

```
mnm comfyui packs rm [OPTIONS] PACK
```

| Argument or option | What it does |
|---|---|
| `PACK` | — |

##### `mnm comfyui packs install`

Fetch the packs' own weights, build the approved packs into ComfyUI's image, and start
ComfyUI once to check each one loads (the layer before stays when one does not).

```
mnm comfyui packs install [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--ca FILE` | A proxy's CA certificate. |
| `--yes` | Do not ask before downloading. |

##### `mnm comfyui packs rollback`

Use the node packs layer before the last install again.

```
mnm comfyui packs rollback [OPTIONS]
```

##### `mnm comfyui packs find`

Which Registry pack provides a node class.

```
mnm comfyui packs find [OPTIONS] NODE_CLASS
```

| Argument or option | What it does |
|---|---|
| `NODE_CLASS` | — |

#### `mnm comfyui workflow`

ComfyUI workflows: import from a link or a file, inspect, remove.

```
mnm comfyui workflow [OPTIONS] COMMAND [ARGS]...
```

##### `mnm comfyui workflow add`

Import a workflow (a Civitai zip may hold several). Nothing is installed or downloaded
beyond the workflow file itself.

```
mnm comfyui workflow add [OPTIONS] REFERENCE
```

| Argument or option | What it does |
|---|---|
| `REFERENCE` | A link (comfy.org, Civitai, HF, GitHub) or a file. |
| `--name STR` | Its name in mnm (one workflow only). |

##### `mnm comfyui workflow ls`

The imported workflows.

```
mnm comfyui workflow ls [OPTIONS]
```

##### `mnm comfyui workflow show`

What a workflow needs: node packs, model files, anything refused.

```
mnm comfyui workflow show [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |

##### `mnm comfyui workflow rm`

Remove an imported workflow (its runs stay in `mnm outputs`).

```
mnm comfyui workflow rm [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `--files` | Also delete the model files mnm fetched for it that no other workflow or instance uses. |

##### `mnm comfyui workflow fields`

The settings a run takes (`key=value` for `mnm comfyui run`), with the workflow's own
values.

```
mnm comfyui workflow fields [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `--all` | More settings too. |

##### `mnm comfyui workflow pick`

Show settings from More settings on top of the web form (and in `fields`).

```
mnm comfyui workflow pick [OPTIONS] NAME KEYS…
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `KEYS` | Keys from `fields --all`. |
| `--unpick` | Back under More settings. |

### `mnm lora`

LoRAs on image instances: add from the Hub, set weights, remove.

```
mnm lora [OPTIONS] COMMAND [ARGS]...
```

#### `mnm lora ls`

The instance's LoRAs: base family, weight, whether every request uses it.

```
mnm lora ls [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | An image instance. |

#### `mnm lora add`

Check the LoRA fits the instance's base model, download it, offer it.

```
mnm lora add [OPTIONS] NAME REFERENCE
```

| Argument or option | What it does |
|---|---|
| `NAME` | An image instance. |
| `REFERENCE` | Hugging Face or Civitai (or Civitai Archive) link, or repo id, of the LoRA. |
| `--name STR` | What the engine page lists it as. |
| `--weight FLOAT` | Strength, 1.0 = as trained. Default: `1.0`. |
| `--always` | Apply to every request (Odysseus's too). |

#### `mnm lora set`

Change a LoRA's default weight or whether every request uses it.

```
mnm lora set [OPTIONS] NAME LORA
```

| Argument or option | What it does |
|---|---|
| `NAME` | An image instance. |
| `LORA` | The LoRA's name (`mnm lora ls`). |
| `--weight FLOAT` | Strength, 1.0 = as trained. |
| `--always / --not-always` | Apply to every request, or only when picked. |

#### `mnm lora rm`

Stop offering a LoRA (the downloaded file stays in the cache).

```
mnm lora rm [OPTIONS] NAME LORA
```

| Argument or option | What it does |
|---|---|
| `NAME` | An image instance. |
| `LORA` | The LoRA's name (`mnm lora ls`). |

## The studio

Songs, the studio's stages, tools and trainers, and voices.

### `mnm song`

Songs: a theme to lyrics, candidates, stems, a mix and an export.

```
mnm song [OPTIONS] COMMAND [ARGS]...
```

#### `mnm song new`

Make a song: lyrics from your theme, candidates to pick from, the picked one's stems, a
mix and an export folder. It stops for you at the lyrics and at the pick unless --auto;
--clean cleans the vocals, --voice sings them in a library voice.

```
mnm song new [OPTIONS] THEME
```

| Argument or option | What it does |
|---|---|
| `THEME` | What the song is about, in a sentence or two. |
| `--length STR` | 30s, 3m, 2m30s (10 s to 10 minutes). Default: `30s`. |
| `--lyrics FILE` | Your own lyrics, a section header like Verse in square brackets above its lines: no lyrics are written for you. |
| `--tags STR` | The style line the song model sings to (else the chat's). |
| `--recording FILE` | A recording of yours. |
| `--as STR` | How the recording is used: complete, vocal2bgm or reference. |
| `--candidates INT RANGE` | Candidates to pick from. Default: `4`. |
| `--tier STR` | fast to explore, full for the keeper. Default: `fast`. |
| `--chat STR` | Who writes the lyrics: the desk's chat, or an instance. Default: `chat`. |
| `--auto` | Straight through: no stop at the lyrics or the pick. |
| `--voice STR` | A library voice the vocals are sung in (`mnm voices ls`). |
| `--shift INT RANGE` | Semitones the voice moves the vocals by. Default: `0`. |
| `--clean STR` | What to do to the separator's vocals: debleed, dereverb, denoise, deess, gate, tune (comma-separated; `mnm song vocals` changes it later). |

#### `mnm song go`

Past the review point the run waits at (the lyrics), and on.

```
mnm song go [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |

#### `mnm song pick`

Pick a candidate: its stems, the mix and the export follow.

```
mnm song pick [OPTIONS] RUN_ID CANDIDATE
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |
| `CANDIDATE` | Which one (1 = the first). |

#### `mnm song voice`

Sing a song's vocals in another voice (or `none`): the vocals as sung are converted,
then the mix and the export follow.

```
mnm song voice [OPTIONS] RUN_ID VOICE
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |
| `VOICE` | A library voice (`mnm voices ls`), or `none` for the model's own. |
| `--shift INT RANGE` | Semitones the voice moves the vocals by. Default: `0`. |

#### `mnm song vocals`

Clean and tune a song's vocals before a voice sings them: made again from the
separator's vocals, then the voice, the mix and the export follow.

```
mnm song vocals [OPTIONS] RUN_ID CLEAN
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |
| `CLEAN` | What to do to the separator's vocals, comma-separated: debleed, dereverb, denoise, deess, gate, tune (done in that order), or `none`. |
| `--key STR` | Tuning's key, as `A minor` (else the structure map's). |
| `--strength INT RANGE` | How far tuning moves a note, 0 to 100 (80). |

#### `mnm song resume`

Carry on a run that failed or was interrupted, from the first item not done.

```
mnm song resume [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |

#### `mnm song redo`

Make one item again and what follows it (the picked candidate: its score, the
structure, its stems, the mix, the export, and the run waits at the pick again; another
candidate: its score and the export, the pick stands). `score` without a number scores
every candidate again (after `mnm studio install`).

```
mnm song redo [OPTIONS] RUN_ID STAGE [NUMBER]
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |
| `STAGE` | lyrics, song, score, structure, stems, vocals, voice, midi, mix or export. |
| `NUMBER` | Which one (a candidate's number; 1 if not said). (optional) |

#### `mnm song export`

The export folder again: the mix (WAV 24-bit 48 kHz, FLAC, MP3 on request), the stems,
song.rpp, provenance.json and disclosure.txt. Never refused for a missing consent.

```
mnm song export [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |
| `--mp3` | An MP3 of the mix too. |
| `--release` | Mark it for release (a note where consent is missing). |
| `--lufs FLOAT` | The mix's loudness, LUFS integrated (else the line kept, -14 at first). |
| `--true-peak FLOAT` | Its true peak ceiling, dBTP (else the line kept, -1 at first). |
| `--no-loudness` | The mix as summed, not normalised. |

#### `mnm song ls`

The song runs, newest first.

```
mnm song ls [OPTIONS]
```

#### `mnm song show`

One song run's steps: each item, its state and what it made.

```
mnm song show [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |

#### `mnm song edit`

Edit a song's take (a range repainted, a section re-sung, extended, covered, an
instrument added or swapped): the song uses the new take, and Undo is `mnm song take`.

```
mnm song edit [OPTIONS] RUN_ID EDIT [VALUES…]
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |
| `EDIT` | repaint, resing, extend, cover, add, swap, cut, crop, duplicate, move, tempo, key |
| `VALUES` | key=value: what the edit takes. (optional) |

#### `mnm song take`

The song's takes (the picked candidate's and every edit's), or the one to use: its
stems, voice, mix and export are made again from it.

```
mnm song take [OPTIONS] RUN_ID [TAKE]
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |
| `TAKE` | An edit's run id, or `candidate` for the picked candidate's take. (optional) |

#### `mnm song master`

Master the song's mix to a reference track, or choose the master its export uses.

```
mnm song master [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |
| `--reference PATH` | The track to sound like (WAV, FLAC, MP3). |
| `--use STR` | Use this master of the song. |
| `--off` | No master: the loudness line. |

#### `mnm song mix`

The stem mixer: show the mix in use, render a new one, or choose one to use.

```
mnm song mix [OPTIONS] RUN_ID [CHANGES…]
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | — |
| `CHANGES` | stem=<dB>[,mute][,solo][,in=<s>][,out=<s>] per stem (a stem not named stays). (optional) |
| `--use STR` | Use this render of the song. |
| `--own` | Use the song's own mix again. |

### `mnm studio`

The studio: its stages, which instance plays each, and the tools image.

```
mnm studio [OPTIONS] COMMAND [ARGS]...
```

#### `mnm studio stage`

The stages and their tiers; with `fast=<instance>` or `full=<instance>`, pick the
instance that plays that tier of the stage (the registry's model stands until you do).

```
mnm studio stage [OPTIONS] [STAGE] [PICKS…]
```

| Argument or option | What it does |
|---|---|
| `STAGE` | A stage: song, stems … (optional) |
| `PICKS` | fast=<instance> or full=<instance> (the song stage: or =workflow:<name>, an imported ComfyUI workflow); `fast=` goes back to the row's. (optional) |

#### `mnm studio models`

Every studio model against this box now: what it needs of the card (or of the RAM on
a box without one), whether this box holds it, the smallest card that does, and its time
for a second of audio (this box's own once measured).

```
mnm studio models [OPTIONS]
```

#### `mnm studio install`

Build the studio tools' image on this box: the candidates' scorer, the structure map,
the voice's similarity, MIDI, mastering and pitch tuning, all on the CPU. The image before
it stays for `rollback`.

```
mnm studio install [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--ca FILE` | A proxy's CA certificate, for a build behind one. |
| `--yes` | Do not ask first. |

#### `mnm studio status`

The tools image: which one is in use, what it carries, and whether this mnm would
build another.

```
mnm studio status [OPTIONS]
```

#### `mnm studio rollback`

Go back to the image before the last install.

```
mnm studio rollback [OPTIONS]
```

#### `mnm studio uninstall`

Remove every studio tools image; the runs, their scores and structure maps stay.

```
mnm studio uninstall [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--yes` | Do not ask first. |

#### `mnm studio trainer`

The trainer's image: Applio, built on this box.

```
mnm studio trainer [OPTIONS] COMMAND [ARGS]...
```

##### `mnm studio trainer install`

Build a trainer's image on this box: for voices, Applio at a pinned release with its
weights checked by sha256; for styles (`--for styles`), ACE-Step 1.5's own trainer at a
pinned commit (its checkpoint is downloaded when a style is trained). PyTorch for this
card, or the processor.

```
mnm studio trainer install [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--cpu` | Build it for the processor (no card: hours to days). |
| `--ca FILE` | A proxy's CA certificate, for a build behind one. |
| `--yes` | Do not ask first. |
| `--for [voices\|styles]` | voices (Applio) or styles (ACE-Step). Default: `voices`. |

##### `mnm studio trainer status`

A trainer's image: which one, for what, and whether this mnm would build another.

```
mnm studio trainer status [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--for [voices\|styles]` | voices (Applio) or styles (ACE-Step). Default: `voices`. |

##### `mnm studio trainer rollback`

Go back to the image before the last install.

```
mnm studio trainer rollback [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--for [voices\|styles]` | voices (Applio) or styles (ACE-Step). Default: `voices`. |

##### `mnm studio trainer uninstall`

Remove every image of a trainer; what it trained and the trainings' logs stay.

```
mnm studio trainer uninstall [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--yes` | Do not ask first. |
| `--for [voices\|styles]` | voices (Applio) or styles (ACE-Step). Default: `voices`. |

#### `mnm studio train`

Train a voice from recordings or a style from songs (the router runs it).

```
mnm studio train [OPTIONS] COMMAND [ARGS]...
```

##### `mnm studio train voice`

Train an RVC voice from a folder of recordings (10 to 60 minutes of the voice alone,
dry, WAV or FLAC is what RVC voices are trained on). The router runs it while nobody uses
the card; a request pauses it, and it resumes from its last checkpoint.

```
mnm studio train voice [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | The voice's name in the library. |
| `--from DIRECTORY` | A folder of the voice's recordings. |
| `--source [own\|generated\|hired\|licensed\|likeness]` | Whose voice: own, hired, licensed, likeness, generated. |
| `--consent FILE` | The singer's consent (only its hash). |
| `--rate INT` | 32000, 40000 or 48000. Default: `40000`. |
| `--epochs INT` | Passes over the recordings. Default: `200`. |
| `--batch INT` | Chunks per step. Default: `8`. |
| `--licence STR` | The terms, in your words. |
| `--now` | Start without the router's idle minute. |

##### `mnm studio train ls`

The trainings: queued, running (with the epoch), paused, done or failed.

```
mnm studio train ls [OPTIONS]
```

##### `mnm studio train show`

One training: its settings, recordings, warnings, progress and outcome.

```
mnm studio train show [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | The training's run id. |

##### `mnm studio train cancel`

Stop a training (running or queued); its folder stays until `mnm clean`.

```
mnm studio train cancel [OPTIONS] RUN_ID
```

| Argument or option | What it does |
|---|---|
| `RUN_ID` | The training's run id. |

##### `mnm studio train style`

Teach ACE-Step a style from a folder of songs (five or more, 30 s to 4 min, WAV or FLAC
keep the sound; a `<song>.txt` beside a song is its lyrics). It downloads ACE-Step's
checkpoint for the build (about 6 GB, once); the router trains while nobody uses the
card, its first epoch measuring the card's real peak; a request pauses it.

```
mnm studio train style [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | The style's name in the library. |
| `--from DIRECTORY` | A folder of the songs. |
| `--caption STR` | What the songs sound like: the style, in words. |
| `--source [own\|licensed\|generated]` | Whose songs: own, licensed, generated. |
| `--trigger STR` | The word that calls the style up (the name if unset). |
| `--variant [base\|turbo]` | The ACE-Step build it plays on. Default: `base`. |
| `--epochs INT` | Passes over the songs. Default: `400`. |
| `--max-seconds FLOAT` | Each song is cut here (30 to 600). Default: `240.0`. |
| `--licence STR` | The terms, in your words. |
| `--now` | Start without the router's idle minute. |

#### `mnm studio styles`

The style library: LoRAs that teach ACE-Step a style.

```
mnm studio styles [OPTIONS] COMMAND [ARGS]...
```

##### `mnm studio styles ls`

The styles: each one's build, trigger word and whose songs it learned from.

```
mnm studio styles ls [OPTIONS]
```

##### `mnm studio styles show`

One style: its files by hash, how it was trained, how to play it.

```
mnm studio styles show [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | The style. |

##### `mnm studio styles add`

Add an ACE-Step LoRA trained elsewhere (PEFT's `adapter_model.safetensors` and
`adapter_config.json`), checked as audio.cpp loads it.

```
mnm studio styles add [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | The style's name in the library. |
| `--adapter DIRECTORY` | A PEFT adapter's folder. |
| `--variant [base\|turbo]` | The build it was trained on. |
| `--trigger STR` | The word its training put in front. |
| `--source [own\|licensed\|generated]` | Whose songs: own, licensed, generated. |
| `--licence STR` | The terms, in your words. |

##### `mnm studio styles rm`

Remove a style from the library (its files too unless `--keep-files`).

```
mnm studio styles rm [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | The style. |
| `--keep-files` | Keep its folder (only the row goes). |

### `mnm voices`

The voice library: the voices a song may be sung in (your own, or someone's you have the rights to), each with its source and, for someone else's, a consent record.

```
mnm voices [OPTIONS] COMMAND [ARGS]...
```

#### `mnm voices add`

Add a voice: its files are copied into the library and kept by their sha256.

```
mnm voices add [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | Its name in the library (`alto`, `me-2026`). |
| `--engine [rvc\|seed_vc\|vevo2\|soulx]` | seed_vc, vevo2 or soulx from a reference; rvc from a checkpoint. Default: `seed_vc`. |
| `--source [own\|generated\|hired\|licensed\|likeness]` | own, generated, hired, licensed or likeness: whose voice it is. |
| `--reference FILE` | 1 to 30 s of the voice alone, a WAV file. |
| `--model FILE` | An RVC checkpoint (.pth). |
| `--index FILE` | Its retrieval index (.index), optional. |
| `--packaged STR` | One of audio.cpp's packaged RVC voices: refused, with why. |
| `--licence STR` | The terms you use it under, in words. |
| `--note STR` | Anything to remember about it. |

#### `mnm voices ls`

The voices in the library.

```
mnm voices ls [OPTIONS]
```

#### `mnm voices show`

One voice: its files by hash, its source and consent, how it was made.

```
mnm voices show [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |

#### `mnm voices rm`

Remove a voice from the library (and its files, unless --keep-files).

```
mnm voices rm [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `--keep-files` | Keep its recordings and checkpoint. |

#### `mnm voices consent`

Record a consent document's sha256 and today's date (the document is never copied).

```
mnm voices consent [OPTIONS] NAME DOCUMENT
```

| Argument or option | What it does |
|---|---|
| `NAME` | — |
| `DOCUMENT` | — |
| `--note STR` | Who signed it, what it covers. |

## Measure and tune

Readings, tuning, the nightly self-check and the engine images.

### `mnm calibration`

Readings from this box and the corrections they make to the planner.

```
mnm calibration [OPTIONS] COMMAND [ARGS]...
```

#### `mnm calibration export`

Write every reading and correction to one file in this folder, to paste into a chat.

```
mnm calibration export [OPTIONS]
```

### `mnm tune`

Tune an instance's settings on this box: real starts, measured on fresh prompts.

```
mnm tune [OPTIONS] [NAME]
```

| Argument or option | What it does |
|---|---|
| `NAME` | The instance to tune. (optional) |
| `--now / --no-now` | Skip the idle minute (still parks by the rules). |
| `--for [chat\|agents]` | What the score weighs: chat (decode only) or agents (long prompts too); default: agents for a coding instance, else decode with the 2k prompt speed. |
| `--minutes INT RANGE` | Wall-clock budget. Default: `45`. |
| `--all` | Every instance the tuner tunes. |
| `--status` | The queue, the tune running, each result. |
| `--cancel` | Unqueue it and stop its tune. |
| `--quality` | Choose the quant instead: each quant that runs here scored against Q8_0 and timed, the most faithful one over the speed floor kept. |
| `--floor FLOAT RANGE` | With --quality: the decode floor (tok/s) for this tune. |
| `--quants STR` | With --quality: the quants to measure (comma-separated). |
| `--keep / --no-keep` | With --quality: keep the quants not chosen on disk. |
| `--yes / --no-yes` | With --quality: download without asking. |

### `mnm nightly`

The nightly self-check: the doctor, every app's doctor, `try --all --speed`, a backup, the disk; one report a night and a notice only when something regressed.

```
mnm nightly [OPTIONS] COMMAND [ARGS]...
```

#### `mnm nightly run`

One nightly run, as the timer runs it: the report under <state>/nightly/<date>.json.

```
mnm nightly run [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--now` | Do not wait for the router's idle minute (it still waits for a tune, a start or stop, or a ComfyUI job to end). |
| `--skip [doctor\|deep\|apps\|try\|backup\|freshness\|disk\|clean\|router]` | Leave a step out, once per step: try, apps, backup, disk, clean, doctor, freshness; deep runs the doctor without --deep. |

#### `mnm nightly show`

A night's report: the verdicts with what mnm did or offers, then each step.

```
mnm nightly show [OPTIONS] [DATE]
```

| Argument or option | What it does |
|---|---|
| `DATE` | A report's name (2026-10-26, 2026-10-26-2); the last. (optional) |

#### `mnm nightly ls`

Every night's report, newest last, with one word: quiet, regression, failed, waited.

```
mnm nightly ls [OPTIONS]
```

#### `mnm nightly set`

The nightly's settings: `clean=<age>` turns on `mnm clean --auto` in the night.

```
mnm nightly set [OPTIONS] [ASSIGNMENT]
```

| Argument or option | What it does |
|---|---|
| `ASSIGNMENT` | clean=30d (or 2w, 12h), clean=off. (optional) |

#### `mnm nightly install`

A systemd user timer that runs `mnm nightly run` every night; it runs the backup too,
so the backup's own timer goes.

```
mnm nightly install [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--at STR` | Every day at this time on the box's clock. Default: `03:00`. |

#### `mnm nightly status`

The timer, and the last night's word.

```
mnm nightly status [OPTIONS]
```

#### `mnm nightly uninstall`

Remove the timer; the reports stay.

```
mnm nightly uninstall [OPTIONS]
```

### `mnm engines`

Engine images (llama.cpp, stable-diffusion.cpp, whisper.cpp, audio.cpp, vLLM): status, check (a newer build upstream?), update (which tests the new build), test, rollback.

```
mnm engines [OPTIONS] COMMAND [ARGS]...
```

#### `mnm engines update`

Pull the newest build of each engine image; the previous build is kept for rollback.

```
mnm engines update [OPTIONS] [WHICH…]
```

| Argument or option | What it does |
|---|---|
| `WHICH` | Engines to update (default: every engine this box has or uses). (optional) |

#### `mnm engines check`

Whether the registry holds a newer build of each engine image, never pulled,
and whether it runs a saved instance the installed build lacks.

```
mnm engines check [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--json` | Machine-readable output. |

#### `mnm engines test`

Test an engine's current build with a copy of its smallest downloaded instance,
against that instance's readings (the ledger's other build may be one a rollback
rejected, so it never sets the bar here).

```
mnm engines test [OPTIONS] ENGINE
```

| Argument or option | What it does |
|---|---|
| `ENGINE` | The engine whose current build to test. |

#### `mnm engines rollback`

Point an engine's tag back at the build the last `mnm engines update` replaced.

```
mnm engines rollback [OPTIONS] TARGET
```

| Argument or option | What it does |
|---|---|
| `TARGET` | llamacpp \| sdcpp \| whispercpp \| audiocpp \| vllm, or an image a recipe names |

## Workspace apps

Install, connect and check the apps that use the router.

### `mnm apps`

Workspace apps: OpenCode, LibreChat and OpenHands installed (pinned, on 127.0.0.1); Goose and Odysseus connected.

```
mnm apps [OPTIONS] COMMAND [ARGS]...
```

#### `mnm apps ls`

Every app: installed or not, running, its version against this mnm's pin, its URL.

```
mnm apps ls [OPTIONS]
```

#### `mnm apps status`

One app: its state, version, URL and where its password or key is.

```
mnm apps status [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | opencode, librechat, open-webui or openhands. |

#### `mnm apps install`

Install the app at this mnm's pin, write its config from the router's manifest, start it.

```
mnm apps install [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | opencode, librechat, open-webui or openhands. |
| `--force` | Replace a config file of yours mnm cannot merge. |

#### `mnm apps start`

Start an installed app (its config as it is).

```
mnm apps start [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | opencode, librechat, open-webui or openhands. |

#### `mnm apps stop`

Stop an app; it stays installed (and stopped across a reboot).

```
mnm apps stop [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | opencode, librechat, open-webui or openhands. |

#### `mnm apps logs`

The app's newest log lines.

```
mnm apps logs [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | opencode, librechat, open-webui or openhands. |
| `-n INT` | How many lines. Default: `200`. |

#### `mnm apps update`

Move to the pin this mnm carries; the installed version is kept for rollback.

```
mnm apps update [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | opencode, librechat, open-webui or openhands. |

#### `mnm apps rollback`

Back to the version the last update replaced.

```
mnm apps rollback [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | opencode, librechat, open-webui or openhands. |

#### `mnm apps uninstall`

Stop and remove the app; asks before deleting its data (under MNM_APPS_DIR).

```
mnm apps uninstall [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | opencode, librechat, open-webui or openhands. |
| `--keep-data` | Keep its accounts, chats and settings. |
| `--yes` | Do not ask before deleting its data. |

#### `mnm apps expose`

Open an installed app to your network (its own login and the router's key required),
or close it again with --off.

```
mnm apps expose [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | opencode, librechat, open-webui or openhands. |
| `--off` | Back on 127.0.0.1 only. |
| `--yes / -y` | Open it without asking (plain http). |

#### `mnm apps connect`

Link an app mnm does not install: Goose's provider file, or Odysseus's .env (a diff
first, then `docker compose up -d` on yes) and the forms to fill in its Settings.

```
mnm apps connect [OPTIONS] NAME
```

| Argument or option | What it does |
|---|---|
| `NAME` | goose or odysseus. |
| `--write / --no-write` | Write the app's file (Goose). Default: `on`. |
| `--force` | Replace a file mnm cannot merge. |
| `--dir PATH` | Odysseus's folder (default ~/odysseus). |
| `--yes` | Write and restart without asking. |

#### `mnm apps user`

Accounts in an app mnm installed (LibreChat).

```
mnm apps user [OPTIONS] COMMAND [ARGS]...
```

##### `mnm apps user add`

An account in mnm's LibreChat (registration is off). Asks for the password.

```
mnm apps user add [OPTIONS] NAME EMAIL
```

| Argument or option | What it does |
|---|---|
| `NAME` | opencode, librechat, open-webui or openhands. |
| `EMAIL` | The account's email (its login). |
| `--name STR` | Shown in LibreChat. |
| `--username STR` | Default: the email's part before @. |

## Keep things safe

Backups, the disk policy, reports for bug reports, notifications.

### `mnm backup`

Recipes, sets, ComfyUI workflows, approvals, readings and the secrets' names in one .tar.gz; never a secret's value or a model file. `--auto` into the backups folder, tested and rotated; `ls`; `schedule`.

```
mnm backup [OPTIONS] COMMAND [ARGS]...
```

| Argument or option | What it does |
|---|---|
| `--to PATH` | The file to write (default: mnm-backup-<time>.tar.gz here). |
| `--auto` | Into MNM_BACKUPS_DIR as mnm-<date>-<time>.tar.gz, restore-tested, then rotated (7 daily, 4 weekly); what the nightly timer runs. |

#### `mnm backup ls`

The backups in MNM_BACKUPS_DIR with their sizes and ages, what the rotation keeps each
as, and the last restore test.

```
mnm backup ls [OPTIONS]
```

#### `mnm backup schedule`

A systemd user timer that runs `mnm backup --auto` every night; no flag shows it.

```
mnm backup schedule [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--at STR` | Every day at this time on the box's clock (03:30). |
| `--off` | Remove the timer. |

### `mnm restore`

Put a backup's files back into this machine's state folder; nothing is planned again
and nothing downloads until an instance's first `mnm start`.

```
mnm restore [OPTIONS] [FILE]
```

| Argument or option | What it does |
|---|---|
| `FILE` | The backup (with --test: the newest in MNM_BACKUPS_DIR unless named). (optional) |
| `--force` | Replace running instances' recipes too. |
| `--test` | Restore into a scratch state folder, read it back, remove it; this machine's state is not touched. Exit 1 when it does not restore. |

### `mnm disk`

What the models folder holds, and the disk policy that keeps it under a limit.

```
mnm disk [OPTIONS] COMMAND [ARGS]...
```

| Argument or option | What it does |
|---|---|
| `--json` | The report as JSON. |

#### `mnm disk keep`

Show the disk policy's limit, or set it (`--models`) and run the policy at once.

```
mnm disk keep [OPTIONS]
```

| Argument or option | What it does |
|---|---|
| `--models STR` | The limit: a share of the models folder's disk (85%, the default), a size of the folder (400G, 1.5T; G is 10^9 bytes, GiB 2^30), or off. |

#### `mnm disk check`

Run the disk policy now: measure, warn, and at the limit evict (what `mnm add`, every
start and the doctor run on their own).

```
mnm disk check [OPTIONS]
```

### `mnm report`

Write report-<time>/ in the current folder: this box's state and logs, secrets removed.

```
mnm report [OPTIONS]
```

### `mnm notify`

Send a test notification; `--ntfy`/`--toast` save the channels in the secrets file.

```
mnm notify [OPTIONS] [TEXT]
```

| Argument or option | What it does |
|---|---|
| `TEXT` | A test message. (optional) |
| `--ntfy STR` | An ntfy topic's URL (https://ntfy.sh/<topic>, or your own ntfy server's); `off` removes it. |
| `--toast STR` | on or off: a Windows toast (WSL only; on there by default). |
