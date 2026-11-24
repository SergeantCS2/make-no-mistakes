# makenomistakes: the guide

This is the one guide to mnm: installing it, running every kind of model it supports, connecting the models to the workspace apps (OpenCode, LibreChat, Open WebUI, OpenHands, Goose, Odysseus) and other apps, changing or removing anything, and uninstalling it. Everything a user needs is on this page, so **Ctrl+F finds it**; [commands.md](commands.md) is the full command reference.

Commands are typed in a terminal: WSL (Ubuntu) on Windows, or a native Debian or Ubuntu machine. `mnm` works from any folder once setup has run.

## Contents

**Getting started**
1. [What mnm does](#1-what-mnm-does)
2. [Install](#2-install)
3. [Native Debian and Ubuntu](#3-native-debian-and-ubuntu)
4. [Check the machine](#4-check-the-machine)
5. [Your first model](#5-your-first-model)
6. [The ideas behind it](#6-the-ideas-behind-it)

**Models**
7. [Every model type, with examples](#7-every-model-type-with-examples)
8. [Pictures in depth](#8-pictures-in-depth)
9. [Video in depth](#9-video-in-depth)
10. [Speech in depth](#10-speech-in-depth)
11. [Coding agents](#11-coding-agents)

**Using them**
12. [Apps](#12-apps)
13. [Other apps](#13-other-apps)
14. [The router](#14-the-router)
15. [Changing an instance](#15-changing-an-instance)
16. [Several models at once: sets, and making things](#16-several-models-at-once-sets-and-making-things)
17. [The web UI](#17-the-web-ui)

**Looking after it**
18. [Updating mnm and the engines](#18-updating-mnm-and-the-engines)
19. [Stopping and removing](#19-stopping-and-removing)
20. [When something goes wrong](#20-when-something-goes-wrong)
21. [Uninstall](#21-uninstall)
22. [Where everything lives](#22-where-everything-lives)
23. [Command reference](#23-command-reference)
24. [What has been checked on real hardware](#24-what-has-been-checked-on-real-hardware)

**Bigger machines**
25. [Large models and many cards](#25-large-models-and-many-cards)

**Running itself**
26. [The nightly](#26-the-nightly)
27. [Your box](#27-your-box)

**Making music**
28. [The studio](#28-the-studio)

---

## 1. What mnm does

You give mnm a Hugging Face (or Civitai) link. It works out what the model is, picks the version and settings that fit your GPU right now, downloads it, and runs it in its own Docker container. Every model then answers at one address, `http://localhost:7801/v1`, under its name: mnm's **router** loads a model when a request needs it. mnm prints the exact URL and the steps to paste it into an app, and installs or connects the workspace apps itself ([12](#12-apps)).

- **Planned for this machine.** mnm's defaults were measured on an RTX 4070 (12 GB) under WSL2. It runs on a native Debian or Ubuntu box too ([3](#3-native-debian-and-ubuntu)), and plans for several cards, an AMD card, a DGX Spark, a server board or no card at all from vendor figures and recorded hardware descriptions until the box has measured itself ([27](#27-your-box)). None of these, a native box included, has been tried on real hardware ([24](#24-what-has-been-checked-on-real-hardware)). Every plan is made against what the GPU and RAM have free at that moment.
- **Official engines in Docker.** Nothing is compiled on your machine; each engine image changes only when you say so (`mnm engines update`). Four optional images are mnm's own, built here when you ask and never pulled ready-made: ComfyUI, from Ubuntu and a pinned release ([18.4](#184-comfyui)), and for the studio its tools image and the two trainers' images, every package and weight pinned by sha256 ([28.3](#283-scores-the-structure-map-stems-and-the-mixer), [28.10](#2810-train-a-voice), [28.11](#2811-teach-ace-step-a-style)).
- **Makes things.** A Run tab per model and a Runs gallery, a Short film pipeline, and ComfyUI workflows imported from comfy.org, Civitai or Hugging Face and run from a form, all kept in one outputs folder ([16.5](#165-making-things-run-runs-and-clean-up), [16.6](#166-pipelines-a-short-film-from-one-idea), [18.4](#184-comfyui)).
- **Makes music.** The studio: a song from a theme, candidates scored to pick from, stems, the vocals cleaned and sung in a voice from your library, edits section by section, a mix, a master and a REAPER project with what made it said; voices and styles trained on this box ([28](#28-the-studio)). Its models have run on the processor only; on a card they are planned from estimates until the box has measured them ([28.12](#2812-the-models-and-your-box)).
- **Explain before refusing.** A model that cannot run here gets an explanation of what it is and which repos to use instead. A model that does not fit is refused with the sizes.
- **Apps pinned, on this machine.** OpenCode, LibreChat, Open WebUI and OpenHands are installed at pinned versions, on 127.0.0.1 only; Goose and Odysseus are connected, never installed. Odysseus's `.env` is changed only after you have seen the diff and said yes ([12.6](#126-odysseus)).

| Type | Engine (Docker image) | What it does |
|---|---|---|
| chat | llama.cpp `server-cuda` (vLLM for safetensors-only chat, coding and vision models) | Text in, text out, tools, thinking |
| coding | llama.cpp | A chat model for coding agents and editors; `--for-agents` plans a 32k context |
| vision | llama.cpp + its projector (`mmproj`) | Chat with pictures in |
| embedding | llama.cpp `--embedding` | Text to vectors: Odysseus's memory and documents |
| rerank | llama.cpp `--rerank` | A query and passages to scores |
| image | stable-diffusion.cpp `master-cuda` | Text to picture: GGUF kits, checkpoints, diffusers folders, single DiT files, LoRAs; repaint and inpaint |
| image edit | stable-diffusion.cpp | Picture plus instruction to picture (FLUX.1 Kontext) |
| video | stable-diffusion.cpp | Text or a first frame to a clip: Wan 2.1/2.2, LingBot, HunyuanVideo 1.5 |
| speech-to-text | whisper.cpp `main-cuda` | Audio to text, OpenAI's transcription route |
| text-to-speech | audio.cpp `full-cuda12` | Text to a WAV: PocketTTS, Qwen3-TTS (Kokoro is refused: the image has no eSpeak-ng) |
| song | audio.cpp | Tags and lyrics to a song: ACE-Step 1.5, HeartMuLa, Stable Audio 3 (instrumentals only); the studio's first stage ([28](#28-the-studio)) |
| stems | audio.cpp | A mix split into stems: HTDemucs (4 or 6), BS-RoFormer and Mel-Band RoFormer; the room dereverb mnm converts ([28.9](#289-clean-and-tune-the-vocals)) |
| voice conversion | audio.cpp | Vocals sung again in a voice from your library: Seed-VC, Vevo2, RVC ([28.5](#285-voices)) |
| music score | audio.cpp | A stem's notes as MIDI: MuScriptor (non-commercial); without one the studio tools' Basic Pitch does it ([28.8](#288-section-tools-tempo-and-key-midi-and-mastering)) |

The tags shown are the ones an NVIDIA card on x86-64 gets. On an AMD card, an ARM64 box or a box without a card, each engine runs the build it publishes for that box, where it has one ([25.10](#2510-backends-cuda-vulkan-rocm-cpu), [25.13](#2513-arm64)).

## 2. Install

### 2.1 The one line

**Under WSL, give it its memory first.** WSL gets half the PC's RAM unless `C:\Users\<you>\.wslconfig` says otherwise; `notepad "$env:USERPROFILE\.wslconfig"` in PowerShell opens it (Notepad offers to create it). Under `[wsl2]`, `memory=` the PC's RAM less 8 GB for Windows, and at least 16 GB, which is what mnm wants; `swap=8GB`:

| PC's RAM | `memory=` | `swap=` |
|---|---|---|
| 16 GB | `16GB` (all of it: WSL takes memory only as it uses it) | `8GB` |
| 32 GB | `24GB` | `8GB` |
| 64 GB | `56GB` | `8GB` |

Then `wsl --shutdown` in PowerShell and open Ubuntu again; `mnm doctor`'s RAM line says what WSL got ([4](#4-check-the-machine)). ComfyUI's Wan 2.2 14B video workflows want more swap, and the doctor says how much when they are installed ([18.4](#184-comfyui)).

On a fresh WSL2 Ubuntu (or a native Debian/Ubuntu box, [section 3](#3-native-debian-and-ubuntu)):

```bash
curl -fsSL https://raw.githubusercontent.com/SergeantCS2/make-no-mistakes/main/install.sh | bash
```

No GitHub account is needed. From an existing clone, `bash ~/makenomistakes/install.sh` does the same.

**Without a terminal** (a script, or an agent's session that runs commands with no tty), the installer cannot ask: sudo must work without a password (`sudo -n true`), and `mnm setup` prints its checks table and **installs nothing** (git, curl, uv and the checkout still come in: the checks need them). To install unattended on the defaults, say so with `MNM_SETUP_YES=1`:

```bash
curl -fsSL https://raw.githubusercontent.com/SergeantCS2/make-no-mistakes/main/install.sh | MNM_SETUP_YES=1 bash
```

It passes `--yes` to `mnm setup`: the checks table installs what is missing (Docker, the NVIDIA runtime, the folders, the secrets file, the engine image), then the web UI and the router are installed as services. On a native box the router stays on this machine and its containers (`MNM_ROUTER_HOST=local`, [3](#3-native-debian-and-ubuntu)) unless an address or a router key is already set: nobody was there to choose your network and make its key. The closing note lists what is left in one order: the re-login (`wsl --shutdown`, or a reboot on a native box), `mnm service install`, `mnm doctor --deep`, then models. At a terminal, `MNM_SETUP_YES=1` runs the guided setup on its defaults without asking.

**A private copy of the repository** asks for a GitHub login, which the installer gets through the GitHub CLI; git then uses that login for every update. Log in with the CLI first (`sudo apt-get update && sudo apt-get install -y gh`, then `gh auth login --hostname github.com --web --git-protocol https`; the installer wants version 2.20 or newer and replaces an older one with GitHub's own), then fetch the script through it, saying where the copy lives:

```bash
gh api -H 'Accept: application/vnd.github.raw' repos/<owner>/<repo>/contents/install.sh | MNM_REPO_URL=https://github.com/<owner>/<repo> bash
```

Under WSL `gh auth login` cannot open a browser ("Failed opening a web browser"): enter the one-time code it prints (`XXXX-XXXX`) at https://github.com/login/device in your Windows browser and wait for "Logged in as …". If it ends with "Too many requests … (slow_down)", GitHub refused the terminal's polling, not the code: `gh auth status` says whether you are logged in; if not, run the login again and wait a few seconds after entering the code.

### 2.2 What the installer does

1. Checks that it runs as your normal user (not root, not under sudo), that `sudo` works (it asks your password once) and that this is an apt system.
2. Installs git, curl and ca-certificates if missing.
3. Installs uv (the Python tool mnm runs with) if missing.
4. Clones the repository into `~/makenomistakes` (or brings an existing checkout up to date) and installs its Python dependencies. The clone needs no account; only when the repository asks for a GitHub login (a private copy) does the installer install the GitHub CLI (from GitHub's own apt repository when the distro's is missing or older than 2.20), log in through it and try again.
5. Runs the **guided setup** (`mnm setup`), below. Without a terminal it runs the checks table instead, which installs only with `MNM_SETUP_YES=1` ([2.1](#21-the-one-line)).
6. Ends with two lines: where the web UI is (or, when setup stopped early, to run `mnm setup` again), and to open a new terminal so `mnm` is found. Without a terminal it ends with the next steps in order instead.

Each step says what it does before it runs, and running the installer again is safe.

### 2.3 The guided setup

Eight sections, each under a ruled heading `[n/8]`. Each one says what it found and asks before it changes anything; Enter takes the default shown in brackets.

| Section | What it does | A typical answer |
|---|---|---|
| 1. This machine | WSL or native, the distro, each card with its PCIe link and what its slot is good for, the driver, RAM (and the bandwidth `mnm doctor --deep` measured), the CPU, the models disk, Docker; then which large models this box can hold ([25.8](#258-what-the-ladder-of-boxes-runs)). On a native box with several cards it asks which ones mnm uses (`MNM_GPU`: `all`, or numbers such as `0,1`; unattended, `all`). On an AMD card that runs both llama.cpp builds it asks `vulkan` or `rocm` (`MNM_BACKEND`; ROCm pulls a 7.7 GB image at step 4; unattended, Vulkan; [25.11](#2511-amd-cards)). On a DGX Spark or an APU it names the one pool instead of VRAM and RAM ("this box has 131 GB of unified memory: the card and the RAM are one") and the large models it holds against it, less 8 GiB kept for the system ([25.12](#2512-unified-memory-the-dgx-spark-and-the-mini-pcs)). Nothing else changes | (nothing asked; several cards on a native box: Enter for `all`) |
| 2. Where models live | The models folder (default `/data/models`) and the free space there, then the outputs folder for what your runs make (default `/data/outputs`). Plan 100 GB for a full set | Enter, Enter |
| 3. Network | Who may reach the router on port 7801. Under WSL's NAT networking nothing is asked: it stays on the PC. On a native box (or WSL's mirrored networking): *this machine and its Docker containers* (recommended) or *your whole network*. Then it offers a **key** for the router (it makes one and shows it once), and a ufw rule when ufw is on | `n` for the key (optional; see [14.4](#144-a-key-for-the-router)) |
| 4. Prerequisites | A table of checks first: Docker Engine, its daemon and group, the NVIDIA container runtime, the models folder, the secrets file, the `mnm` command, the llama.cpp image (about 2 GB; on an arm64 box such as a DGX Spark its arm64 CUDA 13 build, [25.13](#2513-arm64)). On a DGX Spark, DGX OS ships the driver, Docker and NVIDIA's container toolkit, so their rows read OK and nothing is asked. On a native box also the base packages and the **NVIDIA driver**, asked about separately; on an AMD box instead the kernel's amdgpu driver and firmware, the HWE kernel where the card needs it, the `render` and `video` groups, `vulkan-tools` and, on an APU, its GTT kernel parameters ([25.11](#2511-amd-cards)). Then it installs what is missing once you agree; every command is printed before it runs | Enter (yes) |
| 5. Accounts | The Hugging Face token and the Civitai key, both optional. **Hidden as you type**: nothing shows, not even dots. Paste and press Enter, or Enter alone to skip | paste each, or Enter |
| 6. Background services | The web UI (7800) and the router (7801) as systemd user units that start with the machine | Enter (yes) |
| 7. First models | A menu of models that fit your card, with sizes (below). Pick by number or name (`1 5`, `chat memory`, `all`), or `none` | your pick |
| 8. Workspace apps | Optional: OpenCode, LibreChat, Open WebUI, OpenHands, Goose, and Odysseus when `~/odysseus` exists, one paragraph each (what it is, what it needs, its licence, its port). Pick by number or name; for each, *here* (installed by `mnm apps install`, or Goose's file written, or Odysseus's `.env` diff shown and written only on yes) or *remote* (its card for another machine). The projects folder (`MNM_PROJECTS_DIR`, default `~/src`) is asked once, before OpenCode or OpenHands; LibreChat's first account right after it installs; for Open WebUI it says to make the first account on its own page ([12](#12-apps)) | Enter (none), or `1 2` |

The starter menu on a 12 GB card:

| # | Key | Instance(s) | Download | Needs | What |
|---|---|---|---|---|---|
| 1 | `chat` | `qwen3-8b` | ~5.0 GB | 8 GiB | Qwen3 8B: an all-round chat model with tools and thinking (about 76 tok/s on an RTX 4070) |
| 2 | `small` | `tiny` | ~0.4 GB | 2 GiB | Qwen3 0.6B: tiny and fast, for quick tests and small cards |
| 3 | `coding` | `coder8b` | ~5.0 GB (the same file as `qwen3-8b`) | 8 GiB | Qwen3 8B with a 32k context, for Claude Code, Codex, OpenCode and Aider |
| 4 | `coder36` | `coder36` | ~22.1 GB | 12 GiB | Qwen3.6 35B-A3B: a coding agent, 32k a request, with about 10.5 GiB of its experts in RAM (under WSL set `memory=24GB` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`) |
| 5 | `coder38` | `coder38` | ~12.3 GB | 12 GiB | Qwen3.8 27B at 3 bits: the lighter coder, all of it on the card (its MTP head comes with it, used only with `draft=mtp`) |
| 6 | `vision` | `qwen3-vl` | ~6.2 GB | 8 GiB | Qwen3-VL 8B: chat about pictures and screenshots |
| 7 | `memory` | `embed` | ~0.6 GB | 2 GiB | Qwen3 Embedding 0.6B: memory and document search for chat apps |
| 8 | `pictures` | `zimage` | ~10.0 GB | 8 GiB | Z-Image Turbo: text-to-image at 1024x1024 (about 20 s on an RTX 4070) |
| 9 | `speech` | `stt`, `pocket` | ~1.2 GB | 3 GiB | Whisper large-v3-turbo (speech to text) and PocketTTS (26 voices) |
| 10 | `video` | `wan5b` | ~8.0 GB | 10 GiB | Wan 2.2 5B: 2-second clips from text or a picture (2–3 minutes each on an RTX 4070) |

`all` is every row but the large starters below, about 65.8 GB (a file two rows share counts once). *Needs* is the VRAM a row's model wants on the card: a smaller card is offered only what fits it, and `mnm starter` marks a row the card holds but its free VRAM does not right now ("won't fit now: 3.8 GiB free"). Where the cards mnm uses hold 64 GiB together (four 16 GB cards) and the RAM holds them, three large starters follow: `glm47` (GLM-4.7-Flash Q4_K_M, 18.3 GB), `bolt` (Nemotron-3.5-Bolt 74B UD-Q4_K_XL, 50.2 GB) and `ds41` (DeepSeek-V4.1-Flash UD-IQ1_S, 233 GB, 111 GB of it an Engram table on disk; it needs 64 GB of RAM beside those cards and a llama.cpp image with DeepSeek-V4.1). `all` never includes them: pick them by name or number.

> **Section 7 plans against the card as it is at that moment.** Every model is planned against the VRAM free then, and keeps that plan. When another program holds more than 2 GiB of the card, setup says so ("8.2 GiB of the card is in use by Windows and other apps") and offers to look again; what holds the card is your business, mnm only says how much. If you go on, the summary tells you to run `mnm replan --all` later ([15.4](#154-plan-it-again-replan)).

A **summary** closes it: each section marked `[ok]`, `[!!]` (look at it), `[--]` (skipped) or `[XX]` (failed), then *What is left for you*: a reboot or re-login when one is needed, the router's URL for apps, and the next commands.

- **Re-running is safe**, and is how you change an answer: each question shows the current value as its default, and a step already in place is not repeated.
- `mnm setup --dry-run` shows what it would do and asks nothing. `--yes` runs unattended on the defaults (no token, key, model or app is added; a native driver only with `--install-driver`). `--plain` shows only the checks table, which is also what a run without a terminal gets. **Without a terminal, setup installs nothing unless you pass `--yes`**: it prints the checks, says "No terminal, so nothing was installed" and exits 1 when something is missing; `mnm setup --plain --yes` installs what the table finds missing, with nobody to ask.
- **Tokens later.** Add `HF_TOKEN=hf_...` or `CIVITAI_TOKEN=...` to `~/.config/makenomistakes/secrets.env` (mode 600, never in git) at any time; they are read at each request, so nothing needs a restart.

### 2.4 After installing

- **Open a new terminal** so `mnm` is on the PATH (Ubuntu adds `~/.local/bin` at login once it exists).
- **Docker group.** If setup just installed Docker or added you to the `docker` group: under WSL run `wsl --shutdown` from PowerShell and open Ubuntu again; on a native box reboot. Your user and the background services then use Docker without `sudo` (`newgrp docker` only fixes one shell).
- **systemd under WSL.** The services need `[boot]` / `systemd=true` in `/etc/wsl.conf`; setup says so if it is missing.
- **The GPU driver under WSL** is the Windows driver: never install one inside WSL.
- **Shell completion** (optional): `mnm --install-completion` adds it to your shell (bash, zsh or fish); open a new terminal. Then `Tab` completes the commands and their options, and an instance's name after `mnm start`, `stop`, `rm`, `logs`, `show`, `set`, `try`, `run`, `tune`, `replan`, `why`, `open`, `url`, `pin` and the rest.

## 3. Native Debian and Ubuntu

mnm also runs on a plain Debian or Ubuntu install. The commands, the router and the web UI are the same; what differs is the GPU driver, the network, and a few things a minimal install lacks. Setup takes care of each, asking first where it matters.

**What you need:**
- **A distro:** Debian 12 or 13, or Ubuntu 22.04 or 24.04. Derivatives work through their base: Linux Mint, Pop!_OS, Zorin, KDE neon. (Raspberry Pi OS is recognised as Debian, but it is ARM: setup installs Docker there and has no driver route.)
- **A machine:** x86-64 or ARM64. The driver route on Debian is amd64 only; DGX OS on a DGX Spark brings its own driver ([25.13](#2513-arm64)).
- **A graphics card:** an NVIDIA card, Turing (RTX 20xx) or newer for NVIDIA's open driver on Debian (Ubuntu's driver also covers older cards), with 8 GB of VRAM or more for the starter chat model. An AMD card runs on Vulkan or ROCm ([25.11](#2511-amd-cards)). With no card the processor runs the models, slowly, except stable-diffusion.cpp's pictures and clips, which have no CPU build ([25.10](#2510-backends-cuda-vulkan-rocm-cpu)).
- **RAM and disk:** 16 GB of RAM or more, and 100 GB free for a full set of models.
- **sudo** for your user. On a bare Debian, as root: `apt-get install -y sudo && usermod -aG sudo <you>`, then log in again.

**Install** with the line of [2.1](#21-the-one-line); a bare box may lack curl: `sudo apt-get update && sudo apt-get install -y curl` first. If you cannot fetch the script, get the repository onto the machine any other way and run `bash ~/makenomistakes/install.sh`.

On a native box setup also:
1. asks **who may reach the router**, offers a key, and offers the ufw rule when ufw is on (section 3);
2. installs the base packages a minimal image lacks (`ca-certificates curl git gnupg openssl`);
3. asks to install the **NVIDIA driver** when none answers:
   - **Ubuntu:** `sudo ubuntu-drivers install nvidia:580` (Ubuntu's signed driver, R580; the recommended one when 580 is not in the archive);
   - **Pop!_OS:** `sudo apt install system76-driver-nvidia`;
   - **Debian:** NVIDIA's CUDA repository (`cuda-keyring`), then `nvidia-open`, built by DKMS against your kernel's headers. When Debian no longer carries the headers of the kernel you run, it installs the current kernel and its headers instead, and the reboot boots into it;
4. installs **Docker Engine** from Docker's repository for your base distro and the **NVIDIA Container Toolkit**, and adds you to the `docker` group;
5. offers the services and first models; both wait until after the reboot if the driver was just installed.

**Then reboot** if it installed the driver or Docker, and run `mnm setup` again. It finds everything in place and goes on to the services and first models.

**Secure Boot.** A module built by DKMS must be signed by a key your firmware trusts; `mokutil --sb-state` says whether Secure Boot is on.
- **Ubuntu:** the prebuilt driver is signed and loads as is. When it builds by DKMS instead, the install asks for a one-time password. At the next boot a blue *MOK management* screen asks for it: pick **Enroll MOK**, type the password, reboot. The key is `/var/lib/shim-signed/mok/MOK.der`; `sudo mokutil --import` it to try again.
- **Debian:** DKMS signs the module with `/var/lib/dkms/mok.pub` but does not enroll that key. Before the reboot, run `sudo mokutil --import /var/lib/dkms/mok.pub`, choose a password, then pick **Enroll MOK** at the blue screen.

If you skip the enrollment the driver does not load, and `mnm setup` says so. You can also turn Secure Boot off.

**The network.** Under WSL's NAT networking `0.0.0.0` stays on the PC; on a native box it is your whole network. Setup asks who may reach the router (port 7801):
- **This machine and its Docker containers** (`MNM_ROUTER_HOST=local`, recommended). The router listens on 127.0.0.1 and on Docker's bridge gateway (172.17.0.1 by default), which is what `host.docker.internal` means inside a container. Another machine reaches that only with a route to Docker's network; a key closes even that.
- **Every machine on your network** (`MNM_ROUTER_HOST=0.0.0.0`). Set a key (`MNM_API_KEY`); setup makes one, and clients send it as their API key.

Unattended (`mnm setup --yes`, or install.sh's `MNM_SETUP_YES=1` with or without a terminal), setup takes the recommended answer unless `MNM_ROUTER_HOST` or a router key is already set; then the router stays where it is. Either way setup saves the address, so a later run starts from it. To change it later, edit `MNM_ROUTER_HOST` in `~/.config/makenomistakes/secrets.env` and run `mnm service install`.

**ufw.** When ufw is on, it drops what containers send to the host, so Odysseus would never reach the router. Setup offers the one rule that lets Docker's networks in, `sudo ufw allow from 172.16.0.0/12 to any port 7801 proto tcp`, and `uninstall.sh` deletes it again, but only when setup added it.

**host.docker.internal.** Docker Desktop defines it for every container; Docker Engine (WSL and native alike) only when the container is started with `--add-host=host.docker.internal:host-gateway` (compose: `extra_hosts: ["host.docker.internal:host-gateway"]`). If a client container cannot resolve the name, add that line to its compose file.

| | WSL2 | Native |
|---|---|---|
| GPU driver | the Windows one (never install one inside WSL) | installed by `mnm setup`, asked first; reboot |
| Restart after Docker's group | `wsl --shutdown` | reboot |
| systemd | `[boot] systemd=true` in `/etc/wsl.conf` | already there |
| Router on `0.0.0.0` | this PC only (NAT) | your network: use `local` or a key |
| Firewall | Windows' | ufw, when on: setup offers the rule |
| RAM | `memory=` in `.wslconfig`: what WSL has is what it gives (`memory=24GB` shows as 23.5 GiB; `mnm doctor`'s RAM line says "24 GB by .wslconfig"). More buys large models: at 24 GB a box with an RTX 4070 holds GLM-4.7-Flash, Nemotron Bolt UD-Q2_K_XL and, with 1 to 2 GB to spare, Bolt UD-IQ3_XXS and Mistral-Small-4 UD-IQ1_M (36.4 and 37.0 GB of RAM and VRAM together; [25.8](#258-what-the-ladder-of-boxes-runs)). Leave Windows 8 GB or more | the machine's own |
| Disk back after uninstall | the PowerShell shrink script | nothing to do |

**When something is wrong on a native box:** `mnm doctor` names the platform's own remedy for each check. If `nvidia-smi` fails after the reboot, `mnm setup` tells a driver that is installed but not loaded (a reboot, or Secure Boot enrollment) from one that is missing, and `dmesg | grep -i nvidia` shows why the module refused. A driver older than 580 runs chat, embeddings and pictures but not speech-to-text: on Ubuntu, `sudo apt install nvidia-driver-580`, then reboot. Docker's "permission denied": reboot once after setup added you to the `docker` group.

## 4. Check the machine

```bash
mnm doctor          # GPU, VRAM, RAM, the network and the router, Docker, NVIDIA runtime, disk, token
mnm doctor --deep   # also runs a tiny GPU container (about 100 MB, once) and measures the RAM's bandwidth
```

Every line should say OK or WARN, and the command exits 0. A FAIL line says what to do. The checks:

| Check | OK means | WARN or FAIL, and what to do |
|---|---|---|
| NVIDIA GPU | the card, its VRAM and driver | no driver: the Windows driver (WSL) or `mnm setup` (native); whisper needs driver 580+ |
| Card (AMD) | each AMD card's name, gfx target, VRAM or an APU's GTT, and the backend ([25.11](#2511-amd-cards)) | "Vulkan only" where llama.cpp's ROCm image is not built for it |
| AMD driver, Kernel, AMD firmware | amdgpu loaded, a kernel new enough for the card, no firmware failed | FAIL: `mnm setup` (firmware, the HWE kernel); the kernel log may need `sudo dmesg` |
| AMD device files | `/dev/kfd` (ROCm) and the render nodes open | FAIL: `sudo usermod -aG render,video $USER`, or a reboot when the groups were just added |
| Card N BAR, APU memory, AMD cards | Resizable BAR on; an APU's GTT at least half the RAM; each AMD slot's link | WARN: the BIOS setting; the GTT parameters `mnm setup` writes; a card on chipset lanes |
| VRAM in use | what other programs hold is under 2 GiB | **over 2 GiB**: another program holds it. Models started now have that much less of the card; anything added meanwhile was planned around it: `mnm replan --all` when more of the card is free |
| RAM | 16 GiB or more; under WSL what `.wslconfig` gave ("24 GB by .wslconfig") and, when the PC has more, which large models more would hold | raise `memory=` in `.wslconfig` (WSL) |
| Cards in use | (several cards) `MNM_GPU` names the ones mnm uses | WARN while it is unset: mnm uses card 0 only. `MNM_GPU=all` in the secrets file, `mnm service install`, `mnm replan --all` |
| GPU N link | (several cards, native) each card's PCIe link now and at most, and what its slot is good for: the main card on the CPU's lanes, a chipset x4 slot fine for layers and slow for prompt processing | WARN for an x1 riser: layers only (a riser with more lanes would let it hold experts). Under WSL one *PCIe links* line: not readable |
| Main card | (only when wrong) | WARN when the largest card sits on a narrower link than a smaller one: it names the slot to swap it into |
| CPU | cores, threads, NUMA nodes and CCDs (lscpu); the thread counts `mnm tune` tries | — |
| RAM slots | the DIMMs and channels (dmidecode, EDAC or lshw, without sudo) | "unknown without sudo" on most boards (`sudo dmidecode -t memory` lists them); WARN for one channel |
| RAM platform | the row the box's RAM sits on (WSL; a DDR4 or DDR5 desktop, one-CCD Zen apart; a 4-channel workstation; an 8-channel DDR4 EPYC or Threadripper PRO; 8-12 channels of DDR5) and how plans price RAM-held experts now: "priced at 50 GB/s for an EPYC 7532 with 4 CCDs until this box measures", then this box's probe, then its own readings ([25.14](#2514-big-ram-epyc-threadripper-and-intel)) | — |
| RAM channels, Huge pages | (a workstation or server) a DIMM in every channel; transparent huge pages not `never` | WARN for 6 of 8 channels (25% less), for a CPU whose CCDs read less than its channels, for THP `never` |
| NUMA, NUMA balancing, Two sockets, NUMA settings | (several NUMA nodes) the nodes a socket (NPS), each node's CPUs, cores and memory; the page-cache drop `numa=distribute` wants, as a command (root's, never run) | WARN for `numa_balancing=1` (`sudo sysctl -w kernel.numa_balancing=0`), for two sockets (slower than one in mainline llama.cpp), and for each instance keeping weights in RAM with no `numa` setting (`mnm set <name> numa=auto`) |
| Server board | (a server maker's board) what sysfs reads of the BIOS checklist: above-4G decoding, NPS, each card's link | the rest as a remedy: risers at Gen3, PSB locks, BMC fan thresholds (`ipmitool`, never run) |
| Experts in RAM (`--deep`) | each saved model that keeps experts in RAM, what it reads a token there, and the speed the measured bandwidth allows at best | — |
| WSL networking / Network | who can reach the router | mirrored networking puts it on your LAN: a key, or `MNM_ROUTER_HOST=local` |
| Docker, Docker NVIDIA runtime | the engine and the runtime | `mnm setup` installs them |
| Model storage | 50 GiB or more free | free space, or move `MNM_MODELS_DIR` |
| Hugging Face token | set (gated repos) or not (public only) | optional |
| Router | answers on 7801 | `mnm service install` |
| API key | set, or not needed here | advised when the router is on your LAN |
| Router address | (only shown when the running router uses another `MNM_ROUTER_HOST` than the secrets file) | `mnm service install` |
| Firewall | (native, ufw on) mnm's rule is there | the `ufw allow` line above |
| Docker GPU smoke test (`--deep`) | a container saw the GPU | the NVIDIA runtime is not wired into Docker: `mnm setup` |

The web UI's System → Doctor tab shows the same table.

**One screen: `mnm status`.** Once models run, `mnm status` says how the whole box is doing in a dozen lines: the card, the health verdict (the router's `/__mnm/health`, [13.8](#138-outside-monitors); *down* when no router answers), the doctor's worst lines (every FAIL before any WARN, at most three; `mnm doctor` has the rest and what to do), the models by state (loaded with what each holds, parked, stopped by itself with its exit code, off, not on disk), what waits in the router's queue, the tuner and the trainer, the last nightly's word, the web UI's address and, last, whether the hardware changed since plans were made. It exits 1 when the router is down or a doctor check fails, as `mnm doctor` does.

```bash
mnm status          # the box on one screen
mnm open            # the web UI in this machine's browser (17.1)
```

## 5. Your first model

The starter menu ([2.3](#23-the-guided-setup)) is the quickest way: `mnm starter` shows it again any time, and `mnm starter chat memory` adds those two. By hand:

```bash
mnm fit unsloth/Qwen3-8B-GGUF                         # what it is and what fits; nothing downloads
mnm add unsloth/Qwen3-8B-GGUF --name qwen3-8b         # plans, saves, downloads (about 5 GB)
mnm start qwen3-8b                                    # starts; prints the URLs once it answers
mnm try qwen3-8b                                      # sends a test question, prints the answer
mnm stop qwen3-8b                                     # frees the GPU; recipe and files stay
```

You do not have to start a model yourself: the first request to the router loads it ([14](#14-the-router)).

**Finding one.** `mnm what` searches for models and grades each against this box with the planner, exactly as its `mnm add` would plan it: whether it fits, at which quant, how fast, what it needs on the card and in RAM, and the line that adds it. Nothing downloads; the planner reads each header over a few ranged requests (a few seconds for ten rows).

```bash
mnm what --for coder                    # the curated picks for a use, three of them with why
mnm what qwen3 coder --type coder       # a search: the curated rows first, then each source's
mnm what flux --type pictures           # Hugging Face's and Civitai's checkpoints
mnm what qwen3 --vram 24 --ram 64       # graded for another machine, as `mnm fit --vram`
```

- **The order is never mnm's own invention:** first the curated rows (the starter menu's models for the use, all of them run on an RTX 4070, each with its line), then each source in the order it gives: Hugging Face's search (most downloaded first; GGUF repos for chat, coding, vision and memory, the pipeline's tag for pictures and clips), Civitai's checkpoints for pictures (most downloaded, safe-for-work only), and Ollama's library for chat, coder (its *tools* models) and memory (its *embedding* models); not for vision, since mnm runs none of Ollama's projector layers. A coder search keeps the Hub's vision repos too: added for agents, they run text-only. `--for` gives three of the curated rows, sorted by how well they fit: comfortable first, then tight, then offload; a search's hits are never picks, and words with `--for` are refused (search them with `--type`).
- **A quant below Balanced's:** when Balanced's quant does not fit (a coder at the agent's 32k context on a small card), the quants below it are tried, the most faithful first, and the row names the first that fits (*smaller quant*) with `--quant` in its line; a curated row's own quant is never stepped down.
- **Uses:** `chat`, `coder` (graded with the agent profile's context, and its line says `--for-agents`), `vision`, `pictures`, `video`, `memory`; `voice` and `ear` are the curated rows only. A hit the planner reads as another type is left out and counted.
- **A row that does not fit** says why in the planner's words below the table: *does not fit* (no quant fits this box, the RAM beside the card included), *refused* (nothing here runs it: the planner's reason follows), *cannot read* (the source did not answer, refused access, or gave a file whose header cannot be read). Civitai lets a header be read only with a token (`CIVITAI_TOKEN`, [7.12](#712-models-from-civitai-the-civitai-archive-lm-studio-ollama-and-modelscope-and-odd-repo-layouts)). A source that does not answer is one line, and the others still show.
- **The web's Add page** has the same search (*Find a model*): each row that fits has a **Plan it** button that opens its plan with the graded quant, the agent profile and the curated name already chosen, so adding it with the defaults adds what the row graded. With a use and no words it lists that use's curated rows, as `--type` does.

Any of these forms works as a reference: `https://huggingface.co/org/repo`, `org/repo`, `org/repo:Q5_K_M` (one quant), `org/repo/file.gguf` (one file), `org/repo/sub/folder` (a folder in a bigger repo), a Civitai link ([7.12](#712-models-from-civitai-the-civitai-archive-lm-studio-ollama-and-modelscope-and-odd-repo-layouts)).

## 6. The ideas behind it

- **Instance.** One model with its settings, under a name you choose (`--name`, or one taken from the repo). Each instance has its own container; clients reach it through the router by its name, and its engine's own port is on 127.0.0.1 only.
- **Recipe.** The saved plan of an instance: which files, which settings, which port. `mnm show <name>` prints it, with the exact `docker run` command. Change settings with `mnm set`, never by editing the command.
- **Profiles.** `mnm fit` and `mnm add` offer three:
  - **quality**: the best weights that fit comfortably (never below balanced's quant);
  - **balanced**: the usual choice, and the default;
  - **speed**: lighter weights, a short context, or fewer steps for pictures.

  `--profile quality` picks another. `--quant Q5_K_M --ctx 16384 --kv q8_0` picks exact settings yourself.
- **Fit levels.** Each profile says *comfortable* (room to spare), *tight* (fits, little headroom), *offload* (some layers or weights stay in RAM: slower) or *no_fit*.
- **Planned against free VRAM, and kept.** A plan uses the VRAM free when you run `add`. VRAM held by mnm's own running instances counts as free, and the plan names them ("Planned as if qwen3-8b were stopped"). VRAM held by anything else (Windows, Odysseus's own models, whatever else you run) does not, and when that is more than 2 GiB the plan says so in yellow. **The saved recipe keeps that plan** until you plan it again (`mnm replan`), so add models when the card is as free as it will be.
- **Fits now.** `mnm ls` and the web UI give each stopped instance a verdict against the VRAM free right now:
  - *fits*;
  - *near limit*: it starts, with little room to spare;
  - *won't fit*: the verdict names mnm's running instances to stop first;
  - *on request*: a parked model that would not fit now, but the router makes its room: its first request parks what the verdict names (none pinned, none in a running set) and loads it. A start by hand still asks for `--stop-others`;
  - *takes turns*: a member of a swap set whose group-mates hold the card ([16.4](#164-taking-turns---swap));
  - *unknown*: not downloaded, a file unreadable, or Docker not answering;
  - a running instance reads what it holds by its estimate instead, *loaded, 5.1 GiB*.
- **States.** *ready* (loaded and answering), *parked* (added, not loaded: the next request loads it), *stopped* (you stopped it: it stays off until `mnm start`), *exited (code)* (the engine died), *not downloaded*, *evicted* (the disk policy deleted its files: the next start or request downloads them again, [19.1](#191-disk-what-the-models-folder-holds-and-the-disk-policy)), *starting*.
- **Ready.** An instance shows as ready only when its engine answers its health check. Starting a large model can take a minute.
- **Ports.** Each engine listens on 127.0.0.1 at a port from its type's block; clients use the router on 7801. A full block spills into the shared 7870–7899. The web UI is on 7800.

  | Type | Ports |
  |---|---|
  | chat, coding, vision | 7810–7819 |
  | image, image edit | 7820–7829 |
  | speech-to-text, text-to-speech | 7830–7839 |
  | embedding, rerank | 7840–7849 |
  | video | 7860–7869 |
  | song | 7900–7909 |
  | stems | 7910–7919 |
  | voice conversion | 7920–7929 |
  | music score | 7930–7939 |

## 7. Every model type, with examples

`mnm fit <ref>` first is always safe: it reads the model's header over the network and downloads nothing. Readings below were measured on an RTX 4070 (12 GB) under WSL2 ([24](#24-what-has-been-checked-on-real-hardware)).

### 7.1 Chat

```bash
mnm add unsloth/Qwen3-8B-GGUF --name qwen3-8b          # about 5 GB; 75–80 tokens/s on an RTX 4070
mnm add unsloth/Qwen3-0.6B-GGUF --name tiny            # a small one for quick tests (~300 tokens/s)
mnm start qwen3-8b && mnm try qwen3-8b --speed         # prints the decode speed
```

**vLLM for safetensors-only models.** A chat, coding or vision model that exists only as safetensors (no GGUF anywhere in the repo) runs on vLLM instead. vLLM takes a fixed share of the card, sized so one full-context request fits; the plan sets that share from the VRAM free now. A plan never asks for a share that would leave less than 0.5 GiB free at the peak beside what other programs hold now (Windows, a game) and the ~0.35 GiB vLLM keeps outside its share: on an RTX 4070 with Windows holding 1.4 GiB, that is at most 81% of the 12 GiB card (a share of 82% peaked 0.42 GiB from the top); a context that would need more becomes the next one down.

```bash
mnm fit Qwen/Qwen3-8B-AWQ          # a safetensors-only AWQ build: plans on vLLM
mnm add Qwen/Qwen3-8B-AWQ --name qwen-awq
```

A repo with a GGUF stays on llama.cpp. Repos that need their own Python code (`auto_map` in `config.json`) are refused: mnm never runs a repository's code. Claude Code and llama.vscode need llama-server's own APIs, so their cards on a vLLM instance point to a GGUF build instead.

**Mixture-of-experts models** such as `unsloth/Qwen3-30B-A3B-GGUF --quant Q3_K_M` are larger than the GPU. mnm keeps the routed experts of some layers in RAM (`cpu-moe-layers`) and every block's attention on the GPU: slower than an 8B model, but much faster than leaving whole layers in RAM. The fit table says "experts of N/L layers in RAM". On an RTX 4070, Q3_K_M with the experts of 28 of 48 layers in RAM decodes at 28.6 tok/s (the RAM-held experts are computed on the CPU), and the estimate is fitted to that reading. llama.cpp suggests `--load-mode none` for this case; it made no difference there, so mnm keeps llama.cpp's default (`mnm set <name> load=none` tries it).

### 7.2 Coding (for coding agents)

```bash
mnm add unsloth/Qwen3-8B-GGUF --for-agents --name coder8b
```

`--for-agents` saves it as type *coding* with a 32,768-token context (or the model's own, when smaller) and tool calling on. A model named for code (Qwen3-Coder, Devstral, Codestral …) is the coding type by itself; `--type chat` or `--type coding` overrides it. The agents are in [section 11](#11-coding-agents).

### 7.3 Vision (chat about pictures)

```bash
mnm add Qwen/Qwen3-VL-8B-Instruct-GGUF --name qwen3-vl
mnm add unsloth/gemma-3-4b-it-GGUF --name gemma-vision
mnm start qwen3-vl && mnm try qwen3-vl                 # asks about a small test picture
```

The vision projector (`mmproj`) downloads with the model. Qwen-VL instances pass `--image-min-tokens 1024` (the model needs it; `mnm set <name> image-min-tokens=…`).

### 7.4 Embeddings (Odysseus's memory and documents)

```bash
mnm add Qwen/Qwen3-Embedding-0.6B-GGUF --name embed
mnm start embed && mnm try embed                       # prints the vector size (1024)
```

An embedding model computes something for every input token, so its working memory grows fast with the context: balanced plans 2,048 tokens (Odysseus sends at most 900 characters), quality 8,192.

### 7.5 Rerank

```bash
mnm add ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF --name rerank
mnm start rerank && mnm try rerank                     # ranks "fruit" above "car" for "apple"
```

Odysseus has no reranker setting. Use the curl or OpenAI SDK card (`mnm url rerank --for curl`).

### 7.6 Image (text-to-image)

```bash
mnm add unsloth/Z-Image-Turbo-GGUF --name zimage       # about 10 GB with its parts; 1024x1024 in ~20 s
mnm add leejet/FLUX.2-klein-4B-GGUF --name klein       # 4 steps
mnm add LyliaEngine/waiIllustriousSDXL_v170 --name wai # an SDXL (Illustrious) checkpoint, 6.9 GB
mnm add stable-diffusion-v1-5/stable-diffusion-v1-5 --name sd15
mnm add circlestone-labs/Anima --name anima
mnm start zimage && mnm try zimage                     # saves zimage.png in the current folder
```

A picture model has parts: the diffusion model, the text encoders and the VAE. mnm fetches the parts each family needs from known repos, and `mnm show <name>` lists them with sizes. Everything else about pictures is in [section 8](#8-pictures-in-depth).

### 7.7 Image edit

```bash
mnm add QuantStack/FLUX.1-Kontext-dev-GGUF --name kontext   # an instruction editor
mnm start kontext && mnm try kontext --image photo.png      # "make it night" style edits
```

Any picture model can also repaint and inpaint ([8.5](#85-repaint-and-inpaint)).

### 7.8 Video

```bash
mnm add QuantStack/Wan2.2-TI2V-5B-GGUF --name wan5b          # about 8 GB with its parts
mnm start wan5b && mnm try wan5b                             # a 2-second clip, wan5b.webm (2–3 minutes)
mnm try wan5b --image picture.png                            # image-to-video: the picture comes alive
```

The clip has 33 frames at 16 fps unless you change it (`mnm set wan5b frames=81`). Odysseus has no video endpoint type, so clips are made from the engine's own page (`http://localhost:7801/upstream/wan5b/`), the curl card, or its **Run** tab. More in [section 9](#9-video-in-depth).

### 7.9 Speech-to-text

```bash
mnm add ggerganov/whisper.cpp --quant large-v3-turbo-q8_0 --name stt
mnm start stt && mnm try stt --audio clip.wav                # prints the words
```

`--quant` picks the size: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, with an optional `.en` and `-q5_0`/`-q8_0` ending. More in [section 10](#10-speech-in-depth).

### 7.10 Text-to-speech

```bash
mnm add audio-cpp/audio.cpp-gguf/PocketTTS-GGUF/english --name pocket   # 26 voices, about 280 MB
mnm start pocket && mnm try pocket --say "The lighthouse keeper lit the lamp at dusk."   # pocket.wav
```

`mnm show pocket` lists the voices. More in [section 10](#10-speech-in-depth).

### 7.11 What mnm refuses, and why

- A safetensors model that needs its own Python code (`trust_remote_code`), or one whose weights are pickle files (`.ckpt`, `.pt`, `.bin`): both can run code when loaded. The studio reads two kinds of checkpoint itself and runs none of their code: the room dereverb ([28.9](#289-clean-and-tune-the-vocals)) and RVC voices ([28.5](#285-voices)).
- A bare Ollama tag such as `qwen3:8b`: the refusal lists the Hub's GGUF repos of that model (an `ollama.com` link works, [7.12](#712-models-from-civitai-the-civitai-archive-lm-studio-ollama-and-modelscope-and-odd-repo-layouts)).
- A LoRA meant for a chat model.
- A picture repo that holds only part of a model (a bare UNet, a ControlNet), or a single file that bundles its text encoders and VAE.
- A model whose smallest version still does not fit the GPU and RAM. When another program holds the card, the refusal says how much, and that the model may fit when more of the card is free.

Each refusal says what the thing is and what to use instead, often with a list of repos that do run (the web Add page has a *Check fit* button beside each).

### 7.12 Models from Civitai, the Civitai Archive, LM Studio, Ollama and ModelScope, and odd repo layouts

Paste a Civitai link the way you paste a Hugging Face one. Checkpoints and LoRAs work; the file's own header decides the family, whatever the page says:

```bash
mnm fit "https://civitai.com/models/<id>?modelVersionId=<v>"   # the address of the version's page
mnm add "https://civitai.com/models/<id>?modelVersionId=<v>" --name civ
mnm lora add civ "https://civitai.com/models/<lora id>"        # a LoRA; prints its trigger words
```

- `civitai.com`, `civitai.green` and `civitai.red` links, download links and `civitai:<id>[@<version>]` all work. `<id>` and `<v>` stand for the numbers in the page's address (`https://civitai.com/models/12345/name?modelVersionId=67890`): paste the address itself, in quotes. Without `modelVersionId`, the newest version is taken.
- Some downloads need a login: create an API key on civitai.com (Account settings → API keys) and add `CIVITAI_TOKEN=<key>` to `~/.config/makenomistakes/secrets.env` (setup's section 5 does it). It is read at each request.
- Civitai's NSFW flag filters nothing: you pick what you paste.
- Files land under `/data/models/civitai/<version>/` and resume when a download is cut.
- **The Civitai Archive** (`civitaiarchive.com/models/<id>?modelVersionId=<v>`) keeps Civitai's ids, so its links work like Civitai's. A model Civitai has deleted is planned from the archive's copy on Hugging Face, and only when the Hub's checksum of that file matches the one the archive recorded. The plan says so ("Civitai no longer has …: planned from the Civitai Archive's copy on Hugging Face, …"). A deleted model whose only copies are elsewhere (tensorfiles, torrents) is refused, naming them. This works for `mnm lora add` too.

**LM Studio's catalog** (`https://lmstudio.ai/models/<owner>/<name>`, or `lmstudio.ai/<owner>/<name>` as `lms get` takes it): each entry lists its builds, and mnm plans the first GGUF one, a Hugging Face repo, so everything else is as for any Hub model. The plan names the repo ("LM Studio's google/gemma-4-e2b is lmstudio-community/gemma-4-E2B-it-GGUF on Hugging Face").

```bash
mnm fit https://lmstudio.ai/models/google/gemma-4-e2b   # plans lmstudio-community/gemma-4-E2B-it-GGUF
mnm fit lmstudio.ai/models/gemma-4                      # a family page: lists its models to pick from
```

A model LM Studio offers only as MLX (Apple silicon's format) is refused with that reason.

**Ollama's library** (`https://ollama.com/library/<name>:<tag>`, or `ollama.com/<user>/<name>:<tag>`; no tag means `latest`): mnm reads the tag's manifest from Ollama's registry and plans its model layer, a GGUF, like any other: the quant and the context come from its header. The download is the one layer, kept under `/data/models/ollama/` only when its sha256 is the digest Ollama names it by. A tag whose model needs a projector (vision) or an adapter layer is refused with that reason: paste the model's Hugging Face GGUF repo instead. A bare `qwen3:8b`, without `ollama.com`, is not read from Ollama: mnm lists the Hub's GGUF repos of that model.

**ModelScope** (`https://modelscope.cn/models/<org>/<model>`; a file's own link picks that quant): its API answers like the Hub, so a GGUF repo is planned the way a Hugging Face one is, its header read by range. Only GGUF repos: a safetensors model is pointed at its Hugging Face copy. Files land under `/data/models/modelscope/` and are kept only when their sha256 is the one ModelScope lists.

```bash
mnm fit https://ollama.com/library/qwen3:0.6b
mnm add https://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF --quant Q8_0
```

**Where huggingface.co is blocked:** `hf-mirror.com` links read as the Hub's, and `HF_ENDPOINT=https://hf-mirror.com` sends every Hub question and download there. mnm reads it from your shell's environment only, not from the secrets file: `export HF_ENDPOINT=https://hf-mirror.com` (in `~/.bashrc` to keep it), then `mnm service install` so the web UI and the router carry it too. Your `HF_TOKEN` is never sent to a mirror: a gated model needs huggingface.co itself.

Repos laid out unusually work too: a GGUF or a transformers model folder at any depth, a repo holding several model folders (each is a version: `mnm fit <repo>` names them, `--version <folder>` picks one), and a LoRA shipped beside a checkpoint (`mnm lora add <instance> <repo>` finds the small file).

## 8. Pictures in depth

### 8.1 The pieces

- **Checkpoint (base model).** The whole model: it turns a prompt into a picture. Stable Diffusion 1.5 and SDXL are base models; Illustrious, Pony and NoobAI are fine-tunes of SDXL and behave like it. `LyliaEngine/waiIllustriousSDXL_v170` is one of these, in one file.
- **Family.** Which base a checkpoint descends from (SD 1.x, SDXL, FLUX …). mnm reads it from the file's header, not from the name or tags.
- **Two layouts, same weights.** A single file, or a **diffusers folder** (`unet/`, `text_encoder/`, `text_encoder_2/`, `vae/`; Hugging Face's library layout, read directly by the engine).
- **UNet / DiT.** The part that paints, step by step (newer families use a transformer, a DiT).
- **Text encoders.** Turn the prompt into numbers the painter understands. SDXL has two.
- **VAE.** Turns the finished result into pixels. SDXL's own VAE breaks in 16-bit, so mnm adds the standard "fp16-fix" VAE for every SDXL model.
- **LoRA.** A small add-on (20–300 MB) trained on one base family for a style, a character or a concept, applied at a **weight** (1.0 full, 0.5 half). It works only on its family: an SDXL LoRA works on any SDXL checkpoint (Illustrious and Pony included), not on SD 1.5. Many list **trigger words**: put them in the prompt.

### 8.2 Versions

- A repo with several checkpoint files (`…_v150.safetensors`, `…_v170.safetensors`) treats each as a version. `mnm add` takes the highest version number; `mnm fit` lists the others.
- To pick one, paste its file link (`mnm add org/repo/model_v150.safetensors`), or use `--version`. The web Add page has a version picker.
- A repo whose folders each hold a model (an image GGUF repo with `TURBO/` and `BASE/`) offers each as a version.

### 8.3 Every setting, and where it lives

Settings on the instance (`mnm set wai key=value`, or the instance page) are the defaults for every picture, Odysseus's included. The engine page can change any of them for one picture.

| Setting | What it does | Starting point (Illustrious/SDXL) |
|---|---|---|
| Prompt | What to draw. Illustrious works best with comma-separated tags (`1girl, solo, long hair, …`) | per picture |
| Negative prompt (`negative-prompt`) | What to avoid | from the model card: `bad quality,worst quality,worst detail,sketch,censor`; long negatives blur |
| Size (`width`, `height`) | Pixels, multiples of 16. SDXL is trained near one megapixel | 1024x1024, 832x1216 (portrait), 1216x832; SD 1.5: 512x512 |
| Steps (`steps`) | Refinement passes; more is slower and finer, up to a point | 20–30 |
| CFG (`cfg-scale`) | How strictly to follow the prompt; too high looks burnt | 5–7 |
| Sampler / scheduler | The maths of each step: `euler_a` is safe, `dpm++2m` with `karras` sharper | euler_a |
| Seed (`seed`) | Same seed and settings, same picture; empty means random each time | empty |
| CLIP skip (`clip-skip`) | Which text-encoder layer to read; the engine already uses 2 for SDXL | empty |
| Repaint strength (`strength`) | How much a repainted picture changes (8.5): 0.3 keeps its layout, 1.0 redraws it | empty (0.75); 0.4–0.6 to keep the composition |
| Guidance (`guidance`), flow shift (`flow-shift`) | Distilled guidance (FLUX) and the sampler's shift (Wan, some DiTs) | from the family |
| Client size (`client-size`) | `on` lets a client's requested size win over the instance's | off |
| Where things live (`offload`, `vae-tiling`, `text-encoder-cpu`, `vae-cpu`, `diffusion`) | Streamed weights, a tiled decode, encoders on the CPU, a DiT converted to q8_0 while loading | from the plan |
| Batch, hires fix | Pictures per request; generate, upscale, refine (engine page only) | 1; off |

Odysseus's Quality picker does not reach this engine: steps are the instance's own (`mnm set zimage steps=8`).

### 8.4 LoRAs

```bash
mnm lora add wai https://huggingface.co/<org>/<lora-repo>              # weight 1.0, picked per picture
mnm lora add wai <org>/<repo>/<file>.safetensors --weight 0.7 --always  # every picture, Odysseus's too
mnm lora add wai "https://civitai.com/models/<id>"                     # from Civitai
mnm lora ls wai
mnm lora set wai <name> --weight 0.5 --not-always
mnm lora rm wai <name>                                                 # the file stays downloaded
```

- mnm reads the LoRA's header before downloading and refuses one made for another family (SD 1.x, SD 2.x, SDXL, the FLUX families, Anima). When the header does not say, it adds the LoRA with a warning.
- An added LoRA appears on the engine page at once; pick it and its strength per picture there. An *always* LoRA applies to every request (Odysseus's prompts cannot pick a LoRA) from the next start.
- The instance page has the same controls under its **LoRAs** tab. A LoRA is saved only once its download has finished, so a failed download changes nothing.
- The instance's LoRA folder (`<models>/loras/<instance>/`) may also hold your own files, or relative links to LoRA files elsewhere under the models folder. mnm only replaces or removes the links it made.

### 8.5 Repaint and inpaint

Every image instance also takes a picture in: `POST /v1/images/edits` (multipart: `image`, `prompt`, optionally `mask`).

- **Repaint** (img2img): the engine starts from your picture instead of noise and redraws it to the prompt. `strength` decides how much changes.
- **Inpaint**: add a mask of the same size. White is redrawn, black kept.

```bash
mnm try zimage --image input.png                  # repaints input.png, writes zimage.png
mnm try zimage --image input.png --mask mask.png  # redraws only the white part of mask.png
mnm set zimage strength=0.5                        # how much a repaint changes (0.3 subtle … 1.0 all)
mnm url zimage --for curl                          # the same as a one-paste request
```

`--image` and `--mask` must be files in the current folder. Send a picture at the instance's own size (`mnm show <name>`). Odysseus's edit and inpaint tools reach the same route. When the whole picture's decode does not fit the free VRAM, a start turns VAE tiling on and says so: sd.cpp cannot retry the encode a repaint starts with in tiles.

**Z-Image, Qwen-Image, Qwen-Image 2.1 and Krea 2.** sd.cpp would also hand the picture to these families' text encoder as a reference image, which needs a vision part their kits do not ship, and crash (exit 139). The router clears the reference for an `image` instance of these families (`"ref_images": []` in the prompt's `<sd_cpp_extra_args>`), so the picture is only the starting image: a plain repaint.

### 8.6 Editing a picture

An **edit** instance takes a picture and an instruction and returns a new picture, on the same route:
- **FLUX.1 Kontext** (`QuantStack/FLUX.1-Kontext-dev-GGUF`): an edit-trained FLUX.1, planned with the FLUX.1 kit at cfg 1 and 20 steps, typed `image edit`. Its cards and `mnm try` use the edit route; plain `mnm try <name>` sends a flat grey square, `--image` your picture.
- **FLUX.2 klein 4B / 9B** are trained to edit from a reference picture, but mnm types a klein instance `image` like any picture model: its card shows the repaint route, and the engine takes the picture as the starting image and the reference.

The engine answers at the picture's own size: the router drops the `size` Odysseus sends (`client-size=on` passes it through).

### 8.7 Upscaling

`mnm add` fetches the 18 MB ESRGAN upscaler (RealESRGAN x4plus anime 6B) with every picture instance unless `--no-upscaler` is passed (a checkbox on the web Add page). It is used only when asked: the engine page's upscale step and hires fix list it, and `POST /sdcpp/v1/upscale` (`image`, `upscaler`, `repeats`) returns the picture at four times the size.

### 8.8 Where to generate

- **The engine page** gives full control: `http://localhost:7801/upstream/<name>/` (linked on the instance page). Prompt, negative prompt, size, steps, CFG, sampler, scheduler, seed, batch, CLIP skip and LoRAs, with previews and a cancel button; it starts from the instance's defaults.
- **Odysseus.** Add the instance as type Image ([12.6](#126-odysseus)). Every request uses the instance's defaults and always-on LoRAs; the picture comes out at the instance's size whatever Odysseus asks. To get a size in a direct message, end the message with the line the card prints, such as `<sd_cpp_extra_args>{"width":832,"height":1216}</sd_cpp_extra_args>`.
- **Scripts.** `mnm url <name> --for curl` prints a request that saves a PNG; `--for openai-sdk` gives Python.

### 8.9 Families that run here

| Family | Paste | Kit (fetched for you) | Defaults |
|---|---|---|---|
| SD 1.x, SDXL (Illustrious, Pony, NoobAI …) | a checkpoint file or a diffusers folder | SDXL: the fp16-fix VAE | 25 / 20 steps, cfg 7 |
| Z-Image | `unsloth/Z-Image-Turbo-GGUF`, `Comfy-Org/z_image_turbo` | Qwen3-4B, FLUX.1 VAE | turbo 8 steps, cfg 1 |
| FLUX.1 | `city96/FLUX.1-dev-gguf`, `…-schnell-gguf` | CLIP-L, T5, FLUX.1 VAE | dev 20, schnell 4 |
| FLUX.1 Kontext (edit) | `QuantStack/FLUX.1-Kontext-dev-GGUF` | the FLUX.1 kit | 20 steps, cfg 1 |
| FLUX.2 klein 4B / 9B | `leejet/FLUX.2-klein-4B-GGUF`, `unsloth/FLUX.2-klein-9B-GGUF` | Qwen3-4B or Qwen3-8B, FLUX.2 VAE | 4 steps, cfg 1 (base builds 20, cfg 4) |
| Anima | `circlestone-labs/Anima` (nine versions), its fine-tunes, `Bedovyy/Anima-GGUF` | Qwen3-0.6B, Qwen-Image VAE | 30 steps, cfg 4.5, `er_sde` (turbo 8, cfg 1) |
| Qwen-Image 2.1 | `unsloth/Qwen-Image-2.1-GGUF` | Qwen3-VL-8B, its own VAE | 25 steps, cfg 1 |
| Krea 2 | `realrebelai/KREA-2_GGUFs` (Turbo and Base) | Qwen3-VL-4B, Qwen-Image VAE | Turbo 8, cfg 1; Base 28, cfg 4 |
| ERNIE-Image | `unsloth/ERNIE-Image-Turbo-GGUF`, `unsloth/ERNIE-Image-GGUF` | Ministral 3B, FLUX.2 VAE | Turbo 8, cfg 1; base 20, cfg 4 |
| Chroma | `silveroxides/Chroma1-HD-GGUF` | T5, FLUX.1 VAE (no CLIP) | 26 steps, cfg 4 (a negative prompt helps) |

A single safetensors file of any of these (ComfyUI's `split_files/diffusion_models/…`, a fine-tune) plans with the same kit; it loads as stored, or converted to Q8_0 while loading when that is what fits. A diffusers folder of a transformer model (Z-Image, Anima, Qwen-Image …) is not run from the folder; `mnm fit` names the family and the repos that run it. Qwen-Image 2.1, FLUX.2 klein and Krea 2 carry non-commercial or community licenses: read the model page before commercial use.

### 8.10 VRAM on a 12 GB card

The engine works in three phases and frees each one's working memory before the next:
1. **Text encoding**: the encoders turn the prompt into numbers.
2. **Diffusion**: the painter works step by step; its working memory grows with the picture (about 1 GiB at 1024x1024).
3. **Decoding**: the VAE turns the result into pixels. Decoding a whole 1024x1024 picture at once needs about 6.5 GiB of its own, more than any other phase.

So the plan, against the VRAM free at that moment, tries in order: everything on the GPU with a whole decode; a tiled decode (about 0.4 GiB at a time, a moment slower); the text encoders on the CPU (a few seconds per new prompt); weights streamed from RAM (slower, every step re-reads part of them). The engine loads weights at the first request, not at start, so a just-started image instance holds almost no VRAM. On an RTX 4070, whole-picture decodes peaked at 11.3–11.8 GiB of the 12, a tiled Z-Image at about 5.5 GiB, and 4.6–5.2 GiB stayed held after a picture. Run one picture model at a time, beside at most a small chat model.

### 8.11 What does not run here

- **Dedicated inpainting, control and reference models** (FLUX.1 Fill, Controls and Redux, PiD) and Qwen-Image-Edit (20B, too large for 12 GB): named and refused. Repaint, inpaint with a mask and Kontext edits do run.
- **Families too large for 12 GB** (Qwen-Image 20B, FLUX.2 dev), and those that need parts mnm cannot give the engine yet (Lens, Ideogram 4, LLaDA-Image): named and refused with the reason.
- **SD 2.x, SD3 and the SDXL refiner**: recognised and refused.
- **Python in a repo** (`pipeline.py`, custom components): named, never run.
- **A part of a model on its own** (only a UNet, a ControlNet, an adapter, or a diffusers folder missing its encoders or VAE): refused with what it lacks.
- **Gated repos without a token**: accept the terms on the model's page, add a token, and try again.

## 9. Video in depth

### 9.1 The pieces

- **Diffusion model (DiT).** Works on a block of *latent frames* at once, so motion stays consistent. Wan 2.1, Wan 2.2, LingBot-Video, HunyuanVideo 1.5 and LTX-2 are different DiTs.
- **Text encoder.** Wan uses umT5-XXL (3.7 GB at Q4); LingBot a Qwen3-VL-4B; HunyuanVideo a Qwen2.5-VL-7B plus ByT5; LTX-2 a Gemma 3 12B.
- **VAE.** Turns latent frames into pixels and compresses time: four frames make one latent frame (LTX-2: eight), so a clip's frame count should be one more than a multiple of four (33, 49, 81).
- **Family.** Read from the files' tensors the way stable-diffusion.cpp reads them (Wan's versions from its patch embedding), and a GGUF repo's from its name.

### 9.2 Make a clip

```bash
mnm fit QuantStack/Wan2.2-TI2V-5B-GGUF                # what fits, no download
mnm add QuantStack/Wan2.2-TI2V-5B-GGUF --name wan5b   # 3.4 GB + umT5 3.7 GB + the 1.4 GB VAE
mnm start wan5b
mnm try wan5b                                         # queues a clip, polls it, writes wan5b.webm here
mnm try wan5b --image input.png                       # the same, starting from a picture
mnm url wan5b --for curl                              # the same as one paste, and the stitch line
```

`mnm try` prints the clip's frames and rate and where it wrote the file. Each run overwrites `<name>.webm`, so rename a clip you want to keep. A clip takes minutes (the plan's `s/clip` column); on an RTX 4070 wan5b took 155 s for 33 frames at 832x480.

### 9.3 Families that run here

| Family | Paste | Kit | Defaults | On 12 GB |
|---|---|---|---|---|
| Wan 2.2 TI2V 5B | `QuantStack/Wan2.2-TI2V-5B-GGUF`, ComfyUI's `wan2.2_ti2v_5B_fp16.safetensors` | umT5-XXL, Wan 2.2 VAE | 832x480, 33 frames, 16 fps, 20 steps, cfg 6, flow shift 3 | resident at Q4 (about 8.2 GiB, tight: needs the card nearly whole) |
| Wan 2.1 T2V 1.3B | `calcuis/wan-1.3b-gguf` | umT5-XXL, Wan 2.1 VAE | the same | resident |
| Wan 2.1 T2V 14B | `city96/Wan2.1-T2V-14B-gguf` | umT5-XXL, Wan 2.1 VAE | the same | streamed from RAM |
| Wan 2.1 I2V 14B | `city96/Wan2.1-I2V-14B-480P-gguf` | + CLIP-vision H | every request takes a first frame | streamed |
| Wan 2.2 T2V / I2V A14B | `QuantStack/Wan2.2-T2V-A14B-GGUF`, `…-I2V-A14B-GGUF` | umT5-XXL, Wan 2.1 VAE | 8 high-noise steps, then 10 at cfg 3.5 | streamed; RAM decides the quant (Q3 with about 20 GB of RAM) |
| LingBot-Video 1.3B | `robbyant/lingbot-video-dense-1.3b` (its `transformer/` file) | Qwen3-VL-4B, Wan 2.1 VAE | 832x480, 33 frames, cfg 3 | resident |
| HunyuanVideo 1.5 (8B) | ComfyUI's `hunyuanvideo1.5_480p_t2v_fp8_scaled.safetensors` and its siblings | Qwen2.5-VL-7B, ByT5, its own VAE | 832x480, 33 frames, 24 fps, cfg 6 | streamed (FP8 loads as Q8_0) |
| LTX-2.3 (22B, with sound) | `unsloth/LTX-2.3-GGUF` | Gemma 3 12B and its parts | 768x512, 33 frames, 24 fps | refused with about 20 GB of RAM: no quant fits beside its text encoder |

**Text or image to video.** Every family takes a text prompt. To animate a picture, `mnm try <name> --image first.png` sends it as the first frame (`init_image`). The image-to-video builds (Wan 2.1 I2V, Wan 2.2 A14B I2V) need a first frame on every request: their curl card sends `input.png`, and `mnm try` without `--image` sends a flat grey frame. **Use a picture of the clip's aspect ratio** (832x480 for Wan 2.2 5B): the engine resizes without cropping, so another shape is stretched.

### 9.4 Every setting

| Setting | What it does | Starting point |
|---|---|---|
| Frames (`frames`) | The clip's length; one more than a multiple of 4 (LTX-2: 8). VRAM and time grow with it | 33 (2 s at 16 fps); 81 for 5 s |
| Frame rate (`fps`) | The file's timing, not what the model draws | 16 for Wan and LingBot, 24 for HunyuanVideo and LTX-2 |
| Size (`width`, `height`) | Multiples of 16 (LTX-2: 32). Wan is trained at 832x480 and 480x832 | 832x480 |
| Steps (`steps`), high-noise steps (`high-noise-steps`, A14B) | Refinement passes | 20; 10 + 8 for the A14B pair |
| CFG (`cfg-scale`) | Every video family here uses real CFG: a negative prompt counts, each step runs twice | 6 (A14B 3.5, LingBot 3) |
| Flow shift (`flow-shift`), sampler, scheduler, seed, negative prompt | As for pictures | 3; euler; empty; the model card's |
| Streamed weights, tiled VAE, text encoder on the CPU | Where things live when VRAM is short | from the plan |

### 9.5 Where a clip goes, and stitching

- **`mnm try <name>`** writes `<name>.webm` in the current folder.
- **The engine page** (`http://localhost:7801/upstream/<name>/`) makes clips with a preview and a cancel button, and takes a first (and last) frame.
- **Scripts.** The curl card queues a clip on `POST /sdcpp/v1/vid_gen` (202 with a `poll_url`), polls the job until `completed`, and saves `result.b64_json`.
- **Odysseus** has no video endpoint type; a video instance cannot be added there.
- **A set** ([16.3](#163-set-recipes)): a chat model writes the shots, a picture model draws each keyframe, the video model animates each one, a voice reads the lines.

**Stitching.** On a video instance, `mnm url <name> --for curl` also prints the stitch line. It joins `shot*.webm` (name them `shot01.webm`, `shot02.webm` …) and a `voice.wav` into `film.webm` with the ffmpeg inside the whisper.cpp engine image, so nothing is installed:

```bash
printf "file '%s'\n" shot*.webm > shots.txt && docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/w -w /w --entrypoint ffmpeg ghcr.io/ggml-org/whisper.cpp:main-cuda -f concat -safe 0 -i shots.txt -i voice.wav -c:v copy -c:a libopus -shortest film.webm
```

The clips are copied, not re-encoded, so they must share a size and frame rate (clips from one instance do); the film ends with the shorter of the two tracks.

### 9.6 What does not run here

- **Speech-to-video and audio-video models** (Wan 2.2 S2V, MiniMax-H3): named and refused.
- **LTX-2 with about 20 GB of RAM**, and **LTX-2.5** (no kit of its own; do not trust a plan that reads it as LTX-2).
- **Wan VACE's control videos and Wan Fun's control and camera builds**: they plan as plain Wan and generate from text; only the engine page reaches their controls.
- **LoRAs on video models**: `mnm lora add` accepts them (with a warning: their base cannot be read from the header), but none has been tried.

## 10. Speech in depth

### 10.1 Speech-to-text (whisper.cpp)

- **Sizes.** `ggerganov/whisper.cpp` holds `ggml-<size>[-<quant>].bin` files; `--quant large-v3-turbo-q8_0` picks one. large-v3-turbo is close to large-v3 at a third of the time; `base` is fine for short English commands.
- **Settings:** `language` (`auto`, `en` …), `convert` (accept webm/mp3; the image has ffmpeg), `threads`.
- **Memory.** On an RTX 4070 the large-v3-turbo instance held about 0.9 GiB and transcribed an 11 s clip in 0.69 s.
- **The engine needs driver 580 or newer** (CUDA 13); `mnm doctor` warns below that.
- `mnm try stt --audio clip.wav` sends a clip from the current folder; `/v1/models` is 404 on whisper-server by design.

### 10.2 Text-to-speech (audio.cpp)

- **Engine.** audio.cpp's server (`ghcr.io/0xshug0/audio.cpp:full-cuda12`): OpenAI's `POST /v1/audio/speech` in, a WAV out.
- **Package.** The models live in one Hub repo, `audio-cpp/audio.cpp-gguf`, one folder per model; you paste the folder (`audio-cpp/audio.cpp-gguf/PocketTTS-GGUF/english`). PocketTTS's voices travel with it. Packages download as plain files under `<models>/voices/` (the engine refuses the Hub cache's hash-named files).
- **Config.** audio.cpp reads a JSON config, not flags: mnm renders it from the recipe at every start (`<models>/configs/<name>.json`). It turns the engine's own page on (`http://localhost:7801/upstream/<name>/`: type a sentence, pick a voice, clone one from a WAV).

| Family | Package folders | Voices | Size (Q8_0) | Notes |
|---|---|---|---|---|
| **Kokoro** (82M) | `Kokoro-82M-GGUF` | 11 built in (`af_heart` …, American and British English) | 190 MB | **Refused by `mnm add`.** Its English voices need eSpeak-ng, which audio.cpp's official image does not include: every request is answered HTTP 500 "Could not load eSpeak-ng". A Kokoro instance already saved still starts; `mnm try` says what the 500 means; `mnm rm <name> --files` removes it |
| **PocketTTS** (100M) | `PocketTTS-GGUF/<language>` (english, french, german, italian, portuguese, spanish) | the folder's embeddings (26 for English) | 128 MB | One language per package. Refuses any `speed`. About 5x faster than real time on an RTX 4070 |
| **Qwen3-TTS** (0.6B / 1.7B) | `Qwen3-TTS-12Hz-<size>-CustomVoice-GGUF`, `-Base-GGUF`, `-VoiceDesign-GGUF` | 9 speakers (CustomVoice); Base clones a WAV plus its transcript; VoiceDesign takes a description | 0.7 / 1.8 GB | Refuses any `speed` |

**Settings** (`mnm set <name> key=value`, applied at the next start): `voice` (the default when a request names none; must be one of `mnm show`'s), `speed` (Kokoro only), `seed`, `threads`.

**Odysseus** asks for mp3 and sends a `speed` with every request: audio.cpp answers WAV (Odysseus sniffs it and plays it), and the router drops `speed` for PocketTTS and Qwen3-TTS, which would otherwise answer 500.

Other folders of the audio.cpp repo (Chatterbox, Fish Audio, VibeVoice …) and voice models in other formats (Kokoro's ONNX, Coqui, Piper) do not run here; `mnm fit` says so.

## 11. Coding agents

Every chat or coding instance can drive a coding agent. Pick the client on the instance page (Connect tab, **Coding agents**), or in a terminal:

```bash
mnm url coder8b --for claude-code     # or codex, opencode, qwen-code, aider, cline, zed, llama-vscode, continue
```

Each card says where the settings go and gives the block to paste, filled in with the address, model id and context size. **A card never changes the client's default model:** Claude Code starts with its own settings folder (`CLAUDE_CONFIG_DIR=~/.claude-<name>`), so your normal Claude Code and its login stay as they are; Codex CLI, OpenCode, Qwen Code and Zed get a new provider or model named after the instance.

Every client has a **floor**: the context one request must have (its own instructions and tool list come with every request). `mnm add --for-agents` plans for the busiest client (32,768); `mnm add <link> --for <client>` plans for one, and prints "<client> needs N tokens a request; this instance serves M (ctx ÷ parallel)". Each card ends with the same line, checked against the instance as it is now.

| Client | What it talks to | Floor a request | Where the floor comes from |
|---|---|---|---|
| Claude Code | Anthropic API (`/v1/messages`; `ANTHROPIC_BASE_URL` is the origin without `/v1`) | 32,768 | measured: its first request is about 19,300 tokens (Claude Code 2.1.309) |
| Codex CLI | OpenAI Responses API | 16,384 | measured: about 6,200 tokens |
| OpenCode | OpenAI chat API with tools (`limit.context` in opencode.json is the card's) | 32,768 | measured: its first request is about 7,300 to 7,600 tokens |
| Qwen Code | OpenAI chat API with tools (`contextWindowSize` in ~/.qwen/settings.json) | 32,768 | measured: its first request is about 11,400 tokens |
| OpenHands | OpenAI chat API with tools (`--for openhands`; its card comes with the apps) | 32,768 | its docs: at least 22,000, 32,768 recommended |
| Hermes Agent | OpenAI chat API with tools (`--for hermes`) | 65,536 | its docs: refused at startup below 64,000; not in `--for-agents` |
| Goose | OpenAI chat API with tools (`--for goose`) | 32,768 | not stated in its docs |
| Cline (VS Code) | OpenAI chat API with tools | 32,768 | not measured |
| Zed | OpenAI chat API with tools | 16,384 | not measured |
| Aider | OpenAI chat API, text edits | 8,192 | not measured |
| llama.vscode / llama.vim | `/infill` (code completion) | 4,096 | not measured |

**One request at a time, or several.** The planner runs one request at a time (`parallel` 1), so each request gets the whole context. For two agent sessions at once there are two ways, and each card names both: `mnm set coder8b parallel=2 ctx=65536` keeps one copy of the weights and gives each session ctx ÷ 2 (only the KV cache grows), or a second instance beside it in your set (`mnm sets edit desk chat=coder-b`) loads the weights twice and lets each session have a model of its own.

**Does the model call tools right?** `mnm try coder8b --tools` sends five requests the way an agent does: one call, the tool's result sent back, two calls at once, the call streamed in pieces, and once with thinking on. Each step gets a verdict line, the whole *works*, *partial* or *no*, and the instance page shows it on its **Tools** line (Test it has a Tools box too); `mnm try --all` has a Tools column. A model that calls one tool per turn reads "two calls at once: not supported" (slower for an agent, not broken); a model that never thinks reads "with thinking on: not supported".

**Thinking.** Qwen3, Qwen3.6 and Qwen3.8 think by default; their reasoning comes back as `reasoning_content` beside the answer or the tool call. llama-server puts an earlier turn's reasoning back into the prompt when the client sends it (the template's `preserve_thinking`, on by default in llama.cpp), which is what agents want. mnm turns thinking off (`enable_thinking: false` in `chat_template_kwargs`) only in its own short requests: the chat card's 16-token check, `mnm try`, and the curl and Python lines of `mnm url <name> --for curl|openai-sdk`, each with a line saying why. A raw request of your own without it can spend a short `max_tokens` on reasoning alone: the answer's `content` is empty, the reasoning is in `reasoning_content`, and `finish_reason` is `length`. The router then writes one journal line for that model (once per start): "tiny spent all 32 tokens thinking and answered nothing: raise max_tokens, or send …". Raise `max_tokens`, or add `"chat_template_kwargs": {"enable_thinking": false}` to the request. Checked with a capture in front of the router: OpenCode sends each turn's `reasoning_content` back with its tool calls, and Claude Code its thinking blocks, which llama-server turns into `reasoning_content`; Qwen Code, Codex, Aider, Cline and Zed were not checked. The router's own loading notes ("Loading coder36: 12 s …", which a client may keep as reasoning) are taken out of a request's history before the model sees it.

**The prompt cache.** An agent sends the same long beginning with every request (its instructions and tool list: about 19,300 tokens for Claude Code), and llama.cpp keeps it in the slot, so only what is new is read. Two things lose it, and mnm plans for both:
- **A side request** (a title, a subagent) takes the slot. llama.cpp keeps earlier prompts' states in RAM (`--cache-ram`) and brings the conversation back when it returns. llama.cpp's own default is up to 8 GiB of RAM, outside any plan; every mnm plan sizes it instead (`cache-ram=`, MiB): room for four full prompts at the plan's context, at most a quarter of the RAM left after the system's margin and the model's own RAM part, at most 8,192, 0 (off) when not even one prompt fits, and 0 for embedding and rerank. `mnm show` says the cap an instance runs with.
- **An unload** (the idle limit, the router making room for another model, a `keep_alive: 0`, `mnm stop`) loses everything the engine held. A coding instance keeps it (`cache` left on *auto*, or `cache=on` for any chat model; `cache=off` turns it off): as the engine stops, the slot it used last (at least 2,048 tokens) is saved to disk under `/data/models/prompt-cache/<name>/`, and the next start puts the newest back as its last step, before it answers ready: llama.cpp then keeps the longest common beginning, token by token, so a different conversation of the same client still skips its instructions and tool list. Measured on llama.cpp's CPU image, a 4,311-token prefix took 21.3 s to read after a restart and 0.6 s with its 495 MB file put back (0.19 s to read the file); on a hybrid Qwen3.5 15.8 s and 0.3 s. A file is used only by the settings and engine build that wrote it: the settings the container was *started* with, not what `mnm set`, `mnm replan` or `mnm machine adopt` saved since, which apply at the next start (any other file is deleted unread: a slot restored into other weights would load without a word). `cache-disk=` (GiB) caps the folder, the oldest file going first; a slot larger than the cap, or a disk with less free than it, is not saved, and the journal (or `mnm stop`) says so. A save takes the seconds it takes (2.8 s for 495 MB in that measurement, at most two minutes); a request for the model meanwhile waits for the unload, then loads it again.

The saved files hold the conversation: `mnm set <name> cache=off` deletes them, `mnm rm` and the uninstall take them (even with `--keep-models`), and they are never in a backup or a report. The router refuses a client's own `POST …/slots/<id>?action=save|restore` (a 403): only mnm saves and puts them back. `mnm disk` lists each instance's saved prompts, and the disk policy deletes them (oldest first) before any model's files.

`mnm try <name> --agent` measures it on this box: a prefix of about 20,000 tokens (`--prefix N`; at most four fifths of what one request may hold) read cold, again warm, and, when it keeps its slots, after the router parks the instance and puts its saved slot back. Your own saved prompts are set aside while it measures and come back after it; its own file is deleted and the slot it filled erased. Each leg is llama.cpp's own figure; the result is a reading `mnm calibration` shows under *prompt cache*. `mnm replan` sizes both caps again, as it does the context.

**Context is the one setting that matters.** An agent sends its own instructions and tool list with every request. When the context is too small, the card says so and gives the fix (`mnm set coder8b ctx=32768`, then restart). On a 12 GB card, Qwen3-8B Q4_K_M at 32,768 with the KV cache at q8_0 takes about 7 GiB.

**Which model:**
- **Start with Qwen3-8B** (`mnm add unsloth/Qwen3-8B-GGUF --for-agents --name coder8b`): it calls tools reliably for its size.
- **Bigger mixture-of-experts coding models** (Qwen3-Coder-30B-A3B, gpt-oss-20b) run with most of their experts in RAM ([7.1](#71-chat)); gpt-oss-20b is the easiest to fit with about 20 GB of RAM.
- **Two larger coders**, both on the starter menu, each measured on an RTX 4070 (the 35B keeps most of its experts in RAM: under WSL give it `memory=24GB` in `.wslconfig`):
  - `mnm add unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_M --for-agents --name coder36`: 35B, 3B active, the experts of most layers in RAM (20.6 GiB to download); 25–28 tok/s measured.
  - `mnm add unsloth/Qwen3.8-27B-GGUF --quant UD-IQ3_XXS --for-agents --name coder38`: dense 27B, on the card at 32,768 with the KV at q8_0 (10.2 GiB, and its 1.3 GiB MTP head); about 21 tok/s measured.
  - Both ship a vision projector; `--for-agents` (or `--type chat`) runs them text-only, which frees 1.4 GiB of the card. Added as they are, they keep their vision cards. (A `-VL` repo added this way gets a default name without `vl` or `vision`, since Odysseus offers image upload by that token; a `--name` of your own should leave them out too.)
  - Both are hybrid: only one layer in four keeps a KV cache (10 of 40, 16 of 64), the others are linear attention with a small fixed state, so long contexts cost a quarter of what they would on Qwen3. `mnm fit` says so in its first line.
- **Two settings for them, off until set**, each worth one `mnm try <name> --speed` with and without:
  - `mnm set coder38 draft=mtp`: the model's own MTP head drafts tokens it then checks ([15.6](#156-a-draft-model-speculative-decoding)). Reported 1.85-2x on a dense 27B elsewhere, but not on a 12 GB card: on an RTX 4070 the head took coder38 from 10.4 to 11.3 GiB and from 21.4 to 8.5 tok/s (llama.cpp spilling at the card's edge), so `mnm set` refuses a draft that does not fit, or that leaves the card tight when the model alone already is.
  - `mnm set coder36 moe-cache=1024`: keeps 1,024 MiB of the most-used RAM-held experts on the card (`--moe-cache-mib`). Only with experts in RAM (`cpu-moe-layers`); `mnm set` refuses it otherwise. Reported 1.6-2.2x on fast PCIe; on an RTX 4070 under WSL2 it took coder36 from 25.2 to 26.3 tok/s.
- **Code completion (llama.vscode)** needs a fill-in-the-middle model on its own instance: `mnm add ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF --name fim`.

**Known gaps:** Codex's web search and MCP tools do not reach the model (llama-server skips those tool types); small models sometimes answer instead of calling a tool. On an RTX 4070, OpenCode was used with the coding models and Claude Code fixed a bug through the router; the Codex card was run against a processor build of llama-server, and the other cards follow their clients' docs. A coder and an embedding model together are the *coding desk* set ([16.3](#163-set-recipes)).

## 12. Apps

The apps that work with your models, each one optional. mnm **installs** four of them, pinned, on this machine only: OpenCode, LibreChat, Open WebUI and OpenHands. It **connects** two it never installs: Goose (a desktop app) and Odysseus (your own install). Pick them in `mnm setup`'s section 8, on the web's **Add → Get started**, or with `mnm apps` any time. Every app's file names the desk's roles (`coder`, `chat`, `memory` …, [16.2](#162-sets)), so it keeps working when you swap the model behind one ([13.7](#137-workspace-apps-one-file-each)).

### 12.1 Choosing, installing, connecting

| App | What it is | Needs | Licence | Where |
|---|---|---|---|---|
| OpenCode 1.19.4 | A coding agent that acts on your repositories: sessions, MCP, LSP, a web UI and a TUI | One release binary (~50 MB, checked by its sha256), a systemd user unit; works in the projects folder with your own toolchain | MIT | `http://localhost:4096` |
| LibreChat v0.8.7 | A chat app with files (RAG), pictures, voice and mnm's tools; accounts, registration off | Docker: five pinned images (~3 GB), ~2 GB of RAM | MIT | `http://localhost:3080` |
| Open WebUI v0.12.3 | A second chat app: documents (RAG), pictures, voice and mnm's tools; its first account is its admin, sign-up off after it ([12.8](#128-open-webui)) | Docker: one pinned image (~6.4 GB on disk) | the Open WebUI License (BSD-3 with a branding clause) | `http://localhost:3000` |
| OpenHands (Agent Canvas 1.31.0) | A web workspace whose agent edits, runs and tests code in the projects folder, inside its own container | Docker: one pinned image, never the Docker socket; a coder with 32,768 tokens of context | MIT | `http://localhost:8000/canvas` |
| Goose | A desktop and CLI agent, MCP-native | Goose itself (its Windows desktop app, or its CLI here); mnm writes its provider file | Apache-2.0 | its own window |
| Odysseus | The chat front end you installed in `~/odysseus` | Yours; mnm writes its `.env` lines on yes | its own | its own URL |

- **Here or another machine.** For each app setup asks *here* (installed, or its file written) or *remote*: then it prints the app's card for another machine: the router's LAN address (`/v1` and `/mcp`), that the key is required there, and `mnm url --for <app> --from lan` for the whole file. Without a key it prints no LAN address, only the step that makes one. Nothing on this machine changes. Under WSL's NAT networking the router is on this PC only, so the card says so and points at section 3 of setup (the router on your network, with a key) or Tailscale ([13.4](#134-over-tailscale)).
- **The projects folder.** `MNM_PROJECTS_DIR` (default `~/src`, created if missing) is where OpenCode works and what OpenHands mounts at `/projects` (nothing else of your home). Setup asks it once; a change after the services were installed installs them again (their units carry it). Change it later in the secrets file, then `mnm apps install opencode` / `openhands` again.
- **Get started** shows the same list with the same paragraphs: *Install here* runs the install as a job on **System → Apps** (its lines stream there), *Write its file* for Goose, *Connect* for Odysseus (the diff on its own page, written on the button), LibreChat's first account once it is installed, and for Open WebUI the line that says to make its first account on its own page. The projects folder is shown there, not changed.
- **`--yes` installs nothing**, and neither does a dry run. An app that fails is said, left in the summary's to-do list with its reason, and setup goes on.

```bash
mnm apps ls                      # every app: installed or not, running, its version against this mnm's pin, its URL
mnm apps install librechat       # the pinned build, its config from the router's models, started and answering
mnm apps status opencode         # its state, URL and where its password is (never printed)
mnm apps start|stop|logs|update|rollback <app>
mnm apps uninstall <app> [--keep-data]   # asks before deleting its data under MNM_APPS_DIR
mnm apps connect goose|odysseus  # the apps mnm never installs
mnm doctor --app <app>           # the router checked from where the app sits (13.7)
mnm apps expose <app> [--off] [--yes]    # open it to your network over plain http (its own login and the router's key; asks first), or close it (13.3)
```

**What mnm never does:** give an app the Docker socket, run a moving tag (`latest`), put an app on your network without its own login and the router's key (`mnm apps expose`, [13.3](#133-from-another-machine-on-your-network)), or install Odysseus. Each app's data lives under `MNM_APPS_DIR` (default `/data/apps`, one folder per app); `uninstall.sh --keep-apps` keeps it all.

### 12.2 OpenCode

- `mnm apps install opencode` downloads the release at its pin, checks its sha256, writes `~/.config/opencode/opencode.json` (provider `mnm`, mnm's MCP tools; a file of yours is merged, a copy kept) and starts `opencode serve` on 127.0.0.1:4096 as the user unit `mnm-opencode`, in the projects folder.
- **Sign in:** user `opencode`, the password `MNM_OPENCODE_PASSWORD` in `~/.config/makenomistakes/secrets.env` (mnm generates it; the unit reads it from a 0600 file, never its own text). A Windows browser opens `http://localhost:4096`; `opencode attach http://127.0.0.1:4096` joins from a terminal.
- Pick `mnm/coder`. Its file gives each model its context (`limit.context`); OpenCode's first request is about 7,300 to 7,600 tokens, and its docs suggest 16k to 32k.

### 12.3 LibreChat

- `mnm apps install librechat` writes its compose project, `.env` (its secrets generated once, 0600; the router's key when one is set) and `librechat.yaml` under `/data/apps/librechat`, pulls the five pinned images and starts it on 127.0.0.1:3080. Every container runs as you.
- **Registration is off.** Setup asks the first account (email, then the password twice; LibreChat's own script makes it, mnm never keeps the password); later `mnm apps user add librechat EMAIL`.
- A new chat opens on mnm's endpoint with `coder` (or `chat`); files you attach are embedded with `memory`; speech uses the desk's `ear` and `voice`; mnm's tools are in its MCP Servers menu.

### 12.4 OpenHands

- `mnm apps install openhands` runs Agent Canvas 1.31.0 on 127.0.0.1:8000, as you, its home under `/data/apps/openhands`, the projects folder at `/projects`, without the Docker socket: the agent's shell runs inside that container. mnm sets its model (`openai/coder`), base URL, context and mnm's MCP tools through its settings API.
- **Sign in:** the page asks for its session key once per browser: `MNM_OPENHANDS_KEY` in the secrets file. Untick *Send anonymous usage data* at the first visit (it is ticked although mnm asks it not to track).
- It wants 32,768 tokens per request: the card and `mnm doctor --app openhands` say when `coder` has less.

### 12.5 Goose

Goose's documented route on Windows is its desktop app. `mnm apps connect goose` (setup's *here*, or Get started's *Write its file*) writes `~/.config/goose/custom_providers/mnm.json` for Goose's CLI on this machine and prints the `extensions` block; the Windows app reads `%APPDATA%\Block\goose\config\custom_providers\`, so copy the file there (it reaches the router at `localhost:7801` through WSL's forwarding). mnm's tools are a `streamable_http` extension, which Goose sends no key to.

### 12.6 Odysseus

mnm never installs Odysseus. When `~/odysseus` exists, setup's section 8 (or Get started's *Connect*, or `mnm apps connect odysseus`) shows the diff of its `.env` (`EMBEDDING_URL`, `EMBEDDING_MODEL`; secret values hidden), writes it only on yes (yours kept once as `.env.before-mnm`) and runs `docker compose up -d` in its folder, which recreates only the containers whose settings changed. Its endpoints and its MCP Tool Server are forms in its own Settings: the card prints what to enter, as below. `uninstall.sh` never touches it.

Odysseus runs in Docker, so it reaches mnm through `host.docker.internal`, not `localhost`. Every card prints the right URL: `mnm url <name>`, or the instance page's Connect tab.

Chat-like models share one URL; every other model has one of its own under `/upstream/<name>/`.

| Type | Where it goes in Odysseus |
|---|---|
| chat, coding, vision | Settings → **Add Models** → Add Local Models, type **LLM**, `http://host.docker.internal:7801/chat/v1`, once: it lists every chat, coding and vision model by instance name. **Never type Image** for a chat model: Odysseus would send every message of that chat to its image tools |
| image, image edit | Settings → **Add Models** → Add Local Models, type **Image**, `http://host.docker.internal:7801/upstream/<name>/v1` (one endpoint per image instance) |
| speech-to-text | Odysseus 0.20.19 has **no speech-to-text settings page**. Add its `/upstream/<name>/v1` URL under Add Models as type LLM (Odysseus finds one model there, the instance's name: the router lists it whether it is loaded or not); an admin then sets it through Odysseus's API (below). The chat's recorder sends recordings there afterwards |
| text-to-speech | Odysseus 0.20.19 **hides read-aloud** and its text-to-speech settings: nothing in its UI plays speech. The admin API below is the only route; the voice must be one of the model's (`mnm show <name>`) |
| embedding | No UI setting: the two `.env` lines (`EMBEDDING_URL=…`, `EMBEDDING_MODEL=…`), which `mnm apps connect odysseus` writes on yes (or add them to `~/odysseus/.env` yourself, then `docker compose up -d` in `~/odysseus`). Stored memories are re-embedded. Never add the embedding URL under Added Models |
| rerank, video | Odysseus has nowhere to put them; use the curl card or the engine page |

Things worth knowing:
- **Always keep the `/v1`** at the end of the URL. Without it, Odysseus treats the URL as Ollama.
- **The base URL 404s in a browser** by design. `http://localhost:7801/v1/models` is the browser check: it lists every instance you have not stopped, and whether it is loaded.
- **Picture size:** Odysseus asks for 512x512 in direct chat and 1024x1024 from its image tool; the router drops that, so every picture comes out at the instance's own size (`mnm set zimage width=832 height=1216`). `client-size=on` lets the client's size win.
- **Steps:** Odysseus's "quality" choice is ignored; steps are the instance's own (`mnm set zimage steps=8`).
- **Pictures from a chat:** ask plainly ("draw a lighthouse at dusk"). Naming the instance or pasting its URL in the message confuses the model; Odysseus's image tool picks the Image endpoint itself. A chat model that answers "I cannot generate images" was sent no image tool: check the Image endpoint is enabled and the chat endpoint has tools on.
- **With a router key** ([14.4](#144-a-key-for-the-router)), put it in each endpoint's API key field.
- **Engine ports (78xx) are not for Odysseus.** Each engine listens on 127.0.0.1 only, which Odysseus's container cannot reach, so an endpoint on such a port answers nothing: delete it under Settings → **Added Models** and add the router's 7801 address.
- **A stopped instance lists no models** in Odysseus: the router leaves it out until `mnm start`. A parked one is listed, and loads at its first request.
- **Context.** Odysseus asks a local endpoint for a model's context before every message. The router's `/v1/models` gives each model the context one request gets (`context_length`), so Odysseus budgets to that rather than to its default of 128,000 tokens.
- **Names Odysseus hides.** Its chat lists leave out a model whose id (the instance's name) starts with `whisper`, `embed` or `tts-`, or contains `-tts`, `-transcribe` or `bge`, unless the endpoint pins it (`pinned_models`, which only Odysseus's API sets, `PATCH /api/model-endpoints/<id>`; the Add Models form cannot). Give a chat model another name.
- **Speech settings through Odysseus's API** (admin only; the keys are Odysseus's own, `src/settings.py`): with the speech endpoints added, `POST /api/auth/settings` with `{"stt_enabled": true, "stt_provider": "endpoint:<the endpoint's id>", "stt_model": "<the model the card names>"}` for speech-to-text, and `{"tts_provider": "endpoint:<id>", "tts_model": "…", "tts_voice": "…"}` for text-to-speech. The endpoint ids are in `GET /api/model-endpoints`. Every card prints its own line; mnm never calls Odysseus itself.

**Workflows and films from a chat.** The router also serves mnm's runs as MCP tools, which Odysseus's MCP client can call. `mnm mcp` prints the card (the web's **System** page shows it too, with the tools there are now):

- **Settings → Integrations → Add Integration → MCP Tool Server** (an admin): name `mnm`, transport **Streamable HTTP**, URL `http://host.docker.internal:7801/mcp`, **Save**. The form has no header or key field (Odysseus authenticates MCP servers with OAuth only), so it sends no key: with `MNM_API_KEY` set, Odysseus cannot reach `/mcp`. Keyless, `/mcp` answers this machine and its Docker containers, Odysseus's among them.
- **The tools:**
  - one per imported ComfyUI workflow that runs here (`workflow_<name>`), whose arguments are its form's fields: prompt, seed, size and so on;
  - `short_film` (idea, shots, style and the models to use), which runs straight through;
  - `run_status`.
  
  `mnm mcp tools` lists them (and **System** → *Odysseus over MCP*).
- **What a call does:** it starts the run and answers at once with the run's id and its page in mnm's web UI. A run takes minutes. The model then calls `run_status`, which can wait up to 10 minutes. It returns the files, with the pictures as images Odysseus shows; clips and films are linked.
- **A workflow that takes a picture or a clip** can only be given one in mnm's web UI. From a chat it uses the workflow's own.
- **The MCP server is only an address you add** in Odysseus's Settings; mnm changes no file of Odysseus's for it.
- **Another machine:** Odysseus there cannot use `/mcp` (keyless it answers only this machine and its containers, and Odysseus cannot send the key). Another MCP client can: `http://<this machine's address>:7801/mcp` (the router must listen there: [14.5](#145-who-may-reach-it)) with `MNM_API_KEY` sent as `Authorization: Bearer <key>`. `MNM_WEB_URL=http://<address>:7800` in the secrets file makes the links in `run_status` open mnm's web UI on that machine instead of `localhost`. If Docker on this box has handed out a `192.168.…` network (it does once its `172.…` ones are used) that overlaps your LAN, the LAN counts as a container: set a key (and Odysseus loses `/mcp`).
- A workflow name longer than 55 characters becomes a tool name cut short with a hash at the end; `mnm mcp tools` shows each.

### 12.7 From another machine

An app on Windows reaches the router at `localhost:7801` through WSL's forwarding: that is *here*. An app on another PC needs the router on your network and the key ([14.4](#144-a-key-for-the-router), [14.5](#145-who-may-reach-it)): the *remote* card names the router's LAN addresses and says the key is required. The installed apps themselves answer on 127.0.0.1 only; `mnm apps expose` opens one to your network ([13.3](#133-from-another-machine-on-your-network)), and only with the app's own password or accounts on.

### 12.8 Open WebUI

A second chat app beside LibreChat (pick either, or both: each has its own accounts and port).

- `mnm apps install open-webui` writes its compose project under `/data/apps/open-webui` (one container, `mnm-open-webui`, the plain `ghcr.io/open-webui/open-webui:v0.12.3` image pinned by digest, never `:cuda` or `:ollama`: the router serves the models), its `.env` (0600: `WEBUI_SECRET_KEY` generated once and kept, and every line that carries the router's key), and starts it on 127.0.0.1:3000 as you, `no-new-privileges`, without the Docker socket. Its accounts, chats, documents and uploads are in `/data/apps/open-webui/data`. The first start takes a few minutes (its database is made); without a desk `memory` it also downloads its own small embedding model once.
- **The first account is its admin.** No script makes Open WebUI's accounts: open `http://localhost:3000` right after the install and sign up; that first account becomes the admin. Sign-up is off for everyone after it (`ENABLE_SIGNUP=false`; Open WebUI lets only the very first account through whatever that says), and the admin adds others under Admin Panel → Users. `mnm apps user add open-webui` is refused with that sentence; `mnm apps expose open-webui` waits until the account exists.
- **What mnm wires, by desk role** (environment variables Open WebUI reads at the pinned tag; a role the desk lacks is left out, Open WebUI's own default then applies):

  | Role | What Open WebUI does with it |
  |---|---|
  | `chat` (else `coder`) | Its models are the router's `/chat/v1` list (chat, coding and vision models, the desk's roles first); new chats open on `chat` (`DEFAULT_MODELS`). Its Ollama connection is off: the router's Ollama API would list every model twice |
  | `ear` | The microphone: speech to text through the router (`AUDIO_STT_ENGINE=openai`) |
  | `voice` | Read aloud with the voice's default voice (`AUDIO_TTS_ENGINE=openai`; Open WebUI's own `alloy` is not a name audio.cpp knows) |
  | `image` | Image generation on (`IMAGE_GENERATION_ENGINE=openai`); `IMAGE_SIZE` is the instance's own size, which is what comes back (the router keeps it, [8](#8-pictures-in-depth)) |
  | `memory` | Documents and knowledge are embedded through the router (`RAG_EMBEDDING_ENGINE=openai`) |
  | (always) | mnm's MCP tools as a tool server (`TOOL_SERVER_CONNECTIONS`, Streamable HTTP, native in Open WebUI): keyless from its container, the key as a bearer when one is set. The admin sees it under the chat's tools (`mnm`) |

- **Its settings are mnm's at every start** (`ENABLE_PERSISTENT_CONFIG=false`). Open WebUI otherwise copies these variables into its database at the first start and ignores them afterwards; with it off, a desk change reaches Open WebUI with `mnm apps install open-webui` again. The price: a connection or model setting changed in its Admin Settings lasts only until the next restart. Change the desk instead (`mnm sets edit`), or, for a setting mnm does not set, keep it in the Admin Settings knowing it resets.
- `mnm url --for open-webui` prints the same lines for an Open WebUI of your own (`--write --to <its .env>` merges them; the key is named, never printed). `mnm url <name> --for open-webui` is the one-model card of [13](#13-other-apps).
- `mnm doctor --app open-webui` asks from inside its container: the model list, a chat with a tool call, the embedding when `memory` is wired, `/mcp`, the key.
- **Your network:** `mnm apps expose open-webui` (the router's key first, then its first account; it asks, since the sign-in crosses the network in plain http) publishes `0.0.0.0:3000`. Docker publishes past ufw; `--off` closes it.
- `mnm backup` takes no Open WebUI file (its `.env` holds secrets; its settings are rewritten from the desk at install); `mnm apps uninstall open-webui --keep-data` keeps its folder, `.env` included, so a later install signs the same accounts in.

## 13. Other apps

`mnm url <name> --for <app>` prints the card for one app:

| `--for` | What you get |
|---|---|
| `odysseus` (default) | The settings in [12.6](#126-odysseus) |
| `open-webui` | Its admin settings for chat, documents, images and audio |
| `continue` | A `config.yaml` model block |
| `openai-sdk` | A short Python script |
| `curl` | One-paste commands, including the polling script for video |
| `claude-code`, `codex`, `opencode`, `qwen-code`, `aider` | The agent's config or launch line, with the context check ([11](#11-coding-agents)) |
| `cline`, `zed`, `llama-vscode` | Editor settings (Zed: its OpenAI-compatible provider, and its own llama.cpp provider for a llama.cpp instance) |
| `home-assistant` | The Ollama integration's fields (URL, API Key, Model, Context window size); it speaks Ollama's API ([13.6](#136-ollamas-api)) |
| `karakeep` | The `.env` lines: `OPENAI_BASE_URL`, `INFERENCE_TEXT_MODEL` (and `INFERENCE_IMAGE_MODEL` for a vision model), `INFERENCE_CONTEXT_LENGTH`, a longer job timeout; `EMBEDDING_TEXT_MODEL` for an embedding model |
| `frigate` | A `genai:` provider block for `config.yml` (a vision model only), with `context_size` |
| `n8n` | The Ollama credential (its OpenAI credential has no documented base URL) |
| `jetbrains` | AI Assistant's OpenAI-compatible provider: URL, API Key, Tool calling, the context window |
| `raycast` | A `providers.yaml` block with the model's `context` and abilities |
| `obsidian-copilot` | Copilot's custom provider (Base URL, API key, model ID) |
| `brave-leo` | Leo's "bring your own model" form (the full chat completions URL) |

Every card for a chat model ends with the context check: the context a request gets, against what the app needs. The web UI's Connect tab shows the same cards.

- **From Windows apps** use `http://localhost:7801/v1` with the instance's name as the model, or the card's `/upstream/<name>/v1` URL for an app that wants one URL per model.
- `mnm url <name> --direct` prints the engine's own address on 127.0.0.1 (no router: start the instance first; nothing loads it there).
- `mnm sets url <name> --for <app>` prints every member's card at once.
- **Where the client sits** decides the address, one rule for every card: this machine `localhost` ([13.1](#131-from-this-machine)), a container here `host.docker.internal` ([13.2](#132-from-this-machines-containers)), another machine the router's LAN address with the key ([13.3](#133-from-another-machine-on-your-network)). Each card names its client's usual place; `--from here|docker|lan` names another.

### 13.1 From this machine

A client on the box itself (a WSL shell, OpenCode, Goose's CLI, an editor) or a Windows app (WSL forwards Windows' `localhost` to it) uses `http://localhost:7801/v1`. The key is optional here ([14.4](#144-a-key-for-the-router)). `mnm url <name> --for <app> --from here` prints any card this way, even one whose client usually sits in Docker.

### 13.2 From this machine's containers

A client in a Docker container on the box (Odysseus, LibreChat, OpenHands, Open WebUI) uses `http://host.docker.internal:7801/v1`: Docker's bridge gateway, which the container maps with `--add-host host.docker.internal:host-gateway` (`mnm apps install` maps it; Odysseus's compose has it). `127.0.0.1` inside a container is the container itself. `--from docker` prints any card this way. On a native box with ufw on, `mnm setup` adds the rule that lets Docker's networks reach port 7801 ([3](#3-native-debian-and-ubuntu)).

**The names the router answers to.** It answers only a request whose `Host` is `localhost`, `127.0.0.1`, `[::1]`, `host.docker.internal`, or one of its own addresses (Docker's bridge gateway with `MNM_ROUTER_HOST=local`, every address of this machine with `0.0.0.0`, read when the router starts); any other name gets **421 Misdirected Request**, naming the setting, and one journal line per address and name. That stops a web page that points its own name at this machine (DNS rebinding) from talking to the router through your browser, key or no key. A name of your own (a Tailscale name, a name in your LAN's DNS) goes in `MNM_ROUTER_HOSTS` in the secrets file, commas between (`MNM_ROUTER_HOSTS=box.tailnet.ts.net,box.lan`), then `mnm service install` restarts the router with it. `mnm doctor`'s *Router hosts* line lists them.

**Web pages on other sites.** A web page you open can send a request to `http://localhost:7801` from your browser, and a plain-text body needs no permission from the router first. So a router **without a key** refuses every request but a look (`GET`, `HEAD`) that carries an `Origin` other than the names above, whatever its port: **403**, saying why, and one journal line per address and origin. The page cannot read the router's answers either, a look's included: the router names a page's own origin in `Access-Control-Allow-Origin` only when that page may use it, and never sends `Access-Control-Allow-Credentials`. Nothing else is refused: a request without an `Origin` (curl, the OpenAI SDK, Odysseus's server, every app in a container, the web UI's own calls) passes, and a request that carries the key passes from any origin, since the key is the protection. So a browser client served from another site (a web app that calls `localhost:7801` from the page itself, or an app whose window sends its own origin, as Obsidian's `app://obsidian.md` does without Copilot's *Enable CORS*) needs the key ([14.4](#144-a-key-for-the-router)); a page of your own on another name goes in `MNM_ROUTER_HOSTS`, as above, then `mnm service install`. `/mcp` refuses a page on another site key or no key ([12.6](#126-odysseus)).

### 13.3 From another machine on your network

Another PC, a laptop or a phone reaches the router at this machine's LAN address, **with the key**: the router is the door to every model, so off this machine it is never left open.

1. **Let the network in:** `mnm setup`, section 3, *every machine on your network* (`MNM_ROUTER_HOST=0.0.0.0`), with a key (`MNM_API_KEY`). Under WSL's NAT networking (the default) no other machine reaches WSL at all; mirrored networking (`networkingMode=mirrored` in `.wslconfig`) or Tailscale ([13.4](#134-over-tailscale)) does. On a native box with ufw on, let the LAN in: `sudo ufw allow 7801/tcp` (mnm never changes that rule itself).
2. **Print the card for there:** `mnm url --for <app> --from lan` (a workspace app's file) or `mnm url <name> --for <app> --from lan` (one instance): every URL is the router's LAN address, and nothing says `host.docker.internal`. Without a key no card prints a LAN URL: it prints the step that makes one. `mnm setup`'s *remote* answer prints the same card.
3. **Check it from there:** `mnm doctor --remote-script > mnm-check.sh` (or `--shell powershell > mnm-check.ps1` for Windows PowerShell) prints a small script with this machine's address in it and no key. On the other machine:

   ```bash
   read -rs MNM_API_KEY; export MNM_API_KEY; sh mnm-check.sh
   ```

   ```powershell
   $env:MNM_API_KEY = [Net.NetworkCredential]::new('', (Read-Host -AsSecureString 'key')).Password
   powershell -ExecutionPolicy Bypass -File mnm-check.ps1
   ```

   `read -rs` asks for the key without showing it and keeps it out of the shell's history (in a plain `sh`: `stty -echo; read -r MNM_API_KEY; stty echo`); the PowerShell line asks with a hidden prompt too. It asks five things and prints `OK`/`WARN`/`FAIL` for each: `/v1/models` lists the desk's roles, a short chat with `coder`, `/mcp`'s `initialize`, Ollama's port when it has its own, and that a request without the key is refused. The key comes from the shell's environment only (sh hands it to curl on its stdin, never on a command line). `MNM_ROUTER_URL='https://…'` points it elsewhere (a Tailscale name). Needs curl (sh) or nothing at all (PowerShell 5.1 or 7). Under WSL's NAT networking the script names `localhost`, which is where Windows reaches the router.
4. **From the box itself:** `mnm doctor --from lan` asks the same at the router's own LAN address (the bind and the key; only another machine can tell about a firewall in between), `--from docker` from a one-off `curlimages/curl` container on Docker's bridge, `--from here` from 127.0.0.1.

- **What the doctor says:** with the router on the LAN and no key, `mnm doctor`'s *API key* line is a FAIL, and setup's summary says the same; its remedy also says that without a key web pages on other sites are refused, so a browser client served from elsewhere needs the key ([13.2](#132-from-this-machines-containers)). On this machine and its containers only (`MNM_ROUTER_HOST=local`, or WSL's NAT) the key stays optional.
- **The apps on another machine.** OpenCode, LibreChat, Open WebUI and OpenHands listen on 127.0.0.1 only. `mnm apps expose <app>` opens one to your network (OpenCode's unit gets `--hostname 0.0.0.0`, LibreChat, Open WebUI and OpenHands publish `0.0.0.0:<port>`), and only while its own login is on and the router's key is set: OpenCode's password, LibreChat with registration off and at least one account, Open WebUI with its first account made, OpenHands' session key (its page asks a LAN browser for it). `mnm apps expose <app> --off` closes it again, and `mnm apps uninstall` (or `uninstall.sh`) removes it with the app. Every exposed app speaks **plain http**: its login crosses your network in clear text with every request (OpenCode's password is HTTP basic auth, and whoever reads it runs commands on this machine as you; OpenHands' session key drives the agent in your projects folder), so `mnm apps expose` says so and asks first (`--yes` skips the question). Tailscale Serve's `https` ([13.4](#134-over-tailscale)) is the safe way to reach an app from elsewhere. `mnm doctor` lists each app that is open as a warning saying *plain http* (a FAIL if the router's key has gone). mnm never touches ufw for it: for OpenCode (a native process) the command prints `sudo ufw allow <port>/tcp` on a native box with ufw on; LibreChat's, Open WebUI's and OpenHands' ports are published by Docker, which goes past ufw, so the LAN reaches them whatever ufw allows and only `mnm apps expose <app> --off` closes them. LibreChat marks its sign-in cookie secure, so over plain `http://` from another machine a browser may not keep you signed in; Tailscale Serve's `https` does.
- Odysseus on another machine chats through its endpoints with the key as each one's API key; its MCP tools work only from this machine (its form cannot send a key).

### 13.4 Over Tailscale

[Tailscale Serve](https://tailscale.com/kb/1242/tailscale-serve) puts the router on your tailnet (your own devices, nothing else; Funnel, which opens it to the internet, is not for this) with an HTTPS address, without opening a port on your network:

```bash
tailscale serve --bg localhost:7801      # https://<this machine>.<your tailnet>.ts.net/
tailscale serve status                   # what it serves
tailscale serve --bg --https=11434 localhost:11434   # Ollama's port too, with MNM_OLLAMA_PORT=11434
tailscale serve reset                    # stop serving all of it
```

- **Set the key first** (`MNM_API_KEY`, [14.4](#144-a-key-for-the-router)). Tailscale Serve hands every request to the router from this machine, so to the router they all look local: without a key, every device on your tailnet reaches every model, and `/mcp`'s tools too (keyless `/mcp` answers this machine). `mnm doctor`'s *API key* line fails only when the router's own port is on your network, so it cannot see this: set the key yourself. With a key, Odysseus loses `/mcp` (it cannot send one, [12.6](#126-odysseus)).
- **Keep the router local:** Serve reaches it at `localhost:7801`, so `MNM_ROUTER_HOST=local` ([14.5](#145-who-may-reach-it)) is enough, and nothing else on your network reaches it. Under WSL, run `tailscale serve` where Tailscale runs: on Windows it reaches the router through WSL's localhost forwarding; inside WSL, directly.
- **On the phone:** an Ollama app (Reins, Enchanted) takes `https://<this machine>.<your tailnet>.ts.net:11434` (with the Ollama port served), or the plain address for the router's own port; an OpenAI app (Jan and most others) takes `https://<this machine>.<your tailnet>.ts.net/v1`, the instance's name as the model and your key as its API key. An app that cannot send a key cannot use the router once the key is set.
- **Its name:** if a request over Serve gets 421 ("does not answer to the name …"), put `<this machine>.<your tailnet>.ts.net` in `MNM_ROUTER_HOSTS` ([13.2](#132-from-this-machines-containers)) and `mnm service install`.
- The context, the loading notes and the cards work the same over it; a parked model loads at the first message as at home.
- **A client install over Tailscale:** `MNM_ROUTER_URL=https://<this machine>.<your tailnet>.ts.net` on the other machine ([13.5](#135-a-client-only-install)); `mnm url --for <app>` there prints that address. `tailscale funnel` (the internet) is never the way: mnm's cards and docs only ever name Serve.

### 13.5 A client-only install

A second machine (a desktop beside the model server, a laptop) can run mnm's CLI against the box's router without Docker, a GPU or any model: the router and the engines stay on the box, and the client prints the cards and checks the way there.

```bash
# git and uv (https://docs.astral.sh/uv/) on that machine; no Docker, no GPU, no models
git clone https://github.com/SergeantCS2/make-no-mistakes ~/makenomistakes && cd ~/makenomistakes
uv sync
mkdir -p ~/.config/makenomistakes && chmod 700 ~/.config/makenomistakes
printf 'MNM_ROUTER_URL=http://192.168.1.10:7801\nMNM_API_KEY=<the box'"'"'s key>\n' > ~/.config/makenomistakes/secrets.env
chmod 600 ~/.config/makenomistakes/secrets.env
uv run mnm doctor --from lan      # the five checks, against the box
uv run mnm url --for opencode     # OpenCode's file, every URL the box's (--write writes it)
uv run mnm router manifest        # every model the box serves, by role
```

- `MNM_ROUTER_URL` is the box's router as an origin (`http://<address>:7801`, or `https://<box>.<tailnet>.ts.net` over Tailscale Serve); `MNM_API_KEY` is the box's key, kept in this machine's secrets file (mode 0600) like on the box. Without the key the client refuses: the router asks for it off its own machine.
- Only these three work there: `mnm url --for <app>` (a workspace app's file; an instance's own cards come from the box: `mnm url <name> --for <app> --from lan`), `mnm doctor --from lan` and `mnm router manifest`. `mnm add`, `start` and the rest manage the box's engines, which a client has none of: run them on the box. Never set `MNM_ROUTER_URL` on the box itself.
- The web UI stays on the box ([17](#17-the-web-ui)): it listens on 127.0.0.1 there.

### 13.6 Ollama's API

Some apps speak only Ollama's API (Home Assistant's Ollama integration, n8n's Ollama credential, Enchanted, Reins). The router speaks it too, for chat, coding, vision and embedding models: `/api/tags` and `/api/ps` list them (a model's name is the instance's; `name:latest` works too), `/api/show` gives the context a request gets, `/api/chat` and `/api/generate` answer (streamed as Ollama streams, or whole), `/api/embed` embeds, `/api/version` gives mnm's version. Pictures, clips and voices are not listed: ask the OpenAI routes for those.
- **Where:** on the router's own port, `http://localhost:7801` (no `/v1`). Most of these apps assume port 11434: `MNM_OLLAMA_PORT=11434` in the secrets file, then `mnm service install`, adds a second listener there, on the same addresses and with the same key; its root answers "Ollama is running" as Ollama's does. Do not set it while Ollama itself runs on the box: `mnm doctor`'s *Ollama API* line says who holds the port.
- **What the router does not take from an Ollama request:** `options.num_ctx` (the context is the instance's: `mnm set <name> ctx=…`; the journal names it only when a client asks for more than the instance serves), options the engines have no name for, a `format` that is neither `"json"` nor a JSON schema (each named in the journal, with why), and `/api/pull`, `/api/create` or `/api/delete` (models come from `mnm add`; the answer says so).
- **`keep_alive`:** `0` unloads the model once the reply is done, `-1` keeps it loaded (no idle unload, still unloaded for room), a duration (`10m`; under a second counts as one second) is its idle limit; until the router restarts. It never unloads a pinned model or a member of a set in use: those stay loaded, and the journal says why.
- **Timings in the last line:** with llama.cpp, `prompt_eval_duration` and `eval_duration` are llama-server's own; with vLLM, which sends none, the router measures them (request to first word, first word to the end; a reply that is not streamed counts all of it as `eval_duration`). `load_duration` is the router's wait for a parked model. An engine answer the router cannot read ends the reply with Ollama's `error`, saying where.
- **Odysseus:** with the port set, its Cookbook's model scan finds mnm's models there too (`mnm url <name>` says what that gives it); chat with them through its `/chat/v1` endpoint, as [12.6](#126-odysseus) sets it up.

### 13.7 Workspace apps: one file each

The workspace apps (OpenCode, LibreChat, Open WebUI, OpenHands, Goose, Odysseus) get their whole file or card from `mnm url --for <app>`, with no instance name: every chat model it can use, the desk's roles first ([16.2](#162-sets)), so the file names `coder`, `chat`, `memory` and survives a model swap. Each card says which id to pick and why (`coder` for the agents), checks the context a request gets against the app's floor, and says how the key goes.

| `--for` | The file | `--write` writes it to |
|---|---|---|
| `opencode` | `opencode.json` (OpenCode 1.x): provider `mnm` (`@ai-sdk/openai-compatible`, `baseURL` the router on 127.0.0.1, each model's `limit.context`/`limit.output`) and mnm's MCP tools as the remote server `mnm` | `~/.config/opencode/opencode.json`; into a file of yours only `provider.mnm` and `mcp.mnm` change |
| `librechat` | `librechat.yaml`: the custom endpoint `mnm` (the router's `/chat/v1` from Docker, `fetch: true`, `models.default`, `titleModel`), speech for the desk's `ear`/`voice` with `allowedAddresses`, `mcpServers.mnm` and `mcpSettings.allowedAddresses`; the `.env` lines for RAG embeddings with `memory` beside it | `--to PATH` (its own place comes with `mnm apps install librechat`) |
| `open-webui` | A card: the desk's roles it is wired to (chat, speech, pictures, documents, MCP), its context and its first account; `mnm apps install open-webui` writes the settings | nothing: the install sets them from the desk at every start |
| `openhands` | The settings to enter: Custom Model `openai/coder`, Base URL, the MCP server (SHTTP) | nothing: Agent Canvas keeps them encrypted in its own `settings.json` |
| `goose` | `custom_providers/mnm.json` (the full `/v1/chat/completions` URL, `context_limit` per model) and the `extensions` block for `config.yaml` | `~/.config/goose/custom_providers/mnm.json` (the Windows app reads `%APPDATA%\Block\goose\config\custom_providers\`) |
| `odysseus` | `.env` lines `EMBEDDING_URL`/`EMBEDDING_MODEL=memory`, its LLM endpoint (`/chat/v1`) and Image endpoint (`/upstream/image/v1`), the MCP Tool Server card | `--to ~/odysseus/.env` merges the two lines (then `docker compose up -d` in `~/odysseus`) |
| `hermes` | A card only: the custom provider's base URL, `coder`, the key, and the 65,536-token floor Hermes Agent refuses to start below | nothing: mnm writes no file for Hermes Agent; enter the card's values as its custom provider |

In Odysseus, a role id loses what Odysseus reads from a model's name: thinking and tool support come from the family name (`qwen3` …), so the card says to name the instance itself there when that matters (a model swap then touches Odysseus), or to set the endpoint's `supports_tools` by hand; the `/chat/v1` endpoint lists no `memory`, `ear` or `voice` (at `/v1` they would show in its chat list: the role ids lack the `embed`/`whisper`/`tts` tokens it hides by).

`--write` writes a new file whole and replaces a file it wrote itself (it keeps a record under the state folder). A file of yours is merged: only mnm's keys change (a `.env` line by line, comments kept), the file as it was before mnm first changed it is kept once as `<file>.before-mnm`, and the change is shown (only the changed lines; a value whose key names a secret, a URL's login and every known secret's value are hidden). The record names each file by its real path: a relative `--to` and a symlinked config (stow, chezmoi) mean the file they reach, which is written through the link; a file keeps its mode, a new `.env` is 0600. A file it cannot read in its format (OpenCode's JSONC with comments), a YAML file with comments (a merge would lose them), or Goose's provider file written by someone else is left alone; `--force` replaces it, the copy kept.

**Context floors.** OpenHands needs 32,768 tokens per request and Hermes 65,536 (both refuse less); OpenCode's docs suggest 16k to 32k and its file carries the instance's own `limit.context`; LibreChat, Goose and Odysseus set none. A card that falls short says so with the `mnm set <name> ctx=…` that fixes it (the slots of a `parallel` instance counted).

**`mnm doctor --app <app>`** checks the router from where the app sits: OpenCode, Goose and Hermes Agent from this machine; LibreChat, Open WebUI and OpenHands from inside their container (`mnm-librechat-api-1`, `mnm-open-webui`, `mnm-openhands`, as `mnm apps install` names them; with curl, node or python3, whichever it has), or, when that is not running, from a one-off `curlimages/curl:8.11.1` container on Docker's default bridge (the curl project's own image, not a Docker Official Image, about 25 MB, pulled the first time; it says so before asking); Odysseus from inside the container `--container NAME` names, else a one-off. The lines: the model list holds the id its file picks; a chat with that id calls a tool and reads its result; an embedding with `memory` (the apps that embed); a rerank at the full `/v1/rerank` URL when the desk has `rerank`; MCP's `initialize`; the context floor; the key (with `MNM_API_KEY` set, a request without it must be refused, and the file must send it); and whether a file mnm wrote still says what it would write now. Each line is PASS, WARN or FAIL with a remedy; it exits 1 on a FAIL. MCP is asked without the key for Odysseus and Goose (they cannot send one). The key travels in the probe's environment, never on a command line: curl reads it from its stdin, node and python3 from their environment. An answer that is not what an OpenAI client reads is a FAIL saying where; a desk role whose instance is gone is named with the `mnm sets edit` that points it elsewhere.

### 13.8 Outside monitors

Both the router and the web UI answer `GET /__mnm/health` with the same JSON, for a monitor that watches the box: Uptime Kuma, Home Assistant, anything that reads a URL. It needs no key (it names instances, never a model file, an address or a key), is never cached, never journaled and never loads a model, and answers 200 whatever it says:

```json
{"status": "ok", "router": {"up": true, "loaded": ["qwen3-8b"], "dead": [], "busy": 0},
 "web": {"up": true}, "services": {"router": {"active": true, "restarts_last_hour": 0},
 "web": {"active": true, "restarts_last_hour": 0}}, "disk": {"models_pct": 41},
 "nightly": {"last": "<date>", "regressions": 0}, "checked_at": "<date>T03:14:02+00:00",
 "reasons": []}
```

`status` is `ok`, `degraded` (an engine died and is being restarted, or is parked after dying twice in ten minutes; a service restarted more than 3 times in the last hour; the models disk at the disk policy's first warning, 70% by default) or `down` (only the web UI says it: the router does not answer). `reasons` says why, one line each. The router's is the one to watch from another machine (`http://<box>:7801/__mnm/health`; under WSL's NAT only Windows and its containers reach it, [13.3](#133-from-another-machine-on-your-network)); the web UI's listens on 127.0.0.1 only and is the one that can say `down`.

**Uptime Kuma:** a monitor of type *HTTP(s) - Keyword*, URL `http://host.docker.internal:7801/__mnm/health` (Uptime Kuma in Docker on this machine) or the box's address from another one, keyword `"status": "ok"`, interval 60 s. A `degraded` box then reads as down in Kuma, with the JSON (and its `reasons`) in the event.

**Home Assistant:** a REST sensor in `configuration.yaml`:

```yaml
rest:
  - resource: http://192.168.1.20:7801/__mnm/health
    scan_interval: 60
    sensor:
      - name: mnm
        value_template: "{{ value_json.status }}"
        json_attributes: [reasons, router, disk, nightly]
```

An automation on `sensor.mnm` not being `ok` then tells you on your phone. The box's own notices (ntfy, the Windows toast: `mnm notify`) say the same things as they happen.

## 14. The router

### 14.1 One URL for every model

Every model answers through mnm's router on port 7801. A request names its model by the instance's name (the `model` field), and the router:
- **forwards** it when the model is loaded;
- **loads** a parked model first (the request waits 7–60 s; vLLM's first start takes minutes), making room if needed by parking the instances the fit verdict names ([16.1](#161-making-room));
- **refuses** a model you stopped with `mnm stop`: 409 until you start it again. Stopped stays off.

`/chat/v1` lists only the chat-like models; `/upstream/<name>/` reaches one model's own routes (engine pages, `/sdcpp/v1/…`). A model list (`GET /v1/models`) never loads anything, so Odysseus refreshing its lists or a browser tab costs nothing; each chat or embedding model's entry carries the context a request gets (`context_length`), which Odysseus trims its prompts to. Chat replies stream as they are generated.
- **While a parked model loads,** a streamed chat request gets a line every 2 seconds in the reply's thinking ("Loading qwen3-8b: 12 s, about 5.1 GiB on the card …"), then the answer. It keeps Odysseus from giving up on a long load (it stops waiting for a first word after at most 60 s). A load that fails ends the reply with the reason. A client that does not want the lines sends `X-Mnm-Quiet: 1`.

What it did is in its journal, `journalctl --user -u makenomistakes-router` (and in `mnm report`'s folder): one line for each load ("Loading tiny for a request.", "tiny ready in 3.2 s.", and what it parked for the room), each idle unload, each `/mcp` call, and each answer it gives itself that is not a success, with the path, the model and the reason (`/v1/chat/completions POST for qwen3-8b from 127.0.0.1: 409 (qwen3-8b is stopped: …)`). A reply the engine gives is passed on without a line, but for a chat answer that spent every token thinking (empty `content`, `finish_reason` `length`; once per model per start, §11's *Thinking*).

### 14.2 Idle unload

The router unloads a model 10 minutes after its last reply finished, with nothing in flight, and loads it again at its next request. `mnm set <name> idle=1800` changes the wait; `idle=0` never unloads it. It applies at once (no restart). A model you started by hand counts too; its timer starts when the router first sees it. (`idle-sleep`, llama.cpp only, is different: the engine itself gives its VRAM back while idle and the container stays.)

### 14.3 Pinning

`mnm set <name> keep=always` pins a model: the router never unloads it, for idleness or to make room for another, and a load that needs its room is refused with the reason. `keep=auto` unpins it. Pin the one model every request should find loaded. `mnm pin <name>` is the same as `keep=always`; `mnm pin <name> --for 2h` keeps it loaded two hours after each request instead (`idle=7200`, it can still be unloaded for room); `mnm unpin <name>` is `keep=auto` with the ten-minute idle again. A set can be pinned as one ([16.2](#162-sets), *persistent*).

### 14.4 A key for the router

Optional on this machine and its containers, required once the router is on your network. `mnm setup` makes one when you say yes, or add `MNM_API_KEY=<a long random string>` to `~/.config/makenomistakes/secrets.env` (`openssl rand -hex 24` makes one). From the next request on, the router asks every request for it (`Authorization: Bearer <key>` or `x-api-key`), with no restart.
- The router reads it from that file only, never from a shell's environment, so the service, a router run by hand, the cards and `mnm try` agree.
- Put the same string in each client's API key field. Cards say "your MNM_API_KEY" and never print it; the curl lines read `$MNM_API_KEY` from your shell (`export MNM_API_KEY=…` first).
- `mnm doctor` says whether one is set, and FAILs without one wherever port 7801 is on your LAN (WSL's mirrored networking, or a native box with `MNM_ROUTER_HOST=0.0.0.0`): off this machine the key is required ([13.3](#133-from-another-machine-on-your-network)).
- The web UI stays on 127.0.0.1 and needs no key. A browser cannot send the key, so with one set the engine pages open at the engine's own port, `http://127.0.0.1:<port>/`, without the router: start the instance first.
- To remove it: delete the line (`sed -i '/^MNM_API_KEY=/d' ~/.config/makenomistakes/secrets.env`).

### 14.5 Who may reach it

`MNM_ROUTER_HOST` in the secrets file: `0.0.0.0` (the default, every interface: under WSL's NAT that is still only this PC and its containers), `local` (127.0.0.1 plus Docker's bridge gateway: this machine and its containers only), or a comma list of addresses (127.0.0.1 is always added). `mnm service install` applies a change; `mnm doctor` warns when the running router uses another value than the file. Details for a native box: [section 3](#3-native-debian-and-ubuntu).

Without a key, whoever reaches the router can load any model, and through Ollama's API ([13.6](#136-ollamas-api)) also unload one (`keep_alive: 0`), switch off its idle unload (`keep_alive: -1`) or clear a client's `keep_alive` (`POST /__mnm/saved/<name>`, what `mnm pin`/`unpin` send). A pinned model and the members of a set in use stay loaded whatever a client sends. On a LAN, the key ([14.4](#144-a-key-for-the-router)) closes all of it.

### 14.6 A request's own keep_alive

A chat, completion or embedding request may say how long its model stays loaded, as Ollama's API does ([13.6](#136-ollamas-api), the same meaning): `"keep_alive"` in the JSON body, or the `X-Mnm-Keep-Alive` header (which wins). `0` parks the model once the reply is done, unless the router keeps it (pinned, a member of a set in use, other replies in flight: the router's journal says which); `-1` (any negative number) switches off its idle unload; a duration (`5m`, `2h`, `1h30m`, or seconds) is its idle wait. What is saved and what is not: `mnm pin`, `mnm unpin`, `mnm set <name> idle=…` and `mnm set <name> keep=…` are saved in the instance's recipe; a request's `keep_alive` lives in the router's memory only, until the router restarts or until you save `idle` or `keep` again, which wins at once. A recipe's `keep=always` stays whatever a request says, and `-1` is not a pin: the router may still park the model to make room for another (`mnm pin` is). The router takes the field out before the engine sees it; a value it cannot read is a 400 that says what to send.

### 14.7 Metrics (`/metrics`)

`GET http://localhost:7801/metrics` (behind the key, like every route) is one Prometheus page for the box: `mnm_instances{state="loaded|parked|stopped"}`, `mnm_vram_held_bytes{model=…}` (each loaded model's estimate), `mnm_loads_total`, `mnm_parks_total` (for room and idle unloads), `mnm_refusals_total{status=…}` (errors the router answered itself), then each loaded llama.cpp and vLLM model's own counters with `model="<name>"` added (llama.cpp's speculative ones among them). An engine that does not answer within 2 s is left out and reads `mnm_engine_metrics_up{model=…} 0`. Point a Prometheus or Grafana at it.

### 14.8 The manifest (`/__mnm/manifest`)

`GET http://localhost:7801/__mnm/manifest` (behind the key) is what an app needs to know without being told by hand, as JSON (`schema_version` 1): the desk and its roles; whether the key is required; the router's OpenAI base, Ollama and MCP URLs from this machine, from a container (`host.docker.internal`) and from the LAN (none under WSL's NAT networking or `MNM_ROUTER_HOST=local`; `lan_note` says why); and every instance with its roles, type, state (loaded, parked, stopped, not downloaded), the request paths it answers, the context a request gets and the slots that share it, a reply budget, whether its chat template declares tools (read from the model file), vision, and a voice model's voices. `mnm router manifest` prints the same, built without the router; the `--for` files ([13.7](#137-workspace-apps-one-file-each)) are made from it.

### 14.9 The queue: who waits, and for how long

When two requests ask for one model at once, the second waits. The router counts every request that makes something (a chat, a completion, an embedding, a picture, a transcript, speech, a song) from the moment it arrives until its answer ends, on the OpenAI routes and Ollama's:
- **waiting for it to be ready:** past a second for its model: parked and loaded first, its turn behind another start, or the card taken back from a tune or a training;
- **waiting for a slot:** the engine already holds as many requests as it serves at once. llama.cpp serves as many as its slots (`-np`: every plan has one, the whole context for one request; `mnm set <name> parallel=2` splits the context in two), vLLM `--max-num-seqs` (4), and sd.cpp, whisper.cpp and audio.cpp one at a time. The engines take them in order, so the router knows when each one's turn comes.

What waits now is on the Status board's Router panel ("2 waiting on coder (for a slot; 1 answering)"), in the status bar ("· 2 waiting"), under `mnm ls` ("Router: 2 waiting on coder …"), in `/__mnm/busy`'s `queue` (only while something waits) and on `/metrics` (`mnm_requests_waiting{model,for="load|slot"}`, beside `mnm_requests_waited_total` and `mnm_request_wait_seconds_total` since the router started; llama.cpp's own `llamacpp:requests_deferred{model=…}` comes with its page and says the same from inside, but for an instant as one answer ends and the next begins: the router counts a request from when it hands it over until its answer has gone out).

**The readings.** The router keeps each model's requests in windows: a burst with less than two minutes between them, within one container start, cut at an hour whatever is in flight (the rest goes on in the next window). Two minutes after a window's last request its sweep saves it as a reading of this box (`mnm calibration`'s *queue* table, and System → Calibration's): the requests that reached the engine (and how many never did: refused, or the client gone first), the most at once, how many waited for a slot and how long (longest, mean), how many waited for a load, **llama.cpp's own wait** (an answer's time at the engine less the prompt and generation time its own `timings` give: deferred behind others, its prompt cache's copy) and the **context held** (the tokens one request held, prompt and reply, against what one request may hold). A window counts under the settings its container started with (a `mnm set` saved since applies at the next start), and the windows of one settings are summed, so a model given more slots later gets a row of its own. A burst that never reached the engine (a stopped model a client keeps asking) is no reading. A window still open when the router stops is lost.

Nothing is refused for waiting, and no slot count changes by itself. A chat model that often waits for a slot can serve two requests at once (`mnm set <name> parallel=2`); each slot then gets half the context, so the *context held* column says whether its requests would still fit.

## 15. Changing an instance

### 15.1 Settings

```bash
mnm show qwen3-8b                    # every setting, its current value, what it does, the advice
mnm set qwen3-8b ctx=32768 kv=q8_0   # saved; applies on the next start
mnm stop qwen3-8b && mnm start qwen3-8b
```

The instance page's Settings tab has the same settings as a form. A value that does not make sense (an unknown voice, a width that is not a multiple of 16, `speed` on PocketTTS) is refused with the reason. The settings each type has:

| Type | Settings |
|---|---|
| chat, coding, vision | `ctx` (context length), `kv` (f16, q8_0, q4_0: KV cache precision), `gpu-layers`, `flash-attn`, `parallel` (requests at once; each gets ctx / parallel), `threads`, `batch`, `ubatch`, `cpu-moe-layers`, `reasoning` (auto or none), `jinja` (the model's own chat template; needed for tools), `load` (how the weights are read: empty is llama.cpp's own `auto`; `none` reads them into RAM, which llama.cpp suggests when MoE experts stay in RAM; `mmap`, `dio`), `idle-sleep`; vision adds `image-min-tokens` |
| embedding, rerank | `ctx`, `gpu-layers`, `parallel`, `threads`, `batch`, `ubatch`, `pooling`, `load`, `idle-sleep` |
| image, image edit | `width`, `height`, `client-size`, `steps`, `strength`, `cfg-scale`, `sampler`, `scheduler`, `seed`, `clip-skip`, `negative-prompt`, `guidance`, `flow-shift`, `flash-attn`, `diffusion`, `offload`, `vae-tiling`, `text-encoder-cpu`, `vae-cpu`, `threads` |
| video | the image settings without `client-size` and `strength`, plus `frames`, `fps`, `high-noise-steps` |
| speech-to-text | `language`, `convert`, `threads` |
| text-to-speech | `voice`, `speed` (Kokoro only), `seed`, `threads` |
| vLLM (chat, vision) | `ctx`, `kv` (auto or fp8), `parallel` (sequences batched together), `eager` (skip CUDA graphs: less VRAM, slower replies); the share of the card is sized at each start |
| every type (the router's) | `idle` (seconds idle before the router unloads it; 0 never), `keep` (`always` pins it) — these two apply at once |

Chat, coding and vision instances also take a draft model's `draft`, `draft-max` and `draft-p-min` ([15.6](#156-a-draft-model-speculative-decoding)), the prompt cache's `cache`, `cache-ram` and `cache-disk`, and `moe-cache`, a cache on the card of the experts kept in RAM ([11](#11-coding-agents)); every llama.cpp instance takes `numa` ([25.14](#2514-big-ram-epyc-threadripper-and-intel)). Several cards add `experts`, `card-experts`, `gpus` and `split` ([25.7](#257-the-settings)), an AMD card that runs both backends adds `backend` ([25.11](#2511-amd-cards)), and the studio's song, stem and voice instances have their own ([28.1](#281-the-first-song)).

### 15.2 Auto-fix

If an engine dies out of memory at start, mnm replans with the fit model and tries again: at most three attempts in all. It may lower the context, the KV precision or the layers on the GPU (or, for pictures, move parts off the GPU). It saves the change and prints *Changed and saved: …*. To undo, `mnm set` the old value back. A start that would not fit beside mnm's own running instances is refused before anything runs, so auto-fix never shrinks a model just because another one was up.

### 15.3 Start-time upgrades

A start may also improve a recipe, and says so: a chat recipe saved one layer short of "everything on the GPU" gets that layer; an image instance whose whole-picture decode does not fit the free VRAM gets VAE tiling turned on.

### 15.4 Plan it again (`replan`)

A saved recipe keeps the quant, context and offload it was planned with, so it never changes under you. After a hardware change, or after adding models while another program held the card, plan it again:

```bash
mnm replan qwen3-8b              # lists what moves, asks, saves; applies on the next start
mnm replan --all                 # every instance, one after another (asks for each)
mnm replan --all --yes           # every instance, saving without asking
mnm replan qwen3-8b --shrink     # takes a smaller plan too while the saved one still fits
mnm replan qwen3-8b --any-quant  # may pick a bigger quant too (that one downloads)
mnm replan qwen3-8b --vram 24    # plan for a card you do not have yet
```

A re-plan always plans from scratch, but while the saved plan still fits the card it never takes a smaller one: "qwen-awq still fits the card (tight); the plan now would be smaller (max_model_len 16384 → 8192): kept as it is", since a plan sits near thresholds and 0.2 GiB less free one day must not halve a context the next day doubles again. `--shrink` takes the smaller plan; a saved plan that no longer fits is planned smaller without it, and the line before *Saved* says what gets smaller.

Only what the plan sizes to the card moves (context, KV precision, GPU layers, picture size and offloads, vLLM's share, and the prompt cache's RAM and disk caps, [11](#11-coding-agents)); a recipe without those caps gets the plan's even when the re-plan keeps the rest as it is (`cache=on|off` stays yours once set); the name, port, type, profile, quant, LoRAs, swap group and your own settings (seed, negative prompt, sampler, steps, slots, voice, idle) stay, and a coding instance keeps its context. The instance page has a **Re-plan for this machine** button (Settings tab, with *any quant*): it shows the same list and saves only when you say so.

### 15.5 What cannot be set

The quant, the model version and the type: remove and add again. The downloaded files stay when you remove an instance, so only new files download. There is no rename.

```bash
mnm rm qwen3-8b && mnm add unsloth/Qwen3-8B-GGUF --quant Q5_K_M --name qwen3-8b
mnm add unsloth/Qwen3-8B-GGUF --type coding --name coder2      # the same model as another type
```

### 15.6 A draft model (speculative decoding)

A small model of the same family can draft a few tokens that the instance's model checks in one pass: the same reply, often faster (on an RTX 4070, Qwen2.5-7B with a 0.5B draft went from 66 to 92 tok/s in llama.cpp's own discussion). Add the small one as an instance of its own, then:

```bash
mnm add unsloth/Qwen3-0.6B-GGUF --quant Q4_K_M --name qwen3-tiny   # downloaded: a draft needs its file
mnm set qwen3-8b draft=qwen3-tiny     # draft-max=3 and draft-p-min=0.75 by default
mnm start qwen3-8b && mnm try qwen3-8b --speed   # "…; the draft kept N of M drafted tokens (P%)", then whether it gains here
mnm set qwen3-8b draft=               # run it alone again
```

**An MTP head instead:** some repos ship a multi-token-prediction head beside the model (`mnm resolve` lists it as *MTP head*; unsloth's Qwen3.8-27B and Gemma 4 do). mnm downloads it with the model, and `mnm set <name> draft=mtp` drafts with it (`--spec-type draft-mtp`): no second instance, the same vocabulary by construction. *Fits now* adds the head's weights, its own KV and, on a hybrid model (Qwen3.5 and later), a copy of the small recurrent state per drafted token. `mnm show` lists the head under the parts, unused until then.

The draft must be a downloaded llama.cpp chat, coding or vision instance, smaller than the model, with the same vocabulary (llama.cpp's own rule: the same kind, the same begin and end tokens, sizes within 128 tokens, the same token texts); `mnm set` refuses anything else and says which part differs. Its file is kept on the instance, so removing the small instance leaves the draft working (`mnm rm --files` keeps the file). Every layer of the draft goes on the GPU, and its KV cache follows the instance's context: *Fits now* adds both. The planner's tok/s stays the model's own: what a draft gains depends on the text (lists and code gain most, a story at a high temperature least), so `mnm try --speed` is the measure: with a draft it ends with one line, "The draft gains here (…)" or "The draft loses here (27 tok/s against 76 alone, 8% of its tokens kept): `mnm set qwen3-8b draft=` removes it", against the instance's last reading without the draft. `mnm show` lists the draft under the parts.

**What a draft gains.** Measured on an RTX 4070, qwen3-8b with Qwen3-0.6B drafting 16 tokens a step fell from 75.6 to 27.1 tok/s: Qwen3 samples (temperature 0.6 by default), the small model's guesses were kept 8% of the time, and every rejected token was checked for nothing. So by default a draft drafts at most 3 tokens a step (`draft-max`, llama.cpp's own default) and only tokens it gives at least 0.75 (`draft-p-min`; llama.cpp's own is 0), with its KV cache at f16. A draft can win where the reply is predictable: greedy or low-temperature requests (`temperature` 0 to 0.3), code, lists, structured output, a repeated document; on free chat at the model's own sampling, expect little or a loss. Readings taken with a draft are kept apart from the model's own, so a draft never makes the model alone look larger in *Fits now*.

### 15.7 Tuning (`mnm tune`)

`mnm add` plans an instance from what the card and RAM hold; the planner cannot know how fast each setting runs on *this* box. So after `mnm add` a llama.cpp chat, coding or vision instance is queued for **tuning**: the line under the plan says "Queued for tuning: runs when the card is idle". `mnm add … --no-tune` leaves it out.

What a tune does: it starts a copy of the instance (`tune-<name>`, never your own container) with mnm's plan, then with one setting changed at a time (the experts kept in RAM, the expert cache, the micro-batch, the threads, the KV cache, a draft, the layers on the card; where they apply, the backend on an AMD card with both images pulled, each card's experts on several cards and `numa` on several memory nodes), and measures each on fresh prompts: about 2,000 tokens in, 256 out, three times. It keeps a change only when it is at least 3% faster. A chat instance is scored on its reply speed with a little weight on reading the prompt; a coding instance also on a 16,000-token prompt, the size of an agent's first request (`--for chat` or `--for agents` overrides). Up to 24 trials or 45 minutes.

When it runs: only once the router has been idle for a minute, nothing is starting or stopping, and ComfyUI is not running a workflow. It parks what the router would park for a load, and never a pinned model or a member of a set in use (while one of those holds the card, the tune waits). **The first request for any model takes the card back**: the trial stops, your request is served as usual, and the tune carries on later from its last whole trial. The request waits for the job to stop (a trial at once; a training of the studio's up to 30 s, while it writes its checkpoint), then at most 60 s for the card.

```bash
mnm tune tiny --now                 # now, without the idle minute (it still yields to a request)
mnm tune coder36 --now --minutes 20 # a shorter budget
mnm tune --all                      # queue every instance it tunes
mnm tune --status                   # the queue, the tune running, each result
mnm tune tiny --cancel              # unqueue it and stop its tune
```

With the router running, `mnm tune` asks it; without, `--now` runs the tune in your terminal with one line per trial.

Reading the result: `mnm ls`'s Settings cell says **tuned +18%** (the tune's score over mnm's plan, both measured), `mnm show` lists what it chose, `mnm calibration` has a *tuned* table, and the instance page has a **Tuning** tab with every trial. A tune holds for what it was measured on: the engine build, the card, the driver, the RAM, the files. When one changes, `mnm ls` says **tuned for another build** (or card, files, machine) and the tune is queued again; `mnm engines update` queues it when it brings a new build. Your own setting wins: `mnm set <name> ubatch=512` (any setting the tune measured) drops the tune and says so; `mnm tune <name>` tunes again from your settings. `mnm replan` keeps a tune while the new plan still fits with it. vLLM instances are planned, not tuned.

### 15.8 Tuning for quality (`mnm tune --quality`)

A speed tune keeps the quant `mnm add` chose and finds the fastest settings for it. A **quality tune** asks the other question: which of the model's quants answers most like the model itself, and still runs fast enough here? It is never automatic: you ask for it, it shows what it would download, and nothing moves until you say yes.

```bash
mnm tune qwen3-8b --quality                     # the plan, then "Download what is missing and measure?"
mnm tune qwen3-8b --quality --floor 25          # this tune's speed floor, tok/s
mnm tune qwen3-8b --quality --quants Q6_K,Q5_K_M  # these quants, and the current one
mnm tune qwen3-8b --quality --keep              # keep the quants it does not choose
mnm tune qwen3-8b --quality --yes --now         # no question, no idle minute
```

What it measures: each quant is scored against **Q8_0** (or the most faithful quant this box can run, never a 16-bit one to download; an instance already on a more faithful quant, BF16 or unsloth's UD-Q8_K_XL, is its own reference) by llama.cpp's own `llama-perplexity` in the official `full` image of your backend. On a text the model did not train on (WikiText-2's test set for a chat or vision instance, mnm's own source code for a coding one), 24 chunks of 512 tokens, it reads the reference's answers once and then how far each quant's answers drift from them: the **KL divergence** (lower is closer; 0 is the reference itself), how often the most likely next token stayed the reference's, and the perplexity beside them. Each quant's decode is measured too, as a speed tune's trial measures it: a copy at the plan that would serve it, three fresh prompts. Only the quant and where it sits (layers on the card, experts in RAM) move: your context, KV cache, flash attention, batches, slots and threads stay as they are, and the quant the instance runs now is measured exactly as it is saved, so keeping it keeps what was measured. A later `mnm set <name> kv=q8_0` (any change of the KV cache) drops the quality record (its scores were measured with the KV cache it had); a speed tune leaves a quality-tuned instance's KV cache alone.

What it keeps: the quant with the lowest KL divergence whose measured decode is at or over the **floor**: 15 tok/s for a chat or vision instance (`MNM_FLOOR_CHAT`), 20 for a coding one (`MNM_FLOOR_AGENT`); `--floor` sets another for one tune. Two quants whose scores differ by less than twice their combined error are the same quality as far as the text can tell, and the faster of them is kept. When none reaches the floor, nothing changes and the line says which was fastest. When a quant is kept, the instance's recipe moves to it (from its next start or load), `mnm ls` says **quality-tuned**, and a speed tune is queued for its new settings.

Which quants: by default one per bit tier from the reference down to the current quant (at most four), the current one always among them, plus up to two tiers below when the planner reads the current one under the floor. A card busy with other programs while it plans is said beside the plan: a quant that would fit once more of the card is free may be missing. The downloads are shown before anything is fetched, with the disk the reference's log-probabilities need while it runs (about 1.9 GB for Qwen3's vocabulary), and refused when the models disk cannot hold them; a quality tune already queued for the instance is never replaced (`mnm tune <name> --cancel` first), and a speed tune asked meanwhile, from the CLI, the web or the nightly, router or not, is refused with that same line. Saying yes writes the plan into the tuner's queue before anything downloads, so another approval, a speed ask or the nightly sees it at once: while it downloads, `mnm tune --status` says *downloading* with the process doing it, another approval of the same plan is refused until that one ends, and `mnm tune <name> --cancel` asks it to stop once the file under way is in. mnm keeps a ledger of the quants each queued plan measures and of the ones a quality tune downloaded, from before the first byte until the plan ends: after the tune, the quants it downloaded and did not choose are deleted, and the one the instance moved away from, each once no other plan holds it and no other instance or ComfyUI workflow uses it (`--keep` keeps every quant it measured, as your own); a delete that cannot run then (a recipe mnm cannot read, a full disk) is done by the router's next sweep or when the next quality tune ends. A cancelled or failed tune, a download cut short (Ctrl-C included) or a cancel while it downloads deletes only what it downloaded (a failed download says so): a quant that was on disk before is never its to delete, the instance's own included, even when the instance was removed meanwhile with `mnm rm` (without `--files`). An approval that stopped mid-download (the web UI restarted under it, a terminal killed) is ended by the router, its downloads with it. A second approval of the same plan once the first one queued it joins it and downloads nothing. Once its downloads are in, the plan is the router's: a plan the router cannot run (a quant's files gone, the instance changed) ends with what it downloaded, and any other refusal (a router without a tuner, a key it lacks) or a router that does not answer in time leaves it queued, said, for the router's tuner or `mnm tune <name> --cancel`. A cancel is judged when it takes the tuning folder's lock, never on what it saw before: a plan cancelled before a run took it never runs (after the router read its queue, or between the approval and a terminal's `--now`); one the router or the web UI runs is cut there, and a terminal's run goes on, the cancel refused in words: Ctrl-C in that terminal stops the run and leaves the plan queued (a router started later resumes it), and `mnm tune <name> --cancel` then ends it.

When it runs: as a speed tune does (the router's tuner, once the card has been idle a minute; the first request for any model takes the card back and the tune resumes from its last whole step). The approved plan waits in the tuner's queue on disk and the router runs exactly that plan; with no router, `--now` runs it in the terminal (the web UI leaves it queued for the router), and a router started meanwhile leaves it to that terminal; a second terminal's `--now` never runs a tune another process runs, it says where it runs and leaves it there. `mnm set`, `mnm replan` or the web's Re-plan of an instance with a queued quality tune says the tune was planned on the settings before and will be dropped when it runs; `mnm upgrade` refuses until it is cancelled. `mnm tune <name> --cancel` (or the Tuning tab's Cancel) stops it from wherever it runs: a tune nothing runs ends at once; one that runs is marked cancelled and ends with nothing saved (the router's own run stops its step at once, as does the web UI's when its own Tuning tab cancels it; any other run, a terminal's `--now` say, after the trial or step under way), and a request or the nightly coming meanwhile never turns the cancel into a pause; one already saving its result says it is too late; a run stopped before it ended a cancelled tune (Ctrl-C) leaves it to the router's next look or a second cancel. A run that stops while it saves its result (killed, or the router restarted by `mnm update`) leaves the tune to whatever touches it next (a router's look, `mnm tune` or an approval asked again, a cancel): the result is saved by its next run when the save had not landed, and the tune ends when it had, a quality tune's then queueing the speed tune for its new quant as its own end would have; while its recipe cannot be read it stays as it is and says why (while any recipe cannot be read every queued tune waits, since what holds the card cannot be listed, each saying why once), and `mnm tune <name> --cancel` ends it the safe way, from the router too: nothing it may still list is deleted, and when the save had landed, the quant it moved the instance away from stays on disk with nothing listing it (`mnm disk` counts it among the unlisted files) and no speed tune is queued for the new quant (`mnm tune <name>` asks for one). A cancel of an entry mnm cannot read (another version of mnm wrote it) removes it. `mnm tune --status` shows its steps (*Q6_K's quality*, step 4 of 8) and a tune its run is ending (*cancelled, stopping after its step*), or one its run left (*saving, its run stopped (a router's next look settles it, as does `mnm tune <name> --cancel`)*; System says *whether its save landed cannot be read now (why): `mnm tune <name> --cancel` ends it* while its recipe cannot be read, by the disk too). A delete the router cannot do is said once in its journal, not at every tick, and again only when it changes.

Reading the result: `mnm show` has a *quality* line (the kept quant, its score against the reference, its decode and the floor), `mnm calibration` and System → Calibration a *quality* table (every quant measured on this box), and `mnm fit`, `mnm add` and `mnm what` print what this box measured of a repo's quants beside the planner's estimates. On the instance page's **Tuning** tab, **Optimise for speed** queues a speed tune and **Optimise for quality** shows the plan with its downloads and a **Download and measure** button. vLLM, picture, video and speech instances are not tuned.

### 15.9 Why this plan (`mnm why`)

`mnm why <name>` says in words why an instance runs the way it does. It plans the instance again for this box, read-only (it reads the model's header from its source, as `mnm replan` does, and saves nothing), and says:
- **what it runs:** the repo, the quant, the profile, the context (per request, when it serves several at once); an instance added with `--quant`, `--ctx` or `--for-agents` runs *its own settings*, sized from balanced, and the words say which of them a re-plan keeps (its quant, where the engine's quant is a pick; its context, for a coding instance);
- **why:** the profile's own reason (or that it was added with its own quant or context, which a re-plan keeps); what a re-plan keeps that the profile alone would not pick (the quant it downloaded or a quality tune chose, which `mnm replan <name> --any-quant` would trade for the profile's; a coding instance's context); how it fits (comfortably, tightly, only with part of it in RAM) and how much of the card, the RAM or the unified memory it takes of the room its planner judged (a llama.cpp plan keeps WSL's margin out of it; under a learned correction both figures are the card's own), and what that room counts as free; the experts held in RAM; the planner's speed in the instance's own unit (t/s for text, *x realtime* for a voice, seconds of work a second of audio for a song model, s/image or s/clip for pictures and video); and what the other profiles would pick instead. When other programs hold part of the card, that is said first, in `mnm replan`'s words; when a plan for the box now differs from what the instance runs, a line says these reasons are that plan's, not the saved one's;
- **now:** in `mnm replan`'s own words: what a plan for the box as it is now would change (`mnm replan <name>` saves it), or that it is planned as it is, or that a smaller plan is held back (`mnm replan <name> --shrink` takes it, [15.4](#154-plan-it-again-replan)); a tune a re-plan would keep or drop; a quant the repo no longer lists is said with `--any-quant`;
- **more room:** what 4 GiB more (`--more N` for another figure) would change, planned for real on this box's budget grown (its learned correction, and the RAM a unified pool or a box without a card plans in, grow with it), each change beside what it is now; when the profile alone would pick another quant there, the line names it. On several cards an engine that runs on one card (pictures, speech, songs) is planned on its card; for the others more room is said not worked out.

```bash
mnm why qwen3-8b            # the plan in words
mnm why qwen3-8b --more 8   # what 8 GiB more would change
```

A model whose source does not answer, or a card that cannot be read, is said instead of the reasons (exit 1). On several cards the *more room* line says it is not worked out. The instance page's Overview has the same words in its **Why this plan** panel, planned when you click **Explain this plan** ([17.3](#173-an-instances-page)).

## 16. Several models at once: sets, and making things

### 16.1 Making room

12 GB is not much: an 8B chat model takes about 6 GB, a 1024x1024 picture model 6 to 10 GB. So:
- `mnm ls` (or the web Status tab) shows *Fits now* for each stopped instance, against what is free at this moment.
- When something won't fit, the verdict names mnm's running instances that hold the VRAM: *Parking qwen3-8b (about 5.8 GiB) makes room.* A parked one reads *on request* instead: the router parks those at its first request, so there is nothing to do by hand.
  - `mnm start qwen3-vl` then refuses before anything starts.
  - `mnm start qwen3-vl --stop-others` parks exactly those instances, each after the replies it is writing (up to five minutes), then starts. It never parks a pinned instance or a member of a set in use, as the router does not: when only those hold the room, it refuses and names them (`mnm stop` them by hand, or `--anyway`).
  - `--anyway` starts it beside them (it may run out of memory).
  - The web UI has *Make room and start* and *Start anyway*.
- A request through the router does the same make-room step by itself, by the same rules.
- `mnm stop <name> --park` (or `mnm stop --all --park`) unloads and leaves the model loadable: the router loads it at its next request. Plain `mnm stop` means off: the router refuses it (409) and Odysseus lists no model for it.
- VRAM held by other programs is not mnm's to free; mnm says how much is held and plans around it.

### 16.2 Sets

A *set* is a named group of instances with a role each (`mnm workflow …` is another name for `mnm sets …`). It starts and stops its members together, is checked to fit together, and gathers every member's cards for one app on one page. Odysseus and the agents only ever see the instances. To *make* something with a member, use its **Run** tab ([16.5](#165-making-things-run-runs-and-clean-up)). Elsewhere in this guide a *workflow* is an imported ComfyUI workflow ([18.4](#184-comfyui)).

```bash
mnm sets create studio chat=qwen3-8b image=zimage voice=pocket
mnm sets show studio          # each member's verdict and one "fit together" verdict
mnm sets start studio         # the largest first; mnm's other loaded models parked first if they hold the room
mnm sets url studio           # one card set per member, under its role
mnm sets stop studio
mnm sets edit studio voice=pocket --drop image   # change members in place: a new set's checks
mnm sets rm studio            # the instances stay
```

The set's page has the same **Members** form: pick an instance per role, empty a role to drop it. In a set whose members take turns ([16.4](#164-taking-turns---swap)), a member that leaves is stopped and leaves the turns; one that joins must be stopped first.

| Role | Takes | What it is for |
|---|---|---|
| `chat` | chat, coding | the model you talk to |
| `coder` | coding, chat | the model a coding agent drives |
| `vision` | vision | a chat model that takes pictures in |
| `image` | image | text to picture |
| `edit` | image edit, image | picture in, picture out |
| `video` | video | text or a still to a clip |
| `voice` | text-to-speech | reads answers aloud |
| `ear` | speech-to-text | transcribes what you say |
| `memory` | embedding | Odysseus's memory and documents |
| `rerank` | rerank | re-orders retrieved passages |

An instance plays one role per set; a role is given once.

**The desk: roles as model ids.** `mnm sets default desk` makes one set *the desk*: its roles become model ids every client can name. `/v1/models` lists `coder`, `chat`, `memory` … beside the instances (each with `meta.alias_of`, the instance that answers it), a request with `"model": "coder"` goes to the desk's coder (loaded like any parked model), `/upstream/coder/v1` is its own URL, and Ollama's API lists and serves the roles too. `mnm sets edit desk coder=<another>` then swaps the model behind `coder` and every app's config keeps working; the router's journal says *coder is coder36 (the desk's coder).* when the instance behind a role changes. A role of no default set is no id. `mnm sets default` says which set is the desk, `--clear` makes none; `mnm sets ls` marks it, and `mnm sets rm` of the desk clears it. An instance's own name always wins: a role word is refused as a new instance's name (`mnm add … --name coder` says to pick another), and a set whose role an existing instance is named after cannot be (or become, by `mnm sets edit`) the desk; the refusal says how to give that instance another name. `mnm backup` carries the choice. Each role loads at its request and the router parks another role's model to make room (a desk need not fit the card whole: `mnm sets ls` reads *on request* when each member fits alone); only `mnm sets start desk` makes the members hold one another, until `mnm sets stop desk`.

**Persistent sets.** `mnm sets create desk chat=qwen3-8b memory=embed --persistent` (or `mnm sets edit desk --persistent`, `--no-persistent`; the set page's *Make persistent*): while a member runs, nothing unloads it, not to make room for another model (the router, `--stop-others`, a ComfyUI run) and not when it is idle. `mnm stop` still stops it; a request's `keep_alive: 0` leaves it loaded, as it does any member of a set in use. A set whose members take turns cannot be persistent.

**Fit together.** The verdict counts the members that are not running against the VRAM free now: *fits*, *near limit*, *won't fit* (it names the member to leave out: *Leaving out zimage (about 6.3 GiB) lets the rest fit*), or *unknown*. When the whole sum does not fit, a picture or video member whose weights can stream from RAM is priced at its floor (only one member streams, and its weights must fit the RAM less 2 GiB), so the verdict can be *near limit* ("they start, slower"). When mnm's other running models hold the room, `mnm sets start` (and the set page's Start) parks them first, as the router would for a request: the largest first, only as many as the members need, never a pinned model or a member of another set in use, and it says so ("Parked zimage to make room for desk."). What still does not fit is refused, naming what holds the rest; `--no-room` parks nothing and refuses instead, naming what it would park; `--anyway` starts it regardless. The router never parks one member of a set in use to load another; a set is *in use* only between `mnm sets start` and `mnm sets stop` (or `rm`, or `mnm stop --all`): a member the router loaded for a request, or one started by hand, holds nothing.

### 16.3 Set recipes

- **Studio**: `chat=qwen3-8b image=zimage voice=pocket`, created with `--swap` on a 12 GB card. Planned on an RTX 4070, qwen3-8b (ctx 32,768) is estimated at about 7.4 GiB and zimage (its text encoder on the GPU) at about 8.6 GiB: 16.0 GiB together, so `mnm sets show studio` reads *won't fit*. Taking turns on that card, a picture waited about 20 s for zimage to load, and the chat model's next answer came in 4.3 s. The voice, a few hundred MiB, fits beside either.
- **Coding desk**: `coder=coder8b memory=embed`, then `mnm sets default desk`. `mnm sets url desk --for claude-code` prints the coder's Claude Code card and the memory's curl card; `mnm url --for opencode` writes OpenCode's file by role ([13.7](#137-workspace-apps-one-file-each)).
- **Odysseus full set**: `chat= vision= image= edit= voice= ear= memory=`. On 12 GB the chat and vision models fit together only as small quants with short contexts; with an image model beside them, expect *won't fit*, or use `--swap`.
- **A short film**: `chat=qwen3-8b image=zimage video=wan5b voice=pocket`.
  1. Ask the chat model for a shot list on its **Run** tab: a picture prompt, a motion prompt and a spoken line per shot.
  2. Draw each still on the image instance's Run tab (its size is the instance's; set it to the clip's aspect ratio, 832x480, with `mnm set zimage width=832 height=480`).
  3. Animate each: the video instance's Run tab with the still as its first frame (or `mnm run wan5b "the motion" --image still01.png`).
  4. The voice's Run tab reads the lines.
  5. Download the clips and the voice from **Runs**, then run the stitch line ([9.5](#95-where-a-clip-goes-and-stitching)).

  A video model generally runs alone on this card: work in stages (chat, image and voice for 1, 2 and 4; then `mnm stop qwen3-8b zimage pocket` and `mnm start wan5b` for 3), or use `--swap`.

A set is not a pipeline runner: it starts servers. The Run tabs make one thing at a time; pipelines ([16.6](#166-pipelines-a-short-film-from-one-idea)) chain them.

### 16.4 Taking turns (`--swap`)

```bash
mnm sets create turns chat=qwen3-8b image=zimage --swap
mnm sets start turns          # every member's URL answers; the chat model loads now
```

- When the members do not fit together, they take turns: a request that asks for work (a POST: a chat message, a picture, a transcription) to a parked member waits while the router parks the others (largest first, only as many as needed) and loads it, 20–60 s; requests after that are as fast as usual. A model list or a browser GET never loads anything.
- One start at a time on the box: a second request for another parked member waits for the first start, so two members never evict each other. A member being parked to make room first finishes the replies it is writing (up to five minutes).
- A chat that asks for a picture therefore waits twice: once for the picture model, once more for the chat model on its next message. Set your client's request timeout above 60 s.
- `mnm ls` says *takes turns* for a parked member whose group-mates hold the card: that is the swap working.
- `mnm stop <member>` switches that member off (409 until you start it); `mnm sets stop turns` stops them all; `mnm sets rm turns` ends the turns. An instance takes turns in one set at a time.
- The web set form has *Take turns*. It needs the router's service (`mnm service install`).

| Client | Chat member | Image member |
|---|---|---|
| Odysseus | type **LLM**, tools on | type **Image**, at its own URL (never the chat's) |
| Open WebUI | Admin Panel → Settings → Connections → OpenAI API | Admin Panel → Settings → Images, engine OpenAI, the image URL |
| OpenAI SDK / curl | `base_url` = the chat URL | `base_url` = the image URL, `images.generate` |
| Coding agents | the coder member's URL | not used |

### 16.5 Making things: Run, Runs and Clean up

Every chat, vision, picture, edit, video, voice and speech-to-text instance has a **Run** tab on its page: write what you want (a question, a picture, a clip, a line to read), add a picture or a clip when it takes one, press **Run**. The request goes through the router, which loads the model and makes room, exactly as a client's would. Embedding and rerank instances answer numbers, so they have Test it only.

```bash
mnm run qwen3-8b "Write a three-shot film about a lighthouse keeper"
mnm run zimage "a lighthouse at dusk, oil painting"
mnm run zimage "make it night" --image still.png     # repaint (add --mask mask.png to inpaint)
mnm run wan5b "the waves roll in" --image still.png  # image-to-video
mnm run pocket "Hello from makenomistakes."
mnm run stt --audio clip.wav
mnm run zimage "a lighthouse at dusk" --repeat 4      # four pictures, four runs, to pick one
mnm comfyui run wan5b --repeat 3                      # its own prompt, a new seed after the first
```

- **Several at once.** `--repeat N` (1 to 20, on `mnm run` and `mnm comfyui run`) makes N runs one after another, each its own record and folder, and lists the N folders at the end; the gallery shows them side by side. A picture instance draws a new seed for every request unless you set one (`mnm set <name> seed=…`, which the repeat warns about: every picture would be the same); a workflow's runs after the first take `mnm comfyui again <run> --new-seed`'s path, its settings and uploads with a new seed. The first failure stops the repeat (it is kept as a failed run) and says how many were made. The web Run tab has no Repeat field: repeats are the CLI's.

- **What you get.** Each Run is kept: the prompt, the instance's model and size or voice, what you gave it (`input-picture-…`, `input-mask-…`, `input-clip-…`), and what came back (a chat answer whole as `<name>.md`, a PNG, a clip, a WAV). The files live in `/data/outputs/<date>/<run>/` (`MNM_OUTPUTS_DIR`; setup asks), the record in the state folder.
- **Runs** (the web tab, key `5`, or `mnm outputs ls`) shows everything, newest first, filtered by kind and instance, with sizes. **Run again** repeats a run with the same prompt and inputs; **Delete file** and **Delete run** ask nothing more than they say.
- **A run still going** cannot be deleted (it would write its record back). One whose process died (the web UI restarted, a terminal closed) reads *interrupted*: delete it, clean it, or Resume a pipeline.
- **Clean up** (on the Runs page, or `mnm clean`) empties one thing at a time, each with its size: `outputs` (the files; the history stays), `runs` (the history, and the files unless `--keep-files`), `logs` (engine logs kept after `mnm stop`), `workflows` (earlier versions of imported ComfyUI workflows), `imports` (the imported workflows themselves; their runs stay), `comfy` (ComfyUI's uploads, temporary and leftover output files, and the history a running ComfyUI keeps of finished prompts, cleared only without `--older-than`). While a workflow runs, or a short film made with one is going or waiting for you, `comfy` and that workflow are left alone, and the clean says so. `--older-than 30d` keeps the newer ones. Models, recipes, sets and engine images are never touched there.
- **Prune** keeps the folder small: `mnm outputs prune --older-than 30d`, or `--max-gb 20` (the oldest go first).
- `uninstall.sh` never deletes your outputs; it says where they are.

### 16.6 Pipelines: a short film from one idea

A *pipeline* chains Runs for you. The first one is **Short film**: your chat model writes a shot list, your image model draws a still per shot (at the video's size), your voice speaks each shot's line, your video model animates each still, and ffmpeg joins them into one `film.webm`.

```bash
mnm pipeline run "A lighthouse keeper finds a message in a bottle" \
  --chat qwen3-8b --image zimage --video wan5b --voice pocket --shots 3
# … waits at the shot list: read it, then
mnm pipeline go <run>          # draws the stills, then waits again
mnm pipeline redo <run> stills 2   # redraw one still you do not like
mnm pipeline go <run>          # speaks the lines, animates every still, joins the film
```

- **Review points.** It stops after the shot list (on the web you can rewrite any shot, or drop one) and after the stills (redraw any). `--auto` (or *Straight through* on the web) runs without stopping.
- **The order is by model**: every still, then every line, then every clip. The router loads each model once and the video model, the largest, runs last and alone. Expect minutes per clip.
- **Clips from a ComfyUI workflow.** `--video comfy:<workflow>` (or *workflow: …* in the video picker) makes each clip with an imported workflow ([18.4](#184-comfyui)) instead of a video instance: the shot's still goes to its picture input and the motion to its prompt (a text-to-video workflow gets the still's prompt too), the clips are made at the film's own shape, set on the workflow's width, height and frames fields when it has them: 832x480 and 33 frames, sd-server's video defaults, unless `--clip-size WxH` and `--clip-frames N` say otherwise (a workflow's own defaults can be far larger: the Comfy-Org Wan 2.2 5B template's 1280x704x121 took 11 minutes a clip on an RTX 4070, its 832x480x33 48 s). The stills are drawn at that size, the Clips line says the shape ("Clips 1 of 2 (832x480, 33 frames) …") and each clip's seconds, and the film is `film.mp4` when the workflow saves mp4. A video *instance*'s clips are its own shape (`mnm set <name> width=… height=… frames=…`). A Wan 14B workflow makes far better clips than the 5B instance, at minutes a clip.
- **Nothing is lost.** Each item is saved as it is made (`still-01.png`, `line-01.wav`, `clip-01.webm`, `shotlist.json`, `film.webm` in the run's folder). A failure stops the run with the reason; `mnm pipeline resume <run>` (or **Resume**) carries on from the item that failed. A run whose web UI restarted, or whose terminal closed, reads *interrupted*: Resume it. **Make again** on any item remakes it and what was made from it (a new still means a new clip and a new film).
- **The film.** Each clip gets its line, padded with silence and cut at the clip's end, then the clips are joined without re-encoding. ffmpeg runs in the whisper.cpp engine image (or, without it, the ComfyUI image mnm built), offline, seeing only the run's folder. One of them must be on the box (`docker pull ghcr.io/ggml-org/whisper.cpp:main-cuda`, or start a speech-to-text model once); mnm never pulls it by itself here.
- **On the web**: Runs → **Pipelines** has the form (the pickers offer the instances of each type) and every pipeline run; a run's page shows each step with its picture, clip or sound, the review panel, and the buttons. The Runs gallery shows a finished film.
- `mnm pipeline ls`, `mnm pipeline show <run>`; delete one like any run (`mnm outputs rm <run>`).

### 16.7 Told when it ends: notifications

A film, a clip or a download takes minutes. mnm can tell your phone, or Windows, when it ends:

```bash
mnm notify --ntfy https://ntfy.sh/<a topic of your own>   # or your own ntfy server's topic URL
mnm notify "hello"                                       # a test: says what each channel answered
mnm notify --toast off                                   # the Windows toast (WSL: on by default)
```

A notice goes out when `mnm run`, `mnm comfyui run`/`again`, a film's `mnm pipeline` step, `mnm add` (its download), `mnm engines update` or `mnm comfyui install` ends after 20 s or more, and when a start fails; the web UI's downloads, runs, films and engine jobs send the same. A failure carries its own words. A topic that wants a token reads `MNM_NTFY_TOKEN` from the secrets file. A notice never holds anything up: if ntfy or Windows does not answer within 5 s it is dropped (`mnm notify` says why).

## 17. The web UI

http://localhost:7800, a background service (`mnm service status`, `mnm service logs`). It listens on 127.0.0.1 only; under WSL open it from Windows at the same address. Nothing loads from the internet: htmx and the fonts ship with mnm.

### 17.1 The shell

- **The look:** a phosphor terminal in amber. **Green** switches the ink; **FX** turns the scanlines, glow, boot text and typed lines off; **Sound** turns key clicks on (off by default); **Keys** turns the single-key shortcuts off. With *reduce motion* set in your system, or FX off, nothing moves. These live in your browser, not on the box.
- **Tabs:** Status, Add, Sets, Workflows (ComfyUI), Runs, System; keys `1`–`6` switch them.
- **The `mnm>` line** at the bottom (`/` or `Ctrl+K` jumps to it) takes the CLI's verbs: `start qwen3-8b`, `stop --all`, `try zimage`, `logs anima --review`, `add <link>`, `replan qwen3-8b`, `sets start studio`, `run zimage`, `clean`, `engines`, `help`.
- **The status bar** under it shows VRAM, RAM, what ComfyUI holds (when it runs), the router and running jobs on every page, every 10 s.
- **`mnm open`** opens the web UI in this machine's browser (under WSL, Windows' default browser, through `wslview` or `explorer.exe`; on a native box `xdg-open`, only in a desktop session); `mnm open <name>` opens that instance's page. Where nothing can open a page (a box reached over SSH) it prints the address; when nothing answers on the web UI's port it says how to run it.
- With a keyboard, the first `Tab` shows **Skip to the page**; a code block or wide table that scrolls takes the focus, so the arrow keys scroll it. On a phone the page fills the screen and the tables become cards.

### 17.2 Status

The instances grouped by type, each with its state (a split-flap cell that flips when it changed), *Fits now*, Copy URL and its actions: *Start*; *Make room and start* and *Start anyway* for a won't-fit row; *Stop*, or *Switch off* for a parked one so the router stops loading it. **Stop all** stops everything (it asks first). Above them: the machine panel with ten minutes of VRAM history, and the router (up, busy, key); below them, **Recent activity**: the downloads, starts and tests running now and for a while after.

**Use it**, under the machine and the router, is the page's bookmark part: where clients point (the router from this machine and from its containers, and whether it asks a key), the desk's roles each linked to its instance (a client sends the role as its model, [16.2](#162-sets)), and the workspace apps mnm installed, each with its state, its address to open and its login note (where its password is, how its first account is made); the ones not installed or only connected are named with a link to System → Apps. It loads just after the page, since asking each app's state takes Docker or systemd a moment. Who is signed in to an app is the app's own business: mnm reads no app's sessions.

Once ComfyUI is installed, a **ComfyUI** panel sits above the instances: running or stopped, about the VRAM it holds (in no instance's row; ComfyUI reports PyTorch's pool, but the release mnm builds (0.42) keeps buffers outside it, so the figure adds its CUDA context and, once a run has ended, about 1.7 GiB of those buffers, measured against nvidia-smi on an RTX 4070; the hover says how it was made, and planning counts the pool only, what **Free VRAM** gives back), the run it is making (with its step when the web started it) or how long it has been idle and when the router's sweep frees its card (ten minutes after its last run), and **Stop ComfyUI** and **Free VRAM** (refused while a run goes: ComfyUI frees its card between runs only). `mnm ls` ends with the same line.

### 17.3 An instance's page

On top: the state, *Fits now*, the URL to paste with Copy, and Start or Stop. Then a tab each (←/→ moves between them; a link such as `/instances/qwen3-8b#settings` opens one):
- **Overview**: the parts (file, size, where it lives), the readings, the exact `docker run`, the last test and the log review, and **Why this plan** (`mnm why`, [15.9](#159-why-this-plan-mnm-why)), planned when you click **Explain this plan**.
- **Connect**: the cards for each app, and for the coding agents on chat-like types, plus the engine's own address (`--direct`).
- **Run** (`mnm run`): your prompt, kept in Runs ([16.5](#165-making-things-run-runs-and-clean-up)).
- **Test it** (`mnm try`): the card's request; *Speed* for tok/s, *Measure* for the GPU peak; a picture to repaint, edit or animate, a clip to transcribe, a sentence to read (uploads up to 32 MiB, kept in a temporary folder, never in your folders). The PNG, clip or WAV plays on the page.
- **Tuning** on llama.cpp chat, coding and vision instances: the tune's state, its trials and what it saved, with **Optimise for speed**, **Optimise for quality** and **Cancel** (`mnm tune`, [15.7](#157-tuning-mnm-tune), [15.8](#158-tuning-for-quality-mnm-tune---quality)).
- **Settings**: every setting with its advice (*applies now* on the router's two), **Save settings**, **Re-plan for this machine** (optionally *any quant*), and at the end **Remove** and **Remove and delete its files** (each asks first, then says what it freed).
- **LoRAs** on picture and video instances: the list with weight and *always*, and Add.
- **Log**: the last 400 lines, all, or issues only, with Refresh, beside the log review's findings. A stopped instance shows the log it kept ("saved when it was stopped").

### 17.4 Add

- Paste a Hugging Face, Civitai or LM Studio link (or `org/name`) and **Check fit**: what it is, the budget (and a yellow note when another program holds the card), the three profiles; pick one and **Add and download**.
- **Find a model** (`mnm what`, [5](#5-your-first-model)): words, a use, or both; the curated picks first, then Hugging Face's, Civitai's and Ollama's own order, each graded against this box. **Plan it** opens a row's plan as its `mnm add` line would make it. Nothing downloads.
- Options: a version picker, *Add it as* another type, *For coding agents*, *Save without downloading*, *What if* (plan for another amount of free VRAM), *Advanced* (quant, context, KV), *Files and quants* (`mnm resolve`), the upscaler checkbox.
- A refused link shows what it is, what to do and repos to try, each with **Check fit**.
- **Get started** has the starter models that fit this card, one **Add** each (an ordinary add, downloaded in the background).

### 17.5 Sets, Workflows and Runs

**Sets**: create one (a select per role, offering the instances of the right type; *Take turns* for `--swap`), see the fit-together verdict, **Start all**, **Stop all**, **Remove** (its instances stay), and every member's cards for one app. The set's page has the **Members** form: pick another instance for a role, or empty a role to drop it (`mnm sets edit`, [16.2](#162-sets)).

**Workflows** (key `4`): imported ComfyUI workflows: **Import** (a link or a file), each workflow's form, report, model files (**Find**, **Fetch**) and runs, and **Node packs** ([18.4](#184-comfyui)).

**Runs**: the gallery of everything Run tabs, workflows and pipelines made (a workflow run shows its time, VRAM and RAM peaks), with **Run again**, **Delete file**, **Delete run**, **Prune** and **Clean up** ([16.5](#165-making-things-run-runs-and-clean-up)). Its **Pipelines** sub-tab starts a short film and lists them ([16.6](#166-pipelines-a-short-film-from-one-idea)).

**Songs** (Runs → Songs, `/songs`): the studio's page ([28](#28-the-studio)). The form starts a song from a theme (length, candidates, tier, who writes the lyrics, a style line, your own lyrics or a recording, a library voice, straight through); each song's page has the lyrics to edit, the candidates with their scores, players and **Pick**, the structure map, and the panels for **Edits** (a range from the take's waveform), **Stems**, **Vocals**, **Voice**, the **Mixer**, **Master** and **Export**, then every step. A song run in the gallery or Pipelines opens here.

### 17.6 System

- **Overview**: the router (address, up, key) and its **Activity**: its newest 200 journal lines with their times, newest first (each load and its seconds, what it parked for room, idle unloads, every request it refused itself with the reason, every `/mcp` call; a proxied answer that succeeded is not a line). The list starts again when the router restarts; the services' log under it has the rest. Like `/__mnm/busy`, `GET /__mnm/events` answers anyone who reaches the router when no `MNM_API_KEY` is set ([14.4](#144-a-key-for-the-router)): the lines name your instances, the paths asked for and the refusals' reasons, never a key. Then the two services, which settings are set (tokens show *set* or *not set*, never the value), and the services' log.
- **Engines**: each engine's build and who uses it, and the registry's newest with **Check for newer builds** ([18.8](#188-is-there-something-newer)); **Update** (with its self-test and rollback), **Test**, **Roll back**, each behind a confirm.
- **Calibration**: the corrections and readings ([18.3](#183-what-mnm-learns-from-your-card)); **Export JSON**.
- **Doctor**: the checks, and **Run the GPU container test**.
- **ComfyUI**: **Install**, **Start** and **Stop** ComfyUI, its image and what it holds on the card ([18.4](#184-comfyui)); **Link the image models** and the yaml block for a ComfyUI of your own.
- **Machine**: this box as `mnm machine show` prints it, and **Adopt** with its **Dry run** and the download question as a checkbox ([27.4](#274-mnm-machine-adopt)).
- **Disk**: the models folder by instance, by folder and unlisted, the other folders, and the disk policy ([19.1](#191-disk-what-the-models-folder-holds-and-the-disk-policy)).
- **Backup**: the backups, the last restore test and **Back up now** ([18.6](#186-backup-and-restore)).
- **Nightly**: the last night's verdicts and steps, every night, **Run now** and each instance's decode speed over time ([26](#26-the-nightly)).
- **Studio**: the studio tools image (**Install**, **Roll back**, **Uninstall**), the stages with the instance that plays each tier, and every studio model against this box ([28.3](#283-scores-the-structure-map-stems-and-the-mixer), [28.12](#2812-the-models-and-your-box)).
- **Voices**: the voice library: **Add the voice**, a consent's hash, **Remove** ([28.5](#285-voices)).
- **Trainer**: the two trainers' images, **Train** a voice from a folder on this machine or a style from your songs, the style library and the trainings with **Cancel** ([28.10](#2810-train-a-voice), [28.11](#2811-teach-ace-step-a-style)).
- **Apps**: each workspace app with **Install**, **Start**, **Stop**, **Update** and **Roll back**, and its files ([12](#12-apps)).

`mnm update`, `mnm service install` and `mnm setup` stay in a terminal: the first two restart the page itself, and setup needs sudo.

**Jobs** (downloads, starts, tests, engine updates) live in the UI's process: a restart (`mnm update` does one) forgets them, and the pages say so. A half download resumes on the next Start: the Hub's cache and Civitai's `.part` files keep what arrived.

## 18. Updating mnm and the engines

### 18.1 mnm itself

```bash
mnm update                     # git pull, install dependencies, restart the web UI and the router
mnm --version                  # the version and commit
```

`mnm update` is the one command that updates mnm. It pulls the new code (fast-forward only: when you have changed files in the checkout it stops and says so; stash or commit them), installs its locked dependencies and lists the new commits; when the services are installed, the new code rewrites the web UI's and the router's units, with the folders and ports they had, and restarts both. Running instances keep running as they started: restart one (`mnm stop`, then `mnm start`) only when its settings changed or the update says so. If a newer mnm saved a recipe, an older one refuses it with "run `mnm update`".

`mnm update` moves nothing else. The engines update with `mnm engines update` ([18.2](#182-the-engines)). What mnm builds or pins moves when you ask, once the new mnm pins a newer version: ComfyUI with `mnm comfyui install` (the doctor says when its pin moved, [18.4](#184-comfyui)), the studio's tools and trainers with `mnm studio install` and `mnm studio trainer install` (their `status` says whether this mnm would build another), and each workspace app with `mnm apps update <app>` (`mnm apps ls` shows each against this mnm's pin).

### 18.2 The engines

```bash
mnm engines                    # each engine image: build, date, which instances use it
mnm engines check              # a newer build upstream? asks the registries, pulls nothing (18.8)
mnm engines update             # pull newer builds (all, or: mnm engines update sdcpp), each tested
mnm engines test llamacpp      # test the current build again
mnm engines rollback sdcpp     # back to the build the last update replaced
```

- Engine images never change by themselves: only `mnm engines update` pulls a newer build, and it keeps the build it replaced for `mnm engines rollback`.
- Every llama.cpp start passes `--fit off`: newer llama-server builds fit unset settings to the card by themselves, and mnm's plan (layers on the GPU, experts in RAM, context) is the one that decides. A build older than a flag mnm passes (`--fit`, `--moe-cache-mib`, `--spec-type draft-mtp`) stops at once, and `mnm logs` says to run `mnm engines update llamacpp`.
- **Every new build tests itself.** A copy of your smallest downloaded instance on that engine starts on the previous build, answers `mnm try`'s request and stops; the same copy then runs on the new build, on a spare port. Your own instances keep running; when the copy needs their room, the ones the router may park are parked first (each loads again at its next request), never a pinned one or a member of a set in use. The new build fails if it does not start, does not answer, takes more than 15% more VRAM, or runs more than 25% slower; the update then rolls itself back and keeps the new build's log under `~/.local/state/makenomistakes/logs/`. When the previous build fails at the same step, the result says "Not judged" and the new build stays. When only models mnm never parks hold the room, or the copy still does not fit, the test is skipped and says why.
- Restart an instance to run it on the new build. Each engine's image is pulled at its first start (sd.cpp ~2 GB, whisper.cpp and audio.cpp ~2–3 GB each, vLLM ~10 GB).
- **One tag per backend** ([25.10](#2510-backends-cuda-vulkan-rocm-cpu)). Each engine publishes a build per backend: llama.cpp `server-cuda` (CUDA 12), `server-cuda13`, `server-rocm`, `server-vulkan` and `server` (CPU); sd.cpp `master-cuda` and `master-vulkan`; whisper.cpp `main-cuda`, `main-vulkan` and `main`; audio.cpp `full-cuda12`, `full-vulkan` and `full-cpu`; vLLM `vllm-openai` and `vllm-openai-rocm`. A new instance gets the tag of the box's backend; a saved one keeps its tag until `mnm replan` or `mnm machine adopt` moves it and says so (`image: server-cuda -> server-vulkan`). `mnm engines` and System → Engines name each tag's backend and list a tag for another backend while an instance still names it; update, rollback, the self-test and `mnm engines check` work per tag, and a pulled tag still never moves by itself.

### 18.3 What mnm learns from your card

Every start, and every request through the router that generates something, is measured: the VRAM the instance really took, how long it took to load, how fast it answered. The readings stay on the box (`~/.local/state/makenomistakes/measurements/`) and only ever make an estimate safer: a reading above the estimate counts at once, one below needs three readings and lowers it by at most 10%. *Fits now* says "measured on this card" when it uses them, and `mnm fit` prints a "Calibration:" line when an engine's estimates are corrected. A figure counts only for the same engine build, settings and files, and only when nothing else was using the card while it was taken.

**Picture and clip speed.** A picture through the router (and a clip, from the request that queues it to the poll that says it is done) leaves its seconds with the size, steps and frames that made it and the planner's own estimate for them. Per picture family (Z-Image, FLUX.1, Wan 2.2 …) the median of measured over estimated corrects the `s/image` and `s/clip` that `mnm fit` and the web Add page print, with a "Calibration: Z-Image speed: estimates x0.49, measured on this card (3 readings)" line. A family is corrected from three readings on, never from one, and not capped (time is no safety margin). The first picture after a start also moves the weights onto the card, so it counts only while a family has fewer than three warm readings: then the cold ones count with them and the note says so ("4 readings, 3 of them first pictures after a start: those also load the weights, so the correction leans slow"); from three warm readings on they are left out, listed (marked *first*), and the note says how many. The estimates themselves are fitted per family to readings on an RTX 4070 (Z-Image, SDXL, FLUX.1, Wan 2.2 5B). A reading outside 0.25 to 4 times its estimate is left out, and `mnm calibration` says how many were. `mnm calibration` lists the rows under *picture and clip speed*; the speed correction never moves a VRAM figure. A client that sets its own size (`client-size=on`) or a streamed instance (`offload-to-cpu=on`) leaves its seconds without an estimate, so they correct nothing.

```bash
mnm calibration          # the corrections per engine and fit model, and each instance's readings
mnm calibration export   # one file in the current folder, to paste into a chat
```

**The card's own decode figures.** Every card is priced from its row in mnm's device table ([25.9](#259-how-mnm-prices-a-card-it-has-not-seen)); the RTX 4070's row was fitted to readings on an RTX 4070 under WSL2, and every other card's borrows the 4070's decode efficiency and launch term. `mnm calibration` turns this box's decode readings back into its own figures: the speeds of llama.cpp instances with every layer on one card (no draft, no experts in RAM, not a codebook i-quant), and the two speeds each tune measured (mnm's plan and the settings it chose). One model size gives the decode efficiency ("decode efficiency on this RTX 3090: 0.84 from 6 readings (the row's 0.91)"); two sizes apart by half again give the launch term too, the way the 4070's were fitted from Qwen3-8B and Qwen3-0.6B. They are saved under this box (`machine.json`) and every later plan on that card uses them; the plan's "Priced as:" line says "learned on this box". It also learns how fast experts kept in RAM decode, from your MoE models' readings ([18.7](#187-several-cards), *RAM*).

**Per box.** Every reading names the box it was taken on (§27), and every figure above is made of this box's readings only: `mnm calibration` opens with the box in one line and says under each table how many readings of another machine are kept, not used. A reading taken while something outside mnm held the card (a browser's video on Windows, a game) is marked "beside something else (3.1 GiB held, 45% busy)"; it still counts.

**Try every model at once.** `mnm try --all` (add `--speed` for chat models' 400-token rate) goes through every saved instance, largest VRAM estimate first. Before each it parks whatever runs (as `mnm stop --all --park` would: loadable again on request, never off), starts the instance alone and sends its card's own request through the router. One table at the end, also saved as `try-all-<date-time>.txt` in the current folder: the load and answer seconds, the speed (tok/s, s/image, s/clip, x realtime), the VRAM it added from before its load to its request's peak, and the test's figures. An instance it cannot try is a row that says why, and the run goes on: one you stopped (left off: `mnm stop --park <name>` makes it loadable), one not downloaded (never downloaded here), one that will not fit even with everything else parked, one that did not start or answer, and anything else that went wrong on the way (Docker, a park, ComfyUI busy), named. What the router never parks is left as it is and not tried, with a row saying so: a pinned instance (`keep=always`) and the members of a set in use (`mnm try <name>` tries one of them by itself). Every start and request leaves its readings here like any other. A speech-to-text instance is sent a 1.5 s 440 Hz tone (no words come back; the row says it was a tone). Pictures, clips and speech land in the current folder under each instance's name.

### 18.4 ComfyUI

**As mnm's own engine (optional).** Nothing is downloaded for it until you ask:

```bash
mnm comfyui install      # builds ComfyUI's image here: ~9 GB, 10-30 minutes; asks first (--yes does not)
mnm comfyui start        # runs it with no internet, your models read-only
mnm comfyui status       # installed, running, what it holds on the card
mnm comfyui logs | stop | rollback | uninstall
```

- The image is built on this machine from mnm's Dockerfile: Ubuntu, ComfyUI at the release mnm pins (checked by commit), PyTorch at pinned versions (torch 2.14.1, so two installs build the same image), built for the box's backend ([25.10](#2510-backends-cuda-vulkan-rocm-cpu)). The build pulls `ubuntu:24.04` and Docker's Dockerfile frontend from Docker Hub, ComfyUI from GitHub, PyTorch and the rest from PyPI and PyTorch's index; the versions it got (torch and the frontend) are kept, and `mnm comfyui status` shows the image.
- **Before building**, install checks what would make the build useless and refuses with what to do: for a CUDA 13 build, driver 580 or newer and a Turing or newer card (GTX 16xx, RTX 20xx and later: PyTorch's CUDA 13 wheels have nothing for older ones); for every build, Docker's BuildKit (`docker-buildx-plugin`) and room where Docker keeps its images (~12 GB). `--cpu` builds for the CPU instead. A check it cannot read only warns.
- It runs as you, not root, on a Docker network of its own with no route to the internet, and sees only the models folder (read-only), its model-paths file (read-only) and its own folders under the state folder (`input`, `output`, `temp`, `user`, `home`); mnm's own records about it stay outside the container. Nothing outside this machine can reach it. Its files are read-only apart from those folders, and it may run at most 512 processes. mnm's router and its web UI answer nothing that comes from ComfyUI's network, key or no key, so a node pack cannot use your models, start runs or change settings. What else the container can reach was measured: on this machine only what listens on every address (`0.0.0.0`) answers it, at its network's gateway, over TCP or UDP. Programs on `127.0.0.1` refuse it, other containers (Odysseus included) and the internet are unreachable. `mnm doctor`'s *ComfyUI's reach* line names any such port (sshd on a native box, say) and the firewall rule that closes them all.
- When a model needs the card, an idle ComfyUI is told to unload first (`POST /free`). If a workflow is running, the start waits a minute for it, then refuses with "ComfyUI is running a workflow" (Odysseus gets a 503): ask again when it ends. Loading beside it would run out of memory and leave the model with smaller settings for good. When ComfyUI does not answer whether it runs one (it may be streaming a run's weights, or Docker cannot be asked), mnm treats that as a run: nothing is loaded and nothing is marked freed, and the refusal says "ComfyUI did not answer whether it is running a workflow …"; `mnm comfyui status` then reads "unknown" with Docker's or ComfyUI's words, and `mnm comfyui stop` gives its card back. Ten minutes after the last run, whoever started it (the web, the terminal, a film, Odysseus), the router frees ComfyUI's card too; the container stays up.
- A run you stop (Ctrl-C, closing the terminal, a restart of mnm's services) takes its own prompt back from ComfyUI, so nothing keeps the card for work nobody collects. A run that goes quiet is checked every minute: when ComfyUI has nothing left to do, mnm reads what it saved.
- `mnm doctor` adds its lines once it is installed: the image, whether it is the version this mnm builds (after `mnm update` moves the pin: `mnm comfyui install`), whether it was built for another backend than the box runs now (for the CPU while a card is visible, say), the network, the RAM for video workflows (24 GB of WSL memory and a swap file for Wan 2.2 14B), the driver and the card's generation.
- The build prints Docker's own lines as they come, and "N min elapsed, still building" at least once a minute when Docker is quiet (a large wheel downloading). The web's install job shows its last 40 lines. Leave it running: Ctrl-C stops the build, and nothing ComfyUI does works until one finishes.
- After `mnm comfyui install` (a new version) or `rollback`, approved node packs are built again with `mnm comfyui packs install`; both commands say so, and a workflow that needs them says "approved but not built".
- System → **ComfyUI** in the web UI has Install, Start and Stop.

**Importing a workflow.** Paste a link or give a file; nothing is installed or downloaded beyond the workflow file itself:

```bash
mnm comfyui workflow add https://comfy.org/workflows/<id>/          # a template's page
mnm comfyui workflow add "https://civitai.com/models/<id>"          # a Civitai Workflows model (a zip; each workflow in it is imported)
mnm comfyui workflow add https://huggingface.co/<org>/<repo>/blob/main/<file>.json
mnm comfyui workflow add https://github.com/<owner>/<repo>/blob/main/<file>.json
mnm comfyui workflow add ~/Downloads/picture.png                    # a PNG ComfyUI saved carries its workflow
mnm comfyui workflow ls | show <name> | rm <name> [--files]
```

- The report says what it needs: **node packs** (ComfyUI's own, or which pack at which version, installed or not), **model files** (with the folder ComfyUI looks in, and a download link when the workflow names one), **partner nodes** (Comfy's paid cloud: refused), and nodes that **may run code** they are given (named before you approve anything).
- A Civitai zip gives up only its `.json` files and pictures, never a path or anything to run. Some creators require a Civitai account to download: add a key with `mnm setup`.
- Old group nodes are refused: open the workflow once in ComfyUI's editor (it turns them into subgraphs) and import what it saves. An API-format export is refused too: export the workflow itself.
- Importing again under the same name keeps the earlier version; `mnm clean workflows` empties those.

**Running a workflow.** On the web: the **Workflows** tab (key `4`), the workflow's page: its form (the prompt, the negative prompt, the seed, steps, size, frames, pictures to upload, LoRA strengths, and every other setting under *More settings*, each at the workflow's own value), then **Run**. In a terminal:

```bash
mnm comfyui workflow fields wan5b          # the keys a run takes, with the workflow's values
mnm comfyui run wan5b "6.text=a red fox running through snow" 3.steps=20
mnm comfyui run i2v --file 52.image=still.png      # a picture from the current folder
mnm comfyui again <run>                    # the same settings, seed and pictures again
mnm comfyui again <run> --new-seed         # the same with a new seed
mnm comfyui workflow pick wan5b 3.cfg      # show a setting from More settings on top (--unpick)
```

- ComfyUI starts if it is off, and mnm's running models are parked to make room (pinned ones and running sets stay), the way the router makes room for a model: only as many as the run needs, largest first. The need is the larger of the most VRAM an earlier run of the same workflow added (its readings) and the size of its model files, plus 1 GiB; a workflow nobody has measured or sized parks every model, and a CPU build of ComfyUI parks none.
- A run the kernel killed for memory, or one that crashed ComfyUI, says so (with the WSL memory advice on Windows) instead of "lost the connection".
- Progress shows as it runs (which node, step n of m); **Stop the run** interrupts it.
- The pictures, clips and sound it saves are moved into a run in **Runs** (one copy), with the settings used, the seed included. A PNG keeps its workflow inside, so it can be imported again. That means a shared PNG carries your prompt, settings and upload names too: strip its metadata before posting it if that matters.
- A seed left empty is random each run. A workflow run in **Runs** (or under *Its runs*) has **Run again** (the same settings, seed and pictures) and **New seed** (the same with a new seed). A run whose pictures were deleted is refused rather than run with the workflow's own picture: run it from the form. Ten minutes after the last run, ComfyUI gives the card back.
- A number or choice left blank keeps the workflow's own value; a prompt you empty is sent empty.
- *Choose the settings shown above* (under the form) moves the settings you tick out of *More settings* for that workflow; `mnm comfyui workflow fields` marks them *(picked)*. A chat in Odysseus gets them too: each shown setting is an argument of the workflow's MCP tool ([12.6](#126-odysseus)).
- A workflow that cannot run yet says why: a node pack not installed, a model file not where ComfyUI looks (it names the file), a partner node.

**Its model files.** mnm fetches them for you, one copy on disk, linked where ComfyUI looks:

```bash
mnm comfyui models wan5b                   # each file: in place, on disk (an instance's), fetchable, or no link
mnm comfyui models kijai --find            # files without a link: by exact name in known repositories, then on Civitai
mnm comfyui models kijai --set "umt5-xxl-enc-bf16.safetensors=https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
mnm comfyui models wan5b --fetch           # says how many GB, asks, downloads, links
```

- Downloads go where mnm's own do: the Hugging Face cache under `/data/models/hf`, Civitai's files under `/data/models/civitai/<version>/`. A file one of your instances already has is linked, not downloaded again.
- The known repositories are Kijai's WanVideo repos, Comfy-Org's repackaged Wan, HunyuanVideo, Z-Image and Qwen-Image repos, QuantStack's Wan GGUFs and city96's umt5 GGUF. A file found in several says so; pick one with `--set`.
- Then Civitai: a workflow that records a file's sha256 gets Civitai's file of the same bytes, linked at once (under the workflow's name for it). Otherwise Civitai is searched for the file name's words, and a file of exactly that name is only **offered**: two creators can use one name, so check the model's page and confirm it with **Use** (web) or the `--set` line `--find` prints. When Civitai refuses (a wrong key) or does not answer, Find says so rather than "not found".
- A workflow saved on Windows names `WanVideo\x.safetensors`; mnm links it as `WanVideo/x.safetensors`, which is what ComfyUI on Linux lists.
- `.ckpt`, `.pt`, `.pth` and `.bin` files are marked *pickle*: they can run code when loaded. They are named before the download asks, and not otherwise held back: prefer a `.safetensors` file of the same model when there is one.
- The question before a download gives the total (Hugging Face and Civitai files both), and a fetch the models folder's disk cannot hold is refused with the numbers.
- A workflow a run, a film or a song uses (going, waiting at a review, interrupted) or a studio stage plays is refused by `workflow rm` ("<name> is in use: …. Nothing was removed.") and left by `mnm clean imports`.
- `mnm rm --files` never deletes a file a workflow uses. Each file mnm fetches for a workflow is recorded as that workflow's. `mnm comfyui workflow rm <name> --files` (on the web, Remove with *also delete the model files* ticked) deletes the ones no other workflow names and no instance uses; one another workflow names passes to it. Without `--files` the files stay, still recorded as the removed workflow's (so does `mnm clean imports`, which never touches models): `mnm comfyui workflow rm <name> --files` with the same name frees them later. If another workflow cannot be read, every file stays, and the page says so.
- On the web, the workflow's page has the same: each file's place, **Find the files without a link**, Civitai's offers with **Use**, a box to paste a link, and **Fetch** (asks first).

**Custom node packs.** Shared workflows use nodes from packs: Python code from other people. mnm installs a pack only when you add it, at one version, checked by the hash of its zip, built into ComfyUI's image:

```bash
mnm comfyui packs ls                       # the vetted packs, the ones you added, what ComfyUI has now
mnm comfyui packs add --for kijai          # every pack a workflow uses: vetted ones at once, others ask
mnm comfyui packs add comfyui-videohelpersuite comfyui-frame-interpolation
mnm comfyui packs add github:diodiogod/TTS-Audio-Suite@v5.6.0   # a pack off the Registry, at a tag or commit
mnm comfyui packs install                  # fetches their own weights (asks), builds, starts ComfyUI once to check
mnm comfyui packs rollback                 # the layer before, if a new one misbehaves
mnm comfyui packs find VHS_VideoCombine    # which Registry pack has a node
```

- **Vetted by mnm** (one command each): ComfyUI-WanVideoWrapper, VideoHelperSuite (Video Combine), ComfyUI-GGUF, KJNodes, Frame-Interpolation (RIFE), SeedVR2 (upscaling), MMAudio (sound for a clip; its weights are CC-BY-NC, not for commercial use), Matchering (mastering a mix against a reference track; GPL-3.0: what it makes is yours, its code is copyleft). Each at a version mnm checked; the list changes only with mnm.
- **Any other pack** comes from the [Comfy Registry](https://registry.comfy.org) or from GitHub at a release. Its page (or `mnm comfyui packs add <id>`) shows the source, licence, publisher, downloads, pip packages and the download, then asks. Packs or versions the Registry has banned, deleted or flagged are refused.
- **A pack on GitHub** (many audio packs are not on the Registry) is named `github:owner/repo@<tag or commit>`, or by its GitHub link to a release, tag or commit; the web's Node packs box takes the same. mnm asks git which commit the tag names (a branch or `latest` is refused: a branch moves, so the refusal names the repository's newest tags, or its main branch's commit to pin when it has no tags), fetches that commit with git, checks git landed on it, and packs the files itself the same way every time (sorted, no `.git`, no links; git runs with none of your git settings, so no line-ending or Git LFS filter changes a file, no credential helper sends a saved login and git never asks for one), so the approval holds the commit and the sha256 of mnm's own zip of it. A commit named by its sha must be in one of the repository's own branches or tags: GitHub serves a fork's commits through its parent's address too, so one that is not is refused. Its page shows the commit, mnm's sha256, the licence from its LICENSE file, the pip packages from its `requirements.txt` (a line pip would take past the pin, such as a `git+https` URL, is refused here, as the build would), and its stars, open issues and whether it is archived when GitHub's API answers. **Approve** fetches it again and approves only that same commit and sha256: a tag moved since you looked is refused with "look it up again". Nothing else changes: it is built into the same layer and loaded the same way. A workflow saved with a GitHub pack (its nodes carry the repository and the commit) offers that pack at that commit. git submodules are not fetched, and the page says so when a pack has them. One pack per name: a pack of another repository with the same name as one you approved is refused until you `mnm comfyui packs rm` the first (a newer commit of the same repository, or the same repository from the Registry, replaces it). `mnm comfyui packs rm github:owner/repo` removes one by its GitHub name.
- **Nodes that work somewhere else** are refused: ComfyUI's own partner nodes (Comfy's paid cloud), and packs whose node only sends the prompt to a service of their own, such as ACE-Step-ComfyUI's text-to-music node (api.acemusic.ai with a key, or a server of its own beside ComfyUI). ComfyUI here has no internet and nothing beside it, so they cannot work; ComfyUI's own ACE-Step 1.5 nodes run the model here instead ([28.4](#284-songs-from-comfyui-workflows)).
- **What the install does:** the pack's files go into a new image layer; its pip packages install under the versions ComfyUI already has, so no pack replaces torch (a pack that needs another version fails the build, which names it). ComfyUI then starts once; a pack that does not load leaves the layer before in use and names the pack.
- **No internet at run time** still holds: a pack that would download its weights gets them from mnm instead (Frame-Interpolation's RIFE 4.7, linked into the pack's own folder). Other packs' models are a workflow's model files (above).
- Approving a pack does not make its code safe. It runs inside ComfyUI's container: no internet, no secrets, no home folder, the models read-only.
- On the web: **Workflows → Node packs**; a workflow's page has **Add its packs**.
- **Each run's readings.** A workflow run records its time, the GPU memory in use at its peak and how much the run added, the RAM at its peak, and the size of the model files it loads; the CLI prints them after the run and Runs shows them on its card. Before a run, mnm says when the files are larger than the card (ComfyUI keeps part in RAM: slower) or than the card and the free RAM together (it may run out of memory).
- SeedVR2 needs an NVIDIA card: on the CPU build it lists no device and does not load.
- `packs install` refuses while a workflow runs in ComfyUI (it restarts ComfyUI): wait for the run, or stop it. One install runs at a time, across terminals and the web ("Another node packs install is under way …"), each building in its own folder; if ComfyUI's image or its packs changed while it built, it refuses to switch and says to install again, and if the new layer could not be checked (Docker refusing, a failed start) the layer before stays in use. On the web, ComfyUI's jobs and the packs' refuse each other while one runs.
- `--yes` on `packs add` approves a pack mnm did not vet without asking: use it only for packs you already read about.
- KJNodes is held at 1.5.0: the Registry flags 1.5.1 and 1.5.2 (without a reason), and mnm never installs a flagged version.

**Sharing the models with a ComfyUI you installed yourself.** ComfyUI can use mnm's picture models without a second copy:

```bash
mnm comfyui                        # links the files, prints the extra_model_paths.yaml block
mnm comfyui --write ~/ComfyUI      # or adds the block to ComfyUI's own file (keeps a .bak)
```

- The links sit under `/data/models/comfyui/` by ComfyUI's folder names (checkpoints, diffusion_models, vae, text_encoders, loras, upscale_models). Run it again after `mnm add` or `mnm rm`.
- ComfyUI must see `/data/models` at that same path: installed in WSL, or its container mounting the folder there.
- GGUF files (Z-Image and FLUX kits) need the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) custom node; SDXL and SD 1.5 checkpoints, VAEs, LoRAs and upscalers load in plain ComfyUI. Diffusers folders are not linked.
- `bash uninstall.sh` removes the links but not the block in ComfyUI's file: delete the lines between the `makenomistakes` marks yourself.

### 18.5 New hardware

- **Updates by itself:** `mnm doctor`, every new `mnm fit`/`mnm add`, and *Fits now* read the GPU and RAM live.
- **Does not:** a saved recipe keeps its plan, and mnm says the box changed ("the hardware changed since 3 plans were made: …") until you run `mnm machine adopt`, which re-fits every plan to the box as it is now ([27](#27-your-box)). `mnm replan --all` ([15.4](#154-plan-it-again-replan)) re-plans on the same box.
- **Another machine:** install there and bring the old box's state with `mnm backup` and `mnm restore` ([18.6](#186-backup-and-restore)), then `mnm machine adopt`; the old box's readings come along, kept apart ([27.6](#276-where-the-readings-go-and-when-a-box-comes-back)). `mnm fit <ref> --vram 24` plans for a card you do not have yet.
- **Two cards:** mnm plans for, reads and runs on one card, `nvidia-smi`'s card 0, unless `MNM_GPU` says otherwise: `MNM_GPU=1` picks another card, `MNM_GPU=0,1` (or `all`) uses several ([18.7](#187-several-cards)). Set it in the secrets file, which the services read (an `export` in your shell reaches only the commands you run there); `mnm service install` restarts them on it; then `mnm replan --all`. A value that is not a card list is refused wherever it is read (the doctor's GPU line says it), never taken as card 0.

### 18.6 Backup and restore

```bash
mnm backup                              # mnm-backup-<date-time>.tar.gz in the current folder (--to FILE)
mnm restore mnm-backup-<date>-<time>.tar.gz   # on the fresh install, then mnm ls
```

A backup holds what took time to get right: every recipe (the settings you tuned, and each tune's trials), the sets and the desk, the imported ComfyUI workflows, their model links and the node-pack approvals, the readings calibration learns from and the boxes they were taken on, the engine ledger, the studio's stage picks and the voice and style libraries' rows (their recordings and adapters stay in the models folder), mnm's own part of the workspace apps' config files, and the **names** of the variables in your secrets file (never a value). It holds no model file, no run or output, no log. After a restore nothing is planned again: `mnm ls` lists every instance as *not downloaded*, and its first `mnm start` downloads its files (`mnm replan --all` afterwards if the card changed). Set the secrets again (`mnm setup`, or the secrets file); the restore lists their names.

**Automatic backups.** One command keeps a folder of them, each proved to restore:

```bash
mnm backup schedule --at 03:30    # a systemd user timer: mnm backup --auto every night (no flag: shows it; --off removes it)
mnm backup --auto                 # the same now: /data/backups/mnm-<date>-<time>.tar.gz, tested, then rotated
mnm backup ls                     # the backups with their sizes, ages and what each is kept as; the last restore test
mnm restore --test                # the newest backup restored in a scratch state and read back (or name a FILE)
```

`--auto` writes into the backups folder (`MNM_BACKUPS_DIR`, default `/data/backups`; setup makes it like the outputs folder), then restores the new file into a temporary state folder, reads every recipe, set, workflow and reading back with mnm's own stores, and removes it: *restored in a scratch state: OK (18 recipes, 1 set, 3 workflows)*. Your state folder is never touched by the test. A recipe saved here since the backup (or removed since) is a WARN, not a failure: the backup is older. Only then the rotation runs: it keeps the newest backup of each of the last 7 days and the first backup of each of the last 4 weeks, and deletes the other automatic ones (a file you wrote there with `--to` is listed and never deleted). A backup that does not restore deletes nothing older. A failed backup or restore test sends one notification with the reason (`mnm notify`'s channels: ntfy and the Windows toast); a backup that worked says nothing. The timer runs while WSL runs: a night the machine was off runs at its next start (`Persistent=true`). `mnm service status` lists the timer, `mnm report` the backups, and the web's **System → Backup** shows the list, the last restore test and a **Back up now** button. The uninstall removes the timer and keeps the backups folder. The nightly ([26](#26-the-nightly)) runs `mnm backup --auto` itself: `mnm nightly install` removes this timer, and `mnm backup schedule --at` refuses while the nightly's timer is installed (`mnm nightly uninstall` first); keep `mnm backup schedule` for backups without the nightly.

A restore checks the whole file first and writes nothing when anything in it is from a newer mnm (run `mnm update`), when it would replace the recipe of an instance that is running now (`mnm stop` it first, or `--force`), or when a folder in the state folder is a link. Readings taken on this machine since are kept and the backup's added to them, all in time order; every box's readings are kept, each naming the box it was taken on, and the boxes mnm saw (`machine.json`) are merged (this box stays the current one); another box's readings are counted ("kept 12 readings of other machines … for when they return") and never used here (§27). This machine's own engine ledger is kept. No token or key goes in: links are saved without a `?token=` or a signed query in the first place, and every file is scrubbed of the secrets file's values on its way into the archive (a workflow file that changed so gets its hash refreshed on restore, which says so). A backup keeps your cards' ids, so a restored box still knows its cards: never share a backup archive (`mnm report` strips them). Node packs come back as approvals: `mnm comfyui packs add <pack>` fetches each zip again.

### 18.7 Several cards

Several cards are planned, not measured: everything here comes from recorded `nvidia-smi` answers and vendor figures, and no split across real cards has been measured. With `MNM_GPU` one index (the default 0) nothing below applies.

- **Which cards:** `MNM_GPU=0,1` in the secrets file (nvidia-smi's numbers, in the order you want; `all` for every card). `mnm doctor` lists each one with its VRAM and compute capability, and warns about a mix that is a bad idea (below).
- **What runs where:** a llama.cpp instance runs on every chosen card, its layers split between them (`-sm layer`); a vLLM instance runs on all of them only when they are equal (the same model and VRAM: its tensor parallel needs that), else on the first; pictures, video, speech and voices (sd.cpp, whisper.cpp, audio.cpp) and ComfyUI run on the first card.
- **How a model is split, the main card, risers and the settings** (`experts=`, `card-experts=`, `gpus=`, `split=`): [25.5–25.7](#255-many-cards). On one card none of it applies.

What each generation of card can run in mnm's engine images (as the images stood on 2026-10-22; the doctor says it per card):

| Card (compute capability) | llama.cpp | vLLM | ComfyUI | whisper.cpp | sd.cpp |
|---|---|---|---|---|---|
| Pascal: P40, P100, 1080 Ti (6.x) | runs by PTX, no tensor cores, FP32 matmuls on 6.1 | refused (needs 7.5) | refused (CUDA 13) | refused (CUDA 13) | runs by PTX |
| Volta: V100, Titan V (7.0) | runs by PTX | refused | refused | refused | runs by PTX |
| Turing: 2080 Ti, T4 (7.5) | runs | runs, no FP8 | runs | runs | runs |
| Ampere: 3060, 3090, A6000 (8.x) | runs | runs, no FP8 | runs | runs | runs |
| Ada, Hopper: 4070, 4090 (8.9, 9.0) | runs | runs | runs | runs | runs |
| Blackwell: 5090, RTX PRO 6000 (12.0) | runs | runs | runs | runs by PTX | runs by PTX (unverified) |

- **Mixes to avoid:** GeForce cards beside Tesla, Titan or DGX ones (the doctor warns; mnm is never planned for such a mix), and a Pascal or Volta card beside newer ones: it runs only llama.cpp and sd.cpp, without tensor cores, a layer split goes at its pace, and NVIDIA's 580 driver branch is its last, which holds the whole box there.
- **Not supported:** vLLM across unequal cards (pipeline parallel), and llama.cpp `--rpc` workers on another machine. **Not measured:** any speed or VRAM figure of a real split; the cost of a crossing between cards is one constant taken from a third party's reading.
- **The doctor on several cards:** a line per card with its link and what its slot is good for, a riser WARN, the main card, the CPU and the RAM slots ([25.6](#256-the-main-card-slots-and-risers), [4](#4-check-the-machine)).
- **RAM:** `mnm doctor --deep` also measures this box's RAM bandwidth with one copy process per physical core (up to 16), the one-thread figure beside it, and saves both (`<state>/machine.json`). Whole layers kept in RAM are then priced at half the every-core figure (llama.cpp reads 40–55% of peak natively), and experts kept in RAM at that figure times the share llama.cpp reads on its RAM platform (0.35 to 0.5, [25.14](#2514-big-ram-epyc-threadripper-and-intel)); until it is measured, at the platform's own figures (under WSL, 55 GB/s for whole layers, an estimate for dual-channel DDR5, and 30 GB/s for experts, fitted to readings on an RTX 4070 box). Each used expert held in RAM costs 53 µs more. `mnm calibration` learns how fast experts kept in RAM decode from your MoE models' readings, and plans then price them at that rate ([25.9](#259-how-mnm-prices-a-card-it-has-not-seen)).

### 18.8 Is there something newer?

Two questions mnm answers without changing anything.

**A newer engine build:**

```bash
mnm engines check              # each engine: this box's build day, the registry's, newer or current
mnm engines check --json       # the same as data
```

It asks the registry (ghcr.io, Docker Hub) for each tag's newest build with `docker buildx imagetools inspect`: nothing is pulled. "Newer" means a manifest no pull of your build came from, built later. When a saved instance needs a model architecture your llama.cpp build is known to lack (DeepSeek-V4.1-Flash's `deepseek41`, [25](#25-large-models-and-many-cards)), the check says whether the registry's build is from after the merge that brought it: "a newer llama.cpp (built 2026-10-27) runs deepseek41, which ds41 needs …: `mnm engines update llamacpp`". Updating stays yours: `mnm engines update` pulls, tests the new build on a copy of your smallest instance and rolls back by itself when the test fails ([18.2](#182-the-engines)). The doctor's *Engines* line repeats the last check as a plain note (never a warning), System → Engines shows it as a Registry column with a *Check for newer builds* button, and the nightly runs the check every night.

**A newer model in the same family:**

```bash
mnm upgrade qwen3-8b           # what it would become, and the difference
mnm upgrade qwen3-8b --yes     # add it beside the old one
mnm upgrade qwen3-8b --replace # once the new one has proved itself: the old one goes, and its files
```

- **Where the successor comes from.** A table of families mnm knows, read on the Hub on 2026-10-26 (`src/makenomistakes/families.py`): Qwen3 → Qwen3.5 → Qwen3.6 → Qwen3.8 for each size unsloth publishes (Qwen3-8B's newest is Qwen3.5-9B: Qwen3.6 and Qwen3.8 have no 8B or 9B; Qwen3-32B's is Qwen3.8-27B), Gemma 3 → Gemma 4, DeepSeek-V4-Flash → V4.1-Flash-1017, Wan 2.1 → 2.2; Qwen3-Embedding and Reranker, Z-Image and FLUX.2 klein are the newest they know. Then the Hub's own word: a model card's `new_version` ("A newer version of this model is available"), the table on the card's `base_model` (a fine-tune or another publisher's quant finds its family that way), and a "deprecated" tag said as it is. Nothing is guessed from a name.
- **What it shows:** the successor, where it was read, both models planned for this machine now (quant, settings, memory on the card and in RAM, fit, the speed estimate), the download, what is kept and what is dropped, and which sets name the instance. Nothing changes without `--yes`.
- **What is kept:** the type (chat stays chat, coding stays coding), the profile, the context and KV cache (capped at the new model's trained context; when it does not fit beside the new model, or would push its layers into RAM, the profile's own, said), your settings that are not the model's (slots, threads, idle, pinning, the cards, a seed, a negative prompt, a language), LoRAs on the same picture family, the swap group. **Dropped and said:** a draft model (another vocabulary: `mnm set <name> draft=…` again once the new one is downloaded), a tune (the tuner tunes the new model on an idle card), any other setting the new plan decides.
- **`--yes` adds it beside.** If the old instance is loaded it is parked first, after its replies. It is renamed `<name>-old` with its readings, its tools verdict, its tuning trials and its kept log, and stays loadable by that name; the new model is saved under the old name on a new port and downloaded. The desk and your sets name instances, so they now reach the new model: "The desk ('desk'): chat is qwen3-8b, now Qwen3.5-9B". Try it (`mnm try qwen3-8b --speed`, a chat through the desk); `mnm try qwen3-8b-old --speed` is the old one.
- **`--replace`** removes `<name>-old` and the files only it uses (`mnm rm --files`' rules: a file another instance, a draft or a ComfyUI workflow uses stays). Until then both are kept; the nightly reminds you each night ("qwen3-8b-old: is kept beside qwen3-8b since an upgrade", offered `mnm upgrade qwen3-8b --replace`).
- **Refused, with why:** an instance not downloaded (`mnm rm` it and `mnm add` the new model: nothing to keep), `<name>-old` already there (an earlier upgrade's: `--replace` it first), a tune of it running, a successor the planner refuses (the RAM floor's arithmetic, a busy card), one on another engine.
- **Whisper:** `large-v2` → `large-v3` is a newer model; `large-v3` → `large-v3-turbo` is offered as a speed trade, never called better (turbo is several times faster and a little less accurate), and never in the nightly's lines.

## 19. Stopping and removing

| You want to | Command |
|---|---|
| Free the GPU, keep everything | `mnm stop <name>` (or `mnm stop a b`, `mnm stop --all`). It stays off until `mnm start` |
| Free the GPU, keep it loadable | `mnm stop <name> --park` (or `mnm stop --all --park`): the router loads it at its next request |
| Let the router unload it, but load it on the next request | nothing: the idle unload does it ([14.2](#142-idle-unload)) |
| Remove an instance, keep its downloads (re-adding is instant) | `mnm rm <name>` |
| Remove an instance and free its disk space | `mnm rm <name> --files`: deletes the files no other instance uses; shared ones (an SDXL VAE, a text encoder) stay and are named |
| See what fills the disk | `mnm disk` ([19.1](#191-disk-what-the-models-folder-holds-and-the-disk-policy)); `mnm ls --disk` for each instance's files and last use |
| Never let mnm delete a model file by itself | `mnm disk keep --models off` ([19.1](#191-disk-what-the-models-folder-holds-and-the-disk-policy)) |
| Remove a LoRA from an instance | `mnm lora rm <name> <lora>` (the file stays in the cache) |
| Remove a set | `mnm sets rm <name>` (its instances stay) |
| Delete runs and what they made | `mnm outputs rm <run>…` (one), `mnm outputs prune --older-than 30d`, `mnm clean` ([16.5](#165-making-things-run-runs-and-clean-up)) |
| Remove an imported ComfyUI workflow | `mnm comfyui workflow rm <name>` (its runs stay) |
| Withdraw a node pack | `mnm comfyui packs rm <pack>`, then `mnm comfyui packs install` (the layer without it) |
| Remove ComfyUI | `mnm comfyui uninstall`: its container, images and network (imported workflows and runs stay) |
| Stop the web UI and router services | `mnm service uninstall` (instances and models stay; `mnm service install` brings them back) |
| Remove a workspace app | `mnm apps uninstall <app>` (asks before deleting its data; `--keep-data` keeps its accounts, chats and settings) |
| Remove the studio's images | `mnm studio uninstall`, `mnm studio trainer uninstall` (`--for styles` for the style trainer): runs, and what was trained, stay |
| Stop the nightly or the backup timer | `mnm nightly uninstall` (the reports stay), `mnm backup schedule --off` (the backups stay) |
| Remove everything mnm ever put on the machine | [section 21](#21-uninstall) |

`mnm stop` keeps the engine's log, so `mnm logs <name>` (and the web Log tab) still shows why a crashed engine died. `mnm rm` deletes that log.

### 19.1 Disk: what the models folder holds, and the disk policy

```bash
mnm disk                         # by instance, by folder, unlisted; the other folders; the policy
mnm disk --json                  # the same, for a script
mnm ls --disk                    # the instance list, then each one's files and last use
mnm disk keep                    # the limit now
mnm disk keep --models 85%       # the default: 85% of the models folder's disk (as `df` counts it)
mnm disk keep --models 400G      # or a size for the models folder itself (G is 10^9 bytes, GiB 2^30)
mnm disk keep --models off       # mnm never deletes a model file by itself
mnm disk check                   # run the policy now
```

**What `mnm disk` shows** (System → Disk on the web shows the same):

- **By instance:** the files only it lists (*Own*), and those it shares with another instance or a ComfyUI workflow (*Shared*, counted once on disk, named with whoever else uses it), its state (on disk, not downloaded, *evicted* and when) and when it was last used.
- **By folder:** `hf/`, `civitai/`, `ollama/`, `modelscope/`, `voices/`, the link folders (`views/`, `loras/`, `upscalers/`, `comfyui/`) and `configs/`; each ComfyUI workflow's files (they sit in `hf/` or `civitai/` and count there too); anything else in the models folder, marked *not mnm's*.
- **Unlisted:** files in the download folders that no instance, workflow, LoRA or draft lists: what `mnm rm` without `--files` left, a download cut short. They are the first place to look for space; nothing deletes them by itself.
- **The other folders:** the outputs (`MNM_OUTPUTS_DIR`), the state folder and ComfyUI's part of it, the apps (`MNM_APPS_DIR`), and the backups (`MNM_BACKUPS_DIR`, [18.6](#186-backup-and-restore)).
- **The policy:** its limit, how full the disk is, the instances that would go next, in order, and why the others stay; the last 50 evictions.

**Last used** is the newest of: the instance running now; its last start, by you or by the router (each start leaves a reading, and a stopped container keeps its start time); its last request the router measured (it measures a few per build and settings, not every one); and when its recipe file was last written (when it was added, and as a download finishes each file). The router's idle clock knows each reply's time but lives only in its memory, so it is not a source; a model the router keeps loaded is running, and a running model never goes.

**The policy (on by default).** mnm runs it at every `mnm add` and download end, every start (yours, a set's, or the router's for a request), and in `mnm doctor`:

- **From 15 points under the limit** (70% with the default 85%) the doctor's *Model storage* line WARNs, and the same line follows every `mnm add` plan and `mnm ls`: *the models folder's disk is at 71%: at 85% the least recently used instances' files go; `mnm disk` lists them; `mnm disk keep --models off` turns it off.* One notification goes out ([16.7](#167-told-when-it-ends-notifications)) when it first passes that line, and one at 5 points under (80%); each again only after the disk has dropped back below it.
- **At the limit** the least recently used instances' own files go, one instance at a time, until the disk is 5 points under it (80%). The rules are `mnm rm --files`' own: only files no other instance, LoRA, draft or ComfyUI workflow uses, and only in `hf/`, `civitai/`, `ollama/`, `modelscope/` and `voices/`. Never the files of an instance that is running, pinned (`mnm pin`), a member of a set in use or a persistent one, used in the last 24 hours, or the one a start or download is for. Each eviction is one notification, one journal line (the router's or the web UI's), and a line in `mnm disk`.
- **An evicted instance stays saved** (`mnm ls` says *evicted*). Its next `mnm start`, or the first request for it through the router, downloads its files again first: *Re-downloading qwen3-8b (5.0 GB): its files were evicted on <date> by the disk policy*, in the start's progress, the router's loading notes (Odysseus shows them in its thinking panel) and its journal. A download takes minutes for a large model; a client may time out on the first request, and the next one finds it loaded.
- **When what may go would not bring the disk under the limit** (the rest is your own files, shared files, or models in use), nothing is deleted at all, and the doctor, the notification and `mnm disk` say why, naming each instance that stays and the reason. Deleting a model for nothing is never the answer: `mnm disk` shows what else fills the disk.
- **A size limit** (`--models 400G`) measures what the models folder holds instead of the whole disk; the warnings then come at 85% and 95% of it, and eviction stops under 95%.

The limit lives in the state folder (`disk/disk.json`, with the notices sent and the last evictions); `mnm backup` does not carry it, so a restored box starts at the default 85%.

## 20. When something goes wrong

**What the router does by itself.** Every half minute it looks at the container of every model it counts as loaded (one `docker inspect`). An engine that died (its container exited, or was removed while nothing in mnm stopped it) is one journal line with the cause, Docker's exit code and the first thing its log names (`X died: its engine exited (code 137: killed, …); its log says: …`); its log is kept (`mnm logs X` reads it); requests for it answer 503 with that cause while it is started again once, the way a request loads a parked model (room made by the same rules, auto-fix if it runs out of memory): "Restarted X after it died: …", or, when that fails, "Could not restart X after it died: …" and a notice. A second death within ten minutes is a crash loop: no restart, the model parked until you `mnm start X`, one line and one notice. What you stopped, what you parked, a tuning copy and a model mid-start are never restarted. `journalctl --user -u makenomistakes-router` and System → Router's Activity show the lines; the nightly lists the night's deaths and what the router did ([26](#26-the-nightly)); `/metrics` counts them (`mnm_engine_deaths_total`, `mnm_engine_restarts_total`). The two services restart by themselves too (systemd): `mnm service status` and the doctor's *Services* line say how often in the last hour, and more than 3 an hour is a notice (at most one an hour) and a WARN.

Start with these three:

```bash
mnm logs <name> --review   # every warning and error in the engine log, each with what it means
mnm doctor                 # the machine
mnm vram                   # what the GPU holds right now
```

| You see | What it means / what to do |
|---|---|
| A plan with *offload* everywhere, or "N GiB of the card is in use by Windows and other apps" | Another program held the card while it was planned. `mnm replan --all` when more of the card is free |
| *won't fit* on start or in `mnm ls` | Other instances or programs hold the VRAM. The verdict names mnm's instances to stop: `mnm start <name> --stop-others` (or `--anyway`) |
| *on request* in `mnm ls` | Parked, and the router makes its room at its first request (it parks what the line names). Nothing to do; `mnm start <name> --stop-others` loads it now |
| Odysseus lists "no models" for an endpoint | That instance is stopped (`mnm stop` keeps it off: the router refuses it). `mnm start <name>`; parked is fine, it loads at the first request |
| An add refused with "None of its profiles fits right now: … in use by …" | Other programs hold that much of the card now: add it again when more of it is free (`mnm starter video` for wan5b) |
| *Engine log: N known, harmless warnings* after a start | Lines mnm knows need nothing from you (the open-port note, tokenizer quirks llama.cpp corrects, the embedding pooling note). `mnm logs <name> --review` lists them |
| *Changed and saved: …* after a start | Auto-fix: the engine ran out of memory and a smaller plan was saved. Free VRAM and `mnm set` it back, or `mnm replan <name>` |
| `mnm try` says the engine exited with *code 139* | The engine crashed (a segmentation fault). The last log lines print below it; `mnm logs <name>` has the rest, kept even after `mnm stop` |
| *code 137* | Killed. With *OOMKilled* (Docker's flag) it ran out of memory: stop other instances, or give WSL more memory in `.wslconfig`. Without it, something outside the engine killed it: a `docker kill`, the kernel, or another program |
| A request answers **503** "*X died: …. The router is restarting it once*" | The engine died while loaded; the router saw it, kept its log and is starting it again. Ask again in a minute; `mnm logs X --review` says why it died |
| **503** "*X died twice in ten minutes (…): parked*" | A crash loop: the router stopped restarting it. `mnm logs X --review`, fix what it names (often `mnm set X ctx=…` or a smaller quant), then `mnm start X` |
| `mnm service status`: "*restarted N times in the last hour*" | systemd restarted that service by itself (`Restart=`); above 3 an hour the doctor WARNs. `mnm service logs` or `journalctl --user -u makenomistakes-router --since -1h` says how each run ended |
| A request answers **409** | You stopped that instance: `mnm start <name>` |
| A request answers **401** | The router has a key: send it ([14.4](#144-a-key-for-the-router)) |
| An instance named X already exists | `mnm start X` runs it; or pick another `--name`; or `mnm rm X` first |
| Every port mnm gives … instances is taken | `mnm rm` instances you no longer use (their downloads stay) |
| Odysseus: a chat answers with image errors | That URL was added as type Image. Add chat models as LLM |
| Odysseus: the base URL 404s in a browser | Expected; open `/v1/models` instead |
| Odysseus: a voice errors out | The voice is not one of the model's; `mnm show <name>` lists them |
| Odysseus: an endpoint on a port 78xx does not answer | That is an engine's own port, on 127.0.0.1 only, which a container cannot reach: add the 7801 URLs from `mnm url` |
| `permission denied` from Docker right after setup | `wsl --shutdown` and reopen Ubuntu (WSL), or reboot (native), so the docker group applies |
| Odysseus cannot reach `host.docker.internal:7801` (native) | ufw drops it (the rule in [section 3](#3-native-debian-and-ubuntu)), or the container lacks `host-gateway` |
| The web UI shows an error page after an update | `mnm update` restarts it; if you started `mnm serve` yourself, restart that |
| A download warns about "unauthenticated requests" | Harmless: the download is anonymous. Add a token for gated models |
| `gh auth login`: "Failed opening a web browser" or "slow_down" | See [2.1](#21-the-one-line) |
| `mnm: command not found` | Open a new terminal (setup put it in `~/.local/bin`) |
| `mnm comfyui install` refuses or fails | A refusal before the build names what to fix (driver, card generation, BuildKit, disk). A failed build prints its last lines. Behind a proxy pass its certificate: `--ca FILE` |
| ComfyUI exits while starting | `mnm comfyui logs`; a pack that does not load is named (below). `mnm comfyui rollback` goes back to the image before |
| "These packs did not load, so the layer before stays in use: …" | ComfyUI started on the new layer and could not import that pack. `mnm comfyui logs` shows why; `mnm comfyui packs rm <pack>` then `packs install` |
| "ComfyUI is running a workflow: wait…" on `packs install` | The run finishes (or Stop the run on its page), then install again |
| A workflow "cannot run here yet" | It names why: a node pack not installed (`mnm comfyui packs add --for <workflow>`, then `packs install`), a model file not in place (`mnm comfyui models <workflow> --find`, then `--fetch`), or a node that works somewhere else (Comfy's paid cloud, or a pack's own service: it cannot run here) |
| "… is still running" when deleting a run | It is: wait for it, or Stop the run. A run whose process died reads *interrupted* and can be deleted, or Resumed if it is a film |
| A film stops at the stitch | The whisper.cpp image is not pulled: `docker pull ghcr.io/ggml-org/whisper.cpp:main-cuda`, then Resume |
| Odysseus lists no mnm tools (Settings → Integrations) | The router must run (`mnm service status`); the URL is `http://host.docker.internal:7801/mcp`. With `MNM_API_KEY` set the router refuses Odysseus (its MCP form cannot send a key): remove the key, or do without `/mcp`. 403 means a web page from another site asked, 405 a GET |

## 21. Uninstall

```bash
bash ~/makenomistakes/uninstall.sh --dry-run    # lists exactly what would go; changes nothing
bash ~/makenomistakes/uninstall.sh              # shows the list, asks you to type "remove"
```

Run it as your normal user; it asks for sudo where it needs it (the models folder under `/data`, `fstrim`, Docker before a re-login). A dry run never asks.

| Option | Effect |
|---|---|
| `--dry-run` | Print what would be removed; change nothing |
| `--yes` | Do not ask |
| `--keep-models` | Keep the downloaded models (everything else still goes) |
| `--keep-apps` | Keep the workspace apps' data (`MNM_APPS_DIR`: LibreChat's and Open WebUI's accounts and chats, OpenHands' home); their containers, images and OpenCode's unit still go |
| `--docker` | Also remove Docker Engine and the NVIDIA container toolkit. Refused while Odysseus is installed, and while Docker holds anything that is not mnm's |
| `--uv` | Also remove uv, its cache and the Pythons it downloaded |

**What goes:** the web UI and router services, the nightly and backup timers, mnm's containers and ports (ComfyUI's `mnm_comfyui` with them), ComfyUI's `mnm/comfyui` images and its `mnm-comfyui` network, the studio's images (`mnm/studio-tools`, `mnm/studio-trainer`, `mnm/studio-style-trainer`, every tag), the engine images (with the builds kept for rollback) and the GPU check image, vLLM's compile cache (the `mnm-vllm-cache` volume), the workspace apps mnm installed (OpenCode's unit and binary; LibreChat's, Open WebUI's and OpenHands' containers, images and networks; their data in `MNM_APPS_DIR` unless `--keep-apps`), the downloaded models (`hf/`, `voices/`, `civitai/`, `ollama/`, `modelscope/`), the voice configs, mnm's links in `views/`, `loras/`, `upscalers/` and `comfyui/`, the saved prompts (`prompt-cache/`: they hold conversations) and a quality tune's work (`quality/`), both even with `--keep-models`, the recipes, sets, engine records and kept logs, run records, imported ComfyUI workflows, node pack zips and approvals, ComfyUI's own folders (`runs/`, `comfy/`, `comfyui/` in the state folder), the studio's (`studio/` in the state folder: your stage picks, the voice and style libraries' rows, the images' ledgers), the secrets file, the `mnm` command, the download chunk cache, and the checkout itself. Inside each folder it removes only the entries mnm makes, then the folder once nothing else is in it; anything else stays and is listed.

**What stays:** what your runs made (`/data/outputs`, `MNM_OUTPUTS_DIR`; the script names it; `mnm clean outputs` first to delete it); your backups (`/data/backups`, `MNM_BACKUPS_DIR`; the timer goes); your voice library's recordings and checkpoints (`<models>/voice-library`) and your style library's adapters (`<models>/styles`), even when the models go (their rows go with the state folder, and the script names them before you confirm: `mnm backup` first keeps them, and its restore after a reinstall puts them back beside the kept files; without one, move a voice's or style's files out of its folder and add it again); Odysseus (its folder, containers, images, volumes and networks are never touched; `ODYSSEUS_DIR=/path` if it lives elsewhere); Docker unless `--docker` (and even then while Odysseus exists, or while Docker or apt hold anything else); files mnm did not make; the NVIDIA driver on a native box (the system needs it); ufw, except the one rule setup added; git, the GitHub CLI and its login, `/etc/wsl.conf`; uv unless `--uv`.

**Safety:** a folder is never touched when its setting is a relative path, a system folder, your home or above, another user's home, inside or holding Odysseus's folder, or holds a mount point. Any step that fails is reported, the rest still runs, and the script exits 1.

**Before you run it:** copy anything of your own out of the checkout and push any unpushed work: the checkout goes too, and the summary warns about uncommitted changes, stashes and unpushed branches.

**Give the space back to Windows (WSL only).** Deleting files frees space inside WSL, but the disk file Windows sees (`ext4.vhdx`) does not shrink by itself. The script runs `sudo fstrim -v /`, writes `mnm-shrink-wsl-disk.ps1` into your Windows user folder and prints the command:
1. Close every WSL terminal.
2. Open PowerShell as administrator.
3. Run `powershell -ExecutionPolicy Bypass -File "C:\Users\<you>\mnm-shrink-wsl-disk.ps1"`.

It checks it runs as administrator, finds the distribution's `ext4.vhdx`, runs `wsl --shutdown` (Docker and Odysseus stop too, and start again with WSL), compacts the disk with diskpart and prints the size before and after. When the shell cannot reach Windows, the script prints the manual diskpart steps in full instead.

**Start fresh** afterwards with the one line of [2.1](#21-the-one-line). The uninstall asks you to type "remove", so without a terminal it needs `--yes` (a dry run never asks); the installer after it needs `MNM_SETUP_YES=1` to install anything without one.

## 22. Where everything lives

| What | Where |
|---|---|
| The code | `~/makenomistakes` |
| The `mnm` command | `~/.local/bin/mnm` |
| Recipes, sets, run records, the engine ledger, kept engine logs, measurements, the tuner's queue and trials (`tuning/`), the boxes mnm has seen (`machine.json`), the disk policy's limit and evictions (`disk/disk.json`) | `~/.local/state/makenomistakes` (`MNM_STATE_DIR`) |
| ComfyUI | in the state folder: `runs/` (run records), `comfy/` (imported workflows, node pack zips and approvals, the workflows' model links), `comfyui/` (ComfyUI's own folders, its image ledger, its node list) |
| Downloaded models | `/data/models` (`MNM_MODELS_DIR`): the Hugging Face cache layout under `hf/`, voice packages under `voices/`, Civitai files under `civitai/`, Ollama's under `ollama/`, ModelScope's under `modelscope/` |
| Picture views, LoRA and upscaler links, voice configs, ComfyUI's links | `/data/models/views`, `loras`, `upscalers`, `configs`, `comfyui` |
| Saved prompts (`cache=on`) | `/data/models/prompt-cache/<name>/`: a slot file and its sidecar each, owned by you (mode 700), never backed up ([11](#11-coding-agents)) |
| A quality tune's work | `/data/models/quality/<name>/`: its text and the reference's log-probabilities while it runs ([15.8](#158-tuning-for-quality-mnm-tune---quality)) |
| What runs made | `/data/outputs` (`MNM_OUTPUTS_DIR`): one folder per day, one per run. From Windows (WSL): `\\wsl.localhost\Ubuntu-24.04\data\outputs` in Explorer's address bar, with your distro's name from `wsl -l` in place of `Ubuntu-24.04` (`mnm run` prints each run's folder this way too) |
| vLLM's compile cache | the Docker volume `mnm-vllm-cache` |
| The studio | in the state folder's `studio/`: `stages.json` (your stage picks, `mnm studio stage`), `voices/` and `styles/` (the two libraries' rows, a file each), `tools.json`, `trainer.json` and `style-trainer.json` (the images' ledgers, with the image before each), `style-probes.json` (the card peaks style trainings measured); songs, their candidates, edits, mixes and masters, and trainings are runs (`runs/`), their files under the outputs folder (a song's export in its run's `export/`) |
| The voice and style libraries | `/data/models/voice-library/<name>/` (a voice's reference recording or RVC checkpoint and index), `/data/models/styles/<name>/` (a style's adapter); the style trainer's ACE-Step checkpoint is an ordinary download under `hf/` |
| A checkpoint mnm converts | `converted/<stem>/` beside its download (the room dereverb's `model.safetensors` and config); `mnm rm --files` takes it with the download |
| Backups | `/data/backups` (`MNM_BACKUPS_DIR`): `mnm-<date>-<time>.tar.gz`, 7 daily and 4 weekly kept; the timer `~/.config/systemd/user/makenomistakes-backup.{timer,service}`; the last backup and restore test in the state folder's `backup/last.json` |
| The nightly | its reports in the state folder's `nightly/`; the timer `~/.config/systemd/user/makenomistakes-nightly.{timer,service}` ([26](#26-the-nightly)) |
| The workspace apps | their data under `/data/apps/<app>` (`MNM_APPS_DIR`), their ledgers under the state folder's `apps/`, OpenCode's binary there too and its unit `~/.config/systemd/user/mnm-opencode.service`; your repositories in `~/src` (`MNM_PROJECTS_DIR`) |
| Secrets and setup's answers (`HF_TOKEN`, `CIVITAI_TOKEN`, `MNM_API_KEY`, `MNM_MODELS_DIR`, `MNM_OUTPUTS_DIR`, `MNM_ROUTER_HOST`, `MNM_PROJECTS_DIR`, `MNM_OPENCODE_PASSWORD`, `MNM_OPENHANDS_KEY`) | `~/.config/makenomistakes/secrets.env` (mode 600) |
| The web UI and router services | `~/.config/systemd/user/makenomistakes-web.service`, `makenomistakes-router.service` |

Variables, if you want other places or ports: `MNM_MODELS_DIR`, `MNM_STATE_DIR`, `MNM_OUTPUTS_DIR`, `MNM_BACKUPS_DIR` (absolute paths; for the backups see [18.6](#186-backup-and-restore)), `MNM_APPS_DIR` and `MNM_PROJECTS_DIR` (the workspace apps' data and the folder of your code they see, [12](#12-apps)), `MNM_PORT` (7800), `MNM_ROUTER_PORT` (7801), `MNM_ROUTER_HOST`, `MNM_ROUTER_HOSTS` (the names the router answers to, [13.2](#132-from-this-machines-containers)), `HF_TOKEN`, `CIVITAI_TOKEN`, `MNM_API_KEY`, `MNM_GPU` (the card, [18.5](#185-new-hardware)), `MNM_BACKEND` (the backend, [25.10](#2510-backends-cuda-vulkan-rocm-cpu)), `MNM_NTFY_URL`, `MNM_NTFY_TOKEN` and `MNM_TOAST` (notifications, [16.7](#167-told-when-it-ends-notifications)), `MNM_FLOOR_CHAT` and `MNM_FLOOR_AGENT` (the quality tune's floors, [15.8](#158-tuning-for-quality-mnm-tune---quality)), `MNM_WEB_URL` (the web UI's address in MCP links, [12.6](#126-odysseus)), `MNM_OLLAMA_PORT` (Ollama's API on its own port, [13.6](#136-ollamas-api)), `MNM_ROUTER_URL` (a client install only: the box's router, [13.5](#135-a-client-only-install)) and `HF_ENDPOINT` (a Hugging Face mirror, [7.12](#712-models-from-civitai-the-civitai-archive-lm-studio-ollama-and-modelscope-and-odd-repo-layouts)). No model file, token or key is ever stored in the repository.

## 23. Command reference

An overview of every command, with the section that explains it. [commands.md](commands.md) lists every command and every option in full, generated from the CLI itself; `--help` after any command shows that command's.

| Command | Does |
|---|---|
| `mnm setup [--dry-run] [--yes] [--install-driver] [--plain] [--no-pull]` | The guided setup ([2.3](#23-the-guided-setup)) |
| `mnm apps ls\|status\|install\|start\|stop\|logs\|update\|rollback\|uninstall <app>`, `mnm apps user add librechat EMAIL`, `mnm apps connect goose\|odysseus` | The workspace apps ([12](#12-apps)) |
| `mnm apps expose opencode\|librechat\|open-webui\|openhands [--off]` | Open an app to your network (its own login and the router's key required), or close it ([13.3](#133-from-another-machine-on-your-network)) |
| `mnm starter [PICKS…] [--list]` | Add starter models that fit this card |
| `mnm doctor [--deep] [--json]` | Machine checks ([4](#4-check-the-machine)) |
| `mnm status` | The box on one screen: health, the doctor's worst lines, the models by state, the queue, the last nightly, the web UI's address; exit 1 when the router is down or a check fails ([4](#4-check-the-machine)) |
| `mnm doctor --from here\|docker\|lan [--json]` | The router as a client sees it from there: the desk's roles, a chat, MCP, Ollama's port, the key ([13.3](#133-from-another-machine-on-your-network)) |
| `mnm doctor --remote-script [--shell sh\|powershell]` | The same checks as a script for another machine; no key inside ([13.3](#133-from-another-machine-on-your-network)) |
| `mnm doctor --app APP [--container NAME] [--json]` | The router as a workspace app sees it, from where it sits ([13.7](#137-workspace-apps-one-file-each)) |
| `mnm resolve <ref> [--json]` | What a reference is: type, format, files, quants |
| `mnm fit <ref> [--version V] [--vram GiB [--ram GiB] [--gpu NAME]]` | The three profiles, without downloading; `--vram`/`--ram` plan for another machine, its RAM floor included ([25.2](#252-the-ram-floor-worked-through)) |
| `mnm what [WORDS…] [--type USE] [--for USE] [--limit N] [--vram GiB [--ram GiB] [--gpu NAME]]` | Find models and grade each against this box (curated first, then each source's order); `--for` the three curated picks for a use with why; never downloads ([5](#5-your-first-model)) |
| `mnm add <ref> [--name N] [--profile P] [--quant Q] [--ctx C] [--kv K] [--version V] [--type T] [--for-agents \| --for CLIENT] [--no-download] [--no-tune] [--no-upscaler]` | Plan, save, download; a llama.cpp chat, coding or vision instance is queued for tuning unless `--no-tune` ([15.7](#157-tuning-mnm-tune)) |
| `mnm start <name> [--stop-others \| --anyway]` | Start and wait until ready; print the URLs |
| `mnm stop <name>… \| --all [--park]` | Stop (recipe, files and log stay); it stays off, or with `--park` loads at its next request |
| `mnm ls [--disk]` | Every instance: type, port, settings, state, Fits now; `--disk` each one's files and last use |
| `mnm disk [--json]` / `mnm disk keep [--models 85%\|400G\|off]` / `mnm disk check` | What the models folder holds, by instance, folder and unlisted; the disk policy's limit; run it now ([19.1](#191-disk-what-the-models-folder-holds-and-the-disk-policy)) |
| `mnm show <name>` | Recipe, settings with advice, parts, the exact command |
| `mnm why <name> [--more N]` | The plan in words: why this quant, context and place, what a plan now would change, what N GiB more would (4 by default); saves nothing ([15.9](#159-why-this-plan-mnm-why)) |
| `mnm set <name> key=value…` | Change settings (next start; `idle` and `keep` at once); `draft=<instance>` adds a draft model, `draft=mtp` the model's own MTP head ([15.6](#156-a-draft-model-speculative-decoding)); `moe-cache=<MiB>` caches RAM-held experts on the card ([11](#11-coding-agents)); with several cards `experts=auto\|ram\|cards`, `card-experts=a,b`, `gpus=0,1`, `split=tensor` ([25.7](#257-the-settings)) |
| `mnm pin <name> [--for 2h]` / `mnm unpin <name>` | Keep a model loaded (`keep=always`, or `idle` for a while) / let the router unload it ([14.3](#143-pinning)) |
| `mnm notify ["text"] [--ntfy URL\|off] [--toast on\|off]` | Notifications when long things end: test, set the channels ([16.7](#167-told-when-it-ends-notifications)) |
| `mnm url <name> [--for APP] [--from here\|docker\|lan] [--direct]` | The cards to paste; `--from` names where the client sits ([13](#13-other-apps)) |
| `mnm url --for opencode\|librechat\|open-webui\|openhands\|goose\|odysseus\|hermes [--from …] [--write [--to PATH] [--force]]` | A workspace app's whole file, by desk role; `--write` writes or merges it ([13.7](#137-workspace-apps-one-file-each)) |
| `mnm try <name> [--speed] [--measure] [--image F] [--mask F] [--audio F] [--say TEXT]` | Send a test request; pictures, clips and speech land in the current folder |
| `mnm try --all [--speed]` | Every saved instance in turn, each alone; one table, also saved as `try-all-<time>.txt` ([18.3](#183-what-mnm-learns-from-your-card)) |
| `mnm try <name> --agent [--prefix N]` | An agent's long prefix read cold, warm, and with `cache=on` after a park with its saved slot put back; kept as a reading ([11](#11-coding-agents)) |
| `mnm run <name> ["prompt"] [--image F] [--mask F] [--audio F] [--repeat N]` | Your prompt through the router; the result kept as a run ([16.5](#165-making-things-run-runs-and-clean-up)) |
| `mnm outputs ls \| show <run> \| rm <run…> [--keep-files] \| prune [--older-than 30d] [--max-gb N] [--yes]` | The runs and their files; delete or prune by age or size |
| `mnm clean [outputs\|runs\|logs\|workflows\|imports\|comfy] [--older-than 30d] [--keep-files]` | Empty what runs and engines left behind; none: show each with its size |
| `mnm pipeline run "IDEA" --chat C --image I --video V [--voice T] [--shots N] [--style S] [--auto] [--clip-size WxH] [--clip-frames N]` | A short film from one idea ([16.6](#166-pipelines-a-short-film-from-one-idea)) |
| `mnm pipeline go\|resume\|redo\|show\|ls` | Past a review point; carry on after a failure; make one item again; a run's steps; the runs |
| `mnm song new "THEME" [--length 30s] [--lyrics FILE] [--tags TEXT] [--recording FILE --as complete\|vocal2bgm\|reference] [--candidates 4] [--tier fast\|full] [--chat C] [--auto] [--voice NAME [--shift N]] [--clean LIST]` | A song from a theme: the lyrics (a review point), candidates scored to pick from (another), the picked one's stems, its vocals cleaned and sung in a library voice when asked, MIDI, the mix and the export folder ([28.2](#282-lyrics-and-candidates)) |
| `mnm song go\|resume\|show <run>` / `mnm song pick <run> N` / `mnm song redo <run> STAGE [N]` / `mnm song ls` | Past the lyrics; carry on after a failure; a run's steps; pick a candidate; make one item again (`lyrics`, `song`, `score`, `structure`, `stems`, `vocals`, `voice`, `midi`, `mix` or `export`; `score` alone scores every candidate again); the song runs ([28.2](#282-lyrics-and-candidates)) |
| `mnm song edit <run> repaint\|resing\|extend\|cover\|add\|swap\|cut\|crop\|duplicate\|move\|tempo\|key key=value…` / `mnm song take <run> [<edit>\|candidate]` | Edit the song's take: sung again with ACE-Step ([28.7](#287-edits)) or cut, moved, retimed and retuned by ffmpeg ([28.8](#288-section-tools-tempo-and-key-midi-and-mastering)); with no `key=value` an edit says what it takes; `take` lists the takes or uses another (Undo) |
| `mnm song mix <run> [STEM=DB[,mute][,solo][,in=S][,out=S] …] [--use <mix> \| --own]` | The stem mixer: the mix in use, a new render, or another mix to use ([28.3](#283-scores-the-structure-map-stems-and-the-mixer)) |
| `mnm song master <run> [--reference FILE \| --use <master> \| --off]` | Master the mix in use to a reference track with Matchering, use an earlier master, or none ([28.8](#288-section-tools-tempo-and-key-midi-and-mastering)) |
| `mnm song voice <run> NAME\|none [--shift N]` / `mnm song vocals <run> debleed,dereverb,denoise,deess,gate,tune\|none [--key KEY] [--strength 80]` | Sing the vocals in a library voice, or the song model's own again ([28.5](#285-voices)); clean and tune them first ([28.9](#289-clean-and-tune-the-vocals)) |
| `mnm song export <run> [--mp3] [--release] [--lufs -14] [--true-peak -1] [--no-loudness]` | The export folder again: the mix, the stems, MIDI, `song.rpp`, `provenance.json`, `disclosure.txt` ([28.6](#286-exports-and-reaper)) |
| `mnm studio stage [STAGE [fast=<instance>\|full=<instance>\|fast=]]` / `mnm studio models` | The stages registry: each stage's tiers and the instance (or `workflow:<name>` for the song stage) that plays each, picked or put back ([28.1](#281-the-first-song), [28.4](#284-songs-from-comfyui-workflows)); every studio model against this box: what it needs, whether it holds it, its time a second of audio ([28.12](#2812-the-models-and-your-box)) |
| `mnm studio install [--ca FILE] [--yes]` / `mnm studio status\|rollback` / `mnm studio uninstall [--yes]` | The studio tools image, built here (about 3.7 GB): the scorer, the structure map, the speaker similarity, MIDI, mastering and tuning, all on the CPU ([28.3](#283-scores-the-structure-map-stems-and-the-mixer)) |
| `mnm studio trainer install [--for voices\|styles] [--cpu] [--ca FILE] [--yes]` / `mnm studio trainer status\|rollback [--for voices\|styles]` / `mnm studio trainer uninstall [--for voices\|styles] [--yes]` | The two trainers' images, built here: Applio for voices, ACE-Step's own trainer for styles; CUDA 13 or `--cpu` ([28.10](#2810-train-a-voice), [28.11](#2811-teach-ace-step-a-style)) |
| `mnm studio train voice NAME --from DIR --source own\|generated\|hired\|licensed\|likeness [--consent FILE] [--rate 40000] [--epochs 200] [--batch 8] [--licence TEXT] [--now]` | Train an RVC voice from your recordings; the router trains while nobody uses the card, a request pauses it ([28.10](#2810-train-a-voice)) |
| `mnm studio train style NAME --from DIR --caption TEXT --source own\|licensed\|generated [--trigger WORD] [--variant base\|turbo] [--epochs 400] [--max-seconds 240] [--licence TEXT] [--now]` | Teach ACE-Step a style from your songs, its first epoch the card's probe ([28.11](#2811-teach-ace-step-a-style)) |
| `mnm studio train ls` / `mnm studio train show\|cancel <run>` | The trainings (queued, training, paused, done); one's settings, warnings and outcome; cancel one ([28.10](#2810-train-a-voice)) |
| `mnm studio styles ls` / `mnm studio styles show <name>` / `mnm studio styles add NAME --adapter DIR --variant base\|turbo --trigger WORD --source own\|licensed\|generated [--licence TEXT]` / `mnm studio styles rm <name> [--keep-files]` | The style library: a LoRA trained here or elsewhere, played with `mnm set <song> style=<name>` ([28.11](#2811-teach-ace-step-a-style)) |
| `mnm voices add NAME --reference FILE [--engine seed_vc\|vevo2\|soulx] --source own\|generated\|hired\|licensed\|likeness [--licence TEXT] [--note TEXT]` / `mnm voices add NAME --engine rvc --model FILE [--index FILE] --source …` | Add a voice to the library: a reference recording, or an RVC checkpoint and its optional index; copied and kept by sha256 ([28.5](#285-voices)) |
| `mnm voices ls` / `mnm voices show <name>` / `mnm voices rm <name> [--keep-files]` / `mnm voices consent <name> DOCUMENT [--note TEXT]` | The voices, one in full, remove one (and its files), record a consent document's sha256 and date (the document is never copied) ([28.5](#285-voices)) |
| `mnm logs <name> [--tail N] [--review]` | The engine log (400 lines; `--tail 0` all) and what each warning means |
| `mnm rm <name> [--files]` | Remove the instance (and the downloads only it uses) |
| `mnm vram [--seconds S] [--every E]` | GPU memory readings |
| `mnm calibration [export]` | What mnm measured on this box and the corrections it uses (another machine's readings kept, not used), each studio model's music speed ([28.12](#2812-the-models-and-your-box)) and the router's *queue* ([14.9](#149-the-queue-who-waits-and-for-how-long)) |
| `mnm report` | A report folder, `report-<date>-<time>/` in the current folder, with the version, the doctor, `mnm ls`, `mnm machine show`, the engines, the calibration and its export, the outputs, backups, the nightly, ComfyUI's status and log, the studio (the tools' and trainers' status, the stages, the trainings, the songs, the voices and styles), the services, `nvidia-smi`, the router's and web UI's journals and every instance's log; every secret value, card id and MAC address removed before a file is written: the folder to attach to a bug report |
| `mnm lora ls\|add\|set\|rm` | LoRAs on picture instances |
| `mnm sets create\|ls\|show\|start\|stop\|edit\|rm\|url` | Sets of instances with roles; `create … --swap` makes them take turns, `--persistent` keeps them loaded; `edit <set> role=instance… [--drop role]` changes members |
| `mnm sets default [<set> \| --clear]` | The desk: the set whose roles (`coder`, `chat`, `memory` …) are model ids ([16.2](#162-sets)) |
| `mnm replan <name> \| --all [--any-quant] [--vram GiB] [--yes]` | Plan saved instances again for this machine |
| `mnm backup [--to FILE]` / `mnm restore FILE [--force]` | Recipes, sets, workflows, approvals and readings in one file, and back onto a fresh install ([18.6](#186-backup-and-restore)) |
| `mnm backup --auto` / `mnm backup ls` / `mnm backup schedule [--at 03:30\|--off]` / `mnm restore --test [FILE]` | Backups into `MNM_BACKUPS_DIR`, each restore-tested in a scratch state, 7 daily and 4 weekly kept; a nightly timer ([18.6](#186-backup-and-restore)) |
| `mnm nightly run [--now] [--skip STEP]` / `mnm nightly show [DATE]` / `mnm nightly ls` / `mnm nightly set clean=30d\|off` / `mnm nightly install [--at 03:00]` / `mnm nightly status` / `mnm nightly uninstall` | The nightly self-check, its regression rule and its one notice ([26](#26-the-nightly)); `mnm clean --auto` is its clean step |
| `mnm comfyui [--write COMFYUI_FOLDER]` | Share the picture models with ComfyUI |
| `mnm comfyui install [--cpu] [--ca FILE] [--yes] \| start \| stop \| status \| logs [--tail N] \| rollback \| uninstall` | ComfyUI as mnm's optional engine ([18.4](#184-comfyui)) |
| `mnm comfyui workflow add <link\|file> [--name N] \| ls \| show \| fields [--all] \| pick <workflow> KEY… [--unpick] \| rm [--files]` | Import a ComfyUI workflow, see what it needs and the settings it takes; pick the ones shown on top; remove it, with `--files` the model files only it used ([18.4](#184-comfyui)) |
| `mnm comfyui run <workflow> [key=value…] [--file key=path] [--repeat N]` / `mnm comfyui again <run> [--new-seed]` | Run an imported workflow; the outputs become a run; run one again ([18.4](#184-comfyui)) |
| `mnm comfyui models <workflow> [--find] [--set FILE=LINK] [--fetch] [--yes]` | A workflow's model files: where each is; find, link, download ([18.4](#184-comfyui)) |
| `mnm mcp [tools [--schema]]` | The card for Odysseus's MCP Tool Server (Settings → Integrations); the tools the router serves at `/mcp` ([12.6](#126-odysseus)) |
| `mnm comfyui packs ls \| add <pack…> [--version V] [--for <workflow>] [--yes] \| rm <pack> \| install [--yes] \| rollback \| find <Class>` | Custom node packs: vetted, approved (from the Registry, or `github:owner/repo@<tag or commit>`), built into ComfyUI ([18.4](#184-comfyui)) |
| `mnm engines [update [ENGINE…] \| test ENGINE \| rollback ENGINE]` | Engine images, each with the backend its tag is built for ([25.10](#2510-backends-cuda-vulkan-rocm-cpu)); an update tests each new build and rolls a failing one back |
| `mnm engines check [--json]` | Whether the registry holds a newer build of each engine, and whether it runs an instance the installed one lacks; nothing pulled ([18.8](#188-is-there-something-newer)) |
| `mnm upgrade <name> [--yes] [--vram GiB]` / `mnm upgrade <name> --replace [--yes]` | The newest family member of an instance's model, planned with its settings and shown as a diff; `--yes` adds it beside (the old one kept as `<name>-old`); `--replace` removes `<name>-old` and its files ([18.8](#188-is-there-something-newer)) |
| `mnm machine [show [--json]]` / `mnm machine history` / `mnm machine forget <id> [--yes]` | This box as mnm knows it (cards, CPU, RAM, board, platform), what changed since the box before and which instances were planned on another; every box seen; drop one that is not this box ([27](#27-your-box)) |
| `mnm machine adopt [NAME…] [--only NAME] [--yes] [--keep-quants] [--no-tune] [--dry-run]` | Plan every instance (or the named ones) again for this box after its hardware changed: the RAM measured, the quant free, every download listed with its size and asked once, tunes queued; never deletes, never writes the secrets file ([27](#27-your-box)) |
| `mnm router [--host H] [--port P]` | The router by hand (the service runs it) |
| `mnm router manifest` | What the router's `/__mnm/manifest` answers ([14.8](#148-the-manifest-__mnmmanifest)) |
| `mnm update` | Update mnm and restart the web UI and the router |
| `mnm serve [--port P]` / `mnm service install\|status\|logs\|uninstall` | The web UI, by hand or as a service |
| `mnm open [<name>]` | The web UI, or an instance's page, in this machine's browser; the address when nothing here opens a page ([17.1](#171-the-shell)) |
| `mnm tune <name> [--now] [--for chat\|agents] [--minutes N]` / `mnm tune --all` / `mnm tune --status` / `mnm tune <name> --cancel` | Tune an instance's settings on this box: real starts of a copy, measured on fresh prompts; runs when the card is idle and yields to any request ([15.7](#157-tuning-mnm-tune)) |
| `mnm tune <name> --quality [--floor N] [--quants Q6_K,Q5_K_M] [--keep] [--yes] [--now]` | Score the model's quants against Q8_0 with llama.cpp's perplexity tool and keep the most faithful one over the speed floor; shows its downloads first ([15.8](#158-tuning-for-quality-mnm-tune---quality)) |
| `mnm --version` | Version and commit |
| `mnm --install-completion` / `mnm --show-completion` | Shell completion for bash, zsh or fish: commands, options and instance names ([2.4](#24-after-installing)) |

## 24. What has been checked on real hardware

mnm's defaults were measured on one machine, called the **reference box** in this guide: an RTX 4070 (12 GB) under WSL2, with 24 GB of RAM given to WSL. Measured figures in this guide come from there unless they say otherwise; the rest are mnm's estimates, and your box's own readings replace both ([27.2](#272-measured-here-or-estimated-from-a-row)).

**What has run on the reference box**, end to end:

- **Install:** fresh installs with `install.sh`, the guided setup and its starter models.
- **Chat and coding:** Qwen3-0.6B at about 330 tok/s; Qwen3-8B at 75 to 80 tok/s (69 at a 32,768-token context); Qwen3-30B-A3B with the experts of 27 of its 48 layers in RAM at 27 to 33 tok/s; Qwen3.6-35B-A3B at 25 to 28 tok/s with most of its experts in RAM; a model from Ollama's library. The coding models Qwen3.6-35B-A3B and Qwen3.8-27B each passed five of five tool-calling steps (`mnm try <name> --tools`).
- **Coding agents and apps:** OpenCode installed and used with the coding models, and Claude Code fixing a bug through the router; LibreChat, OpenHands and Open WebUI installed; Goose and Odysseus connected. Odysseus chatted with the models, described pictures and drew them, each straight at an instance.
- **Vision, embeddings and rerank.**
- **Pictures:** Z-Image (a 1024 picture in 16.6 s, +7.9 GiB at the peak; a repaint in 12.5 s), FLUX.2 klein 4B, FLUX.1, SD 1.5, Anima, SDXL checkpoints and an SDXL diffusers folder, each at its own size; a Kontext edit (86.8 s at 1024); a Civitai checkpoint (about 21 s a picture) and a LoRA on it.
- **Video:** Wan 2.2 TI2V 5B clips from text and from a picture (33 frames in about 2 minutes, +6.2 GiB at the peak; through ComfyUI, 832x480 and 33 frames in 48 s), and short films from one idea with either's clips.
- **Speech:** speech-to-text on whisper.cpp (an 11 s clip transcribed in 0.69 s) and text-to-speech with PocketTTS (about 4 to 5 times real time).
- **vLLM:** Qwen3-8B-AWQ at about 70 tok/s, Qwen3-4B-AWQ at 113 to 134 tok/s.
- **The router:** every model type loaded on request, parking what held the card; a stopped model refused; the idle unload; a request's `keep_alive`, `mnm pin` and `/metrics`; its key on and off; Ollama's API on port 11434; a killed engine restarted in 7 s, and parked after two deaths.
- **Several models:** sets, and a set that takes turns (`--swap`); `--stop-others` refused against a pinned model; `mnm try --all --speed` over every model; `mnm replan`; the web UI's service across a reboot.
- **ComfyUI:** built and sandboxed (no internet; the router refuses it), imported workflows with their model files, two node packs built in and used.
- **Engine updates:** a new llama.cpp build tested against the old one and kept, a broken one failing its test, the sd.cpp self-test passing; `mnm engines check`.
- **Looking after itself:** the tuner (Qwen3-8B 8% faster, Qwen3.6-35B-A3B's agent score up 116%), automatic backups with their restore test and timer, the disk policy evicting a model and downloading it again, the nightly reporting a slowdown, `mnm upgrade` (Qwen3-8B to Qwen3.5-9B, 68.4 tok/s), the machine fingerprint.
- **Large models' stand-ins:** GLM-4.7-Flash (11.4 tok/s) and Nemotron Bolt with their experts in RAM; DeepSeek-V4.1-Flash refused with the RAM floor's arithmetic ([25.2](#252-the-ram-floor-worked-through)).
- **Sources and the rest:** downloads from Hugging Face, Civitai, Ollama's library and ModelScope; `mnm run --repeat`; notifications as a Windows toast; `mnm report`.

**What has not run on real hardware yet:**

- A native Debian or Ubuntu box, a box with several cards, an AMD card, a DGX Spark, a server board, a box without a card, ARM64. mnm plans these from vendors' figures and documentation and from other people's published readings, and on such a box your own readings are the first measurements. Vulkan has run only on Mesa's software device, the processor backend end to end only in a virtual machine, and ARM64 only under emulation.
- The studio's music models and both trainers on a card: they have run on the processor only, and REAPER has not yet opened an exported project.
- The large models on the bigger boxes that hold them: DeepSeek-V4.1-Flash, DeepSeek-V4-Flash and the rest of [25.8](#258-what-the-ladder-of-boxes-runs)'s ladder.
- Odysseus through the router's one URL (its chat at `/chat/v1`, its image tool, its memory), Odysseus's speech (its build hides read-aloud and has no speech-to-text page) and Odysseus running mnm's tools over MCP.
- The other apps' cards with the apps themselves (Codex, Qwen Code, Aider, Cline, Zed, Home Assistant, Karakeep, Frigate, n8n, JetBrains, Raycast, Obsidian, Brave Leo), and Tailscale Serve.
- The picture families FLUX.2 klein 9B, Qwen-Image 2.1, Krea 2, ERNIE-Image and Chroma (each has drawn on a processor build), every video family but Wan 2.2 TI2V 5B, LoRAs on video models, and Qwen3-TTS.
- These have run on processor builds only, not yet on a card: the prompt cache's saved slots, the router's queue readings, `mnm tune --quality`, `mnm machine adopt` after a real change of hardware, the device table's learner, `mnm what`, `mnm why`, `mnm status` and `mnm open`, and the Anima LoRA; and the web UI's pages in a browser with engines on a card.
- A real uninstall (its dry run has run).

**mnm corrects itself from your box.** Every start, request and tune is a reading kept for the box it was taken on, and the planner corrects its estimates from this box's readings ([18.3](#183-what-mnm-learns-from-your-card), [27.6](#276-where-the-readings-go-and-when-a-box-comes-back)). A box mnm has never seen starts from the estimates and learns its own.

## 25. Large models and many cards

Models larger than one card, planned honestly from the measured machine, and boxes with several cards. **On one card** this has run on the reference box ([24](#24-what-has-been-checked-on-real-hardware): GLM-4.7-Flash and Nemotron Bolt as the large models' stand-ins, the RAM floor's refusals, the tuner); **nothing with several cards, an AMD card, unified memory or a server board has run yet.** For a box mnm has never seen, start at [27](#27-your-box): it says what mnm reads, what it estimates and what a change does; 25.9–25.14 are its details. Every figure below is mnm's own estimate, from the models' recorded headers (checked against llama.cpp's own fitter) and the cards' published figures. Once a large model runs, `mnm tune` finds its fastest settings on your box ([15.7](#157-tuning-mnm-tune)).

### 25.1 How a model larger than the card runs

llama.cpp keeps what does not fit on the cards in RAM and reads it there for every token: a mixture of experts keeps its experts in RAM first (only a few are read per token), then whole layers. mnm plans that split from the measured machine and says it in the plan table ("8.4 GiB on the card + 12.4 GiB in RAM"). RAM is read much more slowly than the card, so the more of a model sits in RAM, the slower it decodes; how fast your RAM feeds the experts is something mnm learns from your MoE models' readings ([18.3](#183-what-mnm-learns-from-your-card)), and `mnm doctor --deep` says the best each saved model could do at the RAM's measured bandwidth ("coder36 reads about N GB a token from RAM (M layers): at B GB/s that is T tok/s at best"; a ceiling, not a plan).

### 25.2 The RAM floor, worked through

Some models are too large for any split. Before any profile mnm checks that the smallest quant can fit at all: its weights (less an Engram table, below), its cache and buffers, and 4 GiB kept for the system (WSL's kernel, Docker, mnm's services, llama.cpp's host buffers) must fit the machine's RAM and VRAM together. RAM and disk are in GB, as they are sold; the cards in GiB, as `nvidia-smi` says them.

DeepSeek-V4.1-Flash's smallest GGUF, unsloth's UD-IQ1_S, on the reference box (an RTX 4070 with 24 GB of RAM under WSL):

| | |
|---|---|
| weights that must sit in RAM or on the card | 122 GB |
| cache and buffers (4,096 tokens) | 0.4 GB |
| kept for the system (4 GiB) | 4.3 GB |
| **needed** | **about 127 GB** |
| the Engram table, read from disk as needed | 111 GB more, on disk only |
| the reference box has | 25 GB of RAM (24 GB by `.wslconfig`) + 12 GiB of VRAM (12.9 GB): about 38 GB |

127 GB against 38: refused, about 89 GB short. `mnm fit` and `mnm add` say the arithmetic and stop:

```text
DeepSeek-V4.1-Flash-1017 UD-IQ1_S needs about 127 GB in RAM and on the card (122 GB of weights,
0.4 GB of cache and buffers, 4.3 GB kept for the system, and 111 GB more stay on disk); this machine
has 25 GB of RAM and 12 GiB of VRAM. The smallest quant that would fit: none. A box with 128 GB of
RAM and these cards runs it.
```

The last sentence is the same sum turned around: 127 GB less what the cards hold, rounded up to the next size a box's RAM comes in (16, 24, 32, 48, 64, 96, 128, 192, 256 GB …). A box with 128 GB and one 12 GiB card holds it (128 + 12.9 ≥ 127), only just; holding a model is not running it fast (25.6). A plan for another machine (`mnm fit <link> --vram 23 --ram 60`) is checked against the RAM you give; without `--ram` it is checked against this machine's.

### 25.3 The Engram table: on disk

DeepSeek-V4.1's GGUFs carry an *Engram* table (111 GB at UD-IQ1_S, 161 GB at the IQ3 quants) that llama.cpp reads from disk a row at a time. mnm downloads it with the model, counts it as disk only ("…, 111 GB of it on disk, read as needed" on `mnm add`'s download line, "disk, read as needed" in `mnm show`'s parts), and starts the model with `--load-mode mmap --lazy-mode on -ot per_layer_token_embd.weight=CPU`. `mnm set <name> load=none` (or `dio`) is refused for such a model: it would read the table into RAM. **Put the models folder on a fast SSD**: every token reads a few rows of it.

### 25.4 Quants, architectures and converters

- **Quants that do not load.** unsloth's UD-Q3_K_XL and larger quants of V4.1 store the table as MXFP8 (ggml type 43 in the file), which llama.cpp does not read: a build before 2026-10-29 refuses the file, and one since loads type 43 as plain FP8 and would read the table wrong. `mnm fit` lists them as not offered and `--quant` names the ones that load, until unsloth re-uploads them.
- **Other converters.** GGUFs of V4.1 written by ik_llama.cpp or antirez's DwarfStar use another layout: refused, since mnm runs official llama.cpp images only. A vision tower named like a model (`*-Vision.gguf`) is never taken for the model.
- **Below 4 bits.** When no 4-bit quant fits, Balanced and Speed step down to the largest smaller quant that does (Nemotron-3.5-Bolt-74B on a 12 GiB card: UD-Q2_K_XL) and say so under the table; Quality never goes below Balanced's pick.
- **A newer architecture than the engine.** DeepSeek-V4.1 (`deepseek41`) needs llama.cpp 74bca39 of 2026-10-25. The plan is made either way and says whether the installed image predates it; a start on an older image fails with "This llama.cpp image is older than DeepSeek-V4.1 …" and the remedy, `mnm engines update llamacpp`, once an official image carries it.
- **What runs.** V4.1 runs as text only; its DSpark draft heads and Bolt's draft head are listed by `mnm show` as unused and are not downloaded.

### 25.5 Many cards

With `MNM_GPU` naming several cards ([18.7](#187-several-cards) says how to choose them and what each engine does with them), a llama.cpp model is split across them:

- **A model that fits the cards whole** gets each card's share of the layers in proportion to its free VRAM (`-ts`). Each card holds its own CUDA context and buffers, which the plan counts. Each token passes every card in turn, so two equal cards hold twice the model, not twice the speed.
- **A mixture of experts that does not fit whole is *placed* per card**: each card holds a run of whole layers and then, in the room left after them, the experts of as many of its own layers as fit (first layers first); the rest of the experts stay in RAM. A card with room left takes another card's layers' experts only where the speed model says that beats RAM (a token then crosses to that card and back). `mnm fit`, `add` and `show` print it per card:

  ```text
  balanced (UD-Q4_K_XL), estimated per card; a box reading replaces it:
    GPU0: blocks 0-16 + experts of 0-9 (8.0 GiB)
    GPU1: blocks 17-39 + experts of 17-29 (8.9 GiB)
    experts of 10-16, 30-39 in RAM (7.8 GiB)
  ```

  The engine gets `--device` (the main card first), `-ts` (each card's layers), `-mg 0` and one `-ot` list (each card's experts, then every other expert to the CPU; an Engram table's rule first). **On one card there is no placement:** no placement line, and the experts kept in RAM are given to llama.cpp with `--n-cpu-moe`.
- **Room per card:** *Fits now*, `--stop-others`, the router's loads, `mnm sets start` and a set's fit-together check judge each card with what each instance holds on it; stopping a model frees only its cards. The doctor's VRAM line, the start-time fixes and the web UI's status bar (one meter per card) read every chosen card.
- **Readings:** `mnm try --measure` on a model across cards samples every card and keeps each card's peak; a load reading needs one card to watch, so a split model's VRAM is read by `--measure` only (its speed always).

### 25.6 The main card, slots and risers

- **The main card** is the card on the widest PCIe link (then the most free VRAM, then the lowest number): it holds the first layers, the intermediate results (`-mg`) and, while a model keeps experts in RAM, the copies of them a long prompt makes over its link. `mnm doctor` and the plan's Cards line show each card's link (`x16 Gen5`) and name the main card.
- **The slot rule.** Put the largest card in the CPU's x16 slot (on most boards the full-length slot nearest the CPU). On AM5 the CPU's lanes split x8/x8 for two cards, and a third or fourth card sits on the chipset at x4, sharing one uplink: fine for layers, slow for prompt processing. The doctor warns when the largest card sits on a narrower link than a smaller one, naming the slot to swap it into.
- **Risers.** A card on an **x1 riser** holds layers only: mnm places it last and puts no experts on it, and the doctor WARNs. A riser with four lanes (or an M.2-to-PCIe x4 adapter) works like a chipset slot.
- Under WSL `nvidia-smi` cannot read the links (`[N/A]`): the doctor says "not readable under WSL" and the card with the most free VRAM is the main one.
- The doctor on a native box with several cards also has *Cards in use* (a WARN suggesting `MNM_GPU=all` while it is unset), a CPU line (cores, NUMA nodes, CCDs, the thread counts `mnm tune` tries), *RAM slots* (the DIMMs and channels where Linux shows them without sudo) and, with `--deep`, *Experts in RAM* ([4](#4-check-the-machine)).

### 25.7 The settings

These exist only while `MNM_GPU` names several cards (on one card `mnm set` refuses them, saying they become settings once `MNM_GPU` names several); they apply at the next start, and `mnm tune` tries the experts one per card ([15.7](#157-tuning-mnm-tune)).

| Setting | Does |
|---|---|
| `experts=auto\|ram\|cards` | Where a placed MoE's experts go: the plan's choice (`auto`, or empty), every expert in RAM (the cards hold the layers only), or each card filled with experts even where a crossing costs more than RAM saves |
| `card-experts=9,7` | How many layers' experts each card holds, one number per card in `mnm show`'s order; placed again from the model's files |
| `gpus=0` / `gpus=0,1` / `gpus=` | Pin an instance to some cards (a model that fits one card leaves the other free); empty is every chosen card. Pinned to other cards than it was placed on, its experts all go to RAM until `mnm replan` |
| `split=tensor` | Every layer split across the cards at once (llama.cpp's experimental `-sm tensor`): needs flash attention, `kv=f16` and every layer on the cards; refused for DeepSeek's architectures (`deepseek2`: GLM-4.7-Flash; `deepseek41`). Faster only on fast slots: read it with `mnm try --speed` |
| `moe-cache=<MiB>` | One number on any box: a cache of the hottest RAM-held experts on the card; llama.cpp splits it across cards like the layers |

### 25.8 What the ladder of boxes runs

The plan's own Balanced pick on each box (from `mnm fit` for that box, over the models' recorded headers). Nothing of V4.1 or V4 has run anywhere, and the coder36 figure on the reference box is the only measured one (GLM-4.7-Flash read 11.4 tok/s there: every other speed is an estimate):

| Box | DeepSeek-V4.1-Flash (763B) | DeepSeek-V4-Flash-0731 (304B) | Nemotron-3.5-Bolt-74B | GLM-4.7-Flash | Qwen3.6-35B-A3B (coder36) |
|---|---|---|---|---|---|
| **12 GB card**, 24 GB RAM (the reference box) | refused: needs about 127 GB | refused: needs about 87 GB | UD-Q2_K_XL, experts in RAM, ~13 tok/s | Q4_K_M, experts in RAM, ~23 tok/s | UD-Q4_K_M, experts of 28–30 of 40 layers in RAM: **26.8–28.3 tok/s measured** |
| **24 GB card** (a 4090-class), 64 GB RAM | refused: needs about 127 GB | UD-IQ1_M, layers in RAM, ~1 tok/s | UD-Q4_K_XL, experts in RAM, ~15 tok/s | Q4_K_M whole on the card, ~166 tok/s | UD-Q4_K_XL whole (tight), ~142 tok/s |
| **2 × RTX 5080** (x8, x8), 128 GB RAM | UD-IQ1_M, the cards hold the blocks, every expert in RAM, ~9 tok/s | holds it (about 87 GB); speed not planned here | experts in RAM, ~18 tok/s | Q4_K_M whole, ~160 tok/s | whole on the cards |
| **DGX Spark** (GB10, 128 GB unified) | refused: needs about 131 GB with the pool's 8 GiB margin, a Spark has under 131 GB | UD-IQ1_M whole in the pool, ~4 tok/s | UD-Q4_K_XL whole, ~15 tok/s | Q4_K_M whole, ~69 tok/s | UD-Q4_K_XL whole, ~49 tok/s (a published Spark reading: MXFP4 62.6) |
| **4 × RTX 5080** (x16, x8, x4, x4), 256 GB RAM | UD-IQ1_S: each card its blocks + the experts of 2–3 of them, 73 GiB of experts in RAM, ~12 tok/s (Balanced alone picks UD-IQ3_S, ~5 tok/s: `--quant UD-IQ1_S`) | holds it | UD-Q4_K_XL whole on the cards (tight), ~51 tok/s | Q4_K_M whole, ~160 tok/s | whole on the cards |

Read the speeds as an order of magnitude: on the large models every token is bound by how fast RAM feeds the experts, which a box measures (`mnm doctor --deep`, then its readings) and mnm learns ([25.9](#259-how-mnm-prices-a-card-it-has-not-seen) says how a token's experts are priced).

**The cheap boxes** (the CPU, RAM and board bought used; a used 3090 is $1,000-1,500 more). `mnm setup` lists them when this box does not hold the top rung; each holds what its RAM and its card hold together (a 24 GiB card on each; a server keeps 8 GiB for its system), worked out from the models' recorded headers. The speeds are `mnm fit`'s estimates for an EPYC 7532 box with an RTX 3090, its RAM-held experts at the platform's 70 GB/s:

| Box | Parts | Holds up to | Estimated |
|---|---|---|---|
| ~$1,000 | EPYC 7532 + H12SSL-i + 8 x 8 GB DDR4 (64 GB) | Bolt-74B UD-Q4_K_XL | Bolt UD-Q4_K_XL ~20 tok/s |
| ~$2,000-2,500 | EPYC 7532 + H12SSL-i + 8 x 32 GB DDR4 (256 GB) | DeepSeek-V4.1-Flash UD-Q2_K_XL | V4.1 UD-Q2_K_XL, every expert in RAM, ~12 tok/s |
| ~$4,000 | the same with 8 x 64 GB (512 GB) | every rung, with room for V4.1 UD-IQ3_XXS | as above |

**What a box holds.** `mnm setup`'s first section ends with the same arithmetic as a ladder: each known large model at one quant, what it needs in RAM and on the cards together (the RAM floor at 4,096 tokens), and whether this box holds it. It is worked out from the recorded headers and shipped with mnm, so setup says it offline.

| Model | Needs (RAM + VRAM) | The reference box (24 GB + 12 GiB) | 128 GB + 2 × 16 GB | 128 GB + 4 × 16 GB | A DGX Spark (one 128 GB pool, 4.3 GB more kept) |
|---|---|---|---|---|---|
| GLM-4.7-Flash Q4_K_M | about 23 GB | holds it | holds it | holds it | holds it |
| Nemotron-3.5-Bolt-74B UD-Q2_K_XL | about 34 GB | holds it | holds it | holds it | holds it |
| Nemotron-3.5-Bolt-74B UD-IQ3_XXS | about 36 GB | holds it (1.7 GB to spare) | holds it | holds it | holds it |
| Mistral-Small-4-119B UD-IQ1_M | about 37 GB | holds it (1.1 GB to spare) | holds it | holds it | holds it |
| Nemotron-3.5-Bolt-74B UD-Q4_K_XL | about 55 GB | short | holds it | holds it | holds it |
| DeepSeek-V4-Flash-0731 UD-IQ1_S | about 87 GB | short | holds it | holds it | holds it (about 91 GB) |
| DeepSeek-V4.1-Flash UD-IQ1_S | about 127 GB (+ 111 GB on disk) | short | holds it | holds it | short: about 131 GB (192 GB of unified memory holds it) |
| DeepSeek-V4.1-Flash UD-Q2_K_XL | about 204 GB (+ 135 GB on disk) | short | short | short (192 GB of RAM holds it) | short |

Each row is the text model, as the large starters add it (`--type chat`); a repo's vision projector is more (Bolt's 1.7 GB, Mistral-Small-4's 0.9 GB). Every quant a plan lands on, a profile or `--quant`, is held to the same floor at the plan's own context, so a row this box falls short of is never planned on it; one it holds by a little runs with most of its experts in the page cache, and the plan keeps 0.5 GiB of a WSL card's top free (the driver spills past it into the PC's memory, and a model that spills crawls: Bolt UD-IQ3_XXS read 2.1 tok/s on the reference box with 11,994 of 12,282 MiB in use).

Under WSL the doctor's RAM line says how much `.wslconfig` gives and, when the PC has more, which of these more memory would hold.

### 25.9 How mnm prices a card it has not seen

*([27.2](#272-measured-here-or-estimated-from-a-row) says the same for every figure in one table.)*

Every speed and size mnm plans is made of figures: the card's memory bandwidth and compute, how much of that bandwidth a decode step reaches (0.91 on the RTX 4070, 0.50 for the codebook i-quants IQ1–IQ3), a launch cost per layer (0.086 ms), what the CUDA context holds (256 MiB), each picture family's speed factor, vLLM's own decode efficiency, the cost of a crossing between cards, and on the CPU side the rates RAM feeds experts and whole layers at. They are kept as a row per card in a **device table**: every GPU at least 1,000 people registered in Hugging Face's hardware table (89 when it was read, on 2026-10-27, laptop parts included) and others besides (the 3090 and 3090 Ti, the 2080 Ti and its 22 GB mod, the Tesla P100 and V100, the Instinct MI50 and MI100, the RX 7900 XTX, the Radeon AI PRO R9700, Strix Halo, the DGX Spark's GB10), each with its bandwidth from the vendor's page the row names.

- **The row is chosen by name**, the most specific match first, whole words only ("RTX 4070 Ti SUPER" before "RTX 4070 Ti" before "RTX 4070"; an RTX 3060 12 GB and 8 GB by the memory nvidia-smi reports). A card the table does not name gets its family's default by compute capability (8.6 Ampere, 8.9 Ada, 12.0 Blackwell, 12.1 GB10, 9.0 Hopper, 7.5 Turing …) or AMD's gfx target, else its vendor's (400 GB/s assumed).
- **Only the 4070's row is measured.** Every other row borrows the 4070's efficiency, launch term and context, so a 3090 is priced at its 936 GB/s with the 4070's 0.91. The plan says so, under the Budget line: "Priced as: RTX 3090: 936 GB/s, priced from its family row (ampere): no reading on this box yet; `mnm try --speed` and `mnm tune` correct it." The doctor's GPU line ends "priced: measured", "family" or "unknown".
- **This box corrects its own row** (from its own readings, never another box's): `mnm calibration` learns the decode efficiency from this box's readings and tunes, and the launch term once two model sizes ran ([18.3](#183-what-mnm-learns-from-your-card)); the next plan on that card says "learned on this box".
- **Experts in RAM.** A token reads only its used experts, priced at their real share (DeepSeek-V4.1's 6 of 384), plus 53 µs for each used expert held in RAM (the small matrix products the CPU runs one by one). The rate RAM feeds them at is the RAM platform's ([25.14](#2514-big-ram-epyc-threadripper-and-intel): 30 GB/s under WSL) until this box's readings say otherwise; both figures were fitted on the reference box to Qwen3-30B-A3B (28.6 tok/s) and give Qwen3.6-35B 28.1 tok/s against the 25.2–28.3 it read there. A `--vram` plan for another machine prices them at 30 GB/s and says so.
- **Whole layers in RAM** are read at half of `mnm doctor --deep`'s every-core bandwidth, else at the RAM platform's figure, else at 55 GB/s (the reference box's).
- **A big MoE on a small card.** When even the dense part of a mixture-of-experts model does not fit the card, the plan may keep every expert in RAM and the dense part of as many layers as fit on the card (llama.cpp's `--cpu-moe -ngl N`, rendered as `--n-cpu-moe <layers>` with `-ngl`), when that is faster than whole layers in RAM; the fit column reads "every expert in RAM, 13/40 layers' dense part on GPU".
- **A crossing between cards** costs 0.56 ms in the row until a box measures it (it came from a Vulkan run and is likely too high for CUDA); a placement across cards says "unmeasured" beside it.
- **Unified memory** (the GB10, [25.12](#2512-unified-memory-the-dgx-spark-and-the-mini-pcs)): one pool, MemAvailable less 8 GiB, every model planned whole, at 273 GB/s with the decode efficiencies other people's Spark readings give (0.72 dense, 0.64 for a mixture of experts) until this box's readings refit them.

### 25.10 Backends: CUDA, Vulkan, ROCm, CPU

The **backend** is which build of each engine runs on this box and how Docker reaches the card. mnm decides it from the cards and says it in the doctor's Machine line (`backend: CUDA 13`), under every plan's Budget line, in `mnm engines` and on System → Engines.

| Backend | Chosen when | What it needs | What runs |
|---|---|---|---|
| CUDA 13 | NVIDIA cards, a 580 driver or newer and every card Turing (7.5) or newer | the NVIDIA driver and container runtime (`mnm setup`) | every engine; on amd64 llama.cpp, sd.cpp and audio.cpp keep their CUDA 12 builds (the CUDA 13.4 llama.cpp image asks a newer driver than the 580 and 590 branches give); on arm64 (a DGX Spark) llama.cpp's and sd.cpp's CUDA 13 builds |
| CUDA 12 | NVIDIA cards with a driver below 580 or a card older than Turing (P100, P40, V100) | the same | llama.cpp, sd.cpp, audio.cpp, vLLM (7.5 and newer only); not whisper.cpp, whose CUDA image is CUDA 13; ComfyUI on PyTorch's CUDA 12.6 wheels |
| Vulkan | AMD, Intel or other cards (the default for them) | Docker and `/dev/dri` (the card's Mesa driver; setup adds the groups and `vulkan-tools`, [25.11](#2511-amd-cards)) | llama.cpp, sd.cpp, whisper.cpp, audio.cpp; not vLLM; ComfyUI on PyTorch's ROCm wheels for an AMD card |
| ROCm | an AMD card whose gfx target llama.cpp's ROCm image is built for, when you choose it (`MNM_BACKEND=rocm`) | Docker, `/dev/kfd` and the card's render node, the `render` and `video` groups | llama.cpp and vLLM on ROCm; sd.cpp, whisper.cpp and audio.cpp on their Vulkan images (none publishes a ROCm one); planned, unverified until a box runs it |
| CPU | no card | Docker | llama.cpp, whisper.cpp, audio.cpp; not sd.cpp (no CPU image) nor vLLM |

- **Choosing one yourself:** `MNM_BACKEND=cuda12|cuda13|rocm|vulkan|cpu` in the secrets file (empty: mnm decides). A backend the cards cannot run is refused with the reason ("MNM_BACKEND=cuda13 cannot run on this box: it has no NVIDIA card, and the CUDA images need one"), in the doctor and at `mnm add`, never a silent fallback. `cpu` runs on any box. `mnm replan --all` (or `mnm machine adopt`) then moves every instance's image.
- **An engine with no build for the box** is refused at `mnm add` and `mnm fit` with the reason, per engine: "stable-diffusion.cpp publishes no CPU image: on this box pictures and clips have no engine"; `mnm machine adopt` keeps such an instance as it was and lists it.
- **The CPU backend** plans every model in RAM: the RAM is the one pool (what is free less 4 GiB for the system), the RAM floor the only check, the profiles by quant as on a card; the recipe runs `-ngl 0`, `-t` the physical cores, `--load-mode none` (the weights copied into memory rather than mapped, so the RAM they use counts as used: [28.12](#2812-the-models-and-your-box)) and `--numa distribute` on a box of several NUMA nodes. Its speed is estimated from the cores and the RAM's bandwidth (`mnm doctor --deep`) until `mnm try --speed` and `mnm tune` measure this box. The doctor's card and Docker lines read "CPU backend: no card".
- **Vulkan's devices** are named by the Vulkan loader's order, which may differ from the PCI order (an iGPU listed first): mnm reads it with `vulkaninfo --summary` where vulkan-tools is installed, else takes the cards' own order. On a box whose only Vulkan device is Mesa's software one (lavapipe), the doctor's `--deep` device check WARNs: "Mesa's software device, slow".
- **ComfyUI** is built for the backend at `mnm comfyui install`: CUDA 13 on PyTorch's cu130 wheels, CUDA 12 on the cu126 wheels with `--disable-cuda-malloc` (Pascal and Volta), AMD on the ROCm 7.2 wheels, the CPU otherwise; the doctor says when the box moved on ("built for CUDA 13, this box runs Vulkan: `mnm comfyui install`").

### 25.11 AMD cards

> **Planned, unverified until an AMD box runs it.** No AMD card has run mnm yet: the probe, the doctor and setup follow AMD's and the kernel's documentation and other people's published readings, Vulkan has run only on Mesa's software device, and every ROCm line is from documentation. Every speed below is an estimate, never a measurement.

**Vulkan first, ROCm beside it.** On an AMD card mnm runs llama.cpp's Vulkan image by default (0.3 GB; every RDNA card, the MI50, Intel's Arc), and its ROCm image (7.7 GB) where the card's gfx target is in that image's list and you chose it. Pictures, speech and voices run their Vulkan images either way (none publishes a ROCm one); vLLM on ROCm is planned (its image is published, mnm has not run it).

| gfx target | Cards | llama.cpp | vLLM | sd.cpp, whisper.cpp, audio.cpp | ComfyUI |
|---|---|---|---|---|---|
| gfx1200, gfx1201 (RDNA 4) | RX 9060, RX 9070 (XT, GRE), Radeon AI PRO R9700 | Vulkan; ROCm when chosen | ROCm, planned | Vulkan | PyTorch's ROCm 7.2 wheels, planned |
| gfx1100, gfx1101 (RDNA 3) | RX 7900 XTX/XT/GRE, PRO W7900, RX 7800 XT | Vulkan; ROCm when chosen | ROCm, planned | Vulkan | ROCm wheels, planned |
| gfx1102 | RX 7600 | Vulkan; ROCm when chosen | no | Vulkan | ROCm wheels, planned |
| gfx1151, gfx1150 (APUs) | Ryzen AI Max+ 395 (Strix Halo), Ryzen AI 9 HX 370 (890M) | Vulkan; ROCm when chosen and measured faster | ROCm, planned | Vulkan | ROCm wheels, planned |
| gfx1030 (RDNA 2) | RX 6800/6900 XT | Vulkan; ROCm when chosen | no | Vulkan | ROCm wheels, planned |
| gfx1031–1036, gfx1103 | RX 6700/6600, 780M | **Vulkan only** | no | Vulkan | no |
| gfx906, gfx900 (GCN 5) | Instinct MI50/MI60, Radeon VII | **Vulkan only** | no | Vulkan | no |
| CDNA (gfx908 and newer) | Instinct MI100 to MI350 | outside mnm's scope (Vulkan runs them as any card) | | | |

**Choosing ROCm.** `MNM_BACKEND=rocm` in the secrets file (setup's first section asks where the card runs both) makes new plans use llama.cpp's ROCm image; `mnm replan --all` moves the saved ones. One instance alone: `mnm set <name> backend=rocm` (or `vulkan`), shown only on a box whose AMD card runs both. When both images are pulled, `mnm tune <name>` tries the other one first (one trial) and keeps the faster: the guides say Vulkan for small and MoE models, ROCm for dense models over about 20B and long prompts. **On an APU** (a Strix Halo) `MNM_BACKEND=rocm` stays on Vulkan until a tune on this box measured ROCm faster; `backend=rocm` on an instance always holds.

**What the box needs** (`mnm setup` does it, each step checked and asked first):

- Native Ubuntu or Debian; the kernel's own `amdgpu` driver with current `linux-firmware` (no ROCm on the host: the engines carry it). RDNA 4 needs kernel 6.12 or newer: on Ubuntu 24.04 setup offers the HWE kernel (`linux-generic-hwe-24.04`) and a reboot. AMD's packaged (DKMS) driver only when the kernel's will not load: setup prints its route, never runs it unread.
- The `render` and `video` groups (`sudo usermod -aG render,video $USER`), then a reboot: the user services keep the groups they started with, as with the docker group.
- `vulkan-tools` (the Vulkan device order); Docker without NVIDIA's container toolkit; `amd-smi` from ROCm's apt repository only when sysfs cannot say a card's link (asked; optional).
- In the BIOS: **Above 4G Decoding** and **Resizable BAR** on (the doctor WARNs when the CPU sees less of the card than it has).
- **An APU** plans in its GTT, the RAM the card maps: set the BIOS's UMA frame buffer to its lowest (512 MB) and give the card the RAM by kernel parameters. Setup computes them from the RAM less 4 GiB and writes them to GRUB after asking (`amdgpu.gttsize=126464 ttm.pages_limit=32374784` on 128 GB), then a reboot and `mnm replan --all`. Plans count that pool once: nothing goes "to RAM" beside it (it is the same memory).
- **Under WSL** an AMD card is refused with the reason (one card, a technical preview through librocdxg, outside the official images).

**The doctor on an AMD box.** The *NVIDIA GPU* and *Docker NVIDIA runtime* lines become *Card* (each card's name, gfx target, VRAM or GTT, the backend, "Vulkan only" where ROCm is not built for it) and *Docker GPU*; then: *AMD driver* (amdgpu loaded), *Kernel* (against RDNA 4's 6.12 and an APU's 6.16.9), *AMD firmware* (a "failed to load firmware" line in the kernel log is a FAIL; the log is often not readable without sudo, then it says the command), *AMD device files* (`/dev/kfd` for ROCm and each render node open; in the groups but not in this login names the reboot), *Card N BAR*, *APU memory* (the GTT against the RAM, with the parameters when it is below half), and on several cards *AMD cards* (each slot's link; a card below x8 is on the chipset's lanes, which throttle peer to peer; the kernel's `CONFIG_HSA_AMD_P2P` and `CONFIG_PCI_P2PDMA` for `split=tensor`). `mnm doctor --deep` runs llama.cpp's own `--list-devices` in the Vulkan image (or `rocminfo` in the ROCm one). The PCIe link is read from the bridge above the card: an AMD card's own node reads its internal switch (Gen 1), never the slot.

**How mnm prices an AMD card** ([25.9](#259-how-mnm-prices-a-card-it-has-not-seen)): the R9700 and RX 9070 family at 640 GB/s, the RX 7900 XTX 960, the XT 800, Strix Halo 215 (measured by others of its 256), the 890M about 90 (estimated), each "unmeasured on this card" until `mnm try --speed` and `mnm tune` read it; the compute behind prompt speed is what llama.cpp reached on an R9700 (about 62 TFLOPS on ROCm, 43 on Vulkan), never the 191 TFLOPS matrix figure. By that arithmetic a Qwen3.8-27B Q4 on one R9700 decodes about 29 tok/s, as the published readings say (29-33); the 190 tok/s quoted for two R9700s is a community vLLM fork with its own quant and drafter, not an official image.

**Quirks the readings name** (to check on a box): a model that hangs while loading needs `mnm set <name> load=none` (`-lm none`; Gemma 4 and gpt-oss-120b on Ubuntu 26.04), which a model with an Engram table cannot use (it keeps the table mapped and read lazily, [25.3](#253-the-engram-table-on-disk)); Vulkan cannot hold one tensor larger than its buffer limit (a 28.8 GB embedding table was killed on a Strix Halo); `-sm row` aborted under concurrent decode, mnm never plans it; the MoE cache beside a tensor split crashed on ROCm: keep `moe-cache=` off with `split=tensor`. Two cards with `split=tensor` need both on the CPU's lanes; never rebind a card between amdgpu and vfio while mnm runs it.

### 25.12 Unified memory: the DGX Spark and the mini PCs

> **Planned, unverified until a Spark or a mini PC runs it.** No DGX Spark and no Strix Halo box has run mnm. What follows comes from NVIDIA's DGX Spark pages, AMD's and the kernel's documentation and other people's published readings.

On a DGX Spark the GPU (the GB10) and the CPU share one 128 GB pool of LPDDR5x; on a Strix Halo or a GEEKOM mini PC the APU's GPU maps the RAM as its GTT ([25.11](#2511-amd-cards)). mnm plans such a box as **one pool**:

- **What is free** is the RAM's MemAvailable less a margin kept for the system: **8 GiB on a Spark**, firmer than the 4 GiB a box with a card keeps, because a Spark whose pool runs out hangs rather than stopping the program (one serving framework at 85% of the memory took a box down, 65% was safe); 4 GiB on an APU, whose GTT the kernel caps. `mnm fit` says it on the Budget line: "83.2 GiB of the 131 GB unified memory for models (what is free less 8 GiB kept for the system): every model planned whole, nothing offloaded".
- **Every model is planned whole.** "RAM" is the same memory, so nothing is offloaded: no experts in RAM, no layers in RAM, no `moe-cache`. The RAM floor ([25.2](#252-the-ram-floor-worked-through)) counts the pool once, with the pool's margin ("needs about 131.3 GB in the unified memory … this machine has 130.6 GB of unified memory").
- **Loading reads the file, never maps it** (`load=none`, `--load-mode none`): a mapped file's page cache would sit in the same pool as the weights copied from it, and the Spark's mmap loads are slow (gpt-oss-120b ready in 22 s without mmap against about 90 s with it). `mnm set <name> load=` changes it as on any box; an Engram table keeps mmap.
- **vLLM's share** is a fraction of the whole pool, never more than 70% until a box measures more safe.
- **Busy** means what the pool holds beyond the system's margin: other programs' memory is the pool's pages. mnm says how much ("30.4 GiB of the unified memory is in use by other programs (8 GiB of it kept for the system anyway)") and plans around it, as on a card.
- **Speed** reads the pool's bandwidth from the card's row: 273 GB/s on a GB10, at 0.72 of it for a dense model and 0.64 for a mixture of experts (from published Spark readings: Qwen3.6-27B Q4_K_M at 11.85 tok/s, gpt-oss-120b at 58.7), Strix Halo 215 GB/s, until `mnm try --speed` and `mnm tune` read this box. On a Spark a Qwen3.8-27B Q4 plans at about 11–12 tok/s, Qwen3.6-35B-A3B at about 49, GLM-4.7-Flash Q4_K_M at about 69.
- **The probe.** A GB10's `nvidia-smi` answers `[N/A]` for its memory; mnm reads the pool from `/proc/meminfo` instead, so *Fits now*, the router's room, `mnm vram` and `mnm try --measure` read the pool: the part mnm plans in is MemTotal less the margin, in use is MemTotal less MemAvailable.
- **The doctor** says the pool instead of VRAM: "Unified memory: 131 GB unified, 98 GB available: models are planned against it whole, 8 GiB kept for the system; 3.0 GiB held by CUDA processes". On a Spark the GPU line reads "NVIDIA GB10 (compute 12.1), unified memory, driver 580.95.05"; a 590 driver is a WARN on one forum post's word (590.48.01 is reported to have broken the memory accounting, 580.126.09 to fix it; unverified). `mnm doctor --deep` measures the pool's bandwidth and never says an expert rate (nothing is kept in RAM beside the pool). The web UI's meter reads *Pool*.
- **Two Sparks** linked by their ConnectX-7 ports run one model only with vLLM over Ray or llama.cpp's RPC across both; mnm runs on one Spark at a time (the doctor's *Several Sparks* line).

What a Spark holds of the large models is the last column of [25.8](#258-what-the-ladder-of-boxes-runs)'s tables: everything up to DeepSeek-V4-Flash at 1 bit, and not DeepSeek-V4.1-Flash, whose smallest quant needs about 131 GB with the margin.

### 25.13 ARM64

A DGX Spark runs Arm cores (`aarch64`), and an engine image must exist for arm64 or mnm refuses that engine with the reason. Every tag was read from the registries with `docker buildx imagetools inspect` on 2026-10-28 (nothing pulled):

| Engine or app | On a Spark (arm64, CUDA 13) |
|---|---|
| llama.cpp | `server-cuda13`: CUDA 13.4 with native code for the GB10 (sm_121); the CUDA 12 image would run it by PTX at a fraction of the speed. It needs the 580 driver branch or newer, which DGX OS ships |
| stable-diffusion.cpp | `master-cuda-spark` (CUDA 13.0, arm64 only) |
| whisper.cpp | **refused**: "whisper.cpp has no arm64 CUDA 13 image yet (the registry, read 2026-10-28: it builds amd64; its arm64 builds are Vulkan and CPU): on this box speech-to-text has no engine." |
| audio.cpp | `full-cuda12` (arm64; CUDA 12.9 runs on the 580 driver; its `full-cuda13` asks CUDA 13.3, which a 590 driver lacks) |
| vLLM | `vllm/vllm-openai:latest` (arm64, CUDA 13.0; its kernels stop at compute 12.0 and run on the GB10 by CUDA's minor-version compatibility, unmeasured) |
| ComfyUI | built on the box as on amd64; PyTorch publishes aarch64 CUDA 13 wheels |
| OpenCode | its `linux-arm64` release, pinned by sha256 like the x64 one |
| LibreChat (with MongoDB, Meilisearch, pgvector and its RAG API), OpenHands, Open WebUI, the doctor's curl probe | multi-platform images that carry arm64: the same pins |

`install.sh` needs nothing different: uv's installer, GitHub's and Docker's apt repositories and NVIDIA's container toolkit all pick the machine's architecture (`dpkg --print-architecture`). llama.cpp's arm64 CPU image has answered a chat under emulation (QEMU, so its speed means nothing); nothing has run on arm64 hardware.

### 25.14 Big RAM: EPYC, Threadripper and Intel

A model larger than the card keeps its routed experts in RAM, and every token reads the active ones from there: the RAM's bandwidth sets the decode speed, and the cheapest box with a lot of fast RAM is a used 8-channel DDR4 server board. mnm prices that RAM by the platform it reads as, like a card by its row. Every figure below is **estimated** from the vendors' documentation and published readings until a box of that kind measures it: no EPYC, Threadripper or Intel server has run mnm yet.

**The platform row.** `mnm doctor` names it on the *RAM platform* line, from the CPU's model name, its L3 groups (each Zen CCD reads through one link: a Zen 1 or 2 part lists two L3 groups a CCD), the sockets and, where `dmidecode` reads them, the DIMMs:

| Platform | Peak a socket | RAM-held experts until measured | Share of the probe | A card gets |
|---|---|---|---|---|
| WSL (the reference box) | — | 30 GB/s (with 53 µs a used expert: fitted on the reference box) | 0.5 | `[N/A]` |
| Desktop, 2 channels of DDR4 | 51 GB/s | 22 GB/s | 0.45 | x16, or x8 + x8 |
| Desktop, 2 channels of DDR5, one-CCD Zen (a Ryzen 5 7600 or 9600) | about 64 GB/s | 28 GB/s | 0.45 | x16, or x8 + x8 |
| Desktop, 2 channels of DDR5 (Intel, two-CCD Zen) | 90-128 GB/s | 45 GB/s | 0.5 | x16, or x8 + x8 |
| Workstation, 4 channels of DDR4 (TR 3000, Xeon E5) | 102 GB/s | 35 GB/s | 0.4 | x16 or x8 each |
| Workstation, 4 channels of DDR5 (TR 7000/9000) | 166 GB/s | 60 GB/s | 0.35 | x16 each |
| Server, 8 channels of DDR4 (EPYC 7002/7003, TR PRO 3000/5000) | 205 GB/s | **70 GB/s with 8 CCDs, 50 with 4, 35 with 2** | 0.4 | x16 Gen4 each |
| Server, 8-12 channels of DDR5 (Genoa, Turin, Sapphire Rapids) | 307-461 GB/s | 120 GB/s | 0.35 | x16 Gen5 each |
| Two sockets | one socket's | one socket's figure: mainline llama.cpp decodes slower across two | — | — |

A plan prices RAM-held experts at the rate learned from this box's own readings first (`mnm calibration`), then at `mnm doctor --deep`'s every-core probe times the row's share, then at the row's default; empty channels and RAM slower than the CPU's rated speed lower the default. Whole blocks kept in RAM (and the CPU backend) read at the probe, else the row's peak, times 0.5. `mnm fit` and `mnm add` print the line ("RAM-held experts priced at 70 GB/s for an EPYC 7532 with 8 CCDs until this box measures"). A server board keeps 8 GiB for its system in the RAM floor, a desktop 4 (both estimates).

**`numa=`, per instance.** An EPYC in NPS4, or two sockets, is several NUMA nodes, and RAM reads fastest from the node a thread runs on (a 4-node Rome went from 1.96 to 4.69 tok/s with `--numa distribute`):

```bash
mnm set ds41 numa=auto        # node:N when its RAM part fits one node's free memory, else distribute
mnm set ds41 numa=distribute  # llama.cpp's own --numa distribute: threads over every node
mnm set ds41 numa=isolate     # --numa isolate: threads on the node llama.cpp starts on
mnm set ds41 numa=node:1      # the container on node 1's CPUs and memory only (Docker's --cpuset-cpus/--cpuset-mems), -t its cores
mnm set ds41 numa=off         # nothing, and a re-plan leaves it so
mnm set ds41 numa=            # cleared
```

The official image has no `numactl` and mnm never runs it: a node binding is Docker's cpuset. A binding the node's memory cannot hold is refused with the arithmetic (on two sockets one node is half the RAM). The CPU backend still plans `distribute` on several nodes by itself; a card's plan sets nothing, and the doctor names every instance keeping weights in RAM across several nodes without a setting. `mnm tune` tries `distribute`, `isolate` and the roomiest node, one trial each, then counts a bound node's cores in its threads stage (on a hybrid Intel CPU it also tries the performance cores alone). `distribute` reads best when the page cache is dropped first, `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`, which needs root: mnm says it and never runs it. Whether `--numa distribute` inside a container's cpuset behaves as on bare metal is unverified.

**A server board.** Nothing more is installed for inference: a native box's steps (§3). `mnm setup` prints the BIOS checklist in its first section on a board whose DMI vendor is a server maker's (Supermicro, ASRock Rack, Gigabyte's MZ/ME/MC lines, Tyan, Dell, HPE, Lenovo), with what it read:

- NPS1, or NPS4 with `numa=` on each instance;
- every channel, one identical DIMM each (6 of 8 reads 25% less; two a channel run slower);
- Above 4G Decoding on (the doctor reads it from the cards' memory windows);
- a riser that logs `link degraded` or PCIe errors forced to Gen3 instead of Auto;
- the BIOS for the CPU: Milan needs 2.0 or later on an H12SSL, Rome on an H11SSL a rev 2.0 board;
- a CPU sold unlocked: Dell and Lenovo fuse EPYC and Threadripper PRO parts to their boards (PSB).

**Quiet fans.** A Supermicro BMC reads slow fans under its low thresholds as stalled and spins them all to full speed in cycles. `sudo apt install ipmitool`, then `sudo ipmitool sensor thresh FAN1 lower 150 250 300` for each fan (on one H12SSL-i, values under 420 RPM did not stick). mnm never runs it.

## 26. The nightly

Every night, with the card idle, mnm checks the whole box and tells you only when something got worse.

```bash
mnm nightly install            # a systemd user timer: mnm nightly run every night at 03:00 (--at 04:15 moves it)
mnm nightly run --now          # one run now (the timer's run, without its idle minute)
mnm nightly show               # the last night: the verdicts with what mnm did or offers, then each step
mnm nightly ls                 # every night with one word: quiet, regression, failed, waited
mnm nightly set clean=30d      # also clean run records and saved logs older than 30 days (clean=off)
mnm nightly status             # the timer and the last night's word
mnm nightly uninstall          # the timer goes; the reports stay
```

**What a night runs**, in order, each step on its own (one that fails never stops the next): `mnm doctor --deep`, every app's `mnm doctor --app` (the apps mnm installed or wrote a file for), `mnm try --all --speed` (every instance alone, as [`try --all`](#183-what-mnm-learns-from-your-card) does), `mnm backup --auto` (the nightly runs the backup: `mnm nightly install` removes `mnm backup schedule`'s own timer and says so), the freshness checks (a newer engine build, a newer member of a model's family, an upgrade's `-old` still kept, [18.8](#188-is-there-something-newer)), `mnm disk check`, `mnm clean --auto` when you turned it on, and last the router's journal since the night before (the engines that died, [20](#20-when-something-goes-wrong)). `--skip try` (or `apps`, `backup`, `disk`, `clean`, `doctor`, `freshness`, `router`) leaves a step out; `--skip deep` runs the doctor without `--deep`. The report is `<state>/nightly/<date>.json`; a second run the same day is `<date>-2.json`.

**It waits for the card.** It starts once the router has been idle a minute, never while a tune runs, a model starts or stops, or ComfyUI runs a workflow; after 30 minutes of waiting it gives up and the night's report says why ("waited"). Between instances it looks again, and stops ("stopped at tiny: the router was in use") when something else took the router. A request that arrives while it tries an instance waits for that one request, as it waits behind any load.

**The rule.** Each chat, coding and vision instance's decode speed is compared with the median of its last five nightly readings on the same engine build and the same settings: more than 15% slower is a regression ("qwen3-8b: 61 tok/s, its last five nightly readings' median 72 (build 334ddc3): 15% slower"). A picture's s/image and a clip's s/clip compare the same way. With fewer than three earlier readings it says "baseline recorded (2 of 5)". A changed setting (`mnm set`) or an engine update starts a new baseline: the rule never calls your own change a regression. Also regressions: an instance that failed to load or answer (one you stopped, one not downloaded, one pinned or in a set in use is left alone, not a fault), a doctor line that turned FAIL since the last night, an app whose doctor passed last night and fails now, a backup or its restore test that failed. The disk policy's evictions are listed, never a regression. An engine that died and was restarted by the router is listed; one whose restart failed, and one parked after two deaths in ten minutes, are regressions. Readings you take by hand (`mnm try`) never move a nightly baseline: the rule reads only the nightly's own reports.

**What mnm does about it.** On its own, each said in one line: a tune queued in the router's tuner when a chat model got slower on the same build (once per regression), an app mnm installed restarted once when its doctor fails (and its doctor run again), a failed backup tried once more at the end. The router's own, listed as its journal tells them: a dead engine restarted, a crash loop parked ("parked after two deaths; `mnm logs X --review`"). Offered, never done: an engine or model update (the freshness step's lines, each with its command: "llama.cpp: has a newer build (built …)", offered `mnm engines update llamacpp`; "qwen3-8b: has a newer family member: unsloth/Qwen3.5-9B-GGUF", offered `mnm upgrade qwen3-8b`), a look at the log, the doctor line's own remedy. A freshness line is never a regression and never a notice by itself; the Hub is asked at most once per repo a night.

**One notice, only when there is something to read.** ntfy and the Windows toast, as `mnm notify` set them: "mnm nightly: 2 regressions" with the figures (ten lines at most, then "and 3 more: `mnm nightly show`"). A quiet night sends nothing; the first nights' "baseline recorded" is only in the report.

**On the web:** System → Nightly shows the last report (or any earlier one by its date), Run now (refused at once, with the reason, while the router is in use) and a graph of decode speed over time for every chat model, one line per engine build; each instance's page has its own graph above its readings. Hover a point for its date, build and settings.

## 27. Your box

mnm runs on one box at a time, and that box can change: a card swapped or added, more RAM, a move from WSL to native Linux, a new machine altogether. This chapter is for a box mnm has never seen. It says, in order, what mnm reads of the box, what it estimates until the box measures, what each change does, and what mnm does on each kind of box. The details per kind of box are in [25.9–25.14](#259-how-mnm-prices-a-card-it-has-not-seen); this chapter points to them.

### 27.1 What mnm reads of the box

mnm keeps a **fingerprint** of the machine. It is read in a fraction of a second, without sudo, by `mnm doctor`, `mnm ls`, the router when it starts and setup's first section, and kept in the state folder's `machine.json`. In words, it holds:

- **the cards:** every card nvidia-smi lists, and every AMD card sysfs shows. For each: its model and memory, its compute capability (NVIDIA) or gfx target (AMD), its PCIe link at most, and the identity it carries (NVIDIA's uuid, AMD's unique id), so a card is the same card in any slot;
- **the driver** version;
- **the CPU:** its model, physical cores, threads, NUMA nodes, its L3 groups, and the vector units llama.cpp uses (AVX2, AVX-512, AMX, SVE);
- **the RAM:** what Linux reports, the channels and speed where Linux lets mnm read the slots, the bandwidth `mnm doctor --deep` measured;
- **the board,** from DMI: its vendor, name and product, never a serial. Under WSL only WSL's virtual board shows;
- **the platform:** WSL or native Linux, the distro and kernel, and the architecture (x86_64 or arm64) the engine images must be built for;
- **the backend** the box runs (CUDA, Vulkan, ROCm or the CPU, [25.10](#2510-backends-cuda-vulkan-rocm-cpu)), beside the fingerprint but never part of its id.

```bash
mnm machine show        # this box in full, what changed since the box before, what was planned elsewhere
mnm machine history     # every box this state folder has seen, newest first, one line each
mnm machine forget ID   # drop a box that is not this one, what was measured on it and its readings (asked)
mnm machine adopt       # plan every instance again for this box (27.4); --dry-run shows it first
```

**Its id** is a short hash of what stays put while the box does: the cards' identities, the CPU model, the RAM in whole GiB, the board, WSL or native, the architecture. Free memory, the time, the driver version, the kernel, the backend and a link's current speed never enter it, and neither does `MNM_GPU` (every listed card counts, used or not). So a reboot or a driver update is the same box. A card swapped, added or removed, more RAM (more DIMMs, or `.wslconfig`'s `memory=` raised), another CPU, another board, or a move from WSL to native Linux is a new one. A second id, `cards_id`, names the cards alone.

**The identities stay on the box.** A card's uuid, an AMD card's unique id, a MAC address and a DMI serial identify your hardware, and mnm treats them like a token. `mnm machine show` masks them ("uuid kept private") unless you ask with `--reveal`; never paste that output, or `--json`'s, anywhere. `mnm report`, `mnm calibration export` and everything else that leaves the box strip them. `mnm backup` keeps them, so a restored box still knows its own cards: a backup archive is yours alone, never shared. Card models, the CPU model, the RAM size and WSL or native are not secret.

**Every plan names its box.** `mnm add`, `mnm replan`, `mnm upgrade`, `mnm machine adopt` and the tuner save the box's id on the recipe (`planned_on`); `mnm show` says it ("planned on: this box").

A box with no NVIDIA card reads as "no NVIDIA card". A box whose nvidia-smi fails, or is missing while a card sits on the PCI bus, is not read at all, so a broken driver never looks like a removed card.

### 27.2 Measured here, or estimated from a row

Every plan is made of figures, and each figure is either measured on this box or estimated from a row of a table until it is. mnm says which, wherever it plans.

| Figure | Until this box measures it | What measures it | Where it is said |
|---|---|---|---|
| The card's decode speed (bandwidth, efficiency, launch cost) | the card's row in the device table ([25.9](#259-how-mnm-prices-a-card-it-has-not-seen)): every card at least 1,000 people registered on Hugging Face, and others; only the RTX 4070's row is measured, the others borrow its efficiencies | `mnm try --speed`, the nightly's `try --all --speed`, `mnm tune`; `mnm calibration` learns the efficiency, and the launch cost once two model sizes ran | the doctor's GPU line ends "priced: measured", "family" or "unknown"; every plan's "Priced as:" line |
| The RAM's bandwidth | the RAM platform's row ([25.14](#2514-big-ram-epyc-threadripper-and-intel)): WSL, a desktop by its channels and DDR, a workstation, a server | `mnm doctor --deep` (one copy process per physical core, up to 16) and `mnm machine adopt` | the doctor's *RAM platform* and *RAM bandwidth* lines |
| The rate RAM-held experts decode at | `mnm doctor --deep`'s probe times the row's share, else the row's default (30 GB/s under WSL) | `mnm calibration`, from this box's MoE readings | "RAM-held experts priced at N GB/s …" under `mnm fit` and `mnm add` |
| What a model holds on the card | mnm's fit model, from the files and the settings | every start and `mnm try --measure` | `mnm ls`'s Fits now, `mnm show` |
| A picture's or a clip's seconds | each family's speed factor | three warm readings of that family | `mnm calibration` |
| The best settings of an instance | mnm's plan | `mnm tune` (queued after every add and every adopt) | "tuned +18%" in `mnm ls`, the Tuning tab |

**This box's readings only.** A figure is never borrowed from another box's readings, however close the cards. A new box starts from the rows and learns its own; each estimate says so ("priced from its family row (ampere): no reading on this box yet"). Every row other than the RTX 4070's, every AMD card, the DGX Spark, every server and workstation platform and the CPU backend's speed are estimates from vendors' pages and other people's published readings: **planned, unverified until a box of that kind runs mnm** (27.7).

### 27.3 When the box changes

The doctor's *Machine* line turns WARN and names what changed, card by card:

```text
Machine   WARN   the hardware changed since 3 plans were made: the cards (RTX 4070 gone, RTX 3090 new); `mnm machine adopt` re-fits them
```

The same sentence is `mnm ls`'s last line, a banner on the web UI's Status board, the router's first journal line after it starts, and one of `/__mnm/health`'s reasons (it never turns the status `degraded`: the models still answer). The nightly says it once a night as a `hardware` verdict, never a regression. With nothing changed the doctor says the box in one line: `this box: RTX 3060 12 GB, Core i5-12400 12 vCPUs, 15.6 GiB RAM, WSL; first seen <date>`. mnm never adopts by itself: you run it.

| What changed | What mnm sees | What `mnm machine adopt` does |
|---|---|---|
| A card swapped for another | a new box: "RTX 4070 gone, RTX 3090 new" | plans every instance for the new card, the quant free to change (a bigger card can take a bigger quant); tunes queued; the new card starts from its row |
| A card added | a new box: "RTX 3060 new"; the `MNM_GPU` line when the setting is set or either box has several cards | the same, and offers the `MNM_GPU` value that uses the cards you want (several cards: [18.7](#187-several-cards), [25.5](#255-many-cards)) |
| A card removed | a new box: "RTX 3060 gone" | the same; a placement whose card is gone keeps every expert in RAM until then (27.5); an instance that cannot run here is listed, kept, never deleted |
| A card moved to another slot | the same box (its identity did not change); `mnm machine show` says "moved from … to …" | nothing needed: placements follow their cards. `MNM_GPU`, a `gpus=` pin and a split that is not a placement still name nvidia-smi's numbers: check them |
| More RAM (DIMMs, or `.wslconfig`'s `memory=`) | a new box: "the RAM (23.5 → 31.4 GiB)" | measures the RAM again, re-plans (the RAM floor and the experts in RAM move), queues tunes |
| Another CPU or board | a new box | the same; a `numa=node:N` binding whose node lists other CPUs is dropped by the re-plan |
| WSL to native Linux (or back) | a new box: "the platform (WSL → native)" | the same (27.8 is the $0 step) |
| A driver update, a reboot, a kernel update | the same box | nothing |
| Another backend (`MNM_BACKEND`, or a driver that crosses 580) | the same box; the doctor's Machine line ends with the backend | `mnm replan --all` (or adopt) moves every instance's image to the box's tag and says so ("image: server-cuda -> server-vulkan"); an engine with no image for the backend is refused with the reason |

**`MNM_GPU` is yours.** When the card at an index `MNM_GPU` names is another card now, the doctor says so ("`MNM_GPU=1` now selects RTX 3090, was RTX 4070") and adopt prints the value that keeps the same cards (by their identities), or `all`. Neither ever writes the secrets file. That file is where every service reads it (no service unit carries it), so the value you set there, then `mnm service install`, is the one the router, the web UI and the nightly use.

### 27.4 `mnm machine adopt`

`mnm machine adopt` puts the plans right for the box as it is now. It runs one sequence, in this order, and says each step as it goes:

1. It waits for the router to be idle (no request in flight, no tune, no start or stop, no ComfyUI job; a minute at most, then "the router is busy: try again"), takes the box's start lock so nothing starts meanwhile, reads the box and says what changed: since the box mnm last read, and since the saved plans were made.
2. When cards came, went or moved, it says what `MNM_GPU` selects now and offers the value that keeps the same cards, or `all`. It never writes the secrets file: setting it is yours.
3. It measures the RAM's bandwidth (as `mnm doctor --deep` does) and keeps it under this box.
4. It plans every instance again for this box, as `mnm replan --all` would, with the quant free to change (`--keep-quants` holds each quant) and no plan withheld for being smaller. Placements across several cards are made again, and each instance's image moves to the box's backend. Each result is printed as `mnm replan` prints it. An instance that cannot run here (no card, too small a card, no engine image for this box) is listed as "not for this box: why" and kept exactly as it was, never deleted. Then it lists every file the new plans need with its size and asks once: "Download 3 files, 14.2 GiB? [y/N]" (nothing downloads without asking). `--yes` takes them all; no keeps the new plans saved with the files not fetched, each named: the next `mnm start` (or a request through the router) downloads them first.
5. It queues a tune for every llama.cpp chat, coding or vision instance whose tune is stale or missing, through the router's tuner (`--no-tune` skips it).
6. It records the adoption in `machine.json`, writes one router journal line, sends one notice ("Adopted RTX 3090 24 GB, …: 3 plans re-fitted, 2 downloads, 3 tunes queued"), and ends with a table: each instance, its plan before → after, the files to fetch, its tune.

```bash
mnm machine adopt --dry-run         # steps 1, 2 and the new plans; nothing saved, measured or downloaded
mnm machine adopt                   # the whole sequence, asking before any download
mnm machine adopt --yes --no-tune   # take the downloads, queue no tunes
mnm machine adopt --only coder36    # one instance first (`--only` repeats; names work too)
```

What adopt never does: delete an instance or a file, write the secrets file, or run by itself. The web's System → Machine has the same Adopt, with its Dry run, and the download question as a checkbox (its size is the last dry run's); `mnm report` carries the last adoptions' tables (`<state>/machine/adopt-*.txt`, the last five). Plain `mnm replan <name>` plans one instance again whenever you want.

### 27.5 Placements follow their cards

A model placed across several cards ([25.5](#255-many-cards)) names each card by its identity: when cards move to other slots, its `--device`, `-ts`, `-mg` and `-ot` follow them to their new indices. When one of its cards is gone, every expert stays in RAM (`--cpu-moe`: slow, but safe on any cards) and the start says "placed on other cards: `mnm machine adopt` places it again".

### 27.6 Where the readings go, and when a box comes back

**What mnm measured stays with the box it measured.** The RAM bandwidth, the rate RAM-held experts decode at, and the card's learned decode efficiency and launch cost are kept under the fingerprint they were measured on. A plan reads this box's figures only; a box that measured nothing yet plans with the rows (27.2) until it does.

**Readings follow the box.** Every reading (a start's VRAM, a request's speed, a picture's seconds) names the box it was taken on, and everything that learns from readings reads this box's only: an instance's measured figure, the engines' corrections, the picture-speed corrections, the expert rate, the card's efficiency, the request meter's "enough readings", the engine self-test's bar, the nightly's baseline, `mnm calibration` and the web's Calibration tab. A card swapped in therefore starts from the estimates, never from the old card's numbers, and learns its own. A reading taken beside something else on the card (a game, a browser holding VRAM) says so ("beside something else (N GiB held, M% busy)") and still counts.

**A box that comes back finds its readings.** The old box's readings stay in the files: put the old card back, or set `.wslconfig` back to what it gave before, and its figures are there again; `mnm machine adopt` re-fits the plans to it. `mnm machine show` and `history` say how many readings each box holds, `mnm show <name>` and the instance page say "N on another machine" when there are any, and `mnm calibration` says "N readings from another machine kept, not used". `mnm calibration export` carries every box's readings, each naming its box (an identity never: it is masked). `mnm backup` keeps every box's readings and `machine.json`, and `mnm restore` merges them, never taking "this box" from the archive ([18.6](#186-backup-and-restore)). The tuner's key names the box too: a tune measured on another box reads "tuned, stale: another box" and is queued again, and a tune cut on one box and resumed on another starts over. `mnm machine forget ID` drops a box you will not see again, with its readings (asked).

### 27.7 Every kind of box

What mnm does on each kind of box, where this guide covers it, and whether a box of that kind has run mnm. Until one has, everything mnm does there is planned from vendors' documentation and other people's published readings, and unverified: your box's own readings are the first measurements of it (27.2).

| Box | What mnm does there | In this guide | Run on that hardware |
|---|---|---|---|
| One NVIDIA card under WSL (an RTX 4070: the reference box) | everything; CUDA images | the whole guide | yes: everything in [24](#24-what-has-been-checked-on-real-hardware) |
| One NVIDIA card on native Debian or Ubuntu | everything; CUDA images; setup offers a packaged NVIDIA driver and asks first | [3](#3-native-debian-and-ubuntu) | not yet |
| A native Linux box with 2 to 4 NVIDIA cards (used 3090s, 5080s, a riser) | layers split by free VRAM, MoE experts placed per card, the main card on the widest link | [25.5–25.7](#255-many-cards), [3](#3-native-debian-and-ubuntu) | not yet |
| An AMD card (an R9700 pair, an RX 7900 XTX, an MI50) or a Strix Halo mini PC | Vulkan images first, ROCm when chosen; the kernel's amdgpu, no ROCm on the host | [25.11](#2511-amd-cards) | not yet (Vulkan has run only on Mesa's software device) |
| A DGX Spark (GB10, 128 GB unified, arm64) | one pool, every model planned whole, arm64 images (llama.cpp's CUDA 13 build); whisper.cpp refused | [25.12](#2512-unified-memory-the-dgx-spark-and-the-mini-pcs), [25.13](#2513-arm64) | not yet (arm64 only under emulation) |
| A server board (EPYC 7002/7003, Threadripper PRO) with lots of RAM | the RAM platform's row, `numa=`, the BIOS checklist | [25.14](#2514-big-ram-epyc-threadripper-and-intel) | not yet |
| A box with no card | the CPU backend: every model in RAM, the official CPU images; no pictures (sd.cpp has no CPU image); on a native box `mnm setup` shows one OK *Graphics card* row instead of the NVIDIA driver's | [25.10](#2510-backends-cuda-vulkan-rocm-cpu) | not yet (end to end only in a virtual machine) |
| An Intel desktop, an Intel Arc card, a Threadripper 7000 workstation | the platform's and the card's rows; Vulkan for Intel's cards | [25.10](#2510-backends-cuda-vulkan-rocm-cpu), [25.14](#2514-big-ram-epyc-threadripper-and-intel) | not yet |

`mnm doctor` (and the web's System → Doctor) checks each kind of box with lines of its own: the cards and their links, the RAM platform and its channels, NUMA, an AMD box's driver, firmware and groups, the unified pool. `mnm report` gathers them with the rest when you want to share them, every secret value, card id and MAC address removed.

### 27.8 Which box next: the $0 step and three tiers

The boxes below hold the large models ([25.8](#258-what-the-ladder-of-boxes-runs)); the prices are for used parts. On every rung the RAM's bandwidth sets how fast a model larger than the card decodes, so the cheapest gains are in the RAM, not the card.

- **$0: your WSL box on native Linux.** The same PC, booted into a native Ubuntu (a second disk, or dual boot), gives mnm all of its RAM rather than what `.wslconfig` gives WSL (32 GB rather than 24, say), and no WSL layer between the CPU and the RAM. Install it with `install.sh` ([3](#3-native-debian-and-ubuntu)); bring the WSL box's state over with `mnm backup` there and `mnm restore` here. The native install is a new box ("the platform (WSL → native)"): its plans read "planned on another box" until `mnm machine adopt`, the models download again (adopt lists them with their sizes) unless you copy the models folder over, and the WSL box's readings come along unused. RAM-held experts are expected to read 2 to 4 times faster natively, but mnm's rows price both near 30 GB/s (the WSL row was fitted on the reference box), so the native box's own readings are what tell.
- **About $1,000: an EPYC 7532 on an H12SSL-i with 8 x 8 GB of DDR4** (64 GB), beside a card you have: Nemotron Bolt at Q4 in RAM and on the card (about 20 tok/s estimated with a 3090).
- **About $2,000–2,500: the same board with 8 x 32 GB** (256 GB): DeepSeek-V4.1-Flash at 2 bits with every expert in RAM (about 12 tok/s estimated), its Engram table on an NVMe.
- **About $4,000: 8 x 64 GB** (512 GB): every rung of the ladder.

A used 3090 is $1,000–1,500 more. Every speed above is an estimate until a box of that kind measures it. Before buying, `mnm fit <link> --vram 24 --ram 256 --gpu "RTX 3090"` plans a model for a box you do not have yet (priced from the table alone, and said so), and `mnm setup`'s first section lists which of the large models this box holds and which more RAM would.

### 27.9 What has run where

Everything in [24](#24-what-has-been-checked-on-real-hardware) ran on the reference box, one RTX 4070 under WSL2, the fingerprint and the doctor's *Machine* line included. Beyond it, the processor backend has run end to end only in a virtual machine with no card, Vulkan only on Mesa's software device, and arm64 only under emulation. No native Debian or Ubuntu box, AMD card, DGX Spark, server board or box of several cards has run mnm yet: there mnm plans from the rows of 27.2, written from vendors' documentation and other people's published readings, until your box measures.

## 28. The studio

mnm makes music on local models with its own web UI as the studio: a song from tags and lyrics, four candidates scored to pick from, the keeper's beats and parts mapped, split into stems and mixed, its vocals cleaned, sung again in a voice from your library, edited section by section, mastered and exported to REAPER with what made it said. Every model is a row in the stages registry (`mnm studio stage`), so a better one drops in without a code change; 28.12 says which of them this box holds. Every timing in this chapter was taken on the processor, on four cores: no studio model or training has run on a card yet, so on a card mnm plans them from estimates until your box's own readings replace them (28.12).

### 28.1 The first song

The studio's models come from audio.cpp's GGUF repo, beside the voices of [§10](#10-speech-in-depth): paste a package's folder. A song model (ACE-Step 1.5) and the separators that split a take into stems are ordinary instances with types of their own: `song`, `stems`, `voice conversion` and `music score`, on ports 7900 to 7939.

```bash
mnm add audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF/turbo --name song   # the fast tier, 6.2 GB
mnm add audio-cpp/audio.cpp-gguf/HTDemucs-6stems-GGUF --name stems6   # six stems, 42 MB
mnm add audio-cpp/audio.cpp-gguf/BS-RoFormer-ep368-GGUF --name vocals # the cleanest vocals, 173 MB
mnm try song                       # a 10 s clip of the card's tags: song.wav here
mnm try stems6 --audio song.wav    # the clip split: stems6-drums.wav … stems6-piano.wav
mnm run song "warm synth pop, female vocal" --lyrics lyrics.txt --repeat 4
```

- **The plan says "estimate".** Nobody has published what ACE-Step or the RoFormers hold on a 12 GB card under audio.cpp, so each family's figure is mnm's estimate until this box has three readings of it (`mnm try <name> --measure`); then the plan says "measured on this box". Another card's time is the RTX 4070's scaled by its TFLOPS, an estimate too. A model runs whole on the card or not at all: nothing offloads.
- **Candidates.** `--repeat 4` on a song makes four runs under one: the Runs gallery shows them nested under the run they were made in, and `mnm outputs show <id>` lists them. The router loads the model once for all four. `--lyrics` reads a text file (`[verse]`, `[chorus]` mark the sections); the song's length is the instance's (`mnm set song duration=30`).
- **Stems.** A separator reads 44.1 kHz stereo; mnm brings any 16-bit WAV to that before it sends it (ACE-Step writes 48 kHz). HTDemucs 6 stems gives drums, bass, other, vocals, guitar and piano in one pass; BS-RoFormer gives the cleanest vocals and an instrumental.
- **Not a chat model.** A music instance answers audio.cpp's task route (`/upstream/<name>/v1/tasks/run`), which no chat client calls: the router leaves it out of `/v1/models` and Ollama's `/api/tags` (Odysseus would offer a song model in its chat picker), and lists it in the manifest with its tasks. A request that names it on another route (a chat, `/v1/audio/speech`) is refused before anything loads: "<name> is a studio model: it answers tasks only, at /upstream/<name>/v1/tasks/run (`mnm try <name>` sends one)". It loads, parks, idles and is healed like any instance, and the router waits for a task as long as its client does (a full-tier song on the processor can take more than an hour).
- **The stages.** `mnm studio stage` lists the registry's rows (a song, the stems) with their two tiers (fast to explore, full for the keeper), licences and sources; `mnm studio stage song fast=song` picks the instance that plays a tier. A pick must be of the kind its stage runs: a song stage takes a song instance, the stems a separator (one of another family keeps every stem it gives), the voice stage a voice conversion instance, the edits an ACE-Step song; anything else is refused with what the stage takes.

**Settings.** A music instance takes `mnm set <name> key=value` (or its page's Settings tab), beside the router's `idle` and `keep` ([14.2](#142-idle-unload), [14.3](#143-pinning)); each applies at the next start, when mnm writes the engine's config from the recipe. An instance shows only the ones its family takes: Seed-VC, Vevo2, HeartMuLa, Stable Audio and MuScriptor run on the engine's own options, so only `seed` (a song model) and `threads` apply to them.

| Key | On | What it does | Default | Range |
|---|---|---|---|---|
| `duration` | ACE-Step | The song's length in seconds when a request names none; a request's own wins (the Song pipeline's `--length`, `mnm try`'s 10 s) | 30 | 1 to 600 (ACE-Step makes 10 s to 10 minutes) |
| `route` | ACE-Step | The operation a request gets when it names none: `text2music` makes a song; `repaint`, `cover`, `cover-nofsq`, `extract`, `lego` and `complete` work on a clip the request sends (the edits of 28.7 name their own) | `text2music` | those seven |
| `steps` | ACE-Step | Diffusion steps; turbo builds cap them at 8, base builds take more | empty: the engine's 8 | 1 to 200 |
| `guidance` | ACE-Step | Diffusion guidance; turbo builds ignore it | empty: the engine's 1.0 | 0 to 30 |
| `mem-saver` | ACE-Step | Releases the engine's staged buffers between a song's phases: less held on the card, a little slower | `off` | `on`, `off` |
| `style` | ACE-Step | A style from the library, trained against this instance's build (base or turbo), merged as the model loads (28.11) | empty: none | a style's name |
| `style-scale` | ACE-Step | How strongly the style is merged: 1 as trained, 0.7 to 0.9 when it overwhelms the prompt, 0 none | 1.0 | 0 to 2 |
| `seed` | a song model | The same seed and words give the same song | empty: a new one each time | 0 and up |
| `overlap` | a separator (HTDemucs, BS-RoFormer, Mel-Band RoFormer) | Passes over the mix the separator blends; fewer is faster and can blur the joins | empty: the package's own (4 for BS-RoFormer) | 1 to 8 |
| `semitones` | RVC | Moves the converted voice; 12 is an octave up. A song's `--shift` other than 0 wins | 0 | -24 to 24 |
| `retrieval` | RVC | How much of the voice's own index is mixed in, when the voice has one; 0 turns it off | 0 | 0 to 1 |
| `threads` | every music instance | The cores the engine works on: on the CPU image (`full-cpu`) the whole task, where empty is one core; on a card only the parts it runs on the processor, where empty is the engine's default | the plan's: the physical cores on the CPU image, empty on a card (a re-plan onto or off the processor takes the plan's value) | 1 and up |

### 28.2 Lyrics and candidates

`mnm song new` is the whole song from a theme: the desk's `chat` writes the lyrics, the song model sings four candidates, you pick one, the separators split it, the stems are summed into a mix and an export folder is written (28.6). It stops for you twice, at the lyrics and at the pick, which follows the candidates' scores (28.3); `--auto` goes straight through and takes the highest-scored candidate, or the first when nothing scored them, and the run's notes say which and why.

```bash
mnm song new "a harbour town at night, the boats coming home" --length 30s
mnm song go <run>          # sing the lyrics as they are (or edit them on the Song page first)
mnm song pick <run> 3      # candidate 3 to its stems, the mix and the export
mnm song show <run>        # every step, the picked candidate marked
mnm song redo <run> song 2 # candidate 2 again (the old one is deleted); the run waits at the pick
mnm song redo <run> score  # score every candidate again (after `mnm studio install`)
mnm song resume <run>      # after a failure or an interrupted run, from the first item not done
```

- **The lyrics.** The chat model answers in a JSON shape mnm checks (sections with a label and their lines, a title, and the style line the song model sings to), asked once more with the reason if it does not fit. On the web's Song page (Runs → Songs) each section is a box you edit, empty to drop it, with one more for a new section; *Replace all* takes the whole text with a header like `[Chorus]` above each section. `--lyrics FILE` gives your own in that same shape and skips the stage; `--tags "…"` fixes the style line either way. `--chat <instance>` picks another writer than the desk's.
- **The candidates.** Each is a child run of the song's run with its own seed (the Runs gallery nests them under it, `mnm outputs show <id>` lists them), made by the instance that plays the song stage's tier (`mnm studio stage song fast=<instance>`; `--tier full` for the keeper's). All four go to the same loaded model, one after another. A pick after the run finished makes the new pick's stems, mix and export.
- **Your recording.** `--recording take.wav --as complete|vocal2bgm|reference` gives the song model a take of yours: `complete` continues it (ACE-Step's `complete` route), `vocal2bgm` writes an accompaniment around a vocal (`complete` with the accompaniment's track classes), `reference` follows it loosely in style (a `cover` at strength 0.3, its length the recording's). The file is kept in the run's folder and recorded by its sha256 in the run and each candidate.
- **Room.** The pipeline asks nothing of the router between stages: the song stage's first request loads the song model, and the router parks the chat model first when the card cannot hold both, by its usual rules; the separators load the same way after the pick.
- **"Run again" on a parent of candidates** (a `mnm run <song> --repeat 4`) makes as many new candidates under a new run; the button says how many. On one candidate of a `mnm song` run it is refused, naming `mnm song redo <song> song N`: outside its song a copy would lose your recording and the song's length.
- **Redo and the pick.** `mnm song redo <run> song N` of a candidate you did not pick keeps the pick, its stems and everything made from it; redoing the picked one opens the pick again. A credit leaves with what it made (the export's models are always the ones whose work it holds).
- **A failed step** stops the run with its reason, and `mnm song …` exits 1, so a script sees it.

### 28.3 Scores, the structure map, stems and the mixer

The studio's tools are an image mnm builds on this box, as it builds ComfyUI's, and runs one job at a time on the CPU: nothing is downloaded until you install it, and a song goes through without it (no scores, REAPER's default tempo; the run says so once).

```bash
mnm studio install            # about 3.7 GB, 10-20 minutes; the image before stays for rollback
mnm studio status             # the image in use, what it carries, whether this mnm builds another
mnm song mix <run>            # the mix in use and the mixes made
mnm song mix <run> vocals=+2,in=1 drums=mute bass=-3,out=4   # render a new mix
mnm song mix <run> --own      # back to the stems summed as made (undo)
```

- **What the image carries.** Every Python package pinned by version and sha256 (CPU PyTorch from its own index; Basic Pitch, Matchering and CREPE for MIDI, mastering and tuning, 28.8 and 28.9), the scorer's, the beat tracker's and the speaker model's (28.5) weights fetched once while it builds and checked by sha256. A job runs `docker run --rm --network none --read-only` as you, with one run's folder mounted read-only, and prints one line mnm reads. System → Studio shows the same and installs, rolls back and uninstalls as jobs; `bash uninstall.sh` removes the image with the rest.
- **The score.** Each candidate is heard by Meta's Audiobox Aesthetics (CC-BY-4.0, a 415 MB checkpoint), which rates four axes from about 1 to 10: enjoyment, usefulness, complexity and production quality. A candidate's score is the mean of enjoyment and production quality (complexity says how busy a mix is, no merit by itself); the page shows all four and marks the highest. The score is a reading on the song instance, this box's only: `mnm calibration` (and System → Calibration) sums them per instance and settings ("6.90 ± 0.40 over 12 candidates"), so a new song model's candidates compare with the old one's. It is relative between candidates of one scorer, never a figure against Suno; your ear decides. SongEval, built for whole songs, is not used: its feature model's weights are CC-BY-NC-4.0, its own licence reads two ways and it is not on PyPI. A scorer that fails on a candidate leaves that one unscored and the song goes on.
- **The structure map.** After the pick the take is mapped once: beats and downbeats by CPJKU's Beat This! (MIT), the tempo from the beats, the bar from the downbeats, and the parts where the music changes (a novelty curve over the take's harmony, timbre and loudness, cut only on downbeats, at least four bars apart), each lettered so a repeated part shares its letter (`A B A`). It is kept as `structure.json` beside the take and drawn under the mix's waveform. The REAPER project takes its tempo, meter and a marker at each part (28.6). It is measured, so check it by ear; `mnm song redo <run> structure` maps it again. (`allin1`, which would label verse and chorus, is not used: it needs `madmom` built from GitHub and NATTEN's old kernels, neither of which builds from PyPI.)
- **Waveforms.** Every candidate, stem and mix on the Song page has its waveform, measured by ffmpeg in the house format's image as each file is made (a run without them gets them from *Draw waveforms*); a click on one plays its file from there.
- **The mixer.** On the Song page (or `mnm song mix`), each stem has a level in dB (-60 to +12), mute, solo, and a fade in and out in seconds. *Render* sums the stems as set into a new mix, a child run of the song (the Runs gallery shows it under the song), and the export uses it from then on; the song's own mix (the stems summed as made) never changes, and *Use* on any mix, or `--own`, chooses which one the export takes. When a stem is soloed only soloed stems sound. A new pick, new stems or a new song mix drop a render of the old stems from use, and Render waits until the song's stems and mix are made. The page says when the export folder was made from another mix than the one in use.

### 28.4 Songs from ComfyUI workflows

A song stage can be played by a ComfyUI workflow instead of an audio.cpp instance: ComfyUI's own ACE-Step 1.5 template, YuE2's, or one from a node pack ([18.4](#184-comfyui)). The stage stays a row: you pick the workflow for a tier, and the Song pipeline sings every candidate through it.

```bash
mnm comfyui workflow add https://github.com/Comfy-Org/workflow_templates/blob/main/templates/audio_ace_step_1_5_split.json --name ace15
mnm comfyui models ace15 --fetch          # its four files, about 9.4 GB
mnm comfyui workflow fields ace15         # Style, Lyrics, Length (s), BPM, Key, Time signature, Language
mnm comfyui run ace15 "94.tags=warm synth pop, female vocal" "94.lyrics=[Verse] …" 94.duration=30 98.seconds=30
mnm studio stage song fast=workflow:ace15 # the Song pipeline's candidates from now on
mnm studio stage song fast=               # back to the registry's audio.cpp instance
```

- **The templates.** ComfyUI's audio templates (ACE-Step 1.5 split and all-in-one, YuE2) are in Comfy-Org's `workflow_templates` repository on GitHub; comfy.org's own download link answers "not found" for them. A template's model files come from Comfy-Org's repositories (`Comfy-Org/ACE-Step-1.5`, `Comfy-Org/YuE2`), which `mnm comfyui models <workflow> --find` knows.
- **The form.** A workflow's music settings are shown by name above the sampling: the style or tags, the lyrics, the length, BPM, key, time signature, language, a repaint range and a stem where a workflow has them. A setting several nodes share (ACE-Step's length is the latent's and the text encoder's) is one setting in the Song pipeline, set on every node that holds it.
- **What the pipeline sets.** For each candidate, the lyrics' style line goes to the workflow's style field, the lyrics (with their `[Verse]` headers) to its lyrics field and the song's length to every length field; the seed is the form's own, random each run, and everything else is the workflow's as imported. A workflow with no style or no lyrics field is refused when the song starts, with what it lacks. A recording (`--recording`) is refused too: a workflow takes no take of yours.
- **The candidates.** Each is the workflow's run, a child of the song's run, so the Runs gallery nests it as it nests an instance's. Its audio is turned into the house format's `song.wav` beside what ComfyUI saved, and the scores, the stems and the export read it as any candidate. The stems stay on audio.cpp's separators. The export credits the workflow and each model file it loads, and the disclosure names them; their licences are their repositories' (YuE2's weights are CC-BY-NC-4.0).
- **Room.** Each candidate's run makes room as every ComfyUI run does: it parks the fewest of mnm's loaded models whose memory it needs (the chat model that wrote the lyrics among them), never a pinned one or a set in use.
- **On the CPU build** a 10 s ACE-Step 1.5 song takes two to three minutes on four cores, and ComfyUI's process reaches about 15.6 GiB of RAM while it sings: a box without a card needs about 20 GB of RAM for it (a machine with 15.7 GiB lost half its runs to the kernel's out-of-memory killer). A candidate lost that way fails with what killed it, and `mnm song resume` makes it again. On a card it has not been timed yet.

### 28.5 Voices

A song's vocals can be sung again in a voice from the library: yours, a generated one, or a singer's you have the rights to. The song model sings the take as usual; after the stems, the voice stage converts the vocals stem into the library voice, and the mix and the export use the converted vocals. With no voice chosen the stage is skipped and the song model's own voice stays (the default).

```bash
mnm add audio-cpp/audio.cpp-gguf/SeedVC-MLX-GGUF --name svc       # the singing converter, 3 GB
mnm voices add alto --reference me.wav --source own                   # 1 to 30 s of your voice alone
mnm voices add singer --reference singer.wav --source hired --licence "studio session"
mnm voices consent singer signed.pdf                                  # its sha256 and today's date
mnm voices ls                                                         # each voice, its source, its consent
mnm song new "a harbour town at night" --voice alto --shift -2        # the vocals in alto, two semitones down
mnm song voice <run> singer                                           # sing a finished song again in that voice
mnm song voice <run> none                                             # the song model's own voice back
```

- **The converters.** Seed-VC (GPL-3.0) and Vevo2 (CC-BY-NC-ND-4.0: not for commercial use) sing from a reference recording, 1 to 30 s of the voice alone, a WAV file (`ffmpeg -i voice.flac voice.wav` for another format); RVC (MIT) sings from a checkpoint you trained or have the rights to (`--engine rvc --model voice.pth [--index voice.index]`). Each is an audio.cpp instance of type *voice conversion*; the stage takes one of the voice's engine that sings: your pick (`mnm studio stage voice fast=<instance>`), else the registry's Seed-VC for the tier (Q8_0 fast, F16 full), else any; a voice whose converter is not added is refused when the song starts, naming the `mnm add`. An RVC instance's own retrieval blend (`mnm set <rvc> retrieval=0.5`) applies when the voice has an index, and its own shift when the song gives none. audio.cpp's packaged RVC voices (`default`, `manthos`, `chocola`, `fraise`) are refused: whose voices they are, and on what terms, is not published. SoulX-Singer voices are kept in the library and refused by the voice stage: they run through a ComfyUI workflow, which the stage does not take.
- **The library.** A voice is a row under the state folder (`studio/voices/<name>.json`): its engine, its files by sha256, whose voice it is (`--source own`, `generated`, `hired`, `licensed`, `likeness`: always said, never assumed), the terms in your words and, for someone else's, a consent document's sha256 and date. `none` is no voice's name (it means the song model's own). The files are copied into `<models>/voice-library/<name>/`, where the converter's container reads them; the document is never copied. System → Voices adds, records consent and removes, as the CLI does. A backup carries the rows, never the recordings: after a restore `mnm voices show` names the files to copy back. `bash uninstall.sh` keeps the recordings, as it keeps your outputs. Wherever a voice sings, its files are checked against the hashes its row recorded: a file replaced by hand, or a row restored over other files, is refused ("… is not what its row recorded …, so mnm cannot say whose it is"), with the way out (`mnm voices rm <name> --keep-files`, move the files out, `mnm voices add <name>` from there); a voice whose files are gone says a restore brings rows, not files. Adding a voice over files a removed row left (`--keep-files`, or the uninstall) names them and asks for a restore or a move.
- **What changes.** The separator's vocals are kept as `vocals-as-sung.wav` beside the stems; `stem-vocals.wav` becomes the converted vocals, and the mixer, the export's stems and the REAPER project use them like any stem. `voice.json` beside them records what sang: the voice as it was then (its files by sha256, whose it is), the converter, the shift and the reading, so a voice removed or replaced in the library later never changes what an old song's export says. Another voice converts the vocals as sung again; `none` puts them back; new stems or a new pick put them back first and sing the new ones again. A mixer render of other vocals is not used again (Render anew). `--shift N` (-12 to 12) moves the vocals by semitones.
- **How alike it sounds.** With the studio tools installed (28.3), the voice stage reads how alike the converted vocals sound to the voice's reference, beside the vocals as sung: a speaker embedding's cosine from SpeechBrain's ECAPA-TDNN (Apache-2.0), -1 to 1, higher meaning more alike ("0.45 alike to alto's reference (as sung 0.08)"). Compare the two numbers, not the size: a short spoken reference reads low against singing even when the conversion took (in one test the same vocals in two voices read 0.14 and 0.45 against their own references, 0.04 and 0.10 against each other's). It is a reading of the voice, never of the singing, and an RVC voice has no reference to read against. The run and the export's `provenance.json` keep it.
- **Consent.** A voice that is not yours or generated is usable without a consent record; the export's disclosure names the voice, whose it is and its consent record, and an export marked `--release` without one carries a note (28.6). mnm records; it never refuses.
- **On the CPU build** Seed-VC took 374 s for 10 s of vocals at its default steps (144 s at 10) and held 3.2 GiB; RVC took 22 s and 2.3 GiB (four cores). On a card it has not been timed yet.
- **Your own RVC voice.** mnm's trainer makes one from your recordings (28.10); an Applio-trained checkpoint you add yourself is read as audio.cpp reads it first (`mnm voices add --engine rvc`): a voice trained on another vocoder (RefineGAN) or other features (SPIN) is refused with the reason, and Applio's way of writing its pitch flag is put right in the library's copy (your file is untouched). The checkpoint is read without running it: a pickle that nests deeper than 64 levels (a checkpoint of weights nests a few) or asks to build anything but weights is refused before it loads, and a damaged zip is named by the member that fails its checksum.
- **One section sung again.** New lyrics on one section is an edit, `mnm song edit <run> resing` (28.7); the library voice then sings the new take's vocals again.

### 28.6 Exports and REAPER

Every song run ends in `export/` inside its run folder (`/data/outputs/<day>/<run>/export/`, which Windows opens as `\\wsl$\Ubuntu-24.04\data\outputs\…`); `mnm song export <run> [--mp3] [--release] [--lufs -14] [--true-peak -1] [--no-loudness]` renders it again from the run record:

| File | What it is |
|---|---|
| `mix.wav`, `mix.flac`, `mix.mp3` (with `--mp3`) | The mix in use (the mixer's render, else the stems summed), at the loudness line, or the master in use as made (28.8): 24-bit 48 kHz WAV, 24-bit FLAC, 320 kb/s MP3 |
| `stems/<name>.wav` and `.flac` | Each stem as the separator gave it, in the house format, not normalised |
| `midi/<name>.mid` | Each pitched stem's notes, when the studio tools read them (28.8) |
| `song.rpp` | A REAPER project: one track per stem with its file beside the project, at the mixer's level and mute; the structure map's tempo, meter and a marker per part; the sample rate, the loudness line, the mix and the lyrics' sections in the project notes |
| `provenance.json` | The models (instance, repo, file, quant, image), the voices and how alike the vocals sound to the voice's reference (28.5), every input by sha256 and where it came from, each candidate with its seed, score and whether it was picked, the mix's levels and the structure map |
| `disclosure.txt` | Made with mnm on which box (the cards and WSL or native, never an identifier), the models that made what the export holds (a credit leaves with what it made: a stage made again, a voice put back, a master turned off), the voice and its consent, the label below, the licences (a master's names Matchering 2.0.6, GPL-3.0-or-later) |

- **The house format.** Every file a stage hands on is a 48 kHz WAV: 24-bit for what is kept, 16-bit at the rate a model reads for what is sent to one (the separators take 44.1 kHz only). The conversions run in ffmpeg inside an engine image already on the box (audio.cpp's, which the song model runs, else whisper.cpp's), with no network and only the run's folder mounted, as the film's stitch does; nothing is pulled. Without one a song's step stops with the `docker pull` to run, and `mnm song resume` carries on.
- **The loudness line.** ffmpeg's `loudnorm` in two passes takes the mix to -14 LUFS integrated with a -1 dBTP true-peak ceiling unless you say otherwise; the stems stay as separated, so they still sum to the mix as made (`mix.wav` in the run folder). The export then measures the `mix.wav` it wrote: `provenance.json`'s `measured` and the project notes ("Loudness of mix.*: … (mix.wav measures …)") are that file's figures, and where loudnorm stopped short of the line (it does on a mix with loud peaks, rather than pass the ceiling) a note says so with the `--lufs` that asks for what it reaches, or `mnm song master` to a track. The line you give (`--lufs`, `--true-peak`, `--no-loudness`) is kept for every later export of the run, the Song page's Export box included. A run whose files were cleaned (`mnm clean outputs`) has nothing to export, and says so.
- **A sum never clips.** Wherever stems are summed (the mix as made, a swap, a key change), the sum is written as float and turned down only as far as the -1 dBTP ceiling needs; the step's line then reads "N stems summed; turned down X dB so it does not clip".
- **REAPER.** Open `song.rpp` from the export folder; the stems are found beside it. REAPER's file format has no published schema: the project is written from public `.rpp` files and read back by mnm's own parser, and it has not yet been opened in REAPER itself. The tempo, meter and part markers are the structure map's (28.3) when the take was mapped, REAPER's default 120 BPM 4/4 and no markers otherwise; the lyrics' sections are listed in the notes rather than placed on a guess. The tracks carry the mix's levels and mutes; its fades are in `mix.*` only.
- **Disclosure.** `disclosure.txt` carries one sentence for Article 50 of the EU AI Act, which asks whoever publishes AI-generated or manipulated audio to say so: "This audio was generated or manipulated with artificial intelligence (disclosed under Article 50 of the EU AI Act)." `--release` marks an export for release: where a voice in it has no consent document recorded (a library voice that is someone else's, 28.5, or a recording you sang around with `complete` or `vocal2bgm`), mnm writes a note into the disclosure, the run record and the page, and exports it anyway. mnm records; it never refuses (what you release is yours to clear).
- **Clean up.** `mnm clean outputs` empties the export with the run's other files; a song waiting at a review point keeps its candidates.

### 28.7 Edits

A finished song can be edited: an edit makes a new take from the one the song uses, with ACE-Step on audio.cpp, and the song then uses the new take. Its structure map, stems, voice, mix and export are made again from it. Nothing is deleted: each edit is a run under the song (the Runs gallery shows it there), and any earlier take can be used again.

```bash
mnm song edit <run> repaint start=12 end=19.5                      # sing that range again (snapped out to whole bars)
mnm song edit <run> repaint part=2 tags="brighter, more drums"      # a whole part of the structure map, in another style
mnm song edit <run> resing section=2 lines="New line one / New line two" part=3   # where the section is sung
mnm song edit <run> extend seconds=20                              # carry the song on past its end
mnm song edit <run> cover strength=0.8                             # near 1 a remaster, lower a new performance
mnm song edit <run> add track=strings tags="warm strings, legato"  # a new track, mixed over the take
mnm song edit <run> swap track=drums tags="brushed jazz drums"     # that stem played again, mixed over the others
mnm song take <run>                                                # the takes: the candidate's and every edit's
mnm song take <run> candidate                                      # back to the picked candidate's take (Undo)
```

- **What each edit takes.** `mnm song edit <run> <edit>` with no `key=value` says what it takes; a key the edit does not take is refused, never ignored (`cover snap=off` is refused: a cover has no range). Each edit sends a new seed and keeps it with the edit.
- **The range.** `start=` and `end=` are seconds into the take, snapped out to whole bars by the structure map's downbeats unless `snap=off` (without the studio tools there is no map, and the range is used as given); `part=N` is a part of the map, from 1, as the Song page's Structure panel lists them with their times. On the Song page a click on the take's waveform plays from there and fills the range of the Repaint and Re-sing forms: the first click the start, the next the end.
- **Re-sing a section** sends the take's lyrics with one section's lines new (`section=N`, from 1, as the lyrics list them; `lines=` one a line, or ` / ` between them on a command line) and repaints the range or part you give, which is where the section is sung: mnm does not place lyrics on the music, so a re-sing without `start=`/`end=` or `part=` is refused. The edit keeps the lyrics it sang, and the edits after it sing those: a second re-sing keeps the first one's lines, and the export's project notes carry them. The lyrics shown for the song stay as written.
- **Extend** is a repaint from the take's end to its new end: audio.cpp answers the longer song (a 10 s take repainted to 14 s came back 14 s long on the CPU image). The style line may change for the new part (`tags=`). A take is 600 s at most (ACE-Step's ten minutes): an Extend past it is refused.
- **Cover or remaster** sings the whole take again held to it by `strength` (0 to 1, 0.5 when not said): near 1 the same song cleaned up, lower a new performance of it, in the style line given.
- **Add an instrument** is ACE-Step's `lego`: it plays one track class (vocals, backing_vocals, drums, bass, guitar, keyboard, percussion, strings, synth, fx, brass, woodwinds) against the take and answers **that track alone**, which mnm mixes over the take for the new take; the track is kept beside it in the edit's folder (`track-strings.wav`). **Swap** takes one stem out (vocals, drums, bass, guitar or piano: the separator's stems stand in for ACE-Step's own `extract`), sums the others, has `lego` play that part again against them, and mixes it over them. A sum that would pass -1 dBTP is turned down to it, and the edit's line says by how much (a clipped take would be split into clipped stems). A swap needs the stems of the take in use (split from it since the take last changed: `mnm song resume <run>` makes them when they are not, `mnm song redo <run> stems` when they are another take's), and where a library voice sang the vocals it sums the vocals as the song model sang them, so the new take never carries the voice. The edit records the stems it summed by hash.
- **The take in use.** `mnm song take <run>` lists the takes and marks the one in use; `mnm song take <run> <edit>` uses another, `candidate` the picked candidate's; `mnm song show <run>` names an edit in use. An edit's take is usable while it is a finished edit of the song, with its files, made from the candidate picked now (each edit records which candidate its takes go back to; removing an older edit with `mnm outputs rm` leaves the later ones usable, and the export notes the gap). A new pick goes back to the candidates, and the old pick's edits stay under the song but cannot be used; the page says why for each take that cannot (another pick's take, its files cleaned); a take in use that cannot be used is refused everywhere with the way back (`mnm song take <run> candidate`, or the candidate's own run id). A library voice (28.5) sings the new take's vocals again; a mixer render of the old take's stems is never used again (Render anew).
- **Which ACE-Step edits.** The stages registry's `edit` row: turbo for the fast tier, base for the full one, as the song's tier says; `mnm studio stage edit fast=<instance>` picks another ACE-Step instance. Without a pick or the row's package, the song's own ACE-Step instance plays it, else any.
- **The export** names the edits that made the take in `provenance.json` (`edits`: each one's run, what it did, on what range or track, and the take it was made from) and credits their models in the disclosure.
- **On the CPU build** a repaint of 3 s inside a 10 s take took 546 s cold and an Extend of a 10 s take to 14 s took 83 s warm (four cores); on a card edits have not been timed yet.
- **The other edits.** The section tools, tempo and key, MIDI and mastering are 28.8; vocal clean-up (de-bleed, dereverb, denoise) and tuning are 28.9.

### 28.8 Section tools, tempo and key, MIDI and mastering

Some edits sing nothing again: they cut, move or retune the take with ffmpeg in the house format (28.6), so they need neither ACE-Step nor the studio tools, take seconds, and land on the beat. Each is an edit like 28.7's (a run under the song, the song uses its take, Undo is `mnm song take`), with no model credited.

```bash
mnm song edit <run> cut part=2                       # take a part out; either side is joined on the beat
mnm song edit <run> crop start=8 end=40              # keep only that range (an intro or outro gone)
mnm song edit <run> duplicate part=3                 # play a part twice (after=N puts the copy after part N)
mnm song edit <run> move part=4 after=0              # a part moved to the start (or at=<seconds>)
mnm song edit <run> tempo bpm=128                    # faster or slower, the key kept (or percent=95)
mnm song edit <run> key semitones=-2                 # down a whole tone, the tempo kept (or to="F minor")
mnm song master <run> --reference ref.flac           # the mix in use mastered to a track you like
mnm song master <run> --off                          # no master: the export's loudness line again
```

- **The section tools** take a part of the structure map or `start=`/`end=` in seconds, snapped to the nearest bar line (never out to whole bars as a repaint is: a cut keeps the music on the beat; `snap=off` uses the seconds as given). Where a duplicated or moved part goes is `after=N` (after part N, 0 for the start) or `at=` seconds, snapped the same way. Every joint is an equal-power crossfade of `fade=` seconds (0.03 when not said). Nothing is sung, so the lyrics stay the take's: a cut verse's words are still in them.
- **Tempo** plays the whole take faster or slower with its key kept, by Rubber Band (ffmpeg's `rubberband` filter, which Ubuntu's ffmpeg in audio.cpp's images carries): `bpm=` is read against the structure map's tempo (without a map give `percent=`), half to double the tempo. **Key** moves the whole take by `semitones=` (-12 to +12) or `to=` a key (`A minor`, `F# major`, `Bb`), read against the key the structure map heard; a key change keeps the mode (from A minor `to="C minor"`, never `to="C major"`, which is the same notes). When the take's own stems are made, each stem but the drums is shifted (the vocals keeping their formants) and they are summed; otherwise the whole mix moves, drums too, and the edit's line says so. Past 1.5x either way, or more than four semitones, Rubber Band smears the attacks: the edit is made, and its line offers a Cover (28.7) with the tempo or key in its tags, which sings the take anew there.
- **The key.** The structure map (28.3) also names the take's key and how clearly it stood out from the next best one (Temperley's key profiles over the chroma of its harmonic part). Keys are hard to hear for a machine: the relative major or minor and a fifth away are the usual confusions, so check it by ear before a `to=`.
- **MIDI.** After the voice stage, each stem but the drums is read by Basic Pitch (Spotify's, Apache-2.0, its ONNX model in the studio tools image) into `midi-<stem>.mid` in the run folder, listed on the Song page's Stems panel; the export copies them into `midi/` and its REAPER project gets a muted MIDI track for each, under the audio. It runs whenever the studio tools are installed; without them the step is skipped and says how (`mnm studio install`, then `mnm song redo <run> midi`). Basic Pitch hears notes, not drums. To use MuScriptor (multi-instrument, CC-BY-NC-4.0: non-commercial) instead, add its package (`mnm add audio-cpp/audio.cpp-gguf/MuScriptor-Small-GGUF`) and pick it: `mnm studio stage midi fast=<instance>` (and `full=` for the full tier); the export's disclosure then carries its licence. `mnm studio stage midi fast=` goes back to Basic Pitch.
- **Master to a reference.** `mnm song master <run> --reference FILE` (any audio ffmpeg reads; the Song page's Master panel takes a FLAC or MP3 up to 32 MiB) has Matchering (GPL-3.0, in the studio tools image) match the mix in use to the reference's loudness, frequency balance, peak and stereo width, then limit it. The master is a run under the song (`master.wav`, the reference kept beside it and recorded by its hash; nothing leaves the box), and the export's `mix.*` are then the master as made: the reference set its loudness, so the loudness line's LUFS does not apply, but its true-peak ceiling does (a master past -1 dBTP, which Matchering's sample-peak limiter leaves possible, is turned down to it, and the line says by how much), and `provenance.json` names the master and the reference. A master stands for the mix it was made from: after a new mix render, new stems, a new pick, take or voice, it is no longer used and the page says to master again. `--use <master run>` takes an earlier one, `--off` none.
- **Where these run.** The section tools, tempo and key run in the house format's ffmpeg (on the CPU, seconds a song). MIDI and the master run in the studio tools image on the CPU: about a second a minute of audio for Basic Pitch and for Matchering at 48 kHz on four cores.

### 28.9 Clean and tune the vocals

Between the stems and the voice (28.5), the song's vocals stage cleans the vocals stem the separator gave, and can tune it. Nothing is done unless asked; a voice from the library then sings the cleaned vocals, and the mix and the export follow.

```bash
mnm add anvuew/dereverb_room                         # once: the dereverb (118 MB; mnm converts it for audio.cpp)
mnm song vocals <run> dereverb,denoise               # take the room off, lower the noise under them
mnm song vocals <run> debleed,deess,gate             # the separator again, softer sibilants, quiet silences
mnm song vocals <run> tune --key "A minor" --strength 60   # toward the key's notes (the studio tools)
mnm song vocals <run> none                           # the separator's vocals back
mnm song new "a harbour town at night" --clean dereverb,denoise --voice alto
```

- **What each does, in this order** (whatever order you write them): `debleed` runs the stems stage's own separator again on the vocals, for what other instruments leaked in; `dereverb` takes a room's reverb off with anvuew's room dereverb (a BS-RoFormer, GPL-3.0, mono: the voice comes back on both channels); `denoise` lowers the noise under the voice with ffmpeg's `afftdn` at the stem's own noise floor (measured, not guessed: a fixed floor took a third of a dB off where the measured one took twelve); `deess` softens sibilants; `gate` quietens the silences between phrases, opening 10 dB above that floor; `tune` comes last (below). The Song page's Vocals panel has a box for each.
- **Always from the separator's vocals.** The first clean-up keeps them as `vocals-raw.wav` beside the stems, and every later one starts from it, so cleaning twice never cleans a cleaned stem; `none` puts them back. New stems, a new pick or a new take start over from the new separation. `vocals.json` beside them says what was done, and the export's disclosure names the dereverb with its licence.
- **The dereverb is a checkpoint mnm converts.** Nobody publishes it as a GGUF for audio.cpp: `mnm add anvuew/dereverb_room` downloads its PyTorch checkpoint and YAML and converts them once into `converted/` beside them (mnm reads the checkpoint without PyTorch and runs none of its code; it was checked against the author's own published example, to 51 dB). `mnm rm dereverb-room --files` removes the converted folder with the download. It is mono, 1.4 GiB while it works, and about 9 s of work a second of audio on four CPU cores (on a card it has not been timed yet). anvuew's Mel-Band dereverb, built for a produced vocal's plate or hall, is not offered: in a test it left a room's reverb as it was.
- **Tuning** moves each sung note toward the nearest note of the key: the key you give (`--key "A minor"`), else the structure map's (28.8), said when the map heard it unsurely. `--strength` (0 to 100, 80 when not said) is how far; a note moves over about 50 ms, so vibrato and slides stay, and nothing robotic is made. It runs in the studio tools image (`mnm studio install`): CREPE hears the pitch, and a pitch-synchronous overlap-add (TD-PSOLA) moves the voice's periods, which keeps its formants. The step's line says how far the notes moved (the median, in cents). It tunes to a key, not to a melody: a blue note or a chromatic run is pulled toward the scale.
- **Before a voice.** A converter (28.5) sings what it is given: cleaned, drier vocals usually convert better. The voice stage keeps what it sang from as `vocals-as-sung.wav`, which is then the cleaned vocals; `mnm song vocals <run> none` puts the separator's back and the voice sings those.

### 28.10 Train a voice

mnm's trainer makes an RVC voice from your recordings: Applio (MIT) built on this machine, run by the router while nobody uses the card, the result a voice in the library (28.5) that sings a song's vocals like any other.

```bash
mnm studio trainer install                            # once: about 9 GB for an NVIDIA card (CUDA 13), 10-30 minutes
mnm studio trainer install --cpu                      # or for the processor: 5.2 GB, and a voice takes hours to days
mnm studio train voice alto --from ~/recordings/alto --source own
mnm studio train voice singer --from ~/takes/singer --source hired --consent signed.pdf --epochs 300
mnm studio train ls                                   # each training: queued, training (epoch 12 of 200), paused, done
mnm studio train show <run>                           # its settings, warnings and outcome
mnm studio train cancel <run>
mnm song voice <run> alto                             # the trained voice sings a song (28.5)
```

- **The recordings.** A folder of the voice alone: 10 to 60 minutes, dry (no reverb), WAV or FLAC, is what RVC voices are trained on. mnm reads every file's header and says what falls short (too few minutes, lossy files, stereo, a rate under the training's) as a warning, never a refusal: a short dataset trains, and sounds less like its singer. The recordings are copied into the training's folder (under the outputs), and the voice keeps each one's sha256 as its dataset. The vocals stage's clean-up (28.9) is the way to dry a recording first.
- **When it runs.** A training is queued; the router's trainer starts it once the router has been idle a minute (`--now` skips the minute), nothing pinned or a set in use holds the card, no tune runs and the nightly holds nothing. It parks every model the router may park (the next request loads them again), then trains. **Any request pauses it**: the training's container is stopped at once and the request is served as usual; once the router has been idle a minute again it goes on from its last checkpoint (one every epoch on the CPU, every five on a card), and the journal says from where. A start or stop by hand (your `mnm start`, a set, `mnm try --all`) or a ComfyUI run pauses it the same way within a few seconds, and the nightly holds the card as it holds a tune, so a long training goes on in the gaps. A training that finishes as a request comes is finished, not paused. A router that stops leaves the training queued; the next one picks it up. On a card, the community's figure is one to four hours for 200 to 400 epochs on 10 to 30 minutes (not yet measured by mnm); on four CPU cores an epoch took about 50 s per minute of recordings.
- **The voice.** At the end the voice joins the library with whose voice it is (`--source`, always said), the consent document's sha256 when you gave one, the training's settings, epochs and the box that trained it, and the recordings' hashes. Its index (for retrieval blending) comes with it when there were enough recordings for one audio.cpp reads; the training's folder keeps only its log. Applio writes the checkpoint's pitch flag in a way audio.cpp refuses ("RVC voice checkpoint missing integer metadata: f0"); the library's copy writes it as audio.cpp reads it.
- **What it runs.** The training runs with no network, read-only, as you, over the training's own folder, with the card (`MNM_GPU`) on the CUDA build. The image is Applio 3.7.1 at a checked commit with every package pinned by sha256, and Applio's pretrained HiFi-GAN models, RMVPE and ContentVec checked by sha256 (`mnm studio trainer status` says what it carries). Only HiFi-GAN voices on ContentVec features are trained: those are what audio.cpp sings. System → Trainer installs it, trains from a folder on this machine and lists the trainings, as the CLI does. `bash uninstall.sh` removes the image and keeps the voices.
- **Other backends.** A box on ROCm, Vulkan or a CUDA 12 driver trains on the processor (`--cpu`).
- **A style.** Teaching ACE-Step a style from your songs is 28.11: another trainer, run the same way.

### 28.11 Teach ACE-Step a style

mnm's style trainer teaches ACE-Step a style from your songs: a LoRA adapter made by ACE-Step 1.5's own trainer (MIT; Side-Step built into it), run by the router while nobody uses the card as a voice's training is (28.10), and played by an ACE-Step instance that names it.

```bash
mnm studio trainer install --for styles                 # once: about 9 GB for an NVIDIA card (CUDA 13), 10-30 minutes
mnm studio train style tape --from ~/music/tape --caption "lo-fi hip hop, dusty drums, warm tape hiss" --source own
mnm studio train style dub --from ~/music/dub --caption "deep dub, spring reverb" --source licensed --variant turbo --epochs 600
mnm studio train ls                                     # styles beside voices: queued, training (epoch 12 of 400), paused, done
mnm studio styles ls                                    # the library: each style's build, trigger word and whose songs
mnm studio styles show tape
mnm set song style=tape style-scale=0.8                 # the ACE-Step instance plays it from its next start
mnm studio styles add mine --adapter ~/lora/final --variant base --trigger "mine" --source own   # a LoRA trained elsewhere
```

- **The songs.** A folder of five or more songs (10 to 20 is the community's figure), 30 seconds to 4 minutes each; WAV or FLAC keep the sound, MP3 and the like train with their artefacts. Longer songs are cut at `--max-seconds` (240 by default: a longer song costs the card more). A `<song>.txt` (or `<song>.lyrics.txt`) beside a song is its lyrics; without one it trains as instrumental. The caption says what the songs sound like; the trigger word (`--trigger`, the style's name unless given) is put in front of every caption, and a song made with the style carries it at the front of its style line, which is how the style is called up. The songs are copied into the training's folder and kept on the style by their hashes, as a voice's recordings are.
- **The build it plays on.** A style is trained against ACE-Step 1.5's base (`--variant base`, the default: upstream rates it the easier to teach) or its turbo, and plays on that build's GGUF only (their noise schedules differ): `mnm set` refuses another build, and says which `mnm add` gives the right one. The checkpoint the training reads (the DiT, its VAE and text encoder, about 6 GB) is downloaded into the models folder when the first style of that build is queued, pinned to a revision; the 5 Hz language model is not needed.
- **The card.** The settings follow the card: on one under 16 GB, upstream's "Tight" preset (rank 32, the 8-bit AdamW optimizer, the text encoder and VAE moved to the processor, gradient checkpointing); 16 GB and 24 GB cards get its larger presets. Upstream's own figure is 16 GB of card for its UI trainer; nobody has published what the tight preset holds on a 12 GB card, so **the first epoch of every training is the probe**: what the card holds as the training's container starts is read first (other programs' share), then, once the songs are preprocessed, the card is read every second until the epoch's checkpoint is saved. The training's own peak is the highest less that share. When its own peak comes within 0.5 GiB of the card's top (under WSL, where the driver spills to system memory rather than run out and a training would crawl for hours; 0.25 GiB on native Linux, where it would run out of memory later), the training stops with the figures and what to change (`--max-seconds 120` halves a song's share, or the style trainer built for the processor). When it is other programs' share that takes it there, the training pauses instead: it keeps its checkpoint and waits until the card has its own peak free again ("5.0 GiB of the card must be free first"), counting what mnm's models hold as free (the training parks them). The own peak is kept for this box and these settings, so the next style's plan says the measured peak and how many hours its epochs take, and a resumed training is not probed again. A plan that cannot fit by its own arithmetic is refused before anything downloads.
- **When it runs.** As a voice's training: the router's idle minute, any request pauses it, a start by hand or a ComfyUI run pauses it, the nightly holds it, and it goes on from its last checkpoint (one every epoch; a cut loses at most the epoch under way). mnm loads the checkpoint's adapter itself and stops in words when it does not load: upstream's own resume drops a LoRA's weights without saying so and trains on from fresh ones.
- **The style.** At the end it joins the library with its build, trigger word, rank, whose songs (`--source`: own, licensed, generated), the training's settings and the songs' hashes. An ACE-Step instance plays it with `mnm set <instance> style=<name>` (audio.cpp merges it into the model as it loads: `ace_step.lora`, at `style-scale` times its alpha over its rank; 1 as trained, 0.7 to 0.9 when it overwhelms the prompt). A song's export credits it in `disclosure.txt`'s models line (its name, how many songs it was trained on and whose they were, and its adapter's sha256): the style the running engine started with, so a `mnm set` that waits for the next start is not credited early. A style an instance names is not removed (`mnm set <instance> style=` first), and a start whose style's files are gone (a restore brings the library's rows, not its adapters) is refused in words; `mnm set <instance> style=<name>` and a song's credit check the adapter against the hash its row recorded. A trigger word starts with a letter or digit, at `mnm studio styles add` as at a training. System → Trainer has the style trainer, a Train a style form and the library. `bash uninstall.sh` removes the image and keeps the adapters (`<models>/styles`).
- **What it does not do.** It trains no LoKr adapters (upstream calls them experimental; audio.cpp loads them without DoRA or Tucker), trains nothing on a CUDA 12, ROCm or Vulkan box's card (`--cpu` trains on the processor, slowly) or on an arm64 processor (no torchcodec wheel), and reads no song's BPM or key into its caption.

### 28.12 The models and your box

```bash
mnm studio models                # every studio model: what it needs, whether this box holds it now, the smallest card that does
mnm calibration                  # "music speed": each model's seconds of work a second of audio, measured on this box
```

- **One stage at a time.** audio.cpp offloads nothing: a music model runs on the card whole or not at all, and the Song pipeline runs one stage after another, so each model needs its room alone (`mnm studio models`' "Needs"). audio.cpp, like sd.cpp and whisper.cpp, runs on one card: on a box with several, a studio model is planned and judged on the first card mnm uses ("15.5 GiB of card 0 (…) free: each studio model runs on that card alone"), never on the cards' sum. Its figure on the card is what its start held until a task's peak is measured (`mnm try <name> --measure`), which then stands for it. By these estimates every seeded tier holds on an RTX 4070 (ACE-Step 1.5 turbo and base at Q8, about 8.3 GiB; the separators 7 to 8 GiB); HeartMuLa 3B at Q8 needs about 11.9 GiB, more than a 12 GiB card offers once its driver and display hold their share (about 0.4 GiB short on the RTX 4070), so the smallest card that holds it is 16 GiB. YuE2, MiniMax Music 3 and ACE-Step 1.5 XL have no audio.cpp package yet: their rows carry published figures (YuE2's int8 ran on a 16 GB card; MiniMax peaked at about 14 GB on a 3090; XL wants 20 GB without offload) and name the smallest card that holds them: 16 GiB for YuE2 and MiniMax, 24 GiB for XL. The card sizes are the common ones (8, 12, 16, 24, 32, 48 and 80 GiB), each less half a GiB for its driver.
- **Its time, measured here.** Every music task audio.cpp answers carries its own `rtf`, the seconds of work a second of audio took. The router keeps it as a reading with the model's file and the task's route (a song, a repaint, a cover, a converter's singing), and from three readings of one group (one file, one route, one set of settings, one engine build) the median is what the plan and `mnm calibration` use instead of the estimate. A first task after a start is timed apart and never counted (on the processor it can take ten times as long), and `mnm set <name> threads=4` starts a new group, so a rate never mixes two speeds. A client's wait (the Song pipeline's, `mnm try`'s) is the estimate's, made longer by this box's rate when it is slower, never shorter. A separator's answer has no `rtf` and is not timed. `mnm fit` says which the time is for each quant ("measured on this box", or an estimate and where it came from). On the processor the estimate is the CPU image's reading (four cores read ACE-Step turbo at 4.9 to 8.3 s of work a second of audio) scaled to this processor's cores, and the box's own readings replace it after three songs.
- **A box without a card.** An instance on the processor's image (the CPU backend: no card, or `MNM_BACKEND=cpu`) sits in RAM, and its room is the RAM's: what is available less what the system keeps (4 GiB on a desktop or under WSL, 8 GiB on a server board, as the plan keeps); an instance on a card's image keeps judging its card, whatever the box's backend says, until `mnm machine adopt` moves it. A request that loads a separator while a song model runs parks the song model first when the RAM cannot hold both, as a card's load would (only what sits in the RAM is parked for it), and `mnm sets start` parks for a set the same way; *Fits now* reads the RAM, and a running instance says how much RAM it uses. A llama.cpp model on the processor counts its whole weights there, and a plan made there loads them with `--load-mode none` (copied into memory, not mapped from the file), since the RAM a mapped file sits in counts as free and a running model would read as holding none of it. On the processor a start's first task reads the model from disk and can take ten times its warm time (546 s against 49 s for 10 s of ACE-Step on four cores): a step that gives up says so, and `mnm song resume` tries again once the engine is free.
- **When the box changes.** `mnm machine adopt` re-plans a music instance as any other ([27.4](#274-mnm-machine-adopt)): its quant and image follow the new box (`full-cpu`, `full-vulkan` or `full-cuda12`, which a CUDA 13 box keeps too), and no tune is queued (the tuner is llama.cpp's). The trainers' images (28.10, 28.11) are built for CUDA 13 or the processor; a ROCm or Vulkan box trains on the processor.

