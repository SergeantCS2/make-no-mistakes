#!/usr/bin/env bash
# Remove makenomistakes from this box: what install.sh and mnm put here, and nothing else.
# Everything lives in main(), so bash has read the whole script before the checkout holding it
# is deleted, and a truncated copy runs nothing. docs/GUIDE.md §21 explains each step.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash ~/makenomistakes/uninstall.sh [--dry-run] [--yes] [--keep-models] [--keep-apps]
                                           [--docker] [--uv]

Removes makenomistakes for a fresh start: the web UI service and any mnm command still
running, the mnm containers (their ports close), the engine images (rollback builds too),
the downloaded models, the workspace apps mnm installed (OpenCode's unit, LibreChat's,
Open WebUI's and OpenHands' containers, their data), recipes, the secrets file, the `mnm` command, mnm's download cache
and this checkout. Then it trims the freed space and writes a PowerShell script that
shrinks the WSL disk, so Windows gets the space back.

Inside a folder it removes only what mnm made, and the folder once empty. Never touched:
Odysseus (its folder, containers, images, volumes, networks), git, the GitHub CLI.

  --dry-run      show what would be removed and change nothing
  --yes          do not ask for confirmation
  --keep-models  keep the downloaded models
  --keep-apps    keep the workspace apps' data (MNM_APPS_DIR: LibreChat's accounts and chats,
                 OpenHands' home); their containers, images and OpenCode's unit still go
  --docker       also remove Docker Engine and the NVIDIA container toolkit, only when Docker
                 holds nothing but mnm's containers and images (never while Odysseus's exist)
  --uv           also remove uv's binary, cache and Python builds (not `uv tool` installs)
USAGE
}

say() { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }

DRY_RUN=0 ASSUME_YES=0 WITH_DOCKER=0 WITH_UV=0 KEEP_MODELS=0 KEEP_APPS=0
DOCKER=()
LEDGER_IDS=""
FAILURES=0
# The repositories of the engine images (engine_images.py IMAGES, every backend's).
ENGINE_REPOS=("ghcr.io/ggml-org/llama.cpp" "ghcr.io/leejet/stable-diffusion.cpp"
  "ghcr.io/ggml-org/whisper.cpp" "ghcr.io/0xshug0/audio.cpp" "vllm/vllm-openai"
  "vllm/vllm-openai-rocm")
# What mnm writes inside each folder: removed entry by entry, never the folder wholesale.
# views/, loras/ and upscalers/ hold mnm's links into hf/ and maybe your own files: only the
# links go. configs/ holds the server configs mnm renders from its recipes, voices/ the plain
# downloads of the voice packages (one `<org>--<repo>/` folder per repo).
MODEL_ENTRIES=(hf configs voices civitai ollama modelscope prompt-cache quality)
LINK_FOLDERS=(views loras upscalers comfyui)
# Every path mnm keeps under its state folder is in this list.
STATE_ENTRIES=(instances workflows desk disk apps runs comfy comfyui swap engines.json .engines.lock ".engines.*.tmp" linger-enabled-by-mnm logs measurements ufw-rule-by-mnm machine.json machine backup engine-check.json nightly watchdog tools tuning studio)
CONFIG_ENTRIES=(secrets.env)
# The web UI and the router (one URL for every model), both written by
# `mnm service install`; OpenCode's, written by `mnm apps install opencode`.
UNIT_NAMES=(makenomistakes-web makenomistakes-router mnm-opencode)
# The nightly backup and the nightly self-check (its reports go with the state folder)
TIMER_UNITS=(makenomistakes-backup.timer makenomistakes-backup.service
  makenomistakes-nightly.timer makenomistakes-nightly.service)
LAUNCHER_MARKER="# Written by \`mnm setup\`"

# Every change goes through here: run it, or under --dry-run only print it.
run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '    would run: %s\n' "$*"
  else
    printf '    %s\n' "$*"
    "$@"
  fi
}

# The folders and port as mnm itself resolves them (environment, secrets file, defaults),
# asked of the checkout's own code while its environment still exists.
mnm_settings() {
  local python="$CHECKOUT/.venv/bin/python"
  [ -x "$python" ] || return 1
  "$python" -c 'from makenomistakes.config import Settings
s = Settings()
print(s.expanded_models_dir()); print(s.expanded_state_dir()); print(s.port)
print(s.router_port); print(s.expanded_outputs_dir()); print(s.expanded_apps_dir())
print(s.expanded_backups_dir())' 2>/dev/null
}

# The fallback when that code cannot run: NAME from the environment, else the secrets file
# (`export`, quotes, a trailing comment and CRLF allowed, the name in any case), else default.
setting() {
  local name=$1 default=$2 value=""
  value="${!name:-}"
  if [ -z "$value" ] && [ -f "$SECRETS_FILE" ]; then
    value="$(sed -n "s/^[[:space:]]*\(export[[:space:]]\+\)\?${name}[[:space:]]*=[[:space:]]*//Ip" \
      "$SECRETS_FILE" | tail -n 1 | sed 's/[[:space:]]\+#.*$//; s/[[:space:]]*$//')" # CR too
    value="${value%\"}" && value="${value#\"}" && value="${value%\'}" && value="${value#\'}"
  fi
  value="${value:-$default}"
  # shellcheck disable=SC2088  # the value holds a literal ~, as the secrets file has it
  case "$value" in "~" | "~/"*) value="$HOME${value#\~}" ;; esac
  printf '%s' "$value"
}

# A path this script may work in: absolute (a relative setting would mean something else here
# than it meant to mnm), existing, not a system folder, not $HOME or above it, not another
# user's home, and neither inside Odysseus's folder nor holding it.
guarded() {
  local path=$1 real home odysseus
  case "$path" in /*) ;; *) return 1 ;; esac
  [ -e "$path" ] || return 1
  real="$(realpath -m -- "$path")"
  case "$real" in
    / | /bin | /boot | /data | /dev | /etc | /etc/* | /home | /lib | /lib64 | /mnt | /mnt/? | \
      /opt | /proc | /root | /run | /sbin | /srv | /sys | /tmp | /usr | /usr/* | /var | /var/*)
      return 1
      ;;
  esac
  home="$(realpath -m -- "$HOME")"
  case "$home/" in "$real"/*) return 1 ;; esac # $HOME itself or a folder above it
  case "$real" in /home/*) case "$real/" in "$home"/*) ;; *) return 1 ;; esac ;; esac
  for odysseus in "$HOME/odysseus" "$ODYSSEUS_DIR"; do
    odysseus="$(realpath -m -- "$odysseus")"
    case "$real/" in "$odysseus"/*) return 1 ;; esac # Odysseus or inside it
    case "$odysseus/" in "$real"/*) return 1 ;; esac # a folder that holds Odysseus
  done
}

# A path this script may delete whole: guarded, and no mount point at or under it. rm would
# reach into the mounted folder, a bind mount on the same filesystem included, which
# `--one-file-system` does not stop.
safe_to_remove() {
  guarded "$1" && ! holds_a_mount "$(realpath -m -- "$1")"
}

# The mount points, one per line. Fails when the table cannot be read or has no root mount
# (an empty file, which would otherwise read as "nothing is mounted anywhere").
mount_points() {
  local point points="" root=0
  [ -r "$MOUNTINFO" ] || return 1
  while read -r _ _ _ _ point _; do
    printf -v point '%b' "$point" # mountinfo writes a space as \040
    if [ "$point" = / ]; then root=1; fi
    points+="$point"$'\n'
  done <"$MOUNTINFO"
  [ "$root" -eq 1 ] || return 1
  printf '%s' "$points"
}

is_mount_point() { # not `mount_points | grep -q`: grep stops early, pipefail sees SIGPIPE
  local points
  points="$(mount_points)" || return 1
  grep -qxF -- "$(realpath -m -- "$1")" <<<"$points"
}

holds_a_mount() { # a mount point at or under the path; a table that cannot be read says yes
  local real=$1 points point
  points="$(mount_points)" || return 0
  while IFS= read -r point; do
    case "$point/" in "$real"/*) return 0 ;; esac
  done <<<"$points"
  return 1
}

failed() { # a step that did not work: said where it happened, counted for the last line
  note "could not $*"
  FAILURES=$((FAILURES + 1))
}

with_sudo_if_needed() { # the parent decides: sudo only where it is not writable
  if [ -w "$(dirname -- "${*: -1}")" ]; then run "$@"; else run sudo "$@"; fi
}

# Delete one thing mnm made (a folder, a file or a link; a link is never followed).
remove_path() {
  local path=$1
  if [ ! -e "$path" ] && [ ! -L "$path" ]; then return 0; fi
  if [ ! -L "$path" ] && ! guarded "$path"; then
    note "kept $path: not something this script deletes"
    return 0
  fi
  if [ ! -L "$path" ] && ! safe_to_remove "$path"; then # mnm's, but rm would reach further
    failed "remove $path: it is or holds a mount point; unmount it and run this again"
    return 0
  fi
  with_sudo_if_needed rm -rf --one-file-system -- "$path" || failed "remove $path"
}

# The entries of a folder that are not mnm's (glob patterns), one indented name per line.
leftovers() {
  local dir=$1 entry name mine
  shift
  while IFS= read -r -d '' name; do
    mine=0
    for entry in "$@"; do
      # shellcheck disable=SC2254  # the entries are glob patterns
      case "$name" in $entry) mine=1 ;; esac
    done
    if [ "$mine" -eq 0 ]; then printf '      %s\n' "$name"; fi
  done < <(find "$dir/" -mindepth 1 -maxdepth 1 -printf '%f\0' 2>/dev/null)
}

# Remove mnm's entries (glob patterns) from a folder, then the folder once nothing else is in
# it; anything else stays and is listed.
clear_folder() {
  local dir=$1 entry path left before=$FAILURES entries=() also=()
  shift
  while [ "$#" -gt 0 ] && [ "$1" != -- ]; do entries+=("$1") && shift; done
  if [ "$#" -gt 0 ]; then shift && also=("$@"); fi # mnm's too, but cleared elsewhere
  if [ ! -e "$dir" ] && [ ! -L "$dir" ]; then return 0; fi
  if ! guarded "$dir"; then
    note "kept $dir: not a folder this script deletes"
    return 0
  fi
  for entry in "${entries[@]}"; do
    for path in "$dir"/$entry; do remove_path "$path"; done
  done
  if [ "$FAILURES" -gt "$before" ]; then return 0; fi # already said; the folder stays with it
  left="$(leftovers "$dir" "${entries[@]}" "${also[@]}")"
  if [ -n "$left" ]; then
    note "kept $dir: it also holds what mnm did not make:"
    printf '%s\n' "$left"
  elif [ -L "$dir" ]; then
    note "kept the link $dir (the folder it points to is empty now)"
  elif is_mount_point "$dir"; then
    note "kept $dir: a mount point (it is empty now)"
  else
    with_sudo_if_needed rmdir -- "$dir" || failed "remove the folder $dir"
  fi
}

remove_service() {
  say "Web UI, router, nightly and backup timers and OpenCode services"
  local name unit removed=0
  for name in "${UNIT_NAMES[@]}"; do
    unit="$UNIT_DIR/$name.service"
    [ -e "$unit" ] || continue
    if command -v systemctl >/dev/null 2>&1; then
      run systemctl --user disable --now "$name" || true
    fi
    run rm -f -- "$unit" "$UNIT_DIR/default.target.wants/$name.service" ||
      failed "remove the unit $unit"
    removed=1
  done
  for name in "${TIMER_UNITS[@]}"; do
    unit="$UNIT_DIR/$name"
    [ -e "$unit" ] || continue
    if command -v systemctl >/dev/null 2>&1; then
      run systemctl --user disable --now "$name" || true
    fi
    run rm -f -- "$unit" "$UNIT_DIR/timers.target.wants/$name" || failed "remove the unit $unit"
    removed=1
  done
  if [ "$removed" -eq 0 ]; then
    note "not installed"
  elif command -v systemctl >/dev/null 2>&1; then
    run systemctl --user daemon-reload || true
  fi
}

# Linger goes off only when `mnm service install` turned it on (its note), and no other user
# service needs it; also after `mnm service uninstall`, which leaves linger as it is.
restore_linger() {
  local others
  # Other user services this account enabled itself (vendor ones do not count), quadlets too.
  others="$(find "$UNIT_DIR" "$HOME/.config/containers/systemd" -mindepth 1 \
    \( -path '*.wants/*' -o -path '*.requires/*' -o -path '*.upholds/*' -o -name '*.container' \) \
    ! -name "${UNIT_NAMES[0]}.service" ! -name "${UNIT_NAMES[1]}.service" \
    ! -name "${UNIT_NAMES[2]}.service" ! -name "${TIMER_UNITS[0]}" \
    ! -name "${TIMER_UNITS[2]}" 2>/dev/null || true)"
  if [ "$LINGER_OURS" -eq 0 ]; then
    note "linger left as it was: no note says mnm turned it on (\`loginctl disable-linger\` does)"
  elif [ -n "$others" ]; then
    note "linger left on: other user services are enabled"
  elif command -v loginctl >/dev/null 2>&1; then
    run loginctl disable-linger || true
  fi
}

# The ufw rule `mnm setup` added on a native box so Docker's containers reach the router:
# deleted only by its note, and only in the exact shape setup writes ("<networks> <port>").
restore_firewall() {
  if [ -z "$UFW_RULE" ]; then
    return 0
  fi
  local networks port
  read -r networks port <<<"$UFW_RULE"
  say "The firewall rule mnm setup added"
  # Only the one rule setup writes: Docker's networks to the router's own port.
  if [ "$networks" != "172.16.0.0/12" ] || [ "$port" != "$ROUTER_PORT" ]; then
    note "the ufw note is not the rule mnm writes ($UFW_RULE); ufw left as it is"
    return 0
  fi
  if ! command -v ufw >/dev/null 2>&1; then
    note "ufw is no longer installed: no rule to delete"
    return 0
  fi
  run sudo ufw delete allow from "$networks" to any port "$port" proto tcp ||
    failed "delete the ufw rule (by hand: sudo ufw delete allow from $networks to any port $port proto tcp)"
}

# This user's processes running the checkout's `mnm` (an `mnm serve` or download started by
# hand): the entry point itself, or a python running it. `tail -f …/mnm` is not mnm.
mnm_pids() {
  local proc args first mnm="$CHECKOUT/.venv/bin/mnm"
  for proc in /proc/[0-9]*; do
    [ -O "$proc" ] && [ "${proc#/proc/}" != "$$" ] || continue
    mapfile -d '' -t args <"$proc/cmdline" 2>/dev/null || continue
    first="${args[0]:-}"
    if [ "$first" = "$mnm" ] ||
      { [ "${args[1]:-}" = "$mnm" ] && [[ ${first##*/} == python* ]]; }; then
      echo "${proc#/proc/}"
    fi
  done
}

stop_processes() {
  say "Running mnm commands"
  local pids
  pids="$(mnm_pids | xargs)"
  if [ -z "$pids" ]; then
    note "none"
    return 0
  fi
  # shellcheck disable=SC2086  # one pid per word
  run kill $pids || true
}

remove_files() {
  say "Recipes, settings and the mnm command"
  clear_folder "$STATE_DIR" "${STATE_ENTRIES[@]}"
  clear_folder "$(dirname -- "$SECRETS_FILE")" "${CONFIG_ENTRIES[@]}"
  if [ -f "$LAUNCHER" ] && sed -n 2p "$LAUNCHER" | grep -qF "$LAUNCHER_MARKER"; then
    run rm -f -- "$LAUNCHER" || failed "remove $LAUNCHER"
  elif [ -e "$LAUNCHER" ]; then
    note "kept $LAUNCHER: mnm setup did not write it"
  fi
  # huggingface_hub's chunk cache (downloads land in the models folder): only its own folder,
  # and none when HF_XET_CACHE moved it, since that may name anything.
  if [ -n "${HF_XET_CACHE:-}" ]; then
    note "kept $HF_XET_CACHE: HF_XET_CACHE names it, so it is yours to clear"
  else
    remove_path "$XET_CACHE"
  fi
}

remove_uv() {
  say "uv (--uv)"
  local uv cache
  if ! uv="$(command -v uv)"; then
    note "not installed"
    return 0
  fi
  cache="$("$uv" cache dir 2>/dev/null || true)"
  if [ "$cache" = "${XDG_CACHE_HOME:-$HOME/.cache}/uv" ]; then # never a folder a setting names
    run "$uv" cache clean || failed "clean uv's cache"
  elif [ -n "$cache" ]; then
    note "kept $cache: UV_CACHE_DIR or uv.toml names it, so it is yours to clear"
  fi
  run "$uv" python uninstall --all || failed "remove uv's Python builds" # uv knows its folder
  if [ "$(dirname -- "$uv")" = "$HOME/.local/bin" ]; then # where install.sh's installer puts it
    run rm -f -- "$uv" "$HOME/.local/bin/uvx" || failed "remove $uv"
  else
    note "kept $uv: a package manager installed it; remove it with that"
  fi
  note "kept: tools from \`uv tool install\` and ~/.config/uv (yours, not mnm's)"
}

remove_checkout() {
  say "The checkout"
  local dir real links
  if ! { [ -d "$CHECKOUT/.git" ] && [ -f "$CHECKOUT/pyproject.toml" ] &&
    grep -q '^name = "makenomistakes"' "$CHECKOUT/pyproject.toml"; }; then
    note "kept $CHECKOUT: not a makenomistakes checkout"
    return 0
  fi
  for dir in "$MODELS_DIR" "$STATE_DIR"; do # a folder kept for its other files, inside it
    real="$(realpath -m -- "$dir")"
    links=()
    if [ "$dir" = "$MODELS_DIR" ]; then mapfile -t links < <(only_mnm_links "$real"); fi
    case "$real/" in "$CHECKOUT"/*)
      if [ -e "$real" ] && { [ "$DRY_RUN" -eq 0 ] ||
        { [ "$dir" = "$MODELS_DIR" ] && [ "$KEEP_MODELS" -eq 1 ]; } || [ -n "$(leftovers "$real" \
        "${MODEL_ENTRIES[@]}" "${STATE_ENTRIES[@]}" "${links[@]}")" ]; }; then
        note "kept $CHECKOUT: it holds $dir, which keeps files mnm did not make"
        return 0
      fi
      ;;
    esac
  done
  remove_path "$CHECKOUT"
}

check_ports() {
  say "Ports"
  local open
  if ! command -v ss >/dev/null 2>&1; then
    note "cannot check (no ss)"
    return 0
  fi
  open="$(ss -Hltn 2>/dev/null | awk '{print $4}' | sed 's/.*://' |
    awk -v web="$WEB_PORT" -v router="$ROUTER_PORT" \
      '($1 >= 7800 && $1 <= 7939) || $1 == web || $1 == router' |
    sort -nu | xargs ||
    true)"
  if [ -z "$open" ]; then
    note "nothing listens on mnm's ports (7800-7939, $WEB_PORT, $ROUTER_PORT)"
  elif [ "$DRY_RUN" -eq 1 ]; then
    note "listening now: $open (they close with the service and the containers)"
  else
    note "still listening: $open (not started by mnm, or a process this script could not stop)"
  fi
}

# What git ignores in the checkout beyond its own environment and caches, at most three names:
# files kept there (notes, recordings) go with the checkout.
ignored_files() {
  local path found="" count=0
  while IFS= read -r path; do
    case "$path" in
      .venv/ | __pycache__/ | */__pycache__/ | .mypy_cache/ | .pytest_cache/ | .ruff_cache/ | .cache/) ;;
      *)
        count=$((count + 1))
        if [ "$count" -le 3 ]; then found="${found:+$found, }$path"; fi
        ;;
    esac
  done < <(git -C "$1" ls-files --others --ignored --exclude-standard --directory 2>/dev/null || true)
  if [ "$count" -gt 3 ]; then found="$found and $((count - 3)) more"; fi
  printf '%s' "$found"
}

# git's word on the checkout: changes, files it ignores, branches and stashes that would go with it.
checkout_notes() {
  local changes ignored unpushed
  if changes="$(git -C "$CHECKOUT" status --porcelain 2>/dev/null)" && [ -n "$changes" ]; then
    note "WARNING: the checkout has changes that are not committed; they go with it"
  fi
  ignored="$(ignored_files "$CHECKOUT")"
  if [ -n "$ignored" ]; then
    note "WARNING: the checkout holds files git ignores ($ignored); copy what you want to keep"
    note "somewhere else first, or they go with the checkout"
  fi
  unpushed="$(git -C "$CHECKOUT" for-each-ref --format='%(refname:short)|%(upstream)|%(upstream:track)' \
    refs/heads 2>/dev/null | awk -F'|' '$2 == "" || $3 ~ /ahead/ {print $1}' | xargs || true)"
  if [ -n "$unpushed" ]; then
    note "WARNING: branches with commits not pushed: $unpushed; they go with the checkout"
  fi
  if [ -n "$(git -C "$CHECKOUT" stash list 2>/dev/null || true)" ]; then
    note "WARNING: the checkout has stashed changes; they go with it"
  fi
}

summary() {
  say "This removes"
  local size
  note "the web UI and router services (ports $WEB_PORT, $ROUTER_PORT),"
  note "running mnm commands,"
  note "the mnm containers, the workspace apps' containers (LibreChat, Open WebUI, OpenHands) and OpenCode's unit"
  note "the engine images mnm runs on any backend and their older builds (other tags of"
  note "${ENGINE_REPOS[*]} stay),"
  note "and the image mnm's GPU check runs (nvidia/cuda:12.8.1-base-ubuntu24.04)"
  if [ -d "$MODELS_DIR/prompt-cache" ]; then
    note "the saved prompts in $MODELS_DIR/prompt-cache (conversations; even with --keep-models)"
  fi
  if [ -d "$MODELS_DIR/quality" ]; then
    note "a quality tune's work in $MODELS_DIR/quality (even with --keep-models)"
  fi
  if [ "$KEEP_MODELS" -eq 1 ]; then
    note "models: kept ($MODELS_DIR)"
  elif guarded "$MODELS_DIR"; then # sized only once it is a folder this may clear
    size="$(cd "$MODELS_DIR" && du -shc -- "${MODEL_ENTRIES[@]}" 2>/dev/null | tail -n 1 |
      cut -f1 || true)"
    note "models: ${MODEL_ENTRIES[*]} in $MODELS_DIR (${size:-0}); links in ${LINK_FOLDERS[*]}"
  else
    note "models: none ($MODELS_DIR is not a folder this script clears)"
  fi
  if [ "$KEEP_APPS" -eq 1 ]; then
    note "the apps' data: kept ($APPS_DIR)"
  else
    note "the apps' data: ${APP_FOLDERS[*]} in $APPS_DIR (accounts, chats, OpenHands' home)"
  fi
  note "recipes, run records and engine records in $STATE_DIR; the secrets file $SECRETS_FILE"
  note "the mnm command $LAUNCHER"
  if [ -z "${HF_XET_CACHE:-}" ]; then note "the download chunk cache $XET_CACHE"; fi
  note "this checkout: $CHECKOUT"
  # git's checks only on a checkout that is its own repository: a copy unpacked from the zip,
  # or one inside another repository (a home folder kept in git), would report that one
  if [ -e "$CHECKOUT/.git" ]; then
    checkout_notes
  fi
  if [ "$WITH_UV" -eq 1 ]; then note "uv's binary, cache and Python builds (--uv)"; fi
  if [ "$WITH_DOCKER" -eq 1 ]; then
    note "Docker Engine and the NVIDIA toolkit, unless Odysseus is installed or Docker holds"
    note "anything else (--docker)"
  fi
  if [ -n "$UFW_RULE" ]; then note "the ufw rule mnm setup added ($UFW_RULE)"; fi
  note "kept: what your runs made, in $OUTPUTS_DIR (\`mnm clean outputs\` first to delete it)"
  note "kept: your backups, in $BACKUPS_DIR (no token and no model file, but your cards' ids: keep them to yourself)"
  note "kept: your voice library's recordings and checkpoints, in $MODELS_DIR/voice-library"
  note "kept: your style library's adapters, in $MODELS_DIR/styles"
  library_rows "$STATE_DIR/studio/voices" voice
  library_rows "$STATE_DIR/studio/styles" style
  note "kept: your projects folder (MNM_PROJECTS_DIR) and OpenCode's own sessions"
  note "never: $ODYSSEUS_DIR or Odysseus's containers, images and volumes; git; gh; the NVIDIA"
  note "driver (on native Linux mnm setup may have installed it: the system needs it)"
}

incomplete() {
  echo "The checkout is incomplete ($CHECKOUT/scripts/uninstall/$1.sh); run" \
    "\`git -C $CHECKOUT checkout -- scripts\` and try again." >&2
  exit 1
}

main() {
  local arg answer="" settings
  for arg in "$@"; do
    case "$arg" in
      --dry-run) DRY_RUN=1 ;;
      --yes) ASSUME_YES=1 ;;
      --docker) WITH_DOCKER=1 ;;
      --uv) WITH_UV=1 ;;
      --keep-models) KEEP_MODELS=1 ;;
      --keep-apps) KEEP_APPS=1 ;;
      -h | --help) usage && exit 0 ;;
      *) printf 'unknown option: %s\n\n' "$arg" >&2 && usage >&2 && exit 2 ;;
    esac
  done
  if [ "$(id -u)" -eq 0 ]; then
    echo "Run this as your normal user, not with sudo; it asks for sudo where it needs it." >&2
    exit 1
  fi
  case "${HOME:-}" in
    /?*) ;;
    *) echo "HOME is not an absolute path (${HOME:-unset}); nothing removed." >&2 && exit 1 ;;
  esac
  if [ ! -d "$HOME" ]; then echo "HOME ($HOME) is not a folder; nothing removed." >&2 && exit 1; fi
  CHECKOUT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  # Read in full now: the checkout is deleted further down.
  # shellcheck source=scripts/uninstall/models.sh
  source "$CHECKOUT/scripts/uninstall/models.sh" || incomplete models
  # shellcheck source=scripts/uninstall/docker.sh
  source "$CHECKOUT/scripts/uninstall/docker.sh" || incomplete docker
  # shellcheck source=scripts/uninstall/windows.sh
  source "$CHECKOUT/scripts/uninstall/windows.sh" || incomplete windows
  # shellcheck source=scripts/uninstall/apps.sh
  source "$CHECKOUT/scripts/uninstall/apps.sh" || incomplete apps
  SECRETS_FILE="$HOME/.config/makenomistakes/secrets.env"
  if settings="$(mnm_settings)" && [ "$(printf '%s\n' "$settings" | wc -l)" -eq 7 ]; then
    { read -r MODELS_DIR && read -r STATE_DIR && read -r WEB_PORT && read -r ROUTER_PORT &&
      read -r OUTPUTS_DIR && read -r APPS_DIR && read -r BACKUPS_DIR; } <<<"$settings"
  else
    MODELS_DIR="$(setting MNM_MODELS_DIR /data/models)"
    STATE_DIR="$(setting MNM_STATE_DIR "$HOME/.local/state/makenomistakes")"
    WEB_PORT="$(setting MNM_PORT 7800)"
    ROUTER_PORT="$(setting MNM_ROUTER_PORT 7801)"
    OUTPUTS_DIR="$(setting MNM_OUTPUTS_DIR /data/outputs)"
    APPS_DIR="$(setting MNM_APPS_DIR /data/apps)"
    BACKUPS_DIR="$(setting MNM_BACKUPS_DIR /data/backups)"
  fi
  ODYSSEUS_DIR="${ODYSSEUS_DIR:-$HOME/odysseus}"
  MOUNTINFO="${MNM_TEST_MOUNTINFO:-/proc/self/mountinfo}" # another mount table can stand in
  UNIT_DIR="$HOME/.config/systemd/user"
  LAUNCHER="$HOME/.local/bin/mnm"
  XET_CACHE="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/xet" # huggingface_hub's
  SHRINK_PS1="$(cat "$CHECKOUT/scripts/uninstall/shrink-wsl-disk.ps1" 2>/dev/null || true)"
  LEDGER_IDS="$(sed -n 's/.*"previous_id": *"\([^"]*\)".*/\1/p' "$STATE_DIR/engines.json" \
    2>/dev/null | xargs || true)" # rollback builds, read before the state folder goes
  RECIPE_IMAGES="$(sed -n 's/^  "image": *"\([^"]*\)".*/\1/p' "$STATE_DIR"/instances/*.json \
    2>/dev/null | xargs || true)" # the engine tags the recipes run
  CONFIG_FILES=() # the voice server configs mnm rendered: <models>/configs/<instance>.json
  local recipe
  for recipe in "$STATE_DIR"/instances/*.json; do
    # only a voice instance renders one; an image instance's name may be the user's file
    if [ -f "$recipe" ] && grep -q '"engine": *"audiocpp"' "$recipe"; then
      CONFIG_FILES+=("$(basename -- "$recipe")")
    fi
  done
  LINGER_OURS=0
  if [ -e "$STATE_DIR/linger-enabled-by-mnm" ]; then LINGER_OURS=1; fi
  UFW_RULE="$(head -n 1 -- "$STATE_DIR/ufw-rule-by-mnm" 2>/dev/null || true)"

  summary
  if [ "$DRY_RUN" -eq 0 ] && [ "$ASSUME_YES" -eq 0 ]; then
    if { : </dev/tty; } 2>/dev/null; then exec </dev/tty; fi
    printf '\nType "remove" to delete all of the above: '
    read -r answer || true
    if [ "$answer" != "remove" ]; then
      echo "Nothing removed."
      exit 1
    fi
  fi

  find_docker
  remove_service
  restore_linger
  restore_firewall
  stop_processes
  remove_containers
  remove_volumes
  remove_images
  remove_models
  remove_app_data
  remove_files
  if [ "$WITH_UV" -eq 1 ]; then remove_uv; fi
  if [ "$WITH_DOCKER" -eq 1 ]; then remove_docker; fi
  remove_checkout
  check_ports
  reclaim_space
  if [ "$DRY_RUN" -eq 1 ]; then
    say "Dry run: nothing was changed."
  elif [ "$FAILURES" -gt 0 ]; then
    say "Done, but $FAILURES step(s) did not work (\"could not\" above); the rest is removed."
    exit 1
  else
    say "makenomistakes is removed. Run \`cd ~\` if this shell was inside the checkout."
  fi
  exit 0
}

main "$@"
