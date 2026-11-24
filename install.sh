#!/usr/bin/env bash
# Install makenomistakes (mnm) on WSL2 Ubuntu, or on a native Debian or Ubuntu box:
#
#   curl -fsSL https://raw.githubusercontent.com/SergeantCS2/make-no-mistakes/main/install.sh | bash
#
# Or clone the repository any way you like and run `bash ~/makenomistakes/install.sh`.
#
# Checks that sudo works, installs git, curl and uv if missing, clones the repository into
# ~/makenomistakes (or updates it), creates the virtualenv, then runs `mnm setup`: at a
# terminal the guided setup (where models live, the router's network and key, Docker, the
# NVIDIA runtime and on native Linux the driver, tokens, the services, a first model), else
# the checks alone: without a terminal nothing is installed past uv, the checkout and, for a
# repository that asks for a login, the GitHub CLI. Each step says what it does before it runs.
# Re-running is safe.
#
# A public repository needs no account to clone. A copy on github.com that asks for a login (a
# private fork) gets one through the GitHub CLI, installed from GitHub's own apt repository when
# it is missing or older than 2.20; git then uses that login for every later update. Fetch the
# script itself through the CLI too, and say where the copy lives:
#
#   gh api -H 'Accept: application/vnd.github.raw' repos/<owner>/<repo>/contents/install.sh |
#     MNM_REPO_URL=https://github.com/<owner>/<repo> bash
#
# Unattended (no terminal, nobody to answer: a script), say so:
#
#   curl -fsSL https://raw.githubusercontent.com/SergeantCS2/make-no-mistakes/main/install.sh |
#     MNM_SETUP_YES=1 bash
#
# MNM_SETUP_YES=1 passes `--yes` to `mnm setup`: without a terminal its checks table then
# installs what is missing (Docker, the NVIDIA runtime, the folders, the secrets file, the
# engine image) and the services step runs; at a terminal the guided setup runs on its
# defaults without asking. sudo must work without a password then (`sudo -n true`).
#
# Everything lives in main() so a truncated download runs nothing, and stdin is re-pointed at
# the terminal so sudo/apt/gh prompts work when the script arrives through a pipe. Because of
# that redirection, bash would otherwise keep reading commands from the keyboard once the
# piped script ends, so main() finishes with an explicit exit.
set -euo pipefail

REPO_URL="${MNM_REPO_URL:-https://github.com/SergeantCS2/make-no-mistakes}"
BRANCH="${MNM_BRANCH:-main}"
TARGET="${MNM_DIR:-$HOME/makenomistakes}"

say() { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }

# apt without a terminal: debconf's own frontend would try to re-open /dev/tty and print
# "dpkg-preconfigure: unable to re-open stdin".
apt_get() { sudo env DEBIAN_FRONTEND=noninteractive apt-get "$@" </dev/null; }

# A repository that asks for a login: git needs one for the clone and for every later update.
# `gh auth status` says whether a usable one is stored (a token missing gh's default scopes
# also counts as "none" and triggers the browser login); if not, the browser login runs and the
# script stops if it does not succeed. `gh auth setup-git` makes git use that login, so no
# token is ever typed or stored by this script.
ensure_github_login() {
  if gh auth status --hostname github.com >/dev/null 2>&1; then
    say "GitHub login found"
  else
    say "Logging in to GitHub"
    if ! gh auth login --hostname github.com --git-protocol https --web; then
      echo "GitHub login did not complete; run 'gh auth login' and start this script again." >&2
      exit 1
    fi
  fi
  gh auth setup-git --hostname github.com
}

GH_MIN_MINOR=20  # gh 2.20+: `auth setup-git --hostname`, `api -H` (Ubuntu 22.04 ships 2.4)

ensure_sudo() {
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is not installed. As root (\`su -\`), run:" >&2
    echo "  apt-get install -y sudo && usermod -aG sudo $(id -un)" >&2
    echo "then log out, log in again and start this script again." >&2
    exit 1
  fi
  if sudo -n true 2>/dev/null; then
    return 0
  fi
  if ! [ -t 0 ]; then
    echo "sudo needs your password, and there is no terminal to type it in. Run this script" >&2
    echo "from a terminal (or as: bash ~/makenomistakes/install.sh)." >&2
    exit 1
  fi
  if ! sudo -v; then
    echo "Your user cannot use sudo. As root (\`su -\`), run: usermod -aG sudo $(id -un)" >&2
    echo "then log out, log in again and start this script again." >&2
    exit 1
  fi
}

gh_is_recent() {
  command -v gh >/dev/null 2>&1 || return 1
  local version major minor
  version="$(gh --version 2>/dev/null | awk 'NR == 1 {print $3}')"
  major="${version%%.*}"
  minor="${version#*.}"
  minor="${minor%%.*}"
  [ "$major" -gt 2 ] 2>/dev/null && return 0
  [ "$major" = "2" ] && [ "$minor" -ge "$GH_MIN_MINOR" ] 2>/dev/null
}

# GitHub's own repository (cli.github.com/packages), as its install docs give it.
install_gh_from_github() {
  local keyring=/etc/apt/keyrings/githubcli-archive-keyring.gpg
  sudo mkdir -p -m 755 /etc/apt/keyrings
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee "$keyring" >/dev/null
  sudo chmod go+r "$keyring"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=$keyring] https://cli.github.com/packages stable main" |
    sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
  apt_get update
  apt_get install -y gh
}

# git that never asks for a password: a repository that wants a login fails at once, and what
# git said is kept in $git_said.
quiet_git() {
  local said
  if said="$(GIT_TERMINAL_PROMPT=0 git "$@" 2>&1)"; then return 0; fi
  git_said=$said
  return 1
}

# Clones the repository, or brings the checkout up to date.
fetch_repo() {
  if [ -e "$TARGET/.git" ]; then  # -e: a worktree's .git is a file
    quiet_git -C "$TARGET" fetch --quiet origin "$BRANCH" &&
      quiet_git -C "$TARGET" checkout --quiet "$BRANCH" &&
      quiet_git -C "$TARGET" pull --quiet --ff-only origin "$BRANCH"
  else
    quiet_git clone --quiet --branch "$BRANCH" "$REPO_URL" "$TARGET"
  fi
}

# What GitHub answers a request that needs a login: over HTTPS it asks for a user name (or a
# password, when the name is known), which git may not ask for; with a login that cannot see
# the repository it says "not found".
asks_for_login() {
  case "$1" in
    *"could not read Username"* | *"could not read Password"* | *"Authentication failed"* | \
      *"Repository not found"* | *"repository '"*"' not found"*) return 0 ;;
  esac
  return 1
}

# The address git fetches from: the checkout's own remote, else the one it clones.
fetch_url() {
  if [ -e "$TARGET/.git" ]; then
    git -C "$TARGET" remote get-url origin 2>/dev/null || printf '%s\n' "$REPO_URL"
  else
    printf '%s\n' "$REPO_URL"
  fi
}

# Without an account first; a repository on github.com that asks for a login gets the GitHub
# CLI's, then git tries once more. Anything else git says stops the script with its words (a
# GitHub login cannot help an SSH address or another host).
get_repo() {
  if [ -e "$TARGET/.git" ]; then say "Updating $TARGET"; else say "Cloning into $TARGET"; fi
  fetch_repo && return 0
  local url
  url="$(fetch_url)"
  if ! asks_for_login "$git_said" || [ "${url#https://github.com/}" = "$url" ]; then
    printf '%s\n' "$git_said" >&2
    echo "git stopped (its words are above); start this script again once that is fixed." >&2
    exit 1
  fi
  say "The repository asks for a GitHub login"
  if ! gh_is_recent; then
    say "Installing the GitHub CLI (from GitHub's apt repository)"
    install_gh_from_github
  fi
  ensure_github_login
  if ! fetch_repo; then
    printf '%s\n' "$git_said" >&2
    echo "git stopped again with that GitHub login (its words are above). Log in with an" >&2
    echo "account that can see $url ('gh auth login', or 'gh auth switch'), then start this" >&2
    echo "script again." >&2
    exit 1
  fi
}

main() {
  if { : </dev/tty; } 2>/dev/null; then exec </dev/tty; else exec </dev/null; fi

  if [ "$(id -u)" -eq 0 ]; then
    echo "Run this as your normal user (not root, not under sudo); it asks for sudo where needed." >&2
    echo "Under root the clone, uv and any GitHub login would all land in /root." >&2
    exit 1
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "mnm installs on Debian and Ubuntu (apt); this system has no apt-get." >&2
    exit 1
  fi
  native=0
  if ! grep -qi microsoft /proc/version 2>/dev/null; then
    native=1
    echo "Native Linux: setup installs the card's driver too when it is missing (the NVIDIA driver asked first; an AMD card's is the kernel's own, with its firmware)."
  fi

  ensure_sudo
  missing=""
  for tool in git curl; do command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"; done
  [ -e /etc/ssl/certs/ca-certificates.crt ] || missing="$missing ca-certificates"
  if [ -n "$missing" ]; then
    say "Installing${missing}"
    apt_get update
    # shellcheck disable=SC2086  # one word per package
    apt_get install -y $missing
  fi
  uv_was_installed=0
  if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv_was_installed=1
  fi

  get_repo

  say "Installing Python dependencies"
  (cd "$TARGET" && uv sync --quiet)

  # Without a terminal `mnm setup` prints the checks and installs nothing unless told
  # `--yes`. mode: guided (a terminal), unattended (no terminal, MNM_SETUP_YES=1), checks (neither).
  setup_args=()
  if [ "${MNM_SETUP_YES:-}" = 1 ]; then setup_args+=(--yes); fi
  if [ -t 0 ]; then
    mode=guided
  else
    setup_args+=(--plain)  # no guided sections without a terminal: the checks table
    if [ "${MNM_SETUP_YES:-}" = 1 ]; then mode=unattended; else mode=checks; fi
  fi
  if [ "$mode" = checks ]; then
    say "Checking prerequisites (mnm setup; no terminal, so nothing is installed)"
  else
    say "Checking and installing prerequisites (mnm setup)"
  fi
  setup_done=1
  if ! (cd "$TARGET" && uv run mnm setup "${setup_args[@]}"); then
    setup_done=0
    [ "$mode" = checks ] ||
      say "mnm setup needs you (see its note above): finish it, then run 'mnm setup' again"
  fi

  case "$mode" in
    checks) checks_only_note ;;
    unattended)
      # The plain setup never asks about the services, so this step installs them.
      services=later
      say "The web UI and the router as background services"
      if (cd "$TARGET" && uv run mnm service install); then services=running; fi
      closing_note "$services" "$native" "$setup_done"
      ;;
    guided)
      # The guided setup asked about the services in its section 6 (and put them off when a
      # reboot or re-login comes first): its answer stands. Its summary just listed what is
      # left; repeating its advice here would only contradict it.
      if systemctl --user is-active --quiet makenomistakes-router 2>/dev/null; then
        web_note="web UI: http://localhost:7800 (running; 'mnm update' restarts it)"
      else
        web_note="web UI: not started (setup's section 6 says when; then 'mnm service install')"
      fi
      echo
      if [ "$setup_done" -eq 1 ]; then
        echo "Done: the summary above lists what is left. $web_note"
      else
        echo "Not finished: deal with setup's note above, then run 'mnm setup' again. $web_note"
      fi
      echo "Open a new terminal so 'mnm' works from any directory; 'mnm update' brings in later changes."
      ;;
  esac
  if [ "$uv_was_installed" -eq 1 ]; then
    echo "uv was just installed: run 'source \$HOME/.local/bin/env' (or open a new shell) so 'uv' is found."
  fi
  exit 0  # stdin is the terminal now; without this, a piped bash would read commands from it
}

# Without a terminal and without MNM_SETUP_YES=1: setup printed the checks and stopped.
checks_only_note() {
  cat <<DONE

Checks only: there was no terminal to ask on, so 'mnm setup' installed nothing (its table
above says what is missing). To install, run this script again at a terminal, or unattended:
  MNM_SETUP_YES=1 bash $TARGET/install.sh
DONE
}

# The unattended setup's closing note (no terminal, so no guided summary came before it).
# One order whatever happened: the re-login, the services, the doctor, then models.
closing_note() {
  local services=$1 native=$2 setup_done=$3 relogin
  if [ "$native" -eq 1 ]; then
    relogin="reboot (a new driver, kernel or group takes effect at boot)"
  else
    relogin="run 'wsl --shutdown' from PowerShell and open Ubuntu again"
  fi
  echo
  if [ "$setup_done" -eq 1 ]; then
    echo "Done. 'mnm setup' put an 'mnm' command in ~/.local/bin (open a new terminal if it is not found)."
  else
    echo "Not finished: deal with setup's note above, then run 'mnm setup --yes' again."
  fi
  if [ "$services" = running ]; then
    echo "The web UI (http://localhost:7800) and the router (7801) run as services."
  else
    echo "The web UI and the router are not services yet (the step above said why)."
  fi
  cat <<DONE
Next, in this order:
  1. If Docker was just installed or you were just added to the docker group: $relogin.
     Your login and the services see the docker group only after that (newgrp is not enough).
  2. mnm service install     # the web UI and the router as services, with the new group
  3. mnm doctor --deep       # everything green?
  4. mnm starter --list      # the models that fit this card; 'mnm starter <pick>' adds them
Later, 'mnm update' pulls, syncs and restarts the services.
DONE
}

main "$@"
