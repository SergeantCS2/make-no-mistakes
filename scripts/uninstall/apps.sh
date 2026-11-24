# shellcheck shell=bash
# uninstall.sh's workspace-app steps: the data `mnm apps` keeps under
# MNM_APPS_DIR, one folder per app, cleared entry by entry like the models folder, and kept
# whole with --keep-apps. Sourced by uninstall.sh, which defines say, note, clear_folder and the
# globals used here (APPS_DIR, KEEP_APPS). The apps' containers (labelled mnm.app), their
# networks and images are docker.sh's; OpenCode's unit is in UNIT_NAMES; its binaries are in
# the state folder. Never touched: the projects folder (MNM_PROJECTS_DIR, the user's code),
# OpenCode's own sessions (~/.local/share/opencode), ~/odysseus.

# The folders `mnm apps install` makes in MNM_APPS_DIR (apps/defs.py, compose.py).
APP_FOLDERS=(opencode librechat openhands open-webui)
# The images the apps run, by the digest mnm pins (apps/defs.py, tag left out: Docker keeps a
# `repo:tag@digest` pull as `repo@digest`), the same as those files.
APP_IMAGES=(
  "ghcr.io/danny-avila/librechat@sha256:c5db3331b845e1f289f8d04c0c77936c4bbe372f76730a804abc1c37e44d23a9"
  "mongo@sha256:098862b1339f031900ca66cf8fef799e616d6324fa41b9a263f2ec899552c1ef"
  "getmeili/meilisearch@sha256:8b57fc3c7f46535ddef3828df1538465ac19d892eb57c9a10da6df0880bd5856"
  "pgvector/pgvector@sha256:8809cfffff0082cf260c9ac752f1dd1afc77f6f0a55c4e6411321e78efc3d9a5"
  "ghcr.io/danny-avila/librechat-rag-api-dev-lite@sha256:18a8dffa9316276b924bf3fd23bfb4544b0f68f8f6b14c048affb35c01721a35"
  "ghcr.io/openhands/agent-canvas@sha256:348253b254600817b627fc05030a135876a408e9988303232d41e9f89fae7d1a"
  "ghcr.io/open-webui/open-webui@sha256:a537bbe1e6b428820f52ed4a029d7fff97062e24d5cf32c365fba36fd0542def"
)

is_app_ref() { # `repo@sha256:…` exactly as pinned (Docker Hub's own images without docker.io/)
  local ref=${1#docker.io/} pinned
  ref=${ref#library/}
  for pinned in "${APP_IMAGES[@]}"; do
    if [ "$ref" = "$pinned" ]; then return 0; fi
  done
  return 1
}

remove_app_data() {
  say "Workspace apps' data"
  if [ "$KEEP_APPS" -eq 1 ]; then
    note "kept $APPS_DIR (--keep-apps): LibreChat's and Open WebUI's accounts and chats, OpenHands' home"
    return 0
  fi
  if [ ! -d "$APPS_DIR" ]; then
    note "no apps folder at $APPS_DIR"
    return 0
  fi
  clear_folder "$APPS_DIR" "${APP_FOLDERS[@]}"
}
