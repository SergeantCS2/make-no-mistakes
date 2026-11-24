# shellcheck shell=bash
# uninstall.sh's models step: the download cache goes whole; in views/, loras/ and upscalers/ only mnm's
# links go (they point into hf/), never the folders whole. Sourced by uninstall.sh, which
# defines run, note, say, failed, remove_path, guarded, holds_a_mount, clear_folder and the
# globals used here (DRY_RUN, KEEP_MODELS, MODELS_DIR, MODEL_ENTRIES, LINK_FOLDERS, CONFIG_FILES);
# `library_rows` is its summary's word on the studio's libraries.

# A link of mnm's: it points into the download cache, Civitai's files or Ollama's.
into_hf() {
  [ -L "$1" ] || return 1
  case "$(realpath -m -- "$1")/" in
    "$(realpath -m -- "$MODELS_DIR/hf")"/* | "$(realpath -m -- "$MODELS_DIR/civitai")"/*) return 0 ;;
    "$(realpath -m -- "$MODELS_DIR/ollama")"/* | "$(realpath -m -- "$MODELS_DIR/modelscope")"/*)
      return 0
      ;;
  esac
  return 1
}

# The files and links in a views/, loras/ or upscalers/ folder that are not mnm's links, one indented name
# per line (nothing for a folder that is not there).
not_mnm_links() {
  local dir=$1 path
  while IFS= read -r -d '' path; do
    if ! into_hf "$path"; then printf '      %s\n' "${path#"$dir"/}"; fi
  done < <(find "$dir" -mindepth 1 ! -type d -print0 2>/dev/null)
}

# views/, loras/ or upscalers/: mnm's links go, then every folder left empty, itself included. Never
# removed whole: your own files and links stay and are listed.
clear_links() {
  local dir=$1 path theirs
  if [ -L "$dir" ]; then
    remove_path "$dir"
    return 0
  fi
  if [ ! -d "$dir" ]; then return 0; fi
  if ! guarded "$dir"; then
    note "kept $dir: not a folder this script clears"
    return 0
  fi
  if holds_a_mount "$(realpath -m -- "$dir")"; then # find would walk into the mounted folder
    failed "clear $dir: it is or holds a mount point; unmount it and run this again"
    return 0
  fi
  theirs="$(not_mnm_links "$dir")"
  while IFS= read -r -d '' path; do
    if into_hf "$path"; then run rm -f -- "$path" || failed "remove the link $path"; fi
  done < <(find "$dir" -mindepth 1 ! -type d -print0 2>/dev/null)
  run find "$dir" -depth -type d -empty -delete || failed "remove the empty folders in $dir"
  if [ -n "$theirs" ]; then
    note "kept in $dir, not mnm's:"
    printf '%s\n' "$theirs"
  fi
}

# The link folders under a models folder that hold nothing but mnm's links: a dry run, which
# leaves the links in place, counts them as gone.
only_mnm_links() {
  local folder
  for folder in "${LINK_FOLDERS[@]}"; do
    if [ -z "$(not_mnm_links "$1/$folder")" ]; then printf '%s\n' "$folder"; fi
  done
}

# What huggingface_hub writes in a cache folder; `hf/` is cleared entry by entry so a models
# folder pointed at a shared place (`MNM_MODELS_DIR=~/Downloads`) keeps the user's other files.
HF_CACHE_ENTRIES=('models--*' 'datasets--*' 'spaces--*' .locks .no_exist xet 'version*.txt' 'tmp*')

# The voice packages: one real folder per repo, `<org>--<repo>/` (huggingface_hub's .cache/
# sits inside it); the Civitai files: one real folder per version id, `<digits>/`. A file or
# link of such a name, or anything else, is the user's and stays. $2 is the regular
# expression a whole name must match (`2024-notes/` in civitai/ is the user's).
owned_folders() {
  local path name
  for path in "$1"/*; do
    name="$(basename -- "$path")"
    if [ -d "$path" ] && [ ! -L "$path" ] && [[ $name =~ $2 ]]; then printf '%s\n' "$name"; fi
  done
}

# What is left in such a folder that is not one of mnm's, one indented name a line.
owned_leftovers() {
  local name
  if [ ! -d "$1" ]; then return 0; fi
  while IFS= read -r -d '' name; do
    if [ ! -d "$1/$name" ] || [ -L "$1/$name" ] || ! [[ $name =~ $2 ]]; then
      printf '      %s\n' "$name"
    fi
  done < <(find "$1/" -mindepth 1 -maxdepth 1 -printf '%f\0' 2>/dev/null)
}

clear_owned() {
  local folder=$1 folders=()
  if [ ! -e "$folder" ] && [ ! -L "$folder" ]; then return 0; fi
  if [ -L "$folder" ]; then
    remove_path "$folder"
    return 0
  fi
  mapfile -t folders < <(owned_folders "$folder" "$2")
  clear_folder "$folder" "${folders[@]}"
}

VOICE_NAME='^[^/]*--[^/]*$'
CIVITAI_NAME='^[0-9]+$'
OLLAMA_NAME="$VOICE_NAME" # `<namespace>--<name>/`, one folder per Ollama model
MODELSCOPE_NAME="$VOICE_NAME" # `<org>--<repo>/`, one folder per ModelScope model

clear_hf_cache() {
  local hf=$1
  if [ ! -e "$hf" ] && [ ! -L "$hf" ]; then return 0; fi
  if [ -L "$hf" ]; then
    note "$hf is a link: the folder it points to, $(readlink -- "$hf"), is cleared, not removed"
  fi
  clear_folder "$hf" "${HF_CACHE_ENTRIES[@]}"
}

remove_models() {
  say "Models"
  # prompt-cache/ holds the saved prompts: conversations an engine wrote, never a model
  # file, so they go even with --keep-models; quality/ a quality tune's text and log-probabilities
  # while it runs, likewise.
  if [ -d "$MODELS_DIR" ]; then
    remove_path "$MODELS_DIR/prompt-cache"
    remove_path "$MODELS_DIR/quality"
  fi
  if [ "$KEEP_MODELS" -eq 1 ]; then
    note "kept $MODELS_DIR (--keep-models)"
    return 0
  fi
  if [ ! -d "$MODELS_DIR" ]; then
    note "no models folder at $MODELS_DIR"
    return 0
  fi
  local folder links=() also=()
  for folder in "${LINK_FOLDERS[@]}"; do clear_links "$MODELS_DIR/$folder"; done
  clear_hf_cache "$MODELS_DIR/hf"
  # configs/ holds the server configs mnm renders from its recipes, one
  # <instance>.json each: only those go (a ComfyUI models folder has a configs/ of its own).
  if [ -L "$MODELS_DIR/configs" ]; then
    remove_path "$MODELS_DIR/configs"
  elif [ "${#CONFIG_FILES[@]}" -gt 0 ]; then
    clear_folder "$MODELS_DIR/configs" "${CONFIG_FILES[@]}"
  fi
  # voices/ holds the voice packages as plain files, one `<org>--<repo>/` folder per repo
  # (huggingface_hub's `.cache/` inside it): those folders go, entry by entry like hf/.
  clear_owned "$MODELS_DIR/voices" "$VOICE_NAME"
  # civitai/ holds Civitai downloads as plain files, one `<version id>/` folder each.
  clear_owned "$MODELS_DIR/civitai" "$CIVITAI_NAME"
  # ollama/ holds Ollama's GGUFs as plain files, `<namespace>--<name>/<tag>/`.
  clear_owned "$MODELS_DIR/ollama" "$OLLAMA_NAME"
  # modelscope/ holds ModelScope's files as plain files, `<org>--<repo>/<revision>/`.
  clear_owned "$MODELS_DIR/modelscope" "$MODELSCOPE_NAME"
  # Cleared, they are gone; a dry run still sees them, and a folder that kept your files stays.
  if [ "$DRY_RUN" -eq 1 ]; then
    mapfile -t links < <(only_mnm_links "$MODELS_DIR")
    if [ -z "$(leftovers "$MODELS_DIR/hf" "${HF_CACHE_ENTRIES[@]}")" ]; then also+=(hf); fi
    if [ -z "$(owned_leftovers "$MODELS_DIR/voices" "$VOICE_NAME")" ]; then also+=(voices); fi
    if [ -z "$(owned_leftovers "$MODELS_DIR/civitai" "$CIVITAI_NAME")" ]; then
      also+=(civitai)
    fi
    if [ -z "$(owned_leftovers "$MODELS_DIR/ollama" "$OLLAMA_NAME")" ]; then also+=(ollama); fi
    if [ -z "$(owned_leftovers "$MODELS_DIR/modelscope" "$MODELSCOPE_NAME")" ]; then
      also+=(modelscope)
    fi
    if [ ! -e "$MODELS_DIR/configs" ] ||
      [ -z "$(leftovers "$MODELS_DIR/configs" "${CONFIG_FILES[@]}")" ]; then also+=(configs); fi
    also+=(prompt-cache quality)
  fi
  clear_folder "$MODELS_DIR" -- "${also[@]}" "${links[@]}"
  # `mnm setup` made /data for /data/models; it goes again only when nothing else is in it.
  local name
  name="$(basename -- "$MODELS_DIR")"
  if [ "$(dirname -- "$MODELS_DIR")" = "/data" ] && [ ! -L /data ] && ! is_mount_point /data &&
    [ -z "$(find /data/ -mindepth 1 -maxdepth 1 ! -name "$name" 2>/dev/null)" ] &&
    { [ "$DRY_RUN" -eq 1 ] || [ ! -e "$MODELS_DIR" ]; }; then
    run sudo rmdir /data || failed "remove /data"
  fi
}

# The voice and style libraries' rows go with the state while their files stay in the models
# folder: named before anything is removed, with the way to keep
# them. A row is whose voice or songs it is, its licence and its consent record.
library_rows() {
  local folder=$1 library=$2 names
  names="$(find "$folder" -maxdepth 1 -name '*.json' -printf '%f\n' 2>/dev/null |
    sed 's/\.json$//' | sort | xargs || true)"
  if [ -n "$names" ]; then
    note "WARNING: the $library library's rows go with the state: $names (each one's source,"
    note "licence and consent record); \`mnm backup\` first keeps them, and \`mnm restore\` of"
    note "it after a reinstall puts them back beside the kept files"
  fi
}
