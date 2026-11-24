# shellcheck shell=bash
# uninstall.sh's Docker steps: the mnm containers, the engine images, and Docker itself when
# nothing else uses it. Sourced by uninstall.sh, which defines run, note, say, failed and the
# globals used here (DRY_RUN, DOCKER, ENGINE_REPOS, LEDGER_IDS, RECIPE_IMAGES, ODYSSEUS_DIR).
# Every Docker question here fails closed: no answer means "something else may use it", and a
# step left undone is reported with the commands to finish it by hand.

# What `mnm setup` installs from Docker's and NVIDIA's apt repositories.
DOCKER_PACKAGES=(docker-ce docker-ce-cli containerd.io docker-buildx-plugin
  docker-compose-plugin docker-ce-rootless-extras nvidia-container-toolkit
  nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1)

# The tags mnm pulls and runs on any backend and architecture (engine_images.py IMAGES: CUDA,
# ROCm, Vulkan, CPU, arm64; setup pulls llama.cpp's Vulkan image beside ROCm, and the doctor's
# `--deep` device check runs the backend's llama.cpp image) and the image the NVIDIA GPU check
# runs (hardware/probe.py GPU_SMOKE_IMAGE, pulled by `mnm doctor --deep`), plus a tag of an
# engine repository that a saved recipe names. Another tag is yours. The list matches those
# constants.
MNM_TAGS=("ghcr.io/ggml-org/llama.cpp:server-cuda" "ghcr.io/ggml-org/llama.cpp:server-cuda13"
  "ghcr.io/ggml-org/llama.cpp:server-rocm" "ghcr.io/ggml-org/llama.cpp:server-vulkan"
  "ghcr.io/ggml-org/llama.cpp:server"
  "ghcr.io/ggml-org/llama.cpp:full-cuda" "ghcr.io/ggml-org/llama.cpp:full-cuda13"
  "ghcr.io/ggml-org/llama.cpp:full-rocm" "ghcr.io/ggml-org/llama.cpp:full-vulkan"
  "ghcr.io/ggml-org/llama.cpp:full"
  "ghcr.io/leejet/stable-diffusion.cpp:master-cuda"
  "ghcr.io/leejet/stable-diffusion.cpp:master-cuda-spark"
  "ghcr.io/leejet/stable-diffusion.cpp:master-vulkan"
  "ghcr.io/ggml-org/whisper.cpp:main-cuda" "ghcr.io/ggml-org/whisper.cpp:main-vulkan"
  "ghcr.io/ggml-org/whisper.cpp:main-vulkan-arm64" "ghcr.io/ggml-org/whisper.cpp:main"
  "ghcr.io/ggml-org/whisper.cpp:main-arm64"
  "ghcr.io/0xshug0/audio.cpp:full-cuda12" "ghcr.io/0xshug0/audio.cpp:full-vulkan"
  "ghcr.io/0xshug0/audio.cpp:full-cpu"
  "vllm/vllm-openai:latest" "vllm/vllm-openai-rocm:latest"
  "nvidia/cuda:12.8.1-base-ubuntu24.04")
# The one volume mnm makes: vLLM's compile caches (recipes/vllm.py CACHE_VOLUME).
MNM_VOLUMES=("mnm-vllm-cache")
# The networks mnm makes: ComfyUI's, internal (comfy/engine.py NETWORK), and
# the default network of each app's compose project (`<project>_default`).
MNM_NETWORKS=("mnm-comfyui" "mnm-librechat_default" "mnm-openhands_default"
  "mnm-open-webui_default")
# The images mnm builds itself (comfy/engine.py IMAGE_REPO; the studio tools,
# studio/tools.py IMAGE_REPO; the trainer, studio/trainer.py IMAGE_REPO): every tag of
# each is mnm's.
MNM_BUILT_REPOS=("mnm/comfyui" "mnm/studio-tools" "mnm/studio-trainer" "mnm/studio-style-trainer")

is_built_tag() {
  local repo
  for repo in "${MNM_BUILT_REPOS[@]}"; do
    case "$1" in "$repo":*) return 0 ;; esac
  done
  return 1
}

is_mnm_tag() { # a recipe file can be edited by hand: it names mnm's tags only in their repos
  case " ${MNM_TAGS[*]} " in *" $1 "*) return 0 ;; esac
  is_built_tag "$1" && return 0
  is_engine_ref "$1" && case " $RECIPE_IMAGES " in *" $1 "*) return 0 ;; esac
  return 1
}

print_docker_cleanup() {
  note "To finish once Docker answers: docker rm --force \$(docker ps -aq --filter label=mnm.instance),"
  note "then docker image rm each ${ENGINE_REPOS[*]} image (docker images --digests)"
  note "and nvidia/cuda:12.8.1-base-ubuntu24.04, the GPU check's."
}

# DOCKER=(docker), or (sudo docker) for a shell not yet in the docker group, when the daemon
# answers; empty when it does not. A dry run never asks for a password.
find_docker() {
  DOCKER=()
  command -v docker >/dev/null 2>&1 || return 0
  DOCKER_WOULD_SUDO=0
  if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
  elif [ "$DRY_RUN" -eq 0 ] && sudo docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  elif [ "$DRY_RUN" -eq 1 ]; then
    DOCKER_WOULD_SUDO=1 # a dry run never prompts; the real run asks Docker with sudo
  fi
}

# Said instead of "could not reach Docker" in a dry run that would have used sudo.
docker_unreachable() {
  if [ "${DOCKER_WOULD_SUDO:-0}" -eq 1 ]; then
    note "would ask Docker with sudo (this shell is not in the docker group): its mnm $1 would go"
  else
    failed "reach Docker: its mnm $1 stay"
    print_docker_cleanup
  fi
}

# Names of the containers without an mnm label (an instance's, or an app's: mnm.app):
# Odysseus's, anyone's; optionally filtered.
foreign_containers() {
  local lines
  lines="$("${DOCKER[@]}" ps -a "$@" \
    --format '{{.Label "mnm.instance"}}{{.Label "mnm.app"}}|{{.Names}}')" || return 1
  printf '%s\n' "$lines" | sed -n 's/^|//p'
}

image_ids() {
  local ids
  ids="$("${DOCKER[@]}" image ls --all --quiet --no-trunc)" || return 1
  printf '%s\n' "$ids" | sort -u
}

is_engine_ref() { # `…/llama.cpp:server-cuda` or `…/llama.cpp@sha256:…`, that repository exactly
  local ref=$1 repo engine
  repo="${ref%@*}"
  if [ "$repo" = "$ref" ]; then repo="${ref%:*}"; fi
  for engine in "${ENGINE_REPOS[@]}"; do
    if [ "$repo" = "$engine" ]; then return 0; fi
  done
  return 1
}

# Sets IMAGE_ENGINE, IMAGE_ENGINE_TAGS and IMAGE_OTHER_TAGS for one image id. mnm's: a tag mnm
# runs, an untagged build of an engine repository (what `mnm engines update` replaced), or a
# build the engine ledger recorded. Any other tag, of any repository, is someone else's.
classify_image() {
  local id=$1 refs ref engine_digest=0 app_digest=0
  IMAGE_ENGINE=0 IMAGE_ENGINE_TAGS=() IMAGE_OTHER_TAGS=()
  refs="$("${DOCKER[@]}" image inspect --format \
    '{{join .RepoTags " "}} {{join .RepoDigests " "}}' "$id")" || return 1
  case " $LEDGER_IDS " in *" $id "*) IMAGE_ENGINE=1 ;; esac
  for ref in $refs; do
    case "$ref" in
      *@*) if is_engine_ref "$ref"; then engine_digest=1; elif is_app_ref "$ref"; then app_digest=1; fi ;;
      *) if is_mnm_tag "$ref"; then
        IMAGE_ENGINE=1 && IMAGE_ENGINE_TAGS+=("$ref")
      else
        IMAGE_OTHER_TAGS+=("$ref")
      fi ;;
    esac
  done
  if [ "$((engine_digest + app_digest))" -gt 0 ] &&
    [ "${#IMAGE_ENGINE_TAGS[@]}${#IMAGE_OTHER_TAGS[@]}" = 00 ]; then
    IMAGE_ENGINE=1
  fi
}

remove_containers() {
  say "Containers"
  if [ "${#DOCKER[@]}" -eq 0 ]; then
    if command -v docker >/dev/null 2>&1; then
      docker_unreachable "containers and images (their ports too)"
    else
      note "Docker is not installed."
    fi
    return 0
  fi
  local ids apps
  if ! ids="$("${DOCKER[@]}" ps -aq --filter label=mnm.instance)" ||
    ! apps="$("${DOCKER[@]}" ps -aq --filter label=mnm.app)"; then
    failed "list Docker's containers: mnm's stay"
    print_docker_cleanup
    return 0
  fi
  ids="$(printf '%s\n%s\n' "$ids" "$apps" | sed '/^$/d' | sort -u)"
  if [ -z "$ids" ]; then
    note "no mnm containers"
    return 0
  fi
  # shellcheck disable=SC2086  # one id per word
  run "${DOCKER[@]}" rm --force $ids || failed "remove the containers $(printf '%s ' $ids)"
}

# Every local build of an engine repository: the tags mnm runs, the builds `mnm engines update`
# replaced (untagged; found by repository digest or in the engine ledger), other tags of them.
remove_images() {
  say "Engine images, the GPU check image and the apps' images"
  if [ "${#DOCKER[@]}" -eq 0 ]; then return 0; fi
  local ids id short users tag found=0 built=0
  if ! ids="$(image_ids)"; then
    failed "list Docker's images: the engine images stay"
    print_docker_cleanup
    return 0
  fi
  for id in $ids; do
    short="${id#sha256:}" && short="${short:0:12}"
    if ! classify_image "$id"; then
      failed "inspect image $short: it stays"
      continue
    fi
    if [ "$IMAGE_ENGINE" -eq 0 ]; then continue; fi
    found=1
    for tag in "${IMAGE_ENGINE_TAGS[@]}"; do
      if is_built_tag "$tag"; then built=1; fi
    done
    if ! users="$(foreign_containers --filter "ancestor=$id")"; then
      failed "tell which containers use ${IMAGE_ENGINE_TAGS[*]:-$short}: it stays"
    elif [ -n "$users" ]; then
      # shellcheck disable=SC2086  # one name per word
      note "kept ${IMAGE_ENGINE_TAGS[*]:-$short}: used by $(printf '%s ' $users)"
    elif [ "${#IMAGE_OTHER_TAGS[@]}" -gt 0 ]; then
      note "kept $short: also tagged ${IMAGE_OTHER_TAGS[*]}; only the engine tags go"
      if [ "${#IMAGE_ENGINE_TAGS[@]}" -gt 0 ]; then
        run "${DOCKER[@]}" image rm "${IMAGE_ENGINE_TAGS[@]}" || failed "untag $short"
      fi
    else
      run "${DOCKER[@]}" image rm --force "$id" || failed "remove the image $short"
    fi
  done
  if [ "$found" -eq 0 ]; then note "no engine images"; fi
  if [ "$built" -eq 1 ]; then # Docker's, not mnm's: your own builds may use them
    note "Docker's build cache and ubuntu:24.04 (what mnm built ComfyUI from) stay; if nothing"
    note "of yours uses them: docker builder prune, then docker image rm ubuntu:24.04"
  fi
}

# mnm's own volumes and networks (MNM_VOLUMES, MNM_NETWORKS), once its containers are gone.
# Another volume or network is not mnm's.
remove_volumes() {
  [ "${#DOCKER[@]}" -gt 0 ] || return 0
  local volume network
  for volume in "${MNM_VOLUMES[@]}"; do
    if "${DOCKER[@]}" volume inspect "$volume" >/dev/null 2>&1; then
      run "${DOCKER[@]}" volume rm "$volume" || failed "remove the volume $volume"
    fi
  done
  for network in "${MNM_NETWORKS[@]}"; do
    if "${DOCKER[@]}" network inspect "$network" >/dev/null 2>&1; then
      run "${DOCKER[@]}" network rm "$network" || failed "remove the network $network"
    fi
  done
}

# What Docker holds that is not mnm's: Odysseus's containers, images, volumes and networks, or
# anyone's. mnm makes only MNM_VOLUMES and MNM_NETWORKS. Returns 1 when Docker did not answer.
foreign_docker() {
  local containers ids id volumes networks
  containers="$(foreign_containers)" || return 1
  ids="$(image_ids)" || return 1
  volumes="$("${DOCKER[@]}" volume ls --quiet)" || return 1
  networks="$("${DOCKER[@]}" network ls --format '{{.Name}}' --filter type=custom)" || return 1
  printf '%s\n' "$containers" | sed '/^$/d; s/^/container /'
  for id in $ids; do
    classify_image "$id" || return 1
    if [ "$IMAGE_ENGINE" -eq 0 ] || [ "${#IMAGE_OTHER_TAGS[@]}" -gt 0 ]; then
      printf 'image %s %s\n' "${id:7:12}" "${IMAGE_OTHER_TAGS[*]}"
    fi
  done
  # one pattern per name (`-e "${A[@]}"` would read every name after the first as a file)
  printf '%s\n' "$volumes" | { grep -vxF -f <(printf '%s\n' "${MNM_VOLUMES[@]}") || true; } |
    sed '/^$/d; s/^/volume /'
  printf '%s\n' "$networks" | { grep -vxF -f <(printf '%s\n' "${MNM_NETWORKS[@]}") || true; } |
    sed '/^$/d; s/^/network /'
}

remove_docker() {
  say "Docker Engine and the NVIDIA container toolkit (--docker)"
  local foreign packages=() pkg context
  if ! command -v docker >/dev/null 2>&1; then
    note "not installed"
    return 0
  fi
  if [ -e "$HOME/odysseus" ] || [ -e "$ODYSSEUS_DIR" ]; then
    note "kept: Odysseus is installed and runs in Docker"
    return 0
  fi
  if [ "${#DOCKER[@]}" -eq 0 ]; then
    failed "remove Docker: it did not answer, so nothing says that nothing else uses it"
    return 0
  fi
  if [ -n "${DOCKER_HOST:-}" ] || ! context="$("${DOCKER[@]}" context show 2>/dev/null)" ||
    [ "$context" != default ]; then
    note "kept: the docker command points at another daemon (DOCKER_HOST or a context), or"
    note "could not say which; its answers say nothing about the system Docker this would remove"
    return 0
  fi
  if ! foreign="$(foreign_docker)"; then
    failed "remove Docker: it did not answer every question, so it stays"
    return 0
  fi
  if [ -n "$foreign" ]; then
    note "kept: Docker still holds things that are not mnm's (Odysseus needs Docker):"
    printf '%s\n' "$foreign" | sed 's/^/      /'
    return 0
  fi
  for pkg in "${DOCKER_PACKAGES[@]}"; do
    if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      packages+=("$pkg")
    fi
  done
  case " ${packages[*]} " in
    *" docker-ce "*) ;;
    *) # Ubuntu's docker.io, a snap, Docker Desktop: not ours, and its data stays with it
      note "kept: Docker was not installed from Docker's apt repository (as mnm setup does it)"
      return 0
      ;;
  esac
  if ! others_apt_takes "${packages[@]}" || ! only_dockers_containerd; then return 0; fi
  # Each step needs the one before: Docker's data goes only once its packages have.
  if ! run sudo apt-get purge -y "${packages[@]}"; then
    failed "purge ${packages[*]}; Docker, its data and its apt source stay"
    return 0
  fi
  if ! run sudo rm -rf /var/lib/docker /var/lib/containerd; then
    failed "remove Docker's data (/var/lib/docker, /var/lib/containerd)"
    return 0
  fi
  # /etc/docker: the files Docker and nvidia-ctk wrote; registry certificates of yours stay.
  run sudo rm -f /etc/docker/daemon.json /etc/docker/key.json ||
    failed "remove /etc/docker/daemon.json"
  if [ -n "$(find /etc/docker -mindepth 1 -maxdepth 1 ! -name daemon.json ! -name key.json \
    2>/dev/null)" ]; then
    note "kept /etc/docker: it holds files of yours"
  elif [ -d /etc/docker ]; then
    run sudo rmdir /etc/docker || failed "remove /etc/docker"
  fi
  run sudo rm -f /etc/apt/sources.list.d/docker.sources /etc/apt/keyrings/docker.asc \
    /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg || failed "remove the apt sources"
  if getent group docker >/dev/null 2>&1; then run sudo groupdel docker || true; fi
  note "Packages those pulled in stay; \`sudo apt-get autoremove\` lists and removes them."
}

# apt would also remove whatever depends on these (nvidia-docker2, a CI runner): ask it first,
# and keep Docker when anything outside mnm's packages would go.
others_apt_takes() {
  local plan other
  if ! plan="$(apt-get -s purge "$@" 2>/dev/null)"; then
    failed "ask apt what purging Docker would remove; Docker stays"
    return 1
  fi
  other="$(printf '%s\n' "$plan" | awk '/^(Purg|Remv) / {print $2}' |
    grep -vxF "$(printf '%s\n' "${DOCKER_PACKAGES[@]}")" | xargs || true)"
  if [ -n "$other" ]; then
    note "kept: purging Docker would also remove $other"
    return 1
  fi
}

# /var/lib/containerd also holds what nerdctl, ctr or Kubernetes made outside Docker (their
# default namespace is "default"); Docker's own is "moby". A dry run never asks for a password.
only_dockers_containerd() {
  local spaces ns held others=""
  local ctr=(sudo ctr)
  if [ "$DRY_RUN" -eq 1 ]; then ctr=(sudo -n ctr); fi
  if ! spaces="$("${ctr[@]}" namespaces ls -q 2>/dev/null)"; then
    if [ "$DRY_RUN" -eq 1 ]; then
      note "containerd's namespaces are checked in the real run (it needs sudo)"
      return 0
    fi
    failed "ask containerd what else it holds; Docker stays"
    return 1
  fi
  for ns in $spaces; do
    if [ "$ns" = moby ]; then continue; fi
    if ! held="$("${ctr[@]}" -n "$ns" containers ls -q && "${ctr[@]}" -n "$ns" images ls -q)"; then
      failed "ask containerd what namespace $ns holds; Docker stays"
      return 1
    fi
    if [ -n "$held" ]; then others+=" $ns"; fi
  done
  if [ -n "$others" ]; then
    note "kept: containerd also holds containers or images outside Docker, in:$others"
    return 1
  fi
}
