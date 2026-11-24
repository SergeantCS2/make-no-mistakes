# shellcheck shell=bash
# uninstall.sh's last step: trim the freed space and hand Windows the script that shrinks the
# WSL disk (shrink-wsl-disk.ps1 next to this file). Sourced by uninstall.sh.

# The steps by hand, for when Windows is out of reach: printed in full, since the checkout
# and its docs are gone by now.
print_manual_shrink() {
  cat <<MANUAL

    To give the space back to Windows by hand: close every WSL terminal, then in an
    administrator PowerShell run \`wsl --shutdown\` and find the disk's folder with

      (Get-ChildItem HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss |
        Where-Object { \$_.GetValue('DistributionName') -eq '$1' }).GetValue('BasePath')

    then run \`diskpart\` and type, with that folder (without a leading \\\\?\\) and the disk
    file (ext4.vhdx unless the same registry key's VhdFileName names another):

      select vdisk file="<folder>\\ext4.vhdx"
      attach vdisk readonly
      compact vdisk
      detach vdisk
MANUAL
}

reclaim_space() {
  say "Giving the space back"
  local distro="${WSL_DISTRO_NAME:-Ubuntu-24.04}" profile="" target
  if command -v fstrim >/dev/null 2>&1; then run sudo fstrim -v / || true; fi
  if [ -z "${WSL_DISTRO_NAME:-}" ] && ! grep -qi microsoft /proc/version 2>/dev/null; then
    return 0
  fi
  if command -v cmd.exe >/dev/null 2>&1 && command -v wslpath >/dev/null 2>&1; then
    profile="$( # cmd.exe warns about a WSL working directory; /mnt/c keeps it quiet
      cd /mnt/c 2>/dev/null || true
      cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r' || true
    )"
  fi
  if [ -z "$profile" ] || [ "$profile" = "%USERPROFILE%" ]; then
    note "Windows is not reachable from this shell (no cmd.exe or wslpath)."
    print_manual_shrink "$distro"
    return 0
  fi
  if [ -z "$SHRINK_PS1" ]; then
    note "scripts/uninstall/shrink-wsl-disk.ps1 was missing from the checkout."
    print_manual_shrink "$distro"
    return 0
  fi
  if ! target="$(wslpath -u "$profile")/mnm-shrink-wsl-disk.ps1"; then
    note "could not turn $profile into a WSL path"
    print_manual_shrink "$distro"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    note "would write $target"
  elif ! printf '%s\n' "${SHRINK_PS1//@DISTRO@/$distro}" 2>/dev/null >"$target"; then
    note "could not write $target"
    print_manual_shrink "$distro"
    return 0
  else
    note "wrote $target"
  fi
  cat <<WINDOWS

    Deleted files free space inside WSL; Windows gets it back once the WSL disk shrinks.
    Close every WSL terminal, then in an administrator PowerShell on Windows run:

      powershell -ExecutionPolicy Bypass -File "$profile\\mnm-shrink-wsl-disk.ps1"
WINDOWS
}
