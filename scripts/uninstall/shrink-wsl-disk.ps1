# Written by makenomistakes' uninstall.sh: give Windows back the space freed inside WSL.
# Run it from an administrator PowerShell. It shuts WSL down (Docker and Odysseus stop too and
# start again with WSL), compacts the distribution's disk, and deletes itself once that worked.
$ErrorActionPreference = 'Stop'
$distro = '@DISTRO@'
$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw 'Run this from an administrator PowerShell (right-click PowerShell, Run as administrator).'
}
$key = Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' |
  Where-Object { $_.GetValue('DistributionName') -eq $distro } | Select-Object -First 1
if (-not $key) { throw "No WSL distribution named $distro" }
# BasePath may carry the \\?\ long-path prefix, which diskpart does not take.
$file = $key.GetValue('VhdFileName'); if (-not $file) { $file = 'ext4.vhdx' }
$vhdx = Join-Path ($key.GetValue('BasePath') -replace '^\\\\\?\\', '') $file
if (-not (Test-Path -LiteralPath $vhdx)) { throw "Disk not found: $vhdx" }
$before = (Get-Item -LiteralPath $vhdx).Length
wsl.exe --shutdown
# The disk is released a moment after WSL stops: wait until it opens with nobody else on it.
$deadline = (Get-Date).AddSeconds(60)
while ($true) {
  try { [IO.File]::Open($vhdx, 'Open', 'ReadWrite', 'None').Close(); break }
  catch {
    if ((Get-Date) -gt $deadline) { throw "WSL still holds $vhdx; run wsl --shutdown, then this again." }
    Start-Sleep -Seconds 2
  }
}
$steps = Join-Path $env:TEMP 'mnm-compact-wsl.txt'
@("select vdisk file=`"$vhdx`"", 'attach vdisk readonly', 'compact vdisk', 'detach vdisk') |
  Set-Content -Encoding Default $steps
diskpart.exe /s $steps
$code = $LASTEXITCODE
if ($code -ne 0) { # a compact that stopped half way can leave the disk attached: detach it
  @("select vdisk file=`"$vhdx`"", 'detach vdisk') | Set-Content -Encoding Default $steps
  diskpart.exe /s $steps | Out-Null
}
Remove-Item $steps
if ($code -ne 0) {
  throw "diskpart failed (exit $code); the disk is unchanged and this script stays. Run it again, or diskpart by hand: select vdisk file=`"$vhdx`", attach vdisk readonly, compact vdisk, detach vdisk."
}
$after = (Get-Item -LiteralPath $vhdx).Length
'{0}: {1:N1} GB -> {2:N1} GB' -f $vhdx, ($before / 1GB), ($after / 1GB)
Remove-Item $PSCommandPath
