"""`mnm doctor --remote-script [--shell sh|powershell]`: `remote_doctor`'s checks as a
script the user pastes on another machine, which needs nothing of mnm's: POSIX sh and curl,
or Windows PowerShell 5.1 / PowerShell 7 alone.

What the box knows goes in (its router's LAN address, the desk's roles, the chat id, Ollama's
port, put on the router's host when the script runs); the key never does. The script reads
`MNM_API_KEY` from its own environment: sh hands it to curl on curl's stdin (`-H @-`, written
by the shell's builtin printf, so it is on no command line), PowerShell keeps it inside its
own process. `MNM_ROUTER_URL` in that environment replaces the address (a Tailscale name,
GUIDE 13.4), Ollama's with it."""

from __future__ import annotations

import json
import shlex
from enum import StrEnum
from urllib.parse import urlsplit

from makenomistakes.bridge.protocol import PATH as MCP_PATH
from makenomistakes.remote_doctor import HELLO, Target, chat_body


class Shell(StrEnum):
    SH = "sh"
    POWERSHELL = "powershell"


def script(target: Target, shell: Shell, need_key: bool = True) -> str:
    """`need_key=False`: the script's address is this machine's (Windows reaching WSL at
    localhost), where the router may answer keyless."""
    return _sh(target, need_key) if shell is Shell.SH else _powershell(target, need_key)


def _body(target: Target) -> str:
    return json.dumps(chat_body(target.chat or "coder"), separators=(",", ":"))


_SH = r"""#!/bin/sh
# mnm's checks from another machine: the box's router as this machine reaches it.
# Run it on that machine (the key typed at a hidden prompt, never into shell history):
#   read -rs MNM_API_KEY; export MNM_API_KEY; sh mnm-check.sh
# (bash or zsh; in a plain sh: stty -echo; read -r MNM_API_KEY; stty echo; export MNM_API_KEY)
# The key is read from this environment only and handed to curl on its stdin (never on a
# command line). MNM_ROUTER_URL='https://…' replaces the address below. Needs curl.
R=${MNM_ROUTER_URL:-}
[ -n "$R" ] || R=__BASE__
R="${R%/}"
# Ollama's own port on the router's host, whichever address R names.
OPORT=__OLLAMA_PORT__
O=''
if [ -n "$OPORT" ]; then
  scheme=${R%%://*}; rest=${R#*://}; host=${rest%%/*}; host=${host##*@}
  case $host in
    *\]) ;;
    *:*) host=${host%:*} ;;
  esac
  O="$scheme://$host:$OPORT"
fi
ROLES=__ROLES__
CHAT=__CHAT__
NEEDKEY=__NEEDKEY__
fails=0
ok() { echo "OK    $1"; }
warn() { echo "WARN  $1"; }
bad() { echo "FAIL  $1"; fails=$((fails + 1)); }
command -v curl >/dev/null 2>&1 || { echo "FAIL  curl is not installed here"; exit 2; }
[ -n "$MNM_API_KEY" ] || warn "MNM_API_KEY is not set in this shell: requests go without it"

# $1 method, $2 url, $3 body (may be empty), $4 key|bare: the body, then the status code
ask() {
  m=$1; u=$2; b=$3; k=$4
  set --
  if [ -n "$b" ]; then set -- --data-raw "$b"; fi
  if [ "$k" = key ] && [ -n "$MNM_API_KEY" ]; then
    printf 'Authorization: Bearer %s\n' "$MNM_API_KEY" | curl -sS -m 300 -X "$m" \
      -H 'Content-Type: application/json' -H @- "$@" -o - -w '\n%{http_code}' "$u" 2>&1
  else
    curl -sS -m 300 -X "$m" -H 'Content-Type: application/json' "$@" -o - \
      -w '\n%{http_code}' "$u" 2>&1
  fi
}
code() { printf '%s\n' "$1" | tail -n 1; }
said() { printf '%s\n' "$1" | sed '$d' | tr '\n' ' ' | cut -c1-200; }

out=$(ask GET "$R/v1/models" "" key); s=$(code "$out")
if [ "$s" = 200 ]; then
  miss=''
  for r in $ROLES; do
    printf '%s' "$out" | grep -Eq "\"id\": ?\"$r\"" || miss="$miss $r"
  done
  if [ -z "$miss" ]; then ok "Models: $R/v1/models lists the desk's roles (${ROLES:-none})"
  else warn "Models: $R/v1/models lacks the desk's$miss"; fi
else
  bad "Models: GET $R/v1/models: $s $(said "$out")"
fi
if [ "$s" != 000 ]; then
  out=$(ask POST "$R/v1/chat/completions" __CHAT_BODY__ key); s=$(code "$out")
  if [ "$s" = 200 ] && printf '%s' "$out" | grep -q '"choices"'; then ok "Chat: $CHAT answered"
  else bad "Chat: POST $R/v1/chat/completions ($CHAT): $s $(said "$out")"; fi
  out=$(ask POST "$R__MCP__" __HELLO__ key); s=$(code "$out")
  if [ "$s" = 200 ] && printf '%s' "$out" | grep -q '"serverInfo"'; then
    ok "MCP: $R__MCP__ answered initialize"
  else bad "MCP: initialize at $R__MCP__: $s $(said "$out")"; fi
  if [ -n "$O" ]; then
    out=$(ask GET "$O/api/tags" "" key); s=$(code "$out")
    if [ "$s" = 200 ]; then ok "Ollama API: $O/api/tags"
    else bad "Ollama API: GET $O/api/tags: $s $(said "$out")"; fi
  fi
fi
out=$(ask GET "$R/v1/models" "" bare); s=$(code "$out")
if [ "$s" = 401 ]; then ok "Key: a request without MNM_API_KEY is refused (401)"
elif [ "$NEEDKEY" = 0 ] && [ "$s" = 200 ]; then ok "Key: none, and none is needed on this PC"
else bad "Key: a request without MNM_API_KEY got $s: the box's router must ask for its key"; fi
if [ "$fails" -gt 0 ]; then echo "$fails failed"; exit 1; fi
echo "all passed"
"""

_PS = r"""# mnm's checks from another machine: the box's router as this machine reaches it.
# Run it on that machine, in PowerShell:
#   $env:MNM_API_KEY = [Net.NetworkCredential]::new('', (Read-Host -AsSecureString 'key')).Password
#   powershell -ExecutionPolicy Bypass -File mnm-check.ps1
# (the key typed at a hidden prompt, never into PowerShell's history)
# The key is read from this environment only and stays inside PowerShell's own process.
# $env:MNM_ROUTER_URL = 'https://…' replaces the address below.
$R = if ($env:MNM_ROUTER_URL) { $env:MNM_ROUTER_URL.TrimEnd('/') } else { '__BASE__' }
# Ollama's own port on the router's host, whichever address $R names.
$OPort = '__OLLAMA_PORT__'
$O = ''
if ($OPort) {
  $b = [UriBuilder]::new($R); $b.Port = [int]$OPort; $b.Path = ''; $b.UserName = ''
  $O = $b.Uri.GetLeftPart([UriPartial]::Authority)
}
$Roles = @(__ROLES__)
$Chat = '__CHAT__'
$NeedKey = $__NEEDKEY__
$Fails = 0
function Ok($t) { Write-Output "OK    $t" }
function Warn($t) { Write-Output "WARN  $t" }
function Bad($t) { Write-Output "FAIL  $t"; $script:Fails++ }
if (-not $env:MNM_API_KEY) { Warn 'MNM_API_KEY is not set in this shell: requests go without it' }

function Ask($Method, $Url, $Body, [switch]$Bare) {
  $h = @{}
  if (-not $Bare -and $env:MNM_API_KEY) { $h['Authorization'] = 'Bearer ' + $env:MNM_API_KEY }
  $p = @{ Uri = $Url; Method = $Method; Headers = $h; UseBasicParsing = $true
          TimeoutSec = 300; ContentType = 'application/json' }
  if ($Body) { $p['Body'] = $Body }
  try {
    $r = Invoke-WebRequest @p
    return @{ Status = [int]$r.StatusCode; Body = [string]$r.Content }
  } catch {
    $resp = $_.Exception.Response
    if ($resp) {
      return @{ Status = [int]$resp.StatusCode; Body = [string]$_.ErrorDetails.Message }
    }
    return @{ Status = 0; Body = $_.Exception.Message }
  }
}
function Said($a) {
  $t = ($a.Body -replace '\s+', ' ')
  "$($a.Status) " + $t.Substring(0, [Math]::Min(200, $t.Length))
}

$a = Ask 'GET' "$R/v1/models" ''
if ($a.Status -eq 200) {
  $ids = @(($a.Body | ConvertFrom-Json).data | ForEach-Object { $_.id })
  $miss = @($Roles | Where-Object { $ids -notcontains $_ })
  if ($miss.Count -eq 0) { Ok "Models: $R/v1/models lists the desk's roles ($($Roles -join ' '))" }
  else { Warn "Models: $R/v1/models lacks the desk's $($miss -join ' ')" }
} else { Bad "Models: GET $R/v1/models: $(Said $a)" }
if ($a.Status -ne 0) {
  $a = Ask 'POST' "$R/v1/chat/completions" '__CHAT_BODY__'
  if ($a.Status -eq 200 -and $a.Body -match '"choices"') { Ok "Chat: $Chat answered" }
  else { Bad "Chat: POST $R/v1/chat/completions ($Chat): $(Said $a)" }
  $a = Ask 'POST' "$R__MCP__" '__HELLO__'
  if ($a.Status -eq 200 -and $a.Body -match '"serverInfo"') {
    Ok "MCP: $R__MCP__ answered initialize"
  }
  else { Bad "MCP: initialize at $R__MCP__: $(Said $a)" }
  if ($O) {
    $a = Ask 'GET' "$O/api/tags" ''
    if ($a.Status -eq 200) { Ok "Ollama API: $O/api/tags" }
    else { Bad "Ollama API: GET $O/api/tags: $(Said $a)" }
  }
}
$a = Ask 'GET' "$R/v1/models" '' -Bare
if ($a.Status -eq 401) { Ok 'Key: a request without MNM_API_KEY is refused (401)' }
elseif (-not $NeedKey -and $a.Status -eq 200) { Ok 'Key: none, and none is needed on this PC' }
else {
  Bad "Key: a request without MNM_API_KEY got $($a.Status): the box's router must ask for it"
}
if ($Fails -gt 0) { Write-Output "$Fails failed"; exit 1 }
Write-Output 'all passed'
"""


def _fill(template: str, values: dict[str, str]) -> str:
    for name, value in values.items():
        template = template.replace(f"__{name}__", value)
    return template


def _sh(target: Target, need_key: bool) -> str:
    return _fill(
        _SH,
        {
            "NEEDKEY": "1" if need_key else "0",
            "BASE": shlex.quote(target.base),  # quoted like every other value
            "OLLAMA_PORT": _ollama_port(target),
            "ROLES": shlex.quote(" ".join(target.roles)),
            "CHAT": shlex.quote(target.chat or "coder"),
            "CHAT_BODY": shlex.quote(_body(target)),
            "MCP": MCP_PATH,
            "HELLO": shlex.quote(json.dumps(HELLO, separators=(",", ":"))),
        },
    )


def _ollama_port(target: Target) -> str:
    """The port alone (digits, no quoting needed): the script puts it on the router's host
    at run time, so `MNM_ROUTER_URL` moves the Ollama check with the router's (else it
    keeps asking `http://localhost:11434`)."""
    if target.ollama is None:
        return ""
    port = urlsplit(target.ollama).port
    return "" if port is None else str(port)


def _ps_quote(text: str) -> str:
    """Inside a single-quoted PowerShell string: a quote doubled (and U+2018-U+201B, which
    PowerShell also ends such a string at, never appear in what mnm puts there)."""
    return text.replace("'", "''")


def _powershell(target: Target, need_key: bool) -> str:
    return _fill(
        _PS,
        {
            "NEEDKEY": "true" if need_key else "false",
            "BASE": _ps_quote(target.base),
            "OLLAMA_PORT": _ollama_port(target),
            "ROLES": ", ".join(f"'{_ps_quote(r)}'" for r in target.roles),
            "CHAT": _ps_quote(target.chat or "coder"),
            "CHAT_BODY": _ps_quote(_body(target)),
            "MCP": MCP_PATH,
            "HELLO": _ps_quote(json.dumps(HELLO, separators=(",", ":"))),
        },
    )
