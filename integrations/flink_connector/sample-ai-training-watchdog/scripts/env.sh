# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Sourced by the other scripts. Loads KEY=value lines from the sample's .env, if there is one,
# without overriding variables already set in the environment, so `DEPLOY_REGION=... scripts/x.sh`
# still wins. Values may be quoted; nothing in the file is expanded or executed. See .env.example.
_dotenv="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [[ -f "$_dotenv" ]]; then
  _assign='^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
  # A quoted value, then optionally a comment: KEY="value"  # note
  _quoted="^'([^']*)'[[:space:]]*(#.*)?\$|^\"([^\"]*)\"[[:space:]]*(#.*)?\$"
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ $_assign ]] || continue
    _key=${BASH_REMATCH[2]}
    _value=${BASH_REMATCH[3]}
    [[ -n "${!_key+set}" ]] && continue
    if [[ "$_value" =~ $_quoted ]]; then
      _value=${BASH_REMATCH[1]}${BASH_REMATCH[3]}
    else
      _value=${_value%%[[:space:]]#*}                   # drop an inline comment
      _value=${_value%"${_value##*[![:space:]]}"}       # and trailing whitespace
    fi
    export "$_key=$_value"
  done <"$_dotenv"
  unset _assign _quoted _line _key _value
fi
unset _dotenv
