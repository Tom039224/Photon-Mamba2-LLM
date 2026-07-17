#!/usr/bin/env bash
# Verify pm-core does not depend on backend-specific crates.
#
# pm-core is the backend-agnostic core of the workspace. Adding a
# dependency on candle-*, cudarc, tt-metal, etc. defeats the whole
# point of the Backend trait and forces a Phase-2 rewrite. CI runs
# this script on every push.

set -euo pipefail

cd "$(dirname "$0")/.."

FORBIDDEN_PATTERN='(^|[[:space:]])(candle[-_]|cudarc|pm[-_]cuda|tt[-_]metal|tt[-_]rs)'

OUT=$(cargo tree -p pm-core --edges normal --prefix none 2>/dev/null \
    | grep -E "$FORBIDDEN_PATTERN" || true)

if [ -n "$OUT" ]; then
    echo "FAIL: pm-core depends on backend-specific crates:" >&2
    echo "$OUT" >&2
    exit 1
fi

echo "OK: pm-core has no backend dependencies."
