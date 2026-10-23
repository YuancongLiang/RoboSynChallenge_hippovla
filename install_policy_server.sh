#!/usr/bin/env bash
# Compatibility entrypoint; the complete runtime ships inside policy/hippovla.
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PACKAGE_ROOT}/policy/hippovla/install_policy_server.sh" "$@"
