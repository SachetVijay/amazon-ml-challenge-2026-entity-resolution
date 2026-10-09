#!/usr/bin/env bash
# Build a minimal, self-contained code zip for one submission version.
#   scripts/make_code_zip.sh v1 path/to/model.txt module1 module2 ...
# -> submissions/<ver>_code.zip containing business_entity_resolution/{README.md,requirements.txt,models/,src/er/}
set -euo pipefail
cd "$(dirname "$0")/.."
VER=$1; MODEL=$2; shift 2
TMP=$(mktemp -d); D=$TMP/business_entity_resolution
mkdir -p "$D/src/er" "$D/models"
cp src/er/__init__.py "$D/src/er/"
for m in "$@"; do cp "src/er/$m.py" "$D/src/er/"; done
if [ -d "$MODEL" ]; then cp "$MODEL"/* "$D/models/"; else cp "$MODEL" "$D/models/model.txt"; fi
cp "package/README_${VER}.md" "$D/README.md"
if [ -f "package/requirements_${VER}.txt" ]; then cp "package/requirements_${VER}.txt" "$D/requirements.txt"; else cp package/requirements.txt "$D/"; fi
rm -f "submissions/${VER}_code.zip"
(cd "$TMP" && zip -qr - business_entity_resolution) > "submissions/${VER}_code.zip"
rm -rf "$TMP"
unzip -l "submissions/${VER}_code.zip"
