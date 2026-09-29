#!/usr/bin/env bash
# Builds an anonymised code bundle for double-blind review:
#   paper/make_anonymous_artifact.sh  ->  paper/build/anonymous-artifact.tar.gz
# Upload it as supplementary material or to an anonymising host
# (e.g. anonymous.4open.science). It contains HEAD without git history.
set -euo pipefail
cd "$(dirname "$0")/.."
out=paper/build
rm -rf "$out/artifact" && mkdir -p "$out/artifact"
git archive HEAD | tar -x -C "$out/artifact"
# Remove the copyright holder's name; everything else is already anonymous.
sed -i 's/^Copyright (c) \([0-9]*\) .*/Copyright (c) \1 Anonymous Authors/' "$out/artifact/LICENSE"
# Refuse to produce a bundle that still identifies the authors.
patterns=(-e "$(git config user.name 2>/dev/null || echo __none__)" -e "$(git config user.email 2>/dev/null || echo __none__)")
if [ -n "${ANON_EXTRA_PATTERNS:-}" ]; then patterns+=(-e "$ANON_EXTRA_PATTERNS"); fi
if grep -rniI "${patterns[@]}" "$out/artifact"; then
  echo "identifying strings found above; not writing the bundle" >&2
  exit 1
fi
tar -czf "$out/anonymous-artifact.tar.gz" -C "$out/artifact" .
echo "wrote $out/anonymous-artifact.tar.gz"
