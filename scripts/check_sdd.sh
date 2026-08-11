#!/usr/bin/env bash
set -euo pipefail

sdd_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sdd_repo_root="$(cd -- "${sdd_script_dir}/.." && pwd)"

required_files=(
  "AGENTS.md"
  ".specify/memory/constitution.md"
  ".specify/skills/api-and-interface-design.md"
  ".specify/skills/code-review-and-quality.md"
  ".specify/skills/git-workflow-and-versioning.md"
  ".specify/skills/test-driven-development.md"
  "specs/README.md"
  "specs/PROJECT_STATE.md"
  "specs/TRACEABILITY.md"
  "specs/001-voice-conversion-engine/spec.md"
  "specs/001-voice-conversion-engine/plan.md"
  "specs/001-voice-conversion-engine/data-model.md"
  "specs/002-voice-training/spec.md"
  "specs/002-voice-training/plan.md"
  "specs/003-webui/spec.md"
  "specs/004-source-separation/spec.md"
  "specs/006-realtime-voice-conversion/spec.md"
  "specs/006-realtime-voice-conversion/plan.md"
  "specs/006-realtime-voice-conversion/data-model.md"
  "specs/006-realtime-voice-conversion/research.md"
  "specs/006-realtime-voice-conversion/r0-assets.json"
  "tests/fixtures/README.md"
  "models/README.md"
  "docs/DESIGN.md"
  "docs/CLI.md"
  "docs/REFACTORING_GUIDE.md"
  "docs/adr/README.md"
  "docs/adr/0001-modular-runtime-and-orchestration-boundaries.md"
  "docs/adr/0002-persistent-streaming-runtime-and-local-audio-boundary.md"
  "docs/adr/0003-retire-legacy-embedded-webserver.md"
  "docs/adr/0004-bounded-offline-file-conversion.md"
  "docs/adr/0005-immutable-flat-index-snapshot.md"
)

failed=0
for relative_path in "${required_files[@]}"; do
  if [[ ! -s "${sdd_repo_root}/${relative_path}" ]]; then
    echo "SDD error: required file is missing or empty: ${relative_path}" >&2
    failed=1
  fi
done

if (( failed )); then
  exit 1
fi

# Codex uses a 32 KiB default budget for the combined project instruction chain.
# Keep the root bridge deliberately small; detailed guidance belongs in the SDD.
agents_bytes="$(wc -c < "${sdd_repo_root}/AGENTS.md")"
if (( agents_bytes > 32768 )); then
  echo "SDD error: AGENTS.md is ${agents_bytes} bytes; keep it below 32768 bytes" >&2
  failed=1
fi

bridge_targets=(
  ".specify/memory/constitution.md"
  ".specify/skills/api-and-interface-design.md"
  ".specify/skills/code-review-and-quality.md"
  ".specify/skills/git-workflow-and-versioning.md"
  ".specify/skills/test-driven-development.md"
  "specs/README.md"
  "specs/PROJECT_STATE.md"
  "specs/TRACEABILITY.md"
  "docs/DESIGN.md"
  "docs/REFACTORING_GUIDE.md"
  "docs/adr/README.md"
)
for target in "${bridge_targets[@]}"; do
  if ! grep -Fq "${target}" "${sdd_repo_root}/AGENTS.md"; then
    echo "SDD error: AGENTS.md does not route to ${target}" >&2
    failed=1
  fi
done

if compgen -G "${sdd_repo_root}/specs/005-*" > /dev/null; then
  echo "SDD error: retired spec 005 must not be recreated" >&2
  failed=1
fi

# Validate repository-relative Markdown links in the active SDD set. Historical
# snapshots/research are excluded; the active Spec 006 research is included because
# its R0 evidence and open qualification blockers govern the accepted ADR.
link_documents=(
  "README.md"
  "AGENTS.md"
  ".specify/memory/constitution.md"
  ".specify/skills/api-and-interface-design.md"
  ".specify/skills/code-review-and-quality.md"
  ".specify/skills/git-workflow-and-versioning.md"
  ".specify/skills/test-driven-development.md"
  "specs/README.md"
  "specs/PROJECT_STATE.md"
  "specs/TRACEABILITY.md"
  "specs/001-voice-conversion-engine/spec.md"
  "specs/001-voice-conversion-engine/plan.md"
  "specs/001-voice-conversion-engine/data-model.md"
  "specs/002-voice-training/spec.md"
  "specs/002-voice-training/plan.md"
  "specs/003-webui/spec.md"
  "specs/004-source-separation/spec.md"
  "specs/006-realtime-voice-conversion/spec.md"
  "specs/006-realtime-voice-conversion/plan.md"
  "specs/006-realtime-voice-conversion/data-model.md"
  "specs/006-realtime-voice-conversion/research.md"
  "tests/fixtures/README.md"
  "models/README.md"
  "docs/DESIGN.md"
  "docs/CLI.md"
  "docs/REFACTORING_GUIDE.md"
  "docs/adr/README.md"
  "docs/adr/0001-modular-runtime-and-orchestration-boundaries.md"
  "docs/adr/0002-persistent-streaming-runtime-and-local-audio-boundary.md"
  "docs/adr/0003-retire-legacy-embedded-webserver.md"
  "docs/adr/0004-bounded-offline-file-conversion.md"
  "docs/adr/0005-immutable-flat-index-snapshot.md"
)
markdown_link_pattern='\]\(([^)]*)\)'

for relative_document in "${link_documents[@]}"; do
  document="${sdd_repo_root}/${relative_document}"
  while IFS= read -r line; do
    remainder="${line}"
    while [[ "${remainder}" =~ ${markdown_link_pattern} ]]; do
      match="${BASH_REMATCH[0]}"
      target="${BASH_REMATCH[1]}"
      remainder="${remainder#*"${match}"}"

      target="${target%%#*}"
      target="${target%%\?*}"
      target="${target#<}"
      target="${target%>}"
      if [[ -z "${target}" || "${target}" == /* ||
            "${target}" =~ ^(https?|mailto): ]]; then
        continue
      fi

      if [[ ! -e "$(dirname -- "${document}")/${target}" ]]; then
        echo "SDD error: broken link in ${relative_document}: ${target}" >&2
        failed=1
      fi
    done
  done < "${document}"
done

if (( failed )); then
  exit 1
fi

echo "SDD contract check passed (${#required_files[@]} required files)."
