#!/usr/bin/env bash
# Copy each round's DeePMD dpdata (deepmd/npy) trees from INVAR-2025/iterXX.finetune/<run>/
# into public_release/INVAR-DART/dataset/<iter>/ (no symlinks).
# Re-run to refresh; existing subdirs under each iter are removed and re-copied.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"

declare -A RUN=(
  [iter00]="dpa_v3.1"
  [iter01]="dpa_v3_1"
  [iter02]="dpa_v3_1"
  [iter03]="dpa_v3_1"
  [iter04]="dpa_v3_1"
  [iter05]="dpa_v3_1"
  [iter06]="dpa_v3_dft_multiTask"
)

copy_dir() {
  local src="$1" dest_parent="$2" name="$3"
  local s="${src}/${name}" d="${dest_parent}/${name}"
  if [[ ! -d "$s" ]]; then
    return 0
  fi
  rm -rf "$d"
  cp -a "$s" "$d"
  echo "  ${name}"
}

for iter in iter00 iter01 iter02 iter03 iter04 iter05 iter06; do
  run="${RUN[$iter]}"
  src="${REPO_ROOT}/${iter}.finetune/${run}"
  dest="${HERE}/${iter}"
  if [[ ! -d "$src" ]]; then
    echo "skip ${iter}: missing ${src}"
    continue
  fi
  mkdir -p "$dest"
  echo "== ${iter} (${run})"
  for name in datasets test new_data new_data_npy; do
    copy_dir "$src" "$dest" "$name"
  done
  for i in 1 2 3 4 5; do
    copy_dir "$src" "$dest" "train_fold_${i}"
  done
  shopt -s nullglob
  for s in "${src}"/new_data_iter*_npy; do
    b="$(basename "$s")"
    copy_dir "$src" "$dest" "$b"
  done
  shopt -u nullglob
done

echo "Done."
