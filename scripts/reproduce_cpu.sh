#!/bin/bash
# Re-run every CPU experiment (the 2017-2020 checkpoints, §7) from the repository root:
#   bash scripts/reproduce_cpu.sh            (about 20 minutes on a 6-core desktop)
# Needs requirements.txt and MobileNetV2 (python scripts/download_data.py mobilenet). Logs go to logs/.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p logs results/exp2_out results/audit
run() { n=$1; shift; s=$(date +%s); python -X utf8 "$@" > "logs/$n.log" 2>&1; echo "[$n] exit $? in $(( $(date +%s) - s ))s"; }
run exp2b_selftest src/exp2b_tinysr18_weightpatch.py --selftest --out results/exp2b_selftest
run audit  src/extract_and_audit.py
run exp2   src/exp2_tinysr18_patch_v1.py
run exp2b  src/exp2b_tinysr18_weightpatch.py --weights legacy/TinySR-18/generator.h5 --lr-filter nearest
run exp3   src/exp3_anime_weightpatch.py
run exp4   src/exp4_anime_eye_circuit.py
run exp5   src/exp5_git_demo.py
run exp6   src/exp6_tinysr18_pair_transfer.py --lr-filter nearest
run exp7   src/exp7_blind_audit.py
run exp7b  src/exp7b_recycle_bot.py
run exp7c  src/exp7c_deep_color.py
run exp7d  src/exp7d_anime_styles.py
run exp8   src/exp8_mobilenet.py
run exp8b  src/exp8b_oracle.py
