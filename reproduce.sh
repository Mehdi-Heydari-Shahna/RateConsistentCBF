#!/bin/bash
# Regenerates every figure, table and number of the paper from source and writes them to results/.
#
#   bash reproduce.sh                 everything (continuous-time examples, excavator runs, tables and figures)
#   bash reproduce.sh pinocchio       Section VII-A, Remarks 4-5, Section VIII, Table I, Figs. 3-4 (about 30 min)
#   bash reproduce.sh excavators      the 38 excavator runs of Section VII-B (several hours; resumable)
#   bash reproduce.sh tables          Tables II-V, Figs. 6-8 and the numbers of Section VII-B from the excavator runs
# (figure files keep the numbering of the first submission: fig2-fig3 are Figs. 3-4, fig5-fig7 are Figs. 6-8)
#
# Environment: PYTHON (default: python), JOBS = parallel excavator runs (default 1).
# The excavator runs (and the Komatsu / HEAP rows of Table I) need the excavator model packages, which are supplied
# separately (README.md, "Excavator models"); without them, Table I takes those rows from the logged results.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=${PYTHON:-python}
JOBS=${JOBS:-1}
RESULTS="$HERE/results"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=${MUJOCO_GL:-egl}

log_run() {          # log_run <folder> <log name> <script and arguments...>
    local dir=$1 log=$2
    shift 2
    echo "== $dir: $* > $log"
    (cd "$dir" && "$PY" "$@" > "$log" 2>&1) || { echo "FAILED: $dir/$log"; tail -20 "$dir/$log"; exit 1; }
}

machines_present() {
    [ -f "$HERE/code/excavators/RCX_TAC/machines/komatsu/machine.py" ] && [ -f "$HERE/code/excavators/RCX_TAC/machines/heap/machine.py" ]
}

need_machines() {
    machines_present || { echo "The excavator model packages komatsu and heap are not part of this repository; copy them into"
                          echo "code/excavators/RCX_TAC/machines/ (README.md, \"Excavator models\")."; exit 2; }
}

pinocchio() {
    local num="$HERE/code/pinocchio/num" sweep="$HERE/code/pinocchio/onedof_sweep" checks="$HERE/code/pinocchio/checks"
    # Table I: algebraic checks on random, constructed and compiled-model instances
    log_run "$num" t1_comparison.log t1_comparison.py
    if machines_present; then
        log_run "$num" t2_komatsu.log t2_mechanism.py komatsu
        log_run "$num" t2_heap.log t2_mechanism.py heap 4
    else
        echo "== excavator model packages not present: the Komatsu and HEAP rows of Table I use the logged results"
        cp "$RESULTS/logs/pinocchio/t2_komatsu_results.json" "$RESULTS/logs/pinocchio/t2_heap_results.json" "$num/"
    fi
    log_run "$num" t7_carrier.log t7_carrier.py
    rm -f "$num/t9_results.json"
    log_run "$num" t9_general.log t9_general.py
    # QP routine of the excavator controllers on nearly dependent constraints, before and after the correction
    log_run "$num" t13_qp_dependent.log t13_qp_dependent.py
    # Section VII-A: six-joint arm (Fig. 2(a)) and scalar example (Figs. 2(b) and 3)
    rm -f "$num/t14_results.json"
    log_run "$num" t14_consistent.log t14_consistent.py
    log_run "$num" t15_onedof.log t15_onedof.py
    log_run "$sweep" sweep_alpha2.log sweep_alpha2.py
    log_run "$checks" arm_onset_corollary.log arm_onset_corollary.py "$num"
    log_run "$checks" rho_bar_arm_narrow.log rho_bar_arm.py --limits narrow
    log_run "$checks" rho_bar_arm_model.log rho_bar_arm.py --limits model
    log_run "$checks" ex_metric.log ex_metric.py
    (cd "$HERE/code/pinocchio" && "$PY" paper_figures.py --out "$RESULTS" && "$PY" paper_numbers.py --out "$RESULTS")
    mkdir -p "$RESULTS/logs/pinocchio"
    cp "$num"/*.log "$num"/*_results.json "$sweep"/sweep_alpha2.log "$sweep"/sweep_alpha2.json "$checks"/*.log "$checks"/*.json "$RESULTS/logs/pinocchio/"
}

excavators() {
    need_machines
    (cd "$HERE/code/excavators/RCX_TAC" && "$PY" batch.py all paper --jobs "$JOBS")
}

tables() {
    need_machines
    "$PY" "$HERE/code/excavators/RCX_TAC/paper/tables.py" --out "$RESULTS"
}

case "${1:-all}" in
    pinocchio) pinocchio ;;
    excavators) excavators ;;
    tables) tables ;;
    all) pinocchio; excavators; tables ;;
    *) echo "usage: bash reproduce.sh [pinocchio|excavators|tables|all]"; exit 2 ;;
esac
echo "done: results in $RESULTS"
