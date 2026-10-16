# RateConsistentCBF

**Tracking Limitations and Rate-Consistent Safety Filtering for Constrained Mechanical Systems**

Mehdi Heydari Shahna · Jouni Mattila

Preprint: [https://doi.org/10.21203/rs.3.rs-11381118/v1](https://doi.org/10.21203/rs.3.rs-11381118/v1)

This repository is the supplementary material of the paper: the source code of every numerical result, the logged
results of the runs behind its figures and tables, the figure trajectories as CSV files, and the comparison video.

Reference-level and effort-level control barrier function (CBF) filters can impose incompatible constraint-approach
rates in mechanical systems. The paper characterizes the resulting velocity-tracking limitation with a
controller-independent lower bound, characterizes the effect of safety corrections on the tracking dissipation of
virtual decomposition control (VDC), and develops a rate-consistent, acceleration-level reference filter.
[RoboCompiler](https://github.com/Mehdi-Heydari-Shahna/RoboCompiler) supplies the shared mechanical interfaces of the
VDC law and both CBF layers. The results are illustrated on a six-joint Pinocchio arm, a scalar example, and two
sampled MuJoCo hydraulic excavator models.

## Repository contents

| Path | Content |
|---|---|
| `code/pinocchio/` | Continuous-time examples (Section VII-A, Remarks 4-5, Section VIII, Table I) |
| `code/excavators/RCX_TAC/` | MuJoCo excavator benchmark and controllers (Section VII-B): compiled interfaces (`core/`), VDC and CBF layers (`rcx/`), soil and site (`env/`), bench (`bench/`), paper scripts (`paper/`) |
| `results/` | Logged results, figure trajectories (CSV), regenerated figures and tables |
| `videos/supplementary_video.mp4` | Comparison video |
| `reproduce.sh` | Regenerates every figure, table and number of the paper into `results/` |
| `requirements.txt` | Pinned Python packages |

## Excavator models

The two excavator machine packages of the paper (Komatsu and HEAP: mechanism records, geometry, actuator and
hydraulic parameters, safety specification and site layout) are not part of this repository. The code accepts any
machine given as a package `code/excavators/RCX_TAC/machines/<name>/` with `machine.py` (a subclass of `Machine`,
`bench/machine.py`), `model.py`, `control_safety_spec.json` and its source records in `source/`.

To run the excavator examples with the models of the paper, copy the folders `komatsu` and `heap` of the separately
supplied model archive into `code/excavators/RCX_TAC/machines/`. The source hash of every logged run
(`results/runs/index.csv`) includes the machine package, so regenerated results can be compared with the logged results
only with these packages.

Without the machine packages, the continuous-time examples and the recomputation of Tables II-V from the logged results
run; the excavator simulations, the videos and the Komatsu and HEAP rows of Table I need the packages (`reproduce.sh`
then takes those rows from the logged results). The logged reports do not list the per-channel declared force-model
error bounds, which belong to the machine records.

## Setup

Use Python 3.12 on Linux (or WSL2 on Windows). Open a terminal in the repository folder and run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep the folder structure intact.

## Run single examples

```bash
# Six-joint arm and one-degree-of-freedom example (Section VII-A)
python code/pinocchio/num/t14_consistent.py
python code/pinocchio/num/t15_onedof.py

# Six approaches to the boom stroke limit (Section VII-B; needs the machine packages)
python code/excavators/RCX_TAC/run.py komatsu hazard --scenario cycles --variant consistent
python code/excavators/RCX_TAC/run.py heap hazard --scenario cycles --variant consistent

# Dig-and-level missions (needs the machine packages)
python code/excavators/RCX_TAC/run.py komatsu mission --variant consistent
python code/excavators/RCX_TAC/run.py heap mission --variant consistent
```

For the excavators, `consistent` selects the proposed acceleration-level first layer and `full` the baseline
first-order first layer. The other variants of the paper are `kinematic_only` (first layer only), `force_only`
(effort layer only), `full_ffraw` (raw difference-quotient feedforward), `accel_b40` (acceleration-level layer with
beta = 40), `conditioned` (conditioning (27)), `consistent_b8`, the gain variants `full_k<k>` and `consistent_k<k>` for
k = 2, 4, 6, 24, and `kinematic_only_k2`; `--payload-scale 1.5` gives the heavier bucket. Add `--duration 1` for a short
execution check, or use `--scenario efence` or `--scenario self` for the workspace scenarios. Simulations run without
a graphical window and write their results to `code/excavators/RCX_TAC/results/<machine>/`; a run replaces the result
of the same name. First runs also generate scene assets.

## Reproduce the paper

```bash
bash reproduce.sh pinocchio     # Section VII-A, Remarks 4-5, Section VIII, Table I, Figs. 3-4   (about 30 min)
bash reproduce.sh excavators    # the 38 excavator runs of Section VII-B (several hours; resumable; needs the machine packages)
bash reproduce.sh tables        # Tables II-V, Figs. 6-8 and the numbers of Section VII-B (needs the machine packages)
```

`bash reproduce.sh` runs all three steps. `JOBS=2 bash reproduce.sh excavators` runs two excavator simulations at a
time; on two cores the 38 runs took 3 to 4 hours (the four missions take most of it). The excavator batch
(`code/excavators/RCX_TAC/batch.py all paper`) skips runs whose report is complete for the present sources, so an
interrupted batch can be started again. Every generated table lists the regenerated value next to the value printed in
the paper (the values that changed from the first submission are listed in `code/excavators/RCX_TAC/paper/tables.py`).
Computation times per controller update depend on the hardware.

Tables II-V and the numbers of Section VII-B can also be recomputed in seconds from the logged results in `results/`,
without running the simulations and without the machine packages:

```bash
python code/excavators/RCX_TAC/paper/tables.py --from-logs results --out check
```

The logged samples carry seven significant digits, so values recomputed from them agree with `results/` at the
printed precision of the paper; a fourth significant digit in `numbers/section_vii_b.md` can differ by one unit.
Without the machine packages, the tables note that the logged reports were not checked against the source hash.

File names, script comments and CSV headers number the figures as in the first submission of the paper; in the paper
each of these figures is one number higher: `fig2` is Fig. 3, `fig3` is Fig. 4, `fig5` is Fig. 6, `fig6` is Fig. 7 and
`fig7` is Fig. 8 (Fig. 2, the methodology diagram, and Fig. 5, the plant stills, have no data files).

| Paper item | Script | Output in `results/` |
|---|---|---|
| Fig. 3(a), Section VII-A (arm) | `code/pinocchio/num/t14_consistent.py`, `checks/arm_onset_corollary.py` | `figures/fig2.pdf`, `csv/fig2a_arm.csv`, `numbers/section_vii_a.*` |
| Fig. 3(b), Fig. 4(b), Example 1 | `code/pinocchio/num/t15_onedof.py` | `figures/fig2.pdf`, `figures/fig3.pdf`, `csv/fig2b_*.csv`, `csv/fig3b_gain.csv` |
| Fig. 4(a) | `code/pinocchio/onedof_sweep/sweep_alpha2.py` | `figures/fig3.pdf`, `csv/fig3a_alpha2.csv` |
| Table I | `num/t2_mechanism.py` (Komatsu, HEAP), `num/t1_comparison.py`, `num/t7_carrier.py`, `num/t9_general.py` | `tables/table1.*` |
| Remarks 4-5 | `code/pinocchio/checks/ex_metric.py` | `numbers/section_vii_a.*` |
| Section VIII (curvature rate) | `code/pinocchio/checks/rho_bar_arm.py`, `num/t14_consistent.py` | `numbers/section_vii_a.*` |
| Tables II-III, Figs. 6-8 | `code/excavators/RCX_TAC/batch.py` (group `cycles`), `paper/tables.py` | `tables/table2.*`, `tables/table3.*`, `figures/fig5.pdf`, `fig6.pdf`, `fig7.pdf`, `csv/fig5_*.csv`, `csv/fig6_*.csv`, `csv/fig7_gain.csv` |
| Table IV | `batch.py` (group `workspace`), `paper/tables.py` | `tables/table4.*` |
| Table V | `batch.py` (group `mission`), `paper/tables.py` | `tables/table5.*` |
| Numbers of Section VII-B | `paper/tables.py` | `numbers/section_vii_b.*` |
| QP routine on nearly dependent constraints | `code/pinocchio/num/t13_qp_dependent.py` | `logs/pinocchio/t13_*` |

The definitions of the table columns (storage, onset and release errors, positive energies of the storage balance,
clearances, binding fractions, RMS errors) are given in `code/excavators/RCX_TAC/paper/metrics.py`.

## Logged results

`results/` holds the results of the runs that produced the paper's figures and tables:

- `runs/<machine>/<run>/report.json` and `run.log`: the report and the console log of every excavator run
  (`runs/index.csv` lists the runs, their status and the hash of the sources they ran with).
- `csv/runs/<machine>/<run>.csv`: the control samples (10 ms) of every excavator run: phase, storage, velocity error,
  powers of the storage balance, binding constraints, QP feasibility and stationarity residual, controller computation
  time, and for the pairs of the scenario the barrier value and the binding flags of both layers; the runs of the
  scenario `cycles` also give the slack of the first-layer constraint of the boom stroke pair at the required velocity
  and J s (columns `zs_<pair>`, `w_<pair>`). The full traces of the runs (about 450 MB) are regenerated by
  `bash reproduce.sh excavators`.
- `csv/fig*.csv`: the trajectories and data points of Figs. 3, 4, 6, 7 and 8 (file numbers: see above).
- `tables/`, `numbers/`: the regenerated Tables I-V and the numbers quoted in Sections VII-A, VII-B and VIII.
- `logs/pinocchio/`: console logs and result files of the continuous-time scripts.

## Notes on the numerical methods

- Arm simulations (Section VII-A) use RK4 steps of 1 ms for both first layers. Changes of the active sets of both
  layers are located by bisection and the integration step is split at the change (`toy.advance_with_events`, shared
  by the first-order and the consistent loop). The scalar example uses RK4 steps of 0.2 ms with the same localization.
- The excavator controllers solve their quadratic programs with the dual active-set method of Goldfarb and Idnani in
  `code/excavators/RCX_TAC/rcx/qp.py`. For a violated constraint that is nearly linearly dependent on the active ones,
  the routine takes the steps of the exact method: a partial step moves the primal and the dual iterate together, and
  the full step that makes the constraint active is taken whenever it is not longer than the partial step (and the
  squared projected normal exceeds 1e-14 of that of the normal). A computed step direction below ten times its
  estimated round-off error is treated as zero, as for an exactly dependent constraint, and after any step for a
  constraint rated dependent the point is accepted only with relative residuals below 1e-8 (active constraints relative
  to |A_i| (|x| + |x - x_u|) + |b_i|, stationarity relative to |H| |x| + |g| + |A|^T mu; otherwise the feasible set is
  rated numerically empty). An earlier version (`code/pinocchio/num/qp_before_fix.py`, used for the first submission of
  the paper) skipped the primal part of every such partial step; its result was then feasible but could differ from the
  minimiser by up to about 3e-4 (relative). When no active constraint blocks the step of a nearly dependent constraint,
  both versions rate the feasible set as numerically empty; that rating is conservative, the controllers count it and
  use their fallback, and it did not occur in the runs of the paper. `code/pinocchio/num/t13_qp_dependent.py` compares
  both versions with 50- and 60-digit references on nearly dependent constraints with large and very small violations
  and on adversarial families. With the corrected routine, three port energies of Table IV (Komatsu) differ from the
  first submission (listed in `code/excavators/RCX_TAC/paper/tables.py`); all other results are unchanged at the
  printed precision.
- The Komatsu e-fence, self-collision and mission runs are sensitive to perturbations of the controller computations at
  the level of round-off. Two versions of the QP routine that differ only in steps along round-off directions give
  solutions of single quadratic programs that differ by up to 1e-10 (relative); along the runs these differences
  changed the Komatsu mission entries of Table V by up to 14 % (port energy of the proposed layer: 95.0 J and 83.5 J)
  and the Komatsu e-fence entries of Table IV by up to 3.5 %. Repeated runs with the shipped code and the pinned
  packages reproduced the logged results exactly on the machine that produced them; on other hardware, round-off
  differences (for example in the linear-algebra library) can change these entries by similar amounts.

## Video

`videos/supplementary_video.mp4` shows the six approaches to the boom stroke limit (`1x`) and the dig-and-level
missions (`4x`) of Komatsu and HEAP, with the baseline on the left and the proposed layer on the right.
`python code/excavators/RCX_TAC/paper/paper_video.py komatsu cycles` renders such a video from the runs `full` and
`consistent` of a scenario recorded with `--render` (for the paper runs:
`python code/excavators/RCX_TAC/batch.py all paper --render`); both need the machine packages.

## Paper and citation

Please cite the paper when using this work:

Mehdi Heydari Shahna and Jouni Mattila. **Tracking Limitations and Rate-Consistent Safety Filtering for Constrained
Mechanical Systems.** Preprint, Research Square, 2026. https://doi.org/10.21203/rs.3.rs-11381118/v1

This work builds on RoboCompiler: Mehdi Heydari Shahna, Joongheon Kim, and Jouni Mattila. **RoboCompiler: Graph-Native
Compilation of Closed-Chain Robots for Consistent Modeling, Control, and Simulation.** 2026.
https://doi.org/10.48550/arXiv.2609.35717

## Acknowledgment

This work was supported by the Post Docs in Companies (PoDoCo) program in Finland.

## License

Released under the Apache License 2.0; see [LICENSE](LICENSE).
