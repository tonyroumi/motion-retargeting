# Prepared CMU ↔ AMASS LAFAN-style experiment

`configs/methods/pan_cmu_amass_lafan_style.yaml` is an **explicit new human/human
experiment**, not a LAFAN1/Dog reproduction. It uses the reference's representation,
three-part network and objective wherever those are physically defined, rather
than running the old generic PAN objective with reference-looking hyperparameters.
No full training was launched to prepare it. Identical convergence is not claimed.

## Use after reviewing the differences below

From an installed checkout with the raw data and SMPL-H assets available:

```sh
# Full data/model preflight and config/provenance snapshot, ZERO optimizer steps:
python scripts/train.py --config configs/methods/pan_cmu_amass_lafan_style.yaml \
  --prepare-only --device cpu

# Actual training command, provided for later use, NOT executed during preparation:
python scripts/train.py --config configs/methods/pan_cmu_amass_lafan_style.yaml
```

Default input roots are `data/raw/amass/CMU` and `data/raw/cmu` (recursive). AMASS
assets default to `data/assets/body_models/smplh/{male,female}/model.npz`. Each
`human_reference_style` block can set `directory`; AMASS can also set `body_models`
to the parent directory containing `smplh`. Use explicit absolute paths when
reading data from another checkout. Neither preflight nor training reads/writes
the old generic `.pt` collection caches. Preflight does construct the full datasets
and model and writes a new log directory, so it is not a cheap inventory command.
Fresh extraction/statistics run again on the next invocation.

## Available data and deliberately chosen split

The inspected local AMASS tree contains **only its CMU subset**: the BMLRub folder
is empty. There are 2082 AMASS clips (2035 distinct clip stems), 2514 raw CMU AMC
clips, and **2005 common clip stems**. The domains are strongly correlated fits of
largely the same source recordings, not independent human/dog motion distributions.
Matching names alone does not prove frame alignment or exact paired ground truth.

The YAML fixes a shared held-out CMU session-ID list. It was selected from roughly
every fifth sorted AMASS actor directory, then conservatively closes the known
multi-actor Justin/Rory sessions **18–23** into one holdout group. Both actors in
those AMASS folders are retained, not deduplicated by ambiguous clip stems. Both
domains must use the same holdout IDs; partial splits of sessions 18–23 are rejected.
This prevents known shared-recording/actor leakage between training and held-out
motions across these domains. It is **not** a claim that every person's identity
across all historical CMU sessions has been independently authenticated.

This is not the reference's action-selected subject-1 split. Numeric CMU filenames
do not supply a verified equivalent aiming/locomotion selection; all available
clips in the selected sessions are used rather than inventing action labels.

| Inventory / projected windows | AMASS-CMU | ASF/AMC CMU |
|---|---:|---:|
| Train / test clips | 1650 / 432 | 2053 / 461 |
| Train / test session IDs present | 76 / 23 | 86 / 26 |
| 64/20 train / test windows | 35352 / 8630 | 35725 / 9355 |
| Too-short clips, train / test | 129 / 31 | 201 / 36 |
| Source rate | 1587 clips at 120 Hz, 495 at 60 Hz | 120 Hz |
| Retained raw motion-channel storage | ~1.93 GiB | ~2.06 GiB |

Window projections use all AMASS pose headers and AMC final frame markers (assuming
standard sequential frame numbering), not full-corpus motion conversion. All 112
used ASF skeletons were parsed and match the fixed topology. All 2082 AMASS headers
have 156 pose values and 16 betas; genders are 1557 male / 525 female, with both body
models available. Bounded real conversion/optimizer checks were run; **not every
motion frame in this larger corpus has been preprocessed or trained**. Actual
preparation records the resulting counts and each clip's path/rate/frame/window
counts in `data_provenance.json`; inspect that file before a long run.

## Matched and adapted physical conventions

- Generic adapters still output coherent **metres**. The experiment reads their
  original motion channels in float64, then converts offsets/root positions to cm
  before network preprocessing. AMASS's installed SMPL-H rest-body model is FP32;
  promoting its output does not recover extra model precision.
- Sample at 30 Hz (120→30 every fourth frame, 60→30 every second), strict reference
  `start + 64 < sampled_frame_count - 1`, stride20, one lookahead displacement.
- Reference LAFAN1 heading comes from authored root-local +Y. That is not forward
  for SMPL/ASF. Here root-local forward is **cross(rest head-up, rest right-minus-left
  hip direction)**, transformed by root rotation and projected onto XZ. This uses
  physical anatomy rather than guessing that both adapters' root +X/+Y axes agree.
- The human experiment uses `atan2` to form a unit yaw quaternion, including an exact
  180° turn. The published quaternion-between/epsilon formula degenerates at that
  antiparallel direction. A truly vertical/undefined forward projection fails
  explicitly, rather than inventing heading. Published LAFAN1/Dog behavior is unchanged.
- Remove heading, canonical +Z facing, positive-w local quaternions, local XYZ
  displacement per sampled frame, within-window yaw increments with the last one
  repeated. No dog heading smoothing is applied to humans.
- Subject/fitted-actor banks; domain-wide **train-plus-test** population mean/std,
  replacing only exact zero std. This intentionally retains the reference's
  test-statistics leakage; it is not a train-only-normalization benchmark.
- Training FK integrates heading-local displacement **inclusively**, without yaw
  restoration or root height. These coordinates are not the original world trajectory.

Coordinate checks on 65 real frames each compare whole-skeleton FK against an
independent original-source FK followed by the documented global axis rotation:
maximum error **8.89e-16 m AMASS / 4.45e-16 m CMU**. These establish coordinate
conversion consistency, not agreement of SMPL fitting with ASF anatomy.

## Topology, masks and objective

The three parts retain LAFAN-style semantics: **legs, trunk, head**. Arms remain
outside correspondence/reconstruction selections, but are visible to the position
discriminator. Root rotation and root velocity remain shared by all parts.

| Quantity | Reference LAFAN1 / Dog | Experiment AMASS / CMU |
|---|---:|---:|
| Rotation joints | 22 / 21 | 22 / 23 |
| Network channels | 92 / 88 | 92 / 96 |
| FK/discriminator joints | 27 / 27 | 22 / 28 |
| Position-discriminator channels | 81 / 81 | 66 / 84 |
| Selected rotation joints | 14 / 19 | 14 / 15 |
| Selected position joints | 17 / 23 | 14 / 18 |
| Latent parts | 3 / 3 | 3 / 3 |

CMU's existing `head_end` is removed from rotation tokens and restored with identity
rotation in FK. Its actual ASF toe and wrist terminal bone vectors add four more
identity-rotation End Sites: **five real endpoints total**, no invented lengths.
The extra CMU neck joint explains its one-joint difference from LAFAN1 human.
Head/toe endpoints join the selected positional loss; wrist endpoints are used by
the full-position discriminator, just as reference arm endpoints are.

The retained 22-joint SMPL-H body has no corresponding head/toe/hand-tip End Sites.
They are **not fabricated from arbitrary lengths or mesh vertices**. This has real
consequences: terminal AMASS wrist orientations have no FK descendants and are
outside the reference-style selected quaternion loss, so those outputs are not
constrained like the reference's End-Site-equipped wrists. Extra hand/tip semantics
would require a separately defined experiment, not an equivalence claim.

The objective is the reference `norm_rec` algebra, with those explicit joint
selections and cm coordinates:

```
selected normalized-quaternion MSE
+ 100 * normalized root XYZ+yaw MSE
+ .01 * selected integrated-position MSE
+ .001 * (selected cycle-position MSE + latent MSE)
+ raw-score L2-GAN loss
+ 1000 * (mapped root-vector MSE + denormalized yaw MSE)
```

There are exactly two cross routes using destination-batch skeletons, no self
cycles, no replay, no legacy height normalization and no five-limb/contact objective.
**One additional singularity rule is explicit:** exact zero root speed maps to the
zero vector, with finite gradients. Every nonzero speed uses the reference formula
unchanged. The published LAFAN/Dog path still retains its original zero-speed NaN.
Differential tests execute upstream `backward_G` on the experimental topologies and
compare each objective term and generator/discriminator gradients away from that
intentional zero-speed exception.

## Model, optimizer and update exposure

The template uses 1 transformer layer / 1 head, embedding32/hidden512/FF256,
zero dropout, kernel15, linear upsampling, three latent parts, batch128, and Adam
G/D learning rates 1e-4 with (0.9,0.999) betas. Seed0 controls Python/NumPy/Torch
initialization and separate shuffled loader generators; the historical reference
seed/initialization is unknown. The extra masked root-offset input column remains
a known skeleton-encoder shape/initialization difference.

`primary_recycle` trains on all full AMASS batches, keeping the CMU iterator alive
across epochs and reshuffling it only on exhaustion. No temporal reversal. It does
not reproduce the upstream extra-first-batch epoch-boundary bookkeeping quirk.
At projected counts: **276 updates / 35328 AMASS samples per epoch**, versus the
current reference human archive's 51 full batches. CMU has 279 full batches per
shuffle. Thus copying only an epoch count would change exposure substantially.

The configured **335001-update cap** is the actual observed `step` counter in
**both installed epoch-5000 reference optimizer files** (learning rate1e-4,
betas0.9/0.999). Important discrepancy: today's 6566-window human archive, batch128
and current training loop imply **255001** updates at that label. The historical
training data/resume history is therefore not established. The template targets
the **observed optimizer counter**, not an invented reconstruction of that history.
At the projected human corpus size it stops during epoch1214, not epoch5001;
5001 is only an epoch ceiling. It checkpoints every33500 updates and on completion.
Matching update count still does not match per-clip exposure or convergence.

Validation retains the existing equally-sized prefix policy: at projected counts,
8630 windows per domain, so 725 trailing CMU held-out windows are not in routine
validation batches. This is not an exhaustive or paired retargeting benchmark.
Config/source revision/data counts/manifests/axes are snapshotted in each new run.
Checkpoints record protocol, selections, statistics and skeleton fingerprints and
reject generic or LAFAN/Dog checkpoints. Mid-epoch saves are identified as such;
resume does **not** restore sampler/RNG state and is not bitwise replay.

## Validation scope

Bounded tests cover both raw adapter types, physical units, heading/half-turn and
zero-speed behavior, splits, real ASF endpoints, domain statistics, exact routes,
reference objective/gradient algebra, finite CPU optimizer updates, checkpoint
boundaries, primary recycling, exact update caps, and headless coordinate plots.
Real checks used prefixes of sessions02 and39, not a full human training run.
The broader targeted suite also retains the LAFAN1/Dog corpus/archive parity tests.
See `docs/visualize.md` for inspecting source, adapter-world and training spaces.
