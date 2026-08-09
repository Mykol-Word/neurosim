# Adaptive controller and dungeon adapter

This document records the current connection between the plastic spiking brain and the dungeon. It is a provisional experimental baseline, not a claim of biological fidelity, general intelligence, or solved navigation.

## General boundary

`AdaptiveController` is task-neutral. It receives only a fixed count of normalized input channels, a fixed count of anonymous output channels, a control timestep, a set of valid output exploration patterns, signed scalar modulation, and seeds. It has no knowledge of pellets, walls, hunger, movement directions, scores, episode endings, or dungeon generation.

The controller builds the default brain as a radius-`10.65` sphere in a `23 x 23 x 23` lattice. This occupies 5,041 cells: 837 sensory neurons, four output neurons, and 4,200 unassigned internal neurons with the default dungeon interface. Sensor and output neurons are selected reproducibly without replacement from all occupied neurons using the brain seed. Their positions have no semantic ordering or distance constraint. Output neurons receive the same seeded polarity and bias policy as every other neuron: 20 percent inhibitory polarity and one bias sampled from the common `30` to `47.5` range. There is no output-only bias or forced output polarity.

Each normalized sensory value is validated in `[0, 1]` and multiplied by the same default drive of `500`. The controller does not apply channel-specific floors, gains, signs, or interpretations. A task can change its observation and action representation without changing controller construction or learning code.

One normally silent external drive channel targets each output neuron. These channels let exploration express a supplied anonymous output pattern; they are not part of the task observation. When no task codebook is supplied, the controller uses one-hot patterns.

The brain uses its own fixed timestep, defaulting to `1/480` second. The controller accumulates neural time independently from the task control rate; the default 120 Hz dungeon therefore advances four neural ticks per control tick. Changing a task's control rate does not change neuron, decay, refractory, or plasticity timing. The controller returns only output activity, spike counts, and the current exploration pattern. The task adapter decides what those outputs mean.

## Dungeon observation

Vision is a circle with a fixed radius in world units, defaulting to 24. It is independent of the viewer window, camera zoom, and rendering. Visibility is radial and does not stop at walls.

The default dungeon places four to eight pellets in every room and can contain at most 160 pellets. The observation reserves 160 slots and never truncates visible food. Live pellets are sorted by distance, with dungeon pellet index breaking ties. Each slot contains `present`, `left`, `right`, `up`, and `down`. Direction values are the corresponding nonnegative relative offset divided by the vision radius and clamped to `[0, 1]`; directions that do not apply are zero. Inactive slots are all zero.

Thirty-two north-up rays report normalized proximity to the first exit from connected navigable floor. Four opponent-coded channels report velocity, and one channel reports normalized time since food relative to the starvation deadline. Score, total remaining food, episode age, and unseen pellets are not exposed. The default dungeon observation has 837 values.

This encoder is deliberately dungeon-specific and supplies semantic food locations rather than learned visual recognition. `BrainAgent` owns it and only passes the resulting normalized vector across the generic controller boundary.

## Dungeon action

The dungeon assigns four ordered controller outputs to `up`, `down`, `left`, and `right`. Opposing directions compete by activity on each axis. An output must meet the configured activity threshold and exceed its opponent; horizontal and vertical winners are independent, so diagonal movement is possible.

The dungeon supplies eight nonzero cardinal and diagonal patterns for exploration. This codebook defines valid actions but does not place outputs near matching sensors or expose action meaning to plasticity. During an exploration period, the selected pattern overrides the current network action and drives its output neurons.

## Learning and exploration

Local causal firing records bounded one-second synaptic eligibility. Co-firing alone does not alter connection strength. The generic controller accepts a signed scalar outcome and applies it to recent eligibility without knowing its source.

The dungeon maps each eaten pellet to `+1`. A wall event contributes `-0.005` and is limited to once every 0.25 seconds so one held collision cannot apply punishment at 120 Hz. Hunger is an observation channel only; it does not schedule exploration or inject noise.

Exploration is seeded motor activity on a fixed schedule. By default, one valid pattern is active for 0.25 seconds every 0.75 seconds, starting at the beginning of an episode. Pattern selection is reproducible from the brain and episode seeds. Its timing and selection are independent of observations, hunger, food, walls, termination, and modulation.

This is a three-factor engineering rule: local source and target activity supplies eligibility, while the task supplies non-local signed modulation. It is gradient-free and has no backpropagation, but the global scalar outcome and externally driven exploration are engineering devices rather than locally biological mechanisms.

## Training and validation

Every training episode uses an ordinary procedural dungeon. Environment seeds advance from the requested first seed; there are no controlled opening episodes or task-specific training layouts.

Validation runs before training, every 25 episodes by default, and after a final partial interval. It uses fixed procedural dungeon seeds with both learning and exploration disabled, so validation never changes connection strengths. Those seeds select the saved checkpoint and are therefore not an untouched test set. The best frozen validation result is retained instead of automatically selecting the final drifting state.

The runner records versioned model, task, timing, backend, precision, seed, controller, dungeon adapter, and starting learned-state fingerprint metadata before episode results. Per-episode JSON includes score, wall hits, exploration and inactive steps, direction changes, movement alignment toward visible food, whole-brain and output firing rates, weight distribution, outgoing entropy, and elapsed time.

The default checkpoint is `runs/brain-<seed>.nsbrain`. It contains a version, brain seed, model fingerprint, and learned strengths. Loading rejects a different topology, interface, seed, or relevant model configuration. Checkpoints made before the generic random controller interface was introduced are therefore incompatible and must be retrained. Transient voltage, spikes, traces, eligibility, and exploration state are intentionally not persisted.

## Reset behavior

Episode resets clear potentials, spikes, activity traces, eligibility, refractory counters, time, and exploration state while preserving learned strengths. Frozen validation performs the same activity reset and additionally disables decay, eligibility recording, modulation, and exploration. Learning state belongs to the generic controller, so direct modulation is also inert while it is frozen.

## Known limitations

The observation exposes semantic food locations and can see food through walls, so direct pursuit can conflict with navigation. Four singleton outputs are less robust than output populations. Random sensor and output placement removes task-specific connectivity guarantees, but it can also make a particular seeded controller difficult to train through a strictly local graph.

The reusable boundary does not remove the need for a task adapter. A new task must still define a normalized observation, an output-to-action decoder, valid exploration patterns, and a signed outcome policy. None of those meanings should enter the generic controller. Learning quality remains an empirical question and must be compared across multiple seeds and an untouched test set beyond checkpoint-selection validation.
