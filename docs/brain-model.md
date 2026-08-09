# Brain model v0

This document fixes the semantics of the first neuron model. The choices are provisional and deliberately small enough to inspect against the C++ reference implementation.

## Topology

The brain occupies selected cells of a finite three-dimensional integer lattice. Empty cells carry no state. Neurons receive stable dense indices after their occupied cells are sorted by flattened grid position.

Two occupied cells are potential neighbors when their Manhattan distance is at most `connection_radius_steps`, which defaults to two. A filled interior therefore has 24 possible neighbors. The path does not need occupied intermediate cells.

Every allowed direction has its own potential connection. `A -> B` and `B -> A` have separate strengths, and either may be zero. A zero-strength connection is a dormant local contact that can grow through later activity; it is not a claim of biological structural growth.

Each neuron has a fixed excitatory or inhibitory polarity. Connection strengths are nonnegative, and the source polarity determines the sign of every outgoing signal. This is a simplified sign constraint, not a complete model of excitatory and inhibitory cell biology.

Input and output channels are ordered lists of arbitrary occupied positions. An input value is an external drive in potential units per second. An output exposes the selected neuron's current spike or recent activity trace. No action decoder, reward, or task meaning belongs to this model.

## State and units

For neuron `i`:

- `v_i` is normalized membrane potential.
- `s_i` is zero or one and records whether it fired on the most recent tick.
- `x_i` is a dimensionless recent-activity trace.
- `r_i` is the number of remaining refractory ticks.
- `b_i` is constant normalized potential drive per second.

For directed connection `i -> j`, `w_ij` is a nonnegative normalized potential jump and `e_ij` is a nonnegative eligibility trace. A signed global plasticity modulator `m` is dimensionless. Time values are seconds. State uses 32-bit floats except spikes, counters, and indices.

## Neuron update

The model is a normalized leaky integrate-and-fire simplification. For a neuron that is not refractory, one forward Euler step computes:

```text
v*_i = v_i
     + dt * ((v_rest - v_i) / tau_m + b_i + external_i + noise_i)
     + sum_j(polarity_j * w_ji * s_j_previous)
```

The synaptic term is an instantaneous potential jump and does not scale with `dt`. External, bias, and optional perturbation drives are continuous and do scale with `dt`.

The optional perturbation is deterministic signed noise sampled independently for every neuron from an explicit seed, tick, and neuron index. Its amplitude is a global non-local control signal. The model provides the mechanism without assigning it biological or task meaning.

If `v*_i >= threshold`, the neuron emits one spike, resets to `v_reset`, and suppresses the next `ceil(refractory_seconds / dt)` ticks. A refractory neuron remains at reset, ignores incoming drive, emits no spike, and decrements its counter. Recurrent signals use the previous tick's spikes, giving every connection one fixed-tick delay.

The implementation requires `dt <= tau_m` as a conservative forward Euler bound. Initial potentials equal rest; spikes, traces, and refractory counters begin at zero.

The leaky integrate-and-fire abstraction follows the standard circuit model summarized in [Gerstner et al., Neuronal Dynamics, section 1.3](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html). It does not model ion channels, conductances, dendrites, axons, morphology, or biological voltage units.

## Local eligibility

After all new spikes are known, activity traces update together:

```text
x_i_new = exp(-dt / tau_trace) * x_i_old + s_i_new
```

When plasticity is enabled, every directed connection first records whether recent source activity preceded a new target spike:

```text
e_ij_new = min(e_max,
    exp(-dt / tau_eligibility) * e_ij_old
    + excitatory_i * s_j_new * x_i_old)
```

This rule is causal and directed: recent activity in `i` followed by a spike in `j` marks `i -> j`, but the same event does not mark `j -> i`. A spike pair only records eligibility; it does not change connection strength by itself. Eligibility is bounded to prevent repeated activity from creating an unbounded later update.

Task-modulated eligibility is currently restricted to excitatory source neurons. Inhibitory strengths retain passive decay but otherwise remain fixed. This avoids treating an inhibitory spike before a rewarded target spike as evidence that stronger inhibition caused the target. A separate inhibitory plasticity or homeostatic rule remains future work.

The local trace is a deliberate engineering simplification of a synaptic eligibility trace. It omits anti-causal timing, separate potentiation and depression windows, cell-type-specific behavior, and biochemical dynamics. Delayed modulation of synaptic eligibility is broadly motivated by work such as [Izhikevich, 2007](https://www.jneurosci.org/content/27/30/8155), but the equation here is not that paper's model.

While plasticity is enabled, connection strength also weakens passively:

```text
w_ij_new = exp(-weakening_rate * dt) * w_ij_old
```

The default weakening rate is `0.0001 / s`, with a half-life of about 6931 simulated seconds. The previous faster rate erased quiet pathways over only a few short episodes. When plasticity is disabled, both eligibility and connection strength are frozen; neuron potentials, spikes, refractory counters, and activity traces still advance normally.

## Global modulation

A separate nonzero event applies one signed global modulator to every eligible connection:

```text
w_raw_ij = max(0, w_ij + learning_rate * m * e_ij)
```

Positive values strengthen eligible connections and negative values weaken them. The default learning rate is `0.05`, eligibility time constant is `1 s`, and maximum eligibility is `1`. A zero modulator leaves eligibility pending. Any nonzero modulator consumes and clears all eligibility after the update.

The eligibility calculation uses only local source and target state. The modulator is a third, global, non-local factor whose meaning belongs to the task or experiment. This supports delayed reward and punishment without gradients or backpropagation, but it is not by itself a biological model of dopamine, pain, hunger, or credit assignment.

## Connection budget

After a modulated update, each source owns a fixed outgoing strength budget `B`:

```text
scale = min(1, B / sum_j(w_raw_ij))
w_ij_new = scale * w_raw_ij
```

When the source is at capacity, strengthening one connection scales its siblings down. Negative modulation and passive weakening can leave budget unused, so neither is canceled by forced normalization. Initial nonzero strengths are seeded reproducibly and normalized to `initial_budget_fraction * B`; the remaining potential contacts start at zero.

The hard outgoing budget is an engineering competition mechanism loosely motivated by homeostatic synaptic scaling, where groups of biological synapses can be adjusted together to stabilize activity. It is not a direct implementation of that process; see [Turrigiano, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2834419/).

The budget limits outgoing strength, not firing rate. This first model may still become quiet or saturated. Adaptive thresholds, neuromodulation, structural changes, or another local homeostatic mechanism should be separate future model choices.

## Tick order

One tick has global phase boundaries:

1. Validate the ordered external inputs.
2. Sample any configured per-neuron perturbation drive.
3. Gather previous-tick spikes through the old connection strengths.
4. Update all potentials, refractory counters, and new spikes.
5. If plasticity is enabled, decay strengths and update eligibility from old source traces and new target spikes.
6. Update all activity traces.
7. Publish the new spikes and increment the tick.

A modulation event occurs outside the fixed neural tick. Updated strengths first affect signal transmission on the following tick. Synchronous update order is part of the model.

## Reset behavior

`reset()` restores resting activity and the model's seeded initial strengths. `reset_activity()` clears potentials, spikes, activity traces, eligibility traces, refractory counters, and time while retaining learned strengths. An explicit state reset is available for tests, replay, and experiment restoration.

Invalid external inputs fail before state changes. If finite inputs still cause numerical overflow during a step, the simulator fails loudly and requires an explicit reset before it can advance again.

## Backend representation

The immutable model stores outgoing and incoming compressed sparse row arrays. Mutable state is stored as separate contiguous arrays for potentials, activity traces, eligibility traces, strengths, refractory counters, and spikes. Every backend must use the same topology, initialization, decay factors, modulation rule, and tick phases.

This layout is ready for one CUDA thread per neuron during dynamics and one thread per source during plasticity without atomics. It does not imply that one small brain is faster on a GPU; CUDA is expected to matter when networks are large or many evolving brains are batched.
