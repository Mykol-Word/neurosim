# Brain model v0

This document fixes the semantics of the first neuron model. The choices are provisional and deliberately small enough to inspect against the C++ reference implementation.

## Topology

The brain occupies selected cells of a finite three-dimensional integer lattice. Empty cells carry no state. Neurons receive stable dense indices after their occupied cells are sorted by flattened grid position.

Two occupied cells are potential neighbors when their Manhattan distance is at most `connection_radius_steps`, which defaults to two. A filled interior therefore has 24 possible neighbors. The path does not need occupied intermediate cells.

Every allowed direction has its own potential connection. `A -> B` and `B -> A` have separate strengths, and either may be zero. A zero-strength connection is a dormant local contact that can grow through later activity; it is not a claim of biological structural growth.

Each neuron has a fixed excitatory or inhibitory polarity. Connection strengths are nonnegative, and the source polarity determines the sign of every outgoing signal. This is a simplified sign constraint, not a complete model of excitatory and inhibitory cell biology.

Input and output channels are ordered lists of arbitrary occupied positions. An input value is a normalized external drive in potential units per second. An output exposes the selected neuron's current spike or recent activity trace. No action decoder, reward, or task meaning belongs to this model.

## State and units

For neuron `i`:

- `v_i` is normalized membrane potential.
- `s_i` is zero or one and records whether it fired on the most recent tick.
- `x_i` is a dimensionless recent-activity trace.
- `r_i` is the number of remaining refractory ticks.
- `b_i` is constant normalized potential drive per second.

For directed connection `i -> j`, `w_ij` is a nonnegative normalized potential jump. Time values are seconds. State uses 32-bit floats except spikes, counters, and indices.

## Neuron update

The model is a normalized leaky integrate-and-fire simplification. For a neuron that is not refractory, one forward Euler step computes:

```text
v*_i = v_i
     + dt * ((v_rest - v_i) / tau_m + b_i + external_i)
     + sum_j(polarity_j * w_ji * s_j_previous)
```

The synaptic term is an instantaneous potential jump and does not scale with `dt`. External and bias drives are continuous and do scale with `dt`.

If `v*_i >= threshold`, the neuron emits one spike, resets to `v_reset`, and suppresses the next `ceil(refractory_seconds / dt)` ticks. A refractory neuron remains at reset, ignores incoming drive, emits no spike, and decrements its counter. Recurrent signals use the previous tick's spikes, giving every connection one fixed-tick delay.

The implementation requires `dt <= tau_m` as a conservative forward Euler bound. Initial potentials equal rest; spikes, traces, and refractory counters begin at zero.

The leaky integrate-and-fire abstraction follows the standard circuit model summarized in [Gerstner et al., Neuronal Dynamics, section 1.3](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html). It does not model ion channels, conductances, dendrites, axons, morphology, or biological voltage units.

## Plasticity

After all new spikes are known, activity traces update together:

```text
x_i_new = exp(-dt / tau_trace) * x_i_old + s_i_new
```

Each local directed connection then computes one symmetric coactivity value:

```text
c_ij = s_i_new * x_j_new
     + s_j_new * x_i_new
     - s_i_new * s_j_new

w_raw = max(0,
    exp(-weakening_rate * dt) * w_ij
    + learning_rate * c_ij)
```

The subtraction counts a simultaneous spike pair once. Nearby firing in either temporal order strengthens the connection, while inactivity exponentially weakens it. The rule uses only source activity, target activity, and the local weight. It has no gradient, backpropagation, target, loss, or global task signal.

This is a custom symmetric trace-based Hebbian rule, not spike-timing-dependent plasticity. Hebbian locality and joint activity are described in [Gerstner et al., Neuronal Dynamics, sections 19.1 and 19.2](https://neuronaldynamics.epfl.ch/online/Ch19.S1.html). Biological experiments show that real plasticity can instead depend asymmetrically on precise spike timing, initial strength, and cell type, as in [Bi and Poo, 1998](https://pubmed.ncbi.nlm.nih.gov/9852584/).

## Connection budget

After raw updates, each source owns a fixed outgoing strength budget `B`:

```text
scale = min(1, B / sum_j(w_raw_ij))
w_ij_new = scale * w_raw_ij
```

When the source is at capacity, strengthening one connection scales its siblings down. Inactivity can leave budget unused, so global weakening is not canceled by forced normalization. Initial nonzero strengths are seeded reproducibly and normalized to `initial_budget_fraction * B`; the remaining potential contacts start at zero.

The hard outgoing budget is an engineering competition mechanism loosely motivated by homeostatic synaptic scaling, where groups of biological synapses can be adjusted together to stabilize activity. It is not a direct implementation of that process; see [Turrigiano, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2834419/).

The budget limits outgoing strength, not firing rate. This first model may still become quiet or saturated. Adaptive thresholds, neuromodulation, structural changes, or another local homeostatic mechanism should be separate future model choices.

## Tick order

One tick has global phase boundaries:

1. Validate the ordered external inputs.
2. Gather previous-tick spikes through the old connection strengths.
3. Update all potentials, refractory counters, and new spikes.
4. Update all activity traces.
5. Update every directed strength and enforce each source budget.
6. Publish the new spikes and increment the tick.

Updated strengths first affect signal transmission on the following tick. Synchronous update order is part of the model.

## Reset behavior

`reset()` restores resting activity and the model's seeded initial strengths. `reset_activity()` clears potentials, spikes, traces, refractory counters, and time while retaining learned strengths. An explicit state reset is available for tests, replay, and experiment restoration.

Invalid external inputs fail before state changes. If finite inputs still cause numerical overflow during a step, the simulator fails loudly and requires an explicit reset before it can advance again.

## Backend representation

The immutable model stores outgoing and incoming compressed sparse row arrays. Mutable state is stored as separate contiguous arrays for potentials, traces, strengths, refractory counters, and spikes. The CPU and CUDA backends use the same topology, initialization, decay factors, and tick phases.

This layout is ready for one CUDA thread per neuron during dynamics and one thread per source during plasticity without atomics. It does not imply that one small brain is faster on a GPU; CUDA is expected to matter when networks are large or many evolving brains are batched.
