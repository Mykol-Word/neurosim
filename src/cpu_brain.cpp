#include "neurosim/cpu_brain.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace neurosim
{

using namespace std;

namespace
{

// returns whether a scalar is finite
bool finite(float value)
{
    return isfinite(value);
}

// returns the signed effect of one source neuron's outgoing signal
float polarity_sign(NeuronPolarity polarity)
{
    return polarity == NeuronPolarity::Excitatory ? 1.0f : -1.0f;
}

// mixes a neuron, tick, and perturbation seed into a stable random value
uint64_t mix_noise(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

// returns one deterministic signed perturbation drive
float noise_drive(BrainPerturbation perturbation, uint64_t tick, uint32_t neuron)
{
    if (perturbation.noise_drive_amplitude == 0.0f)
    {
        return 0.0f;
    }

    const uint64_t keyed = perturbation.noise_seed
        ^ ((tick + 1U) * 0xd1b54a32d192ed03ULL)
        ^ ((static_cast<uint64_t>(neuron) + 1U) * 0x94d049bb133111ebULL);
    const float unit = static_cast<float>(mix_noise(keyed) >> 40U) * (1.0f / 16777216.0f);
    return (unit * 2.0f - 1.0f) * perturbation.noise_drive_amplitude;
}

}

// creates a simulator with fresh state from a compiled model
CpuBrainSimulator::CpuBrainSimulator(BrainModel model)
    : model_(move(model))
{
    validate_brain_model(model_);
    reset();
}

// restores activity and learned weights to their initial values
void CpuBrainSimulator::reset()
{
    state_ = make_brain_state(model_);
    synaptic_drives_.assign(model_.positions.size(), 0.0f);
    next_spikes_.assign(model_.positions.size(), 0U);
    requires_reset_ = false;
}

// clears transient activity while preserving learned connection strengths
void CpuBrainSimulator::reset_activity()
{
    if (requires_reset_)
    {
        throw logic_error("brain numerical failure requires a full state reset");
    }

    const float resting_potential = model_.config.dynamics.resting_potential;
    fill(state_.membrane_potentials.begin(), state_.membrane_potentials.end(), resting_potential);
    fill(state_.activity_traces.begin(), state_.activity_traces.end(), 0.0f);
    fill(state_.connection_eligibilities.begin(), state_.connection_eligibilities.end(), 0.0f);
    fill(state_.refractory_ticks.begin(), state_.refractory_ticks.end(), 0U);
    fill(state_.spikes.begin(), state_.spikes.end(), uint8_t{0});
    fill(synaptic_drives_.begin(), synaptic_drives_.end(), 0.0f);
    fill(next_spikes_.begin(), next_spikes_.end(), uint8_t{0});
    state_.tick = 0U;
}

// replaces the complete mutable state after validating it
void CpuBrainSimulator::reset(BrainState state)
{
    validate_brain_state(model_, state);
    state_ = move(state);
    synaptic_drives_.assign(model_.positions.size(), 0.0f);
    next_spikes_.assign(model_.positions.size(), 0U);
    requires_reset_ = false;
}

// advances every neuron and optionally records plasticity by one fixed tick
void CpuBrainSimulator::step(
    span<const float> input_drives,
    BrainPerturbation perturbation,
    bool plasticity_enabled)
{
    advance(input_drives, 1U, perturbation, plasticity_enabled, {});
}

// advances multiple fixed ticks after validating shared inputs once
BrainAdvanceSummary CpuBrainSimulator::advance(
    span<const float> input_drives,
    uint32_t tick_count,
    BrainPerturbation perturbation,
    bool plasticity_enabled,
    span<float> output_activity)
{
    if (requires_reset_)
    {
        throw logic_error("brain simulator must be reset after a failed numerical step");
    }

    if (input_drives.size() != model_.input_neurons.size())
    {
        throw invalid_argument("brain input count does not match the configured input channels");
    }

    if (tick_count == 0U)
    {
        throw invalid_argument("brain advance tick count must be positive");
    }

    if (!output_activity.empty() && output_activity.size() != model_.output_neurons.size())
    {
        throw invalid_argument("brain output activity count does not match the configured outputs");
    }

    for (const float input : input_drives)
    {
        if (!finite(input))
        {
            throw invalid_argument("brain input drives must be finite");
        }
    }

    if (!finite(perturbation.noise_drive_amplitude)
        || perturbation.noise_drive_amplitude < 0.0f)
    {
        throw invalid_argument("brain perturbation amplitude must be finite and nonnegative");
    }

    if (tick_count > numeric_limits<uint64_t>::max() - state_.tick)
    {
        throw overflow_error("brain tick counter overflow");
    }

    BrainAdvanceSummary summary;

    for (uint32_t tick = 0; tick < tick_count; ++tick)
    {
        const BrainAdvanceSummary tick_summary = advance_tick(
            input_drives,
            perturbation,
            plasticity_enabled);
        summary.neuron_spikes += tick_summary.neuron_spikes;
        summary.output_spikes += tick_summary.output_spikes;
    }

    for (uint32_t output = 0; output < output_activity.size(); ++output)
    {
        output_activity[output] = state_.activity_traces[model_.output_neurons[output]];
    }

    return summary;
}

// advances one already-validated fixed tick and counts its spikes
BrainAdvanceSummary CpuBrainSimulator::advance_tick(
    span<const float> input_drives,
    BrainPerturbation perturbation,
    bool plasticity_enabled)
{
    BrainAdvanceSummary summary;

    const NeuronDynamicsConfig& dynamics = model_.config.dynamics;
    fill(synaptic_drives_.begin(), synaptic_drives_.end(), 0.0f);

    for (uint32_t source = 0; source < model_.positions.size(); ++source)
    {
        if (state_.spikes[source] == 0U)
        {
            continue;
        }

        const float signal = polarity_sign(model_.polarities[source]);

        for (uint32_t edge = model_.outgoing_offsets[source];
            edge < model_.outgoing_offsets[source + 1U]; ++edge)
        {
            const uint32_t target = model_.outgoing_targets[edge];
            synaptic_drives_[target] += signal * state_.connection_strengths[edge];
        }
    }

    for (uint32_t neuron = 0; neuron < model_.positions.size(); ++neuron)
    {
        next_spikes_[neuron] = 0U;

        if (state_.refractory_ticks[neuron] > 0U)
        {
            --state_.refractory_ticks[neuron];
            state_.membrane_potentials[neuron] = dynamics.reset_potential;
            continue;
        }

        float external_drive = 0.0f;
        const int32_t input_slot = model_.input_slots_by_neuron[neuron];

        if (input_slot >= 0)
        {
            external_drive = input_drives[static_cast<size_t>(input_slot)];
        }

        const float potential = state_.membrane_potentials[neuron];
        const float continuous_drive = (dynamics.resting_potential - potential)
                / dynamics.membrane_time_constant_seconds
            + model_.bias_drives[neuron]
            + external_drive
            + noise_drive(perturbation, state_.tick, neuron);
        const float next_potential = potential
            + dynamics.fixed_time_step_seconds * continuous_drive
            + synaptic_drives_[neuron];

        if (!finite(next_potential))
        {
            requires_reset_ = true;
            throw runtime_error("brain dynamics produced a non-finite membrane potential");
        }

        if (next_potential >= dynamics.firing_threshold)
        {
            next_spikes_[neuron] = 1U;
            ++summary.neuron_spikes;
            state_.membrane_potentials[neuron] = dynamics.reset_potential;
            state_.refractory_ticks[neuron] = model_.refractory_ticks;
        }
        else
        {
            state_.membrane_potentials[neuron] = next_potential;
        }
    }

    const PlasticityConfig& plasticity = model_.config.plasticity;

    if (plasticity_enabled)
    {
        for (uint32_t edge = 0; edge < model_.outgoing_targets.size(); ++edge)
        {
            const float strength = state_.connection_strengths[edge] * model_.weight_decay;
            const float eligibility = state_.connection_eligibilities[edge]
                * model_.eligibility_decay;

            if (!finite(strength) || strength < 0.0f
                || !finite(eligibility) || eligibility < 0.0f)
            {
                requires_reset_ = true;
                throw runtime_error("brain plasticity produced invalid local state");
            }

            state_.connection_strengths[edge] = strength;
            state_.connection_eligibilities[edge] = eligibility;
        }

        for (uint32_t target = 0; target < model_.positions.size(); ++target)
        {
            if (next_spikes_[target] == 0U)
            {
                continue;
            }

            for (uint32_t incoming = model_.incoming_offsets[target];
                incoming < model_.incoming_offsets[target + 1U]; ++incoming)
            {
                const uint32_t edge = model_.incoming_edges[incoming];
                const uint32_t source = model_.edge_sources[edge];

                if (model_.polarities[source] == NeuronPolarity::Inhibitory)
                {
                    continue;
                }

                const float eligibility = state_.connection_eligibilities[edge]
                    + state_.activity_traces[source];

                if (!finite(eligibility) || eligibility < 0.0f)
                {
                    requires_reset_ = true;
                    throw runtime_error("brain plasticity produced invalid local state");
                }

                state_.connection_eligibilities[edge] = min(
                    plasticity.maximum_eligibility,
                    eligibility);
            }
        }
    }

    for (uint32_t neuron = 0; neuron < model_.positions.size(); ++neuron)
    {
        state_.activity_traces[neuron] = state_.activity_traces[neuron] * model_.trace_decay
            + static_cast<float>(next_spikes_[neuron]);

        if (!finite(state_.activity_traces[neuron]))
        {
            requires_reset_ = true;
            throw runtime_error("brain dynamics produced a non-finite activity trace");
        }
    }

    state_.spikes.swap(next_spikes_);

    for (const uint32_t output : model_.output_neurons)
    {
        summary.output_spikes += state_.spikes[output];
    }

    ++state_.tick;
    return summary;
}

// applies one signed global event to eligible local connections
void CpuBrainSimulator::apply_plasticity_modulation(float modulator)
{
    if (requires_reset_)
    {
        throw logic_error("brain simulator must be reset after a failed numerical step");
    }

    if (!finite(modulator))
    {
        throw invalid_argument("brain plasticity modulator must be finite");
    }

    if (modulator == 0.0f)
    {
        return;
    }

    const PlasticityConfig& plasticity = model_.config.plasticity;

    for (uint32_t source = 0; source < model_.positions.size(); ++source)
    {
        float strength_sum = 0.0f;

        for (uint32_t edge = model_.outgoing_offsets[source];
            edge < model_.outgoing_offsets[source + 1U]; ++edge)
        {
            const float delta = plasticity.learning_rate * modulator
                * state_.connection_eligibilities[edge];
            const float raw_strength = state_.connection_strengths[edge] + delta;

            if (!finite(delta) || !finite(raw_strength))
            {
                requires_reset_ = true;
                throw runtime_error("brain modulation produced a non-finite connection strength");
            }

            const float strength = max(0.0f, raw_strength);
            state_.connection_strengths[edge] = strength;
            strength_sum += strength;
        }

        if (!finite(strength_sum))
        {
            requires_reset_ = true;
            throw runtime_error("brain modulation produced a non-finite outgoing strength sum");
        }

        if (strength_sum > plasticity.outgoing_strength_budget)
        {
            const float scale = plasticity.outgoing_strength_budget / strength_sum;

            for (uint32_t edge = model_.outgoing_offsets[source];
                edge < model_.outgoing_offsets[source + 1U]; ++edge)
            {
                state_.connection_strengths[edge] *= scale;
            }
        }
    }

    fill(state_.connection_eligibilities.begin(), state_.connection_eligibilities.end(), 0.0f);
}

// replaces learned strengths and clears transient activity
void CpuBrainSimulator::load_connection_strengths(span<const float> strengths)
{
    validate_connection_strengths(model_, strengths);
    BrainState state = make_brain_state(model_);
    state.connection_strengths.assign(strengths.begin(), strengths.end());
    reset(move(state));
}

// returns a host copy of the learned connection strengths
vector<float> CpuBrainSimulator::copy_connection_strengths() const
{
    return state_.connection_strengths;
}

// returns a full host snapshot for diagnostics outside the hot path
BrainState CpuBrainSimulator::copy_state() const
{
    return state_;
}

// returns the immutable compiled model
const BrainModel& CpuBrainSimulator::model() const
{
    return model_;
}

// returns the cpu execution backend
BrainBackend CpuBrainSimulator::backend() const
{
    using enum BrainBackend;
    return Cpu;
}

// returns the current mutable state as read-only data
const BrainState& CpuBrainSimulator::state() const
{
    return state_;
}

// returns whether one ordered output neuron fired this tick
bool CpuBrainSimulator::output_spike(uint32_t output_channel) const
{
    if (output_channel >= model_.output_neurons.size())
    {
        throw out_of_range("brain output channel is out of range");
    }

    return state_.spikes[model_.output_neurons[output_channel]] != 0U;
}

// returns the recent activity trace of one ordered output neuron
float CpuBrainSimulator::output_activity(uint32_t output_channel) const
{
    if (output_channel >= model_.output_neurons.size())
    {
        throw out_of_range("brain output channel is out of range");
    }

    return state_.activity_traces[model_.output_neurons[output_channel]];
}

// creates one cpu simulator for the requested cpu execution configuration
unique_ptr<BrainSimulator> make_cpu_brain_simulator(
    BrainModel model,
    const BrainExecutionConfig& execution)
{
    using enum BrainBackend;

    if (execution.backend != Cpu)
    {
        throw invalid_argument("cpu brain factory received a non-cpu execution request");
    }

    if (execution.cuda_device != 0)
    {
        throw invalid_argument("cpu brain execution cannot select a cuda device");
    }

    return make_unique<CpuBrainSimulator>(move(model));
}

}
