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
    next_spikes_.assign(model_.positions.size(), 0U);
    requires_reset_ = false;
}

// clears transient activity while preserving learned connection strengths
void CpuBrainSimulator::reset_activity()
{
    vector<float> learned_strengths = move(state_.connection_strengths);
    state_ = make_brain_state(model_);
    state_.connection_strengths = move(learned_strengths);
    validate_brain_state(model_, state_);
    next_spikes_.assign(model_.positions.size(), 0U);
    requires_reset_ = false;
}

// replaces the complete mutable state after validating it
void CpuBrainSimulator::reset(BrainState state)
{
    validate_brain_state(model_, state);
    state_ = move(state);
    next_spikes_.assign(model_.positions.size(), 0U);
    requires_reset_ = false;
}

// advances every neuron and optionally records plasticity by one fixed tick
void CpuBrainSimulator::step(
    span<const float> input_drives,
    BrainPerturbation perturbation,
    bool plasticity_enabled)
{
    if (requires_reset_)
    {
        throw logic_error("brain simulator must be reset after a failed numerical step");
    }

    if (input_drives.size() != model_.input_neurons.size())
    {
        throw invalid_argument("brain input count does not match the configured input channels");
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

    if (state_.tick == numeric_limits<uint64_t>::max())
    {
        throw overflow_error("brain tick counter overflow");
    }

    const NeuronDynamicsConfig& dynamics = model_.config.dynamics;
    fill(next_spikes_.begin(), next_spikes_.end(), uint8_t{0});

    for (uint32_t neuron = 0; neuron < model_.positions.size(); ++neuron)
    {
        if (state_.refractory_ticks[neuron] > 0U)
        {
            --state_.refractory_ticks[neuron];
            state_.membrane_potentials[neuron] = dynamics.reset_potential;
            continue;
        }

        float synaptic_drive = 0.0f;

        for (uint32_t incoming = model_.incoming_offsets[neuron];
            incoming < model_.incoming_offsets[neuron + 1U]; ++incoming)
        {
            const uint32_t edge = model_.incoming_edges[incoming];
            const uint32_t source = model_.edge_sources[edge];
            synaptic_drive += polarity_sign(model_.polarities[source])
                * state_.connection_strengths[edge]
                * static_cast<float>(state_.spikes[source]);
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
            + synaptic_drive;

        if (!finite(next_potential))
        {
            requires_reset_ = true;
            throw runtime_error("brain dynamics produced a non-finite membrane potential");
        }

        if (next_potential >= dynamics.firing_threshold)
        {
            next_spikes_[neuron] = 1U;
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
        for (uint32_t source = 0; source < model_.positions.size(); ++source)
        {
            const float old_source_activity = state_.activity_traces[source];
            const bool excitatory = model_.polarities[source] == NeuronPolarity::Excitatory;

            for (uint32_t edge = model_.outgoing_offsets[source];
                edge < model_.outgoing_offsets[source + 1U]; ++edge)
            {
                const uint32_t target = model_.outgoing_targets[edge];
                const float strength = state_.connection_strengths[edge] * model_.weight_decay;
                const float eligibility = state_.connection_eligibilities[edge]
                        * model_.eligibility_decay
                    + static_cast<float>(excitatory) * static_cast<float>(next_spikes_[target])
                        * old_source_activity;

                if (!finite(strength) || strength < 0.0f
                    || !finite(eligibility) || eligibility < 0.0f)
                {
                    requires_reset_ = true;
                    throw runtime_error("brain plasticity produced invalid local state");
                }

                state_.connection_strengths[edge] = strength;
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
    ++state_.tick;
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

// returns the immutable compiled model
const BrainModel& CpuBrainSimulator::model() const
{
    return model_;
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

}
