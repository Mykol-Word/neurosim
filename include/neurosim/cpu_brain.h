#pragma once

#include "neurosim/brain.h"

#include <cstdint>
#include <span>
#include <vector>

namespace neurosim
{

using namespace std;

// advances one brain with the readable cpu reference implementation
class CpuBrainSimulator
{
public:
    // creates a simulator with fresh state from a compiled model
    explicit CpuBrainSimulator(BrainModel model);

    // restores activity and learned weights to their initial values
    void reset();

    // clears transient activity while preserving learned connection strengths
    void reset_activity();

    // replaces the complete mutable state after validating it
    void reset(BrainState state);

    // advances every neuron and optionally records plasticity by one fixed tick
    void step(
        span<const float> input_drives,
        BrainPerturbation perturbation = {},
        bool plasticity_enabled = true);

    // applies one signed global event to eligible local connections
    void apply_plasticity_modulation(float modulator);

    // returns the immutable compiled model
    const BrainModel& model() const;

    // returns the current mutable state as read-only data
    const BrainState& state() const;

    // returns whether one ordered output neuron fired this tick
    bool output_spike(uint32_t output_channel) const;

    // returns the recent activity trace of one ordered output neuron
    float output_activity(uint32_t output_channel) const;

private:
    BrainModel model_;
    BrainState state_;
    vector<uint8_t> next_spikes_;
    bool requires_reset_ = false;
};

}
