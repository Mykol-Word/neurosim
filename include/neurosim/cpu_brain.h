#pragma once

#include "neurosim/brain_simulator.h"

#include <cstdint>
#include <span>
#include <vector>

namespace neurosim
{

using namespace std;

// advances one brain with the readable cpu reference implementation
class CpuBrainSimulator final : public BrainSimulator
{
public:
    // creates a simulator with fresh state from a compiled model
    explicit CpuBrainSimulator(BrainModel model);

    // restores activity and learned weights to their initial values
    void reset();

    // clears transient activity while preserving learned connection strengths
    void reset_activity() override;

    // replaces the complete mutable state after validating it
    void reset(BrainState state);

    // advances every neuron and optionally records plasticity by one fixed tick
    void step(
        span<const float> input_drives,
        BrainPerturbation perturbation = {},
        bool plasticity_enabled = true);

    // advances multiple fixed ticks after validating shared inputs once
    BrainAdvanceSummary advance(
        span<const float> input_drives,
        uint32_t tick_count,
        BrainPerturbation perturbation,
        bool plasticity_enabled,
        span<float> output_activity) override;

    // applies one signed global event to eligible local connections
    void apply_plasticity_modulation(float modulator) override;

    // replaces learned strengths and clears transient activity
    void load_connection_strengths(span<const float> strengths) override;

    // returns a host copy of the learned connection strengths
    vector<float> copy_connection_strengths() const override;

    // returns a full host snapshot for diagnostics outside the hot path
    BrainState copy_state() const override;

    // returns the immutable compiled model
    const BrainModel& model() const override;

    // returns the cpu execution backend
    BrainBackend backend() const override;

    // returns the current mutable state as read-only data
    const BrainState& state() const;

    // returns whether one ordered output neuron fired this tick
    bool output_spike(uint32_t output_channel) const;

    // returns the recent activity trace of one ordered output neuron
    float output_activity(uint32_t output_channel) const;

private:
    // advances one already-validated fixed tick and counts its spikes
    BrainAdvanceSummary advance_tick(
        span<const float> input_drives,
        BrainPerturbation perturbation,
        bool plasticity_enabled);

    BrainModel model_;
    BrainState state_;
    vector<float> synaptic_drives_;
    vector<uint8_t> next_spikes_;
    bool requires_reset_ = false;
};

// creates one cpu simulator for the requested cpu execution configuration
unique_ptr<BrainSimulator> make_cpu_brain_simulator(
    BrainModel model,
    const BrainExecutionConfig& execution);

}
