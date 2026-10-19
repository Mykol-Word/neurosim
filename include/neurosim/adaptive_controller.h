#pragma once

#include "neurosim/brain_genome.h"
#include "neurosim/brain_simulator.h"

#include <cstdint>
#include <span>
#include <vector>

namespace neurosim
{

using namespace std;

// configures task-neutral brain construction, input encoding, and exploration
struct AdaptiveControllerConfig
{
    GridShape brain_shape{23, 23, 23};
    float brain_radius = 10.65f;
    float brain_time_step_seconds = 1.0f / 480.0f;
    float sensory_drive = 500.0f;
    float exploration_drive = 600.0f;
    float inhibitory_fraction = 0.2f;
    float minimum_bias_drive = 30.0f;
    float maximum_bias_drive = 47.5f;
    float exploration_interval_seconds = 0.75f;
    float exploration_duration_seconds = 0.25f;
    vector<vector<float>> exploration_patterns;
    BrainConfig brain;
};

// reports anonymous output activity and task-blind exploration for one control step
struct AdaptiveControllerStatus
{
    vector<float> output_activity;
    vector<float> exploration_outputs;
    uint64_t brain_spikes = 0;
    uint64_t output_spikes = 0;
    bool exploring = false;
};

// adapts arbitrary normalized inputs into anonymous outputs with one plastic brain
class AdaptiveController
{
public:
    // constructs a reproducible seeded or explicit task-neutral brain
    AdaptiveController(
        uint32_t input_count,
        uint32_t output_count,
        float control_time_step_seconds,
        AdaptiveControllerConfig config = {},
        uint64_t seed = 1,
        BrainExecutionConfig execution = {},
        BrainSimulatorFactory simulator_factory = nullptr,
        const BrainGenome* genome = nullptr);

    // advances one control step from normalized task-owned inputs
    const AdaptiveControllerStatus& step(span<const float> normalized_inputs);

    // applies one task-owned signed outcome to recent local eligibility
    void apply_modulation(float value);

    // clears activity and restarts an independent exploration stream
    void reset_episode(uint64_t exploration_seed);

    // enables or disables task-blind output exploration
    void set_exploration_enabled(bool enabled);

    // enables or freezes every connection-strength adaptation path
    void set_learning_enabled(bool enabled);

    // replaces learned strengths after validating them against this controller
    void load_connection_strengths(span<const float> strengths);

    // returns a host copy of the learned connection strengths
    vector<float> copy_connection_strengths() const;

    // returns a full host snapshot for diagnostics outside the hot path
    BrainState copy_state() const;

    // returns the fixed construction seed
    uint64_t seed() const;

    // returns the normalized sensory channel count
    uint32_t input_count() const;

    // returns the anonymous output channel count
    uint32_t output_count() const;

    // returns the immutable generic controller configuration
    const AdaptiveControllerConfig& config() const;

    // returns the most recent neural and exploration state
    const AdaptiveControllerStatus& status() const;

    // returns whether task-blind output exploration is enabled
    bool exploration_enabled() const;

    // returns whether connection-strength adaptation is enabled
    bool learning_enabled() const;

    // returns the immutable compiled brain model
    const BrainModel& model() const;

    // returns the explicit execution backend
    BrainBackend backend() const;

private:
    AdaptiveControllerConfig config_;
    BrainExecutionConfig execution_;
    uint64_t seed_ = 0;
    uint64_t exploration_seed_ = 0;
    uint64_t control_tick_ = 0;
    uint64_t next_exploration_tick_ = 0;
    uint64_t exploration_interval_ticks_ = 0;
    uint64_t exploration_duration_ticks_ = 0;
    uint64_t exploration_ticks_remaining_ = 0;
    uint64_t exploration_event_ = 0;
    uint32_t input_count_ = 0;
    uint32_t output_count_ = 0;
    uint32_t exploration_pattern_ = 0;
    double control_time_step_seconds_ = 0.0;
    double pending_brain_time_seconds_ = 0.0;
    bool exploration_enabled_ = true;
    bool learning_enabled_ = true;
    vector<float> brain_inputs_;
    unique_ptr<BrainSimulator> simulator_;
    AdaptiveControllerStatus status_;
};

}
