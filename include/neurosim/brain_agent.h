#pragma once

#include "neurosim/adaptive_controller.h"
#include "neurosim/environment.h"

#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace neurosim
{

using namespace std;

// stores one visible pellet as nonnegative relative-position signals
struct PelletObservation
{
    float present = 0.0f;
    float left = 0.0f;
    float right = 0.0f;
    float up = 0.0f;
    float down = 0.0f;
};

// stores the complete normalized local observation for one environment tick
struct AgentObservation
{
    vector<PelletObservation> pellets;
    vector<float> wall_proximity;
    array<float, 4> velocity{};
    float hunger = 0.0f;
    uint32_t visible_pellets = 0;
};

// configures dungeon sensing, action decoding, and signed outcome mapping
struct BrainAgentConfig
{
    uint32_t wall_ray_count = 32;
    float output_activity_threshold = 0.5f;
    float wall_feedback_interval_seconds = 0.25f;
    float food_modulator = 1.0f;
    float wall_modulator = -0.005f;
    AdaptiveControllerConfig controller;
};

// reports the most recent dungeon action and generic controller state
struct BrainAgentStatus
{
    Action action = Action::None;
    Action network_action = Action::None;
    Action exploration_action = Action::None;
    array<float, 4> output_activity{};
    uint64_t brain_spikes = 0;
    uint64_t output_spikes = 0;
    uint32_t visible_pellets = 0;
    float applied_modulator = 0.0f;
    bool exploring = false;
};

// creates a bounded local observation without rendering dependencies
AgentObservation observe_environment(
    const Environment& environment,
    uint32_t pellet_slots,
    uint32_t wall_ray_count);

// controls one environment with a seeded plastic spiking brain
class BrainAgent
{
public:
    // constructs a random sparse brain compatible with one environment configuration
    BrainAgent(
        const EnvironmentConfig& environment_config,
        BrainAgentConfig config = {},
        uint64_t seed = 1,
        BrainExecutionConfig execution = {},
        BrainSimulatorFactory simulator_factory = nullptr);

    // observes the current environment and returns one simultaneous movement action
    Action decide(const Environment& environment);

    // schedules consequences from the environment transition caused by the last action
    void observe_transition(const StepResult& transition);

    // clears transient activity and exploration while preserving learned strengths
    void reset_episode(uint64_t episode_seed);

    // enables or freezes outcome-driven connection updates
    void set_learning_enabled(bool enabled);

    // enables or disables task-blind output exploration
    void set_exploration_enabled(bool enabled);

    // replaces learned strengths after validating them against this brain
    void load_connection_strengths(span<const float> strengths);

    // returns a host copy of the learned connection strengths
    vector<float> copy_connection_strengths() const;

    // returns a full host snapshot for diagnostics outside the hot path
    BrainState copy_state() const;

    // returns the fixed construction seed for this brain
    uint64_t seed() const;

    // returns the number of fixed pellet observation slots
    uint32_t pellet_slots() const;

    // returns the immutable agent configuration
    const BrainAgentConfig& config() const;

    // returns the most recently encoded normalized observation
    const AgentObservation& observation() const;

    // returns the most recent action, exploration, and neural status
    const BrainAgentStatus& status() const;

    // returns whether outcome-driven connection updates are enabled
    bool learning_enabled() const;

    // returns whether task-blind output exploration is enabled
    bool exploration_enabled() const;

    // returns the task-neutral adaptive controller
    const AdaptiveController& controller() const;

    // returns the immutable compiled brain model
    const BrainModel& model() const;

    // returns the explicit execution backend
    BrainBackend backend() const;

private:
    EnvironmentConfig environment_config_;
    BrainAgentConfig config_;
    uint64_t seed_ = 0;
    uint32_t pellet_slots_ = 0;
    uint32_t sensory_input_count_ = 0;
    uint64_t wall_feedback_interval_ticks_ = 0;
    uint64_t wall_feedback_cooldown_ = 0;
    AgentObservation observation_;
    vector<float> normalized_inputs_;
    vector<Vec2> wall_ray_directions_;
    vector<pair<float, float>> wall_ray_intervals_;
    vector<pair<float, uint32_t>> visible_pellets_;
    AdaptiveController controller_;
    BrainAgentStatus status_;
};

}
