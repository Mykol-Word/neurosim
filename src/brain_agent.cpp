#include "neurosim/brain_agent.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace neurosim
{

using namespace std;

namespace
{

using enum Action;

inline constexpr uint32_t PELLET_CHANNELS = 5;
inline constexpr uint32_t VELOCITY_CHANNELS = 4;
inline constexpr uint32_t OUTPUT_CHANNELS = 4;
inline constexpr array<Action, 8> MOVEMENT_ACTIONS{
    Up,
    Down,
    Left,
    Right,
    Up | Left,
    Up | Right,
    Down | Left,
    Down | Right
};

// stores one ray interval inside a navigable floor rectangle
struct RayInterval
{
    float minimum;
    float maximum;
};

// stores one visible pellet before stable distance ordering
struct VisiblePellet
{
    uint32_t index;
    Vec2 offset;
    float distance_squared;
};

// returns whether a simultaneous action contains one direction
bool has_action(Action actions, Action direction)
{
    return (static_cast<uint8_t>(actions) & static_cast<uint8_t>(direction)) != 0U;
}

// creates the dungeon's valid anonymous output patterns
vector<vector<float>> movement_patterns()
{
    using enum Action;

    const array<Action, OUTPUT_CHANNELS> outputs{Up, Down, Left, Right};
    vector<vector<float>> patterns;
    patterns.reserve(MOVEMENT_ACTIONS.size());

    for (const Action action : MOVEMENT_ACTIONS)
    {
        vector<float> pattern(OUTPUT_CHANNELS, 0.0f);

        for (uint32_t output = 0; output < OUTPUT_CHANNELS; ++output)
        {
            pattern[output] = static_cast<float>(has_action(action, outputs[output]));
        }

        patterns.push_back(move(pattern));
    }

    return patterns;
}

// supplies the dungeon action codebook without changing generic defaults
BrainAgentConfig prepare_agent_config(BrainAgentConfig config)
{
    if (config.controller.exploration_patterns.empty())
    {
        config.controller.exploration_patterns = movement_patterns();
    }

    return config;
}

// converts a positive duration to the first fixed tick meeting it
uint64_t duration_ticks(double seconds, double fixed_time_step)
{
    const double ticks = ceil(seconds / fixed_time_step - 1.0e-12);

    if (!isfinite(ticks) || ticks < 1.0
        || ticks > static_cast<double>(numeric_limits<uint64_t>::max()))
    {
        throw invalid_argument("agent duration cannot be represented as fixed ticks");
    }

    return static_cast<uint64_t>(ticks);
}

// returns the maximum pellet count promised by one dungeon configuration
uint32_t maximum_pellet_count(const EnvironmentConfig& environment_config)
{
    const uint64_t count = static_cast<uint64_t>(environment_config.dungeon.max_rooms)
        * environment_config.dungeon.max_pellets_per_room;

    if (count == 0U || count > numeric_limits<uint32_t>::max())
    {
        throw invalid_argument("agent pellet capacity is not representable");
    }

    return static_cast<uint32_t>(count);
}

// returns the flattened sensory input count for one controller configuration
uint32_t input_count(uint32_t pellet_slots, uint32_t wall_ray_count)
{
    const uint64_t count = static_cast<uint64_t>(pellet_slots) * PELLET_CHANNELS
        + wall_ray_count
        + VELOCITY_CHANNELS
        + 1U;

    if (count > numeric_limits<uint32_t>::max())
    {
        throw invalid_argument("agent input count is not representable");
    }

    return static_cast<uint32_t>(count);
}

// validates dungeon-specific sensing, decoding, and outcome parameters
void validate_agent_config(const BrainAgentConfig& config)
{
    if (config.wall_ray_count < 4U || config.wall_ray_count % 4U != 0U)
    {
        throw invalid_argument("agent wall ray count must be a positive multiple of four");
    }

    if (!isfinite(config.output_activity_threshold) || config.output_activity_threshold < 0.0f)
    {
        throw invalid_argument("agent output threshold must be finite and nonnegative");
    }

    const array<float, 2> positive_values{
        config.wall_feedback_interval_seconds,
        config.food_modulator
    };

    for (const float value : positive_values)
    {
        if (!isfinite(value) || value <= 0.0f)
        {
            throw invalid_argument("agent outcome values must be finite and positive");
        }
    }

    if (!isfinite(config.wall_modulator) || config.wall_modulator >= 0.0f)
    {
        throw invalid_argument("agent wall modulator must be finite and negative");
    }

}

// returns whether one ray overlaps an axis interval and narrows its parameter range
bool intersect_ray_axis(
    float origin,
    float direction,
    float minimum,
    float maximum,
    float& near_time,
    float& far_time)
{
    if (abs(direction) <= numeric_limits<float>::epsilon())
    {
        return origin >= minimum && origin <= maximum;
    }

    float first = (minimum - origin) / direction;
    float second = (maximum - origin) / direction;

    if (first > second)
    {
        swap(first, second);
    }

    near_time = max(near_time, first);
    far_time = min(far_time, second);
    return near_time <= far_time;
}

// returns the first distance at which a ray leaves connected navigable floor
float navigable_ray_distance(
    const Environment& environment,
    Vec2 direction)
{
    const float radius = environment.config().vision_radius;
    const float inset = environment.config().player_size * 0.5f + NAVIGATION_EPSILON;
    const Vec2 origin = environment.state().position;
    vector<RayInterval> intervals;
    intervals.reserve(environment.dungeon().floor_rects.size());

    for (const Rect& floor : environment.dungeon().floor_rects)
    {
        const Rect navigable = floor.inset(inset);

        if (navigable.width() <= 0.0f || navigable.height() <= 0.0f)
        {
            continue;
        }

        float near_time = 0.0f;
        float far_time = radius;

        if (!intersect_ray_axis(
                origin.x,
                direction.x,
                navigable.min_x,
                navigable.max_x,
                near_time,
                far_time)
            || !intersect_ray_axis(
                origin.y,
                direction.y,
                navigable.min_y,
                navigable.max_y,
                near_time,
                far_time))
        {
            continue;
        }

        intervals.push_back({max(0.0f, near_time), min(radius, far_time)});
    }

    sort(intervals.begin(), intervals.end(), [](RayInterval left, RayInterval right)
    {
        return left.minimum < right.minimum;
    });

    float connected_end = -1.0f;

    for (const RayInterval interval : intervals)
    {
        if (connected_end < 0.0f)
        {
            if (interval.minimum <= NAVIGATION_EPSILON)
            {
                connected_end = interval.maximum;
            }

            continue;
        }

        if (interval.minimum > connected_end + NAVIGATION_EPSILON)
        {
            break;
        }

        connected_end = max(connected_end, interval.maximum);
    }

    if (connected_end < 0.0f)
    {
        throw logic_error("agent observation origin is outside navigable floor");
    }

    return min(radius, connected_end);
}

// appends every normalized observation channel in stable model input order
void flatten_observation(
    const AgentObservation& observation,
    vector<float>& normalized_inputs)
{
    fill(normalized_inputs.begin(), normalized_inputs.end(), 0.0f);
    size_t input = 0;

    for (const PelletObservation& pellet : observation.pellets)
    {
        normalized_inputs[input++] = pellet.present;
        normalized_inputs[input++] = pellet.left;
        normalized_inputs[input++] = pellet.right;
        normalized_inputs[input++] = pellet.up;
        normalized_inputs[input++] = pellet.down;
    }

    for (const float wall : observation.wall_proximity)
    {
        normalized_inputs[input++] = wall;
    }

    for (const float velocity : observation.velocity)
    {
        normalized_inputs[input++] = velocity;
    }

    normalized_inputs[input++] = observation.hunger;

    if (input != normalized_inputs.size())
    {
        throw logic_error("agent observation does not match its brain input count");
    }
}

// decodes four anonymous output values through the dungeon action schema
Action decode_action(span<const float> outputs, float threshold)
{
    using enum Action;

    if (outputs.size() != OUTPUT_CHANNELS)
    {
        throw logic_error("dungeon action decoder requires four outputs");
    }

    Action action = None;
    const float up = outputs[0];
    const float down = outputs[1];
    const float left = outputs[2];
    const float right = outputs[3];

    if (up >= threshold && up > down)
    {
        action = action | Up;
    }
    else if (down >= threshold && down > up)
    {
        action = action | Down;
    }

    if (left >= threshold && left > right)
    {
        action = action | Left;
    }
    else if (right >= threshold && right > left)
    {
        action = action | Right;
    }

    return action;
}

}

// creates a bounded local observation without rendering dependencies
AgentObservation observe_environment(
    const Environment& environment,
    uint32_t pellet_slots,
    uint32_t wall_ray_count)
{
    if (pellet_slots == 0U)
    {
        throw invalid_argument("agent observation requires at least one pellet slot");
    }

    if (wall_ray_count < 4U || wall_ray_count % 4U != 0U)
    {
        throw invalid_argument("agent observation wall rays must be a positive multiple of four");
    }

    AgentObservation observation;
    observation.pellets.resize(pellet_slots);
    observation.wall_proximity.resize(wall_ray_count);
    const float radius = environment.config().vision_radius;
    const float radius_squared = radius * radius;
    const EnvironmentState& state = environment.state();
    const Dungeon& dungeon = environment.dungeon();
    vector<VisiblePellet> visible;
    visible.reserve(dungeon.pellets.size());

    for (uint32_t index = 0; index < dungeon.pellets.size(); ++index)
    {
        if (state.pellet_alive[index] == 0U)
        {
            continue;
        }

        const Vec2 offset = dungeon.pellets[index].position - state.position;
        const float squared = length_squared(offset);

        if (squared <= radius_squared)
        {
            visible.push_back({index, offset, squared});
        }
    }

    sort(visible.begin(), visible.end(), [](const VisiblePellet& left, const VisiblePellet& right)
    {
        return left.distance_squared == right.distance_squared
            ? left.index < right.index
            : left.distance_squared < right.distance_squared;
    });

    if (visible.size() > pellet_slots)
    {
        throw runtime_error("visible pellet count exceeds the agent observation capacity");
    }

    observation.visible_pellets = static_cast<uint32_t>(visible.size());

    for (uint32_t slot = 0; slot < visible.size(); ++slot)
    {
        const Vec2 offset = visible[slot].offset;
        PelletObservation& pellet = observation.pellets[slot];
        pellet.present = 1.0f;
        pellet.left = clamp(-offset.x / radius, 0.0f, 1.0f);
        pellet.right = clamp(offset.x / radius, 0.0f, 1.0f);
        pellet.up = clamp(-offset.y / radius, 0.0f, 1.0f);
        pellet.down = clamp(offset.y / radius, 0.0f, 1.0f);
    }

    constexpr float FULL_TURN = 2.0f * numbers::pi_v<float>;

    for (uint32_t ray = 0; ray < wall_ray_count; ++ray)
    {
        const float angle = -numbers::pi_v<float> * 0.5f
            + FULL_TURN * static_cast<float>(ray) / static_cast<float>(wall_ray_count);
        const Vec2 direction{cos(angle), sin(angle)};
        const float distance = navigable_ray_distance(environment, direction);
        observation.wall_proximity[ray] = clamp(1.0f - distance / radius, 0.0f, 1.0f);
    }

    const float maximum_speed = environment.config().max_speed;
    observation.velocity = {
        clamp(-state.velocity.x / maximum_speed, 0.0f, 1.0f),
        clamp(state.velocity.x / maximum_speed, 0.0f, 1.0f),
        clamp(-state.velocity.y / maximum_speed, 0.0f, 1.0f),
        clamp(state.velocity.y / maximum_speed, 0.0f, 1.0f)
    };
    observation.hunger = clamp(
        static_cast<float>(state.ticks_since_food)
            / static_cast<float>(environment.starvation_ticks()),
        0.0f,
        1.0f);
    return observation;
}

// constructs a random sparse brain compatible with one environment configuration
BrainAgent::BrainAgent(
    const EnvironmentConfig& environment_config,
    BrainAgentConfig config,
    uint64_t seed)
    : environment_config_(environment_config),
      config_(prepare_agent_config(move(config))),
      seed_(seed),
      pellet_slots_(maximum_pellet_count(environment_config_)),
      sensory_input_count_(input_count(pellet_slots_, config_.wall_ray_count)),
      controller_(
          sensory_input_count_,
          OUTPUT_CHANNELS,
          environment_config_.fixed_time_step,
          config_.controller,
          seed_)
{
    validate_agent_config(config_);
    const double environment_step = static_cast<double>(environment_config_.fixed_time_step);
    wall_feedback_interval_ticks_ = duration_ticks(
        config_.wall_feedback_interval_seconds,
        environment_step);
    normalized_inputs_.resize(sensory_input_count_);
    reset_episode(0U);
}

// observes the current environment and returns one simultaneous movement action
Action BrainAgent::decide(const Environment& environment)
{
    using enum Action;

    if (environment.state().termination != TerminationReason::None)
    {
        throw logic_error("brain agent cannot decide for a terminal environment");
    }

    if (environment.config().fixed_time_step != environment_config_.fixed_time_step
        || environment.config().vision_radius != environment_config_.vision_radius
        || environment.config().max_speed != environment_config_.max_speed
        || environment.config().starvation_seconds != environment_config_.starvation_seconds)
    {
        throw invalid_argument("brain agent environment configuration changed after construction");
    }

    if (wall_feedback_cooldown_ > 0U)
    {
        --wall_feedback_cooldown_;
    }

    observation_ = observe_environment(environment, pellet_slots_, config_.wall_ray_count);
    flatten_observation(observation_, normalized_inputs_);
    const AdaptiveControllerStatus& controller_status = controller_.step(normalized_inputs_);

    status_ = {};
    status_.visible_pellets = observation_.visible_pellets;
    status_.brain_spikes = controller_status.brain_spikes;
    status_.output_spikes = controller_status.output_spikes;
    status_.exploring = controller_status.exploring;

    for (uint32_t output = 0; output < OUTPUT_CHANNELS; ++output)
    {
        status_.output_activity[output] = controller_status.output_activity[output];
    }

    status_.network_action = decode_action(
        controller_status.output_activity,
        config_.output_activity_threshold);
    status_.exploration_action = controller_status.exploring
        ? decode_action(controller_status.exploration_outputs, 0.5f)
        : None;
    status_.action = controller_status.exploring
        ? status_.exploration_action
        : status_.network_action;

    return status_.action;
}

// schedules consequences from the environment transition caused by the last action
void BrainAgent::observe_transition(const StepResult& transition)
{
    float modulator = config_.food_modulator * static_cast<float>(transition.pellets_eaten);
    const bool wall_feedback = transition.hit_wall && wall_feedback_cooldown_ == 0U;

    if (wall_feedback)
    {
        modulator += config_.wall_modulator;
        wall_feedback_cooldown_ = wall_feedback_interval_ticks_;
    }

    if (controller_.learning_enabled() && modulator != 0.0f)
    {
        controller_.apply_modulation(modulator);
        status_.applied_modulator = modulator;
    }
}

// clears transient activity and exploration while preserving learned strengths
void BrainAgent::reset_episode(uint64_t episode_seed)
{
    controller_.reset_episode(episode_seed);
    wall_feedback_cooldown_ = 0U;
    observation_ = {};
    fill(normalized_inputs_.begin(), normalized_inputs_.end(), 0.0f);
    status_ = {};
}

// enables or freezes outcome-driven connection updates
void BrainAgent::set_learning_enabled(bool enabled)
{
    controller_.set_learning_enabled(enabled);
}

// enables or disables task-blind output exploration
void BrainAgent::set_exploration_enabled(bool enabled)
{
    controller_.set_exploration_enabled(enabled);
}

// replaces learned strengths after validating them against this brain
void BrainAgent::load_connection_strengths(span<const float> strengths)
{
    controller_.load_connection_strengths(strengths);
}

// returns the fixed construction seed for this brain
uint64_t BrainAgent::seed() const
{
    return seed_;
}

// returns the number of fixed pellet observation slots
uint32_t BrainAgent::pellet_slots() const
{
    return pellet_slots_;
}

// returns the immutable agent configuration
const BrainAgentConfig& BrainAgent::config() const
{
    return config_;
}

// returns the most recently encoded normalized observation
const AgentObservation& BrainAgent::observation() const
{
    return observation_;
}

// returns the most recent action, exploration, and neural status
const BrainAgentStatus& BrainAgent::status() const
{
    return status_;
}

// returns whether outcome-driven connection updates are enabled
bool BrainAgent::learning_enabled() const
{
    return controller_.learning_enabled();
}

// returns whether task-blind output exploration is enabled
bool BrainAgent::exploration_enabled() const
{
    return controller_.exploration_enabled();
}

// returns the task-neutral adaptive controller
const AdaptiveController& BrainAgent::controller() const
{
    return controller_;
}

// returns the cpu brain simulator for recording and inspection
const CpuBrainSimulator& BrainAgent::simulator() const
{
    return controller_.simulator();
}

}
