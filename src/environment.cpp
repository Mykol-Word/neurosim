#include "neurosim/environment.h"

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

// stores one closed movement interval along a single axis
struct Interval
{
    float minimum;
    float maximum;
};

// validates movement and timing configuration values
void validate_environment_config(const EnvironmentConfig& config)
{
    validate_dungeon_config(config.dungeon, config.player_size);

    if (!isfinite(config.fixed_time_step) || config.fixed_time_step <= 0.0f)
    {
        throw invalid_argument("fixed time step must be finite and positive");
    }

    if (!isfinite(config.acceleration) || config.acceleration <= 0.0f)
    {
        throw invalid_argument("acceleration must be finite and positive");
    }

    if (!isfinite(config.braking) || config.braking <= 0.0f)
    {
        throw invalid_argument("braking must be finite and positive");
    }

    if (!isfinite(config.max_speed) || config.max_speed <= 0.0f)
    {
        throw invalid_argument("maximum speed must be finite and positive");
    }

    if (!isfinite(config.starvation_seconds) || config.starvation_seconds <= 0.0)
    {
        throw invalid_argument("starvation duration must be finite and positive");
    }

    if (!isfinite(config.time_limit_seconds) || config.time_limit_seconds <= 0.0)
    {
        throw invalid_argument("time limit must be finite and positive");
    }
}

// converts a positive duration to the first fixed tick meeting it
uint64_t seconds_to_ticks(double seconds, float fixed_time_step)
{
    const double exact_ticks = seconds / static_cast<double>(fixed_time_step);
    const double adjusted_ticks = exact_ticks - max(1.0, exact_ticks) * 1.0e-12;
    const double ticks = ceil(adjusted_ticks);

    if (ticks < 1.0 || ticks > static_cast<double>(numeric_limits<uint64_t>::max()))
    {
        throw invalid_argument("duration cannot be represented as simulation ticks");
    }

    return static_cast<uint64_t>(ticks);
}

// returns whether a combined action contains one movement input
bool has_action(Action actions, Action action)
{
    return (static_cast<uint8_t>(actions) & static_cast<uint8_t>(action)) != 0;
}

// maps simultaneous movement inputs to one unit direction
Vec2 action_direction(Action action)
{
    using enum Action;

    constexpr uint8_t VALID_ACTIONS = static_cast<uint8_t>(Up)
        | static_cast<uint8_t>(Down)
        | static_cast<uint8_t>(Left)
        | static_cast<uint8_t>(Right);
    const uint8_t action_bits = static_cast<uint8_t>(action);

    if ((action_bits & ~VALID_ACTIONS) != 0)
    {
        throw invalid_argument("unknown environment action");
    }

    const Vec2 direction{
        static_cast<float>(has_action(action, Right)) - static_cast<float>(has_action(action, Left)),
        static_cast<float>(has_action(action, Down)) - static_cast<float>(has_action(action, Up))
    };
    const float direction_length = length(direction);
    return direction_length == 0.0f ? Vec2{} : direction / direction_length;
}

// moves a vector toward a target without exceeding a maximum change
Vec2 move_toward(Vec2 current, Vec2 target, float maximum_delta)
{
    const Vec2 difference = target - current;
    const float distance = length(difference);

    if (distance <= maximum_delta || distance == 0.0f)
    {
        return target;
    }

    return current + difference * (maximum_delta / distance);
}

// resolves one movement axis against connected navigable intervals
float resolve_axis(
    const Dungeon& dungeon,
    Vec2 current,
    float target,
    float player_size,
    bool horizontal,
    bool& blocked)
{
    vector<Interval> intervals;
    intervals.reserve(dungeon.floor_rects.size());
    const float inset = player_size * 0.5f + NAVIGATION_EPSILON;
    const float cross_axis = horizontal ? current.y : current.x;

    for (const Rect& floor : dungeon.floor_rects)
    {
        const Rect navigable = floor.inset(inset);

        if (navigable.width() <= 0.0f || navigable.height() <= 0.0f)
        {
            continue;
        }

        const float cross_minimum = horizontal ? navigable.min_y : navigable.min_x;
        const float cross_maximum = horizontal ? navigable.max_y : navigable.max_x;

        if (cross_axis < cross_minimum - NAVIGATION_EPSILON
            || cross_axis > cross_maximum + NAVIGATION_EPSILON)
        {
            continue;
        }

        intervals.push_back(horizontal
            ? Interval{navigable.min_x, navigable.max_x}
            : Interval{navigable.min_y, navigable.max_y});
    }

    sort(intervals.begin(), intervals.end(), [](const Interval& left, const Interval& right)
    {
        return left.minimum < right.minimum;
    });

    vector<Interval> merged;
    merged.reserve(intervals.size());

    for (const Interval interval : intervals)
    {
        if (merged.empty() || interval.minimum > merged.back().maximum + NAVIGATION_EPSILON)
        {
            merged.push_back(interval);
        }
        else
        {
            merged.back().maximum = max(merged.back().maximum, interval.maximum);
        }
    }

    const float current_axis = horizontal ? current.x : current.y;

    for (const Interval interval : merged)
    {
        if (current_axis < interval.minimum - NAVIGATION_EPSILON
            || current_axis > interval.maximum + NAVIGATION_EPSILON)
        {
            continue;
        }

        const float result = clamp(target, interval.minimum, interval.maximum);
        blocked = abs(result - target) > NAVIGATION_EPSILON;
        return result;
    }

    throw logic_error("player is outside the navigable dungeon geometry");
}

// returns whether an axis-aligned swept player square crosses a point
bool swept_square_contains(Vec2 point, Vec2 start, Vec2 end, float half_size)
{
    return point.x >= min(start.x, end.x) - half_size
        && point.x <= max(start.x, end.x) + half_size
        && point.y >= min(start.y, end.y) - half_size
        && point.y <= max(start.y, end.y) + half_size;
}

}

// creates an environment and resets it to the supplied seed
Environment::Environment(EnvironmentConfig config, uint64_t seed)
    : config_(move(config))
{
    validate_environment_config(config_);
    reset(seed);
}

// regenerates and resets the environment from a seed
void Environment::reset(uint64_t seed)
{
    reset(generate_dungeon(config_.dungeon, config_.player_size, seed));
}

// resets the environment from validated external geometry
void Environment::reset(Dungeon dungeon)
{
    validate_environment_config(config_);
    validate_dungeon(dungeon, config_.player_size);
    dungeon_ = move(dungeon);
    state_ = {};
    state_.position = dungeon_.spawn;
    state_.pellet_alive.assign(dungeon_.pellets.size(), 1U);
    state_.pellets_remaining = static_cast<uint32_t>(dungeon_.pellets.size());
}

// advances the environment by one fixed tick
StepResult Environment::step(Action action)
{
    using enum TerminationReason;

    if (state_.termination != None)
    {
        throw logic_error("cannot step a terminal environment before reset");
    }

    const Vec2 direction = action_direction(action);
    const Vec2 target_velocity = direction * config_.max_speed;
    const float rate = direction == Vec2{} ? config_.braking : config_.acceleration;
    state_.velocity = move_toward(state_.velocity, target_velocity, rate * config_.fixed_time_step);

    const Vec2 start = state_.position;
    const Vec2 target = start + state_.velocity * config_.fixed_time_step;
    bool blocked_x = false;
    bool blocked_y = false;
    const float resolved_x = resolve_axis(dungeon_, start, target.x, config_.player_size, true, blocked_x);
    const Vec2 after_x{resolved_x, start.y};
    const float resolved_y = resolve_axis(dungeon_, after_x, target.y, config_.player_size, false, blocked_y);
    const Vec2 end{resolved_x, resolved_y};

    if (blocked_x)
    {
        state_.velocity.x = 0.0f;
    }

    if (blocked_y)
    {
        state_.velocity.y = 0.0f;
    }

    state_.position = end;
    uint32_t eaten = 0;
    const float half_size = config_.player_size * 0.5f;

    for (size_t index = 0; index < dungeon_.pellets.size(); ++index)
    {
        if (state_.pellet_alive[index] == 0U)
        {
            continue;
        }

        const Vec2 pellet = dungeon_.pellets[index].position;

        if (!swept_square_contains(pellet, start, after_x, half_size)
            && !swept_square_contains(pellet, after_x, end, half_size))
        {
            continue;
        }

        state_.pellet_alive[index] = 0U;
        --state_.pellets_remaining;
        ++state_.score;
        ++eaten;
    }

    ++state_.elapsed_ticks;

    if (eaten > 0)
    {
        state_.ticks_since_food = 0;
    }
    else
    {
        ++state_.ticks_since_food;
    }

    if (state_.pellets_remaining == 0)
    {
        state_.termination = Cleared;
    }
    else if (state_.ticks_since_food >= starvation_ticks())
    {
        state_.termination = Starved;
    }
    else if (state_.elapsed_ticks >= time_limit_ticks())
    {
        state_.termination = TimeLimit;
    }

    if (!is_finite(state_.position) || !is_finite(state_.velocity))
    {
        throw runtime_error("simulation produced non-finite player state");
    }

    return {
        eaten,
        state_.score,
        state_.pellets_remaining,
        elapsed_seconds(),
        seconds_since_food(),
        state_.termination
    };
}

// returns the immutable environment configuration
const EnvironmentConfig& Environment::config() const
{
    return config_;
}

// returns the immutable dungeon geometry
const Dungeon& Environment::dungeon() const
{
    return dungeon_;
}

// returns the current mutable episode state as read-only data
const EnvironmentState& Environment::state() const
{
    return state_;
}

// returns elapsed episode time in seconds
double Environment::elapsed_seconds() const
{
    return static_cast<double>(state_.elapsed_ticks) * static_cast<double>(config_.fixed_time_step);
}

// returns elapsed time since the most recent pellet in seconds
double Environment::seconds_since_food() const
{
    return static_cast<double>(state_.ticks_since_food) * static_cast<double>(config_.fixed_time_step);
}

// returns the starvation deadline in fixed ticks
uint64_t Environment::starvation_ticks() const
{
    return seconds_to_ticks(config_.starvation_seconds, config_.fixed_time_step);
}

// returns the episode time limit in fixed ticks
uint64_t Environment::time_limit_ticks() const
{
    return seconds_to_ticks(config_.time_limit_seconds, config_.fixed_time_step);
}

// returns the stable text name for a termination reason
const char* termination_name(TerminationReason reason)
{
    using enum TerminationReason;

    switch (reason)
    {
    case None:
        return "running";
    case Cleared:
        return "cleared";
    case Starved:
        return "starved";
    case TimeLimit:
        return "time_limit";
    }

    return "unknown";
}

}
