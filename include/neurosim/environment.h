#pragma once

#include "neurosim/dungeon.h"

#include <cstdint>
#include <vector>

namespace neurosim
{

using namespace std;

// identifies simultaneous cardinal movement inputs
enum class Action : uint8_t
{
    None = 0,
    Up = 1 << 0,
    Down = 1 << 1,
    Left = 1 << 2,
    Right = 1 << 3
};

// combines simultaneous movement inputs
constexpr Action operator|(Action left, Action right)
{
    return static_cast<Action>(static_cast<uint8_t>(left) | static_cast<uint8_t>(right));
}

// identifies why an episode stopped
enum class TerminationReason : uint8_t
{
    None,
    Cleared,
    Starved,
    TimeLimit
};

// configures dungeon generation, movement, and episode timing
struct EnvironmentConfig
{
    DungeonConfig dungeon;
    float fixed_time_step = 1.0f / 120.0f;
    float player_size = 1.4f;
    float acceleration = 30.0f;
    float braking = 42.0f;
    float max_speed = 12.0f;
    double starvation_seconds = 15.0;
    double time_limit_seconds = 300.0;
};

// stores all mutable state for one environment episode
struct EnvironmentState
{
    Vec2 position;
    Vec2 velocity;
    vector<uint8_t> pellet_alive;
    uint32_t score = 0;
    uint32_t pellets_remaining = 0;
    uint64_t elapsed_ticks = 0;
    uint64_t ticks_since_food = 0;
    TerminationReason termination = TerminationReason::None;
};

// summarizes the state transition produced by one action
struct StepResult
{
    uint32_t pellets_eaten = 0;
    uint32_t score = 0;
    uint32_t pellets_remaining = 0;
    double elapsed_seconds = 0.0;
    double seconds_since_food = 0.0;
    TerminationReason termination = TerminationReason::None;
};

// owns deterministic dungeon state and advances one fixed simulation tick
class Environment
{
public:
    // creates an environment and resets it to the supplied seed
    explicit Environment(EnvironmentConfig config = {}, uint64_t seed = 1);

    // regenerates and resets the environment from a seed
    void reset(uint64_t seed);

    // resets the environment from validated external geometry
    void reset(Dungeon dungeon);

    // advances the environment by one fixed tick
    StepResult step(Action action);

    // returns the immutable environment configuration
    const EnvironmentConfig& config() const;

    // returns the immutable dungeon geometry
    const Dungeon& dungeon() const;

    // returns the current mutable episode state as read-only data
    const EnvironmentState& state() const;

    // returns elapsed episode time in seconds
    double elapsed_seconds() const;

    // returns elapsed time since the most recent pellet in seconds
    double seconds_since_food() const;

    // returns the starvation deadline in fixed ticks
    uint64_t starvation_ticks() const;

    // returns the episode time limit in fixed ticks
    uint64_t time_limit_ticks() const;

private:
    EnvironmentConfig config_;
    Dungeon dungeon_;
    EnvironmentState state_;
};

// returns the stable text name for a termination reason
const char* termination_name(TerminationReason reason);

}
