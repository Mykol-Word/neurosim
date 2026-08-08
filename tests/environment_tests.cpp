#include "neurosim/environment.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace neurosim;

namespace
{

// fails a test when a boolean condition is false
void check(bool condition, const string& message)
{
    if (!condition)
    {
        throw runtime_error(message);
    }
}

// fails a test when two floating-point values differ beyond tolerance
void check_near(float actual, float expected, float tolerance, const string& message)
{
    if (abs(actual - expected) > tolerance)
    {
        throw runtime_error(message + ": expected " + to_string(expected)
            + ", received " + to_string(actual));
    }
}

// creates a minimal one-room dungeon for focused environment tests
Dungeon make_test_dungeon(
    Rect floor,
    Vec2 spawn,
    vector<Vec2> pellet_positions)
{
    Dungeon dungeon;
    dungeon.seed = 77;
    dungeon.rooms.push_back({floor.center(), floor, {floor}});
    dungeon.floor_rects.push_back(floor);
    dungeon.spawn = spawn;
    dungeon.bounds = floor.expanded(1.0f);

    for (const Vec2 position : pellet_positions)
    {
        dungeon.pellets.push_back({position});
    }

    return dungeon;
}

// returns fast and isolated movement settings for environment tests
EnvironmentConfig make_test_config()
{
    EnvironmentConfig config;
    config.fixed_time_step = 0.1f;
    config.player_size = 1.0f;
    config.acceleration = 10.0f;
    config.braking = 20.0f;
    config.max_speed = 4.0f;
    config.starvation_seconds = 100.0;
    config.time_limit_seconds = 100.0;
    return config;
}

// verifies seeded generation determinism, validity, and variation
void test_generation_is_deterministic_and_varied()
{
    const DungeonConfig config;
    const Dungeon first = generate_dungeon(config, 1.4f, 123456);
    const Dungeon second = generate_dungeon(config, 1.4f, 123456);
    check(first == second, "same seed did not reproduce identical dungeon geometry");

    bool found_different_geometry = false;
    bool found_compound_room = false;
    size_t minimum_room_count = first.rooms.size();
    size_t maximum_room_count = first.rooms.size();

    for (uint64_t seed = 1; seed <= 32; ++seed)
    {
        const Dungeon dungeon = generate_dungeon(config, 1.4f, seed);
        validate_dungeon(dungeon, 1.4f);
        minimum_room_count = min(minimum_room_count, dungeon.rooms.size());
        maximum_room_count = max(maximum_room_count, dungeon.rooms.size());

        if (dungeon != first)
        {
            found_different_geometry = true;
        }

        for (const Room& room : dungeon.rooms)
        {
            found_compound_room = found_compound_room || room.floor_rects.size() > 1;
        }
    }

    check(found_different_geometry, "different seeds did not vary dungeon geometry");
    check(found_compound_room, "generator did not produce any compound room shapes");
    check(minimum_room_count < maximum_room_count, "generator did not vary room counts across sampled seeds");
}

// verifies acceleration, maximum speed, and idle braking
void test_acceleration_speed_cap_and_braking()
{
    using enum Action;

    EnvironmentConfig config = make_test_config();
    const Dungeon dungeon = make_test_dungeon({0.0f, 0.0f, 100.0f, 20.0f}, {10.0f, 10.0f}, {{90.0f, 10.0f}});
    Environment environment(config, 1);
    environment.reset(dungeon);

    environment.step(Right);
    check_near(environment.state().velocity.x, 1.0f, 0.0001f, "first action did not accelerate");

    for (int index = 0; index < 10; ++index)
    {
        environment.step(Right);
    }

    check_near(environment.state().velocity.x, config.max_speed, 0.0001f, "velocity exceeded or missed maximum speed");
    environment.step(None);
    check_near(environment.state().velocity.x, 2.0f, 0.0001f, "no-op did not apply braking");
    environment.step(None);
    check_near(environment.state().velocity.x, 0.0f, 0.0001f, "braking did not stop at zero");
}

// verifies continuous collision clamping prevents wall tunneling
void test_wall_collision_clamps_without_tunneling()
{
    using enum Action;

    EnvironmentConfig config = make_test_config();
    config.fixed_time_step = 1.0f;
    config.acceleration = 100.0f;
    config.max_speed = 100.0f;
    const Dungeon dungeon = make_test_dungeon({0.0f, 0.0f, 10.0f, 10.0f}, {5.0f, 5.0f}, {{1.0f, 1.0f}});
    Environment environment(config, 1);
    environment.reset(dungeon);
    environment.step(Right);

    check_near(environment.state().position.x, 9.5f - NAVIGATION_EPSILON, 0.0002f,
        "continuous collision did not clamp to the wall");
    check_near(environment.state().velocity.x, 0.0f, 0.0001f, "blocked velocity component was not cleared");
    check(is_navigable(environment.dungeon(), environment.state().position, config.player_size),
        "collision left the player outside navigable geometry");
}

// verifies swept pickup collects each crossed pellet only once
void test_swept_pickup_and_no_double_eat()
{
    using enum Action;

    EnvironmentConfig config = make_test_config();
    config.fixed_time_step = 1.0f;
    config.acceleration = 20.0f;
    config.max_speed = 20.0f;
    const Dungeon dungeon = make_test_dungeon(
        {0.0f, 0.0f, 40.0f, 10.0f},
        {2.0f, 5.0f},
        {{8.0f, 5.0f}, {14.0f, 5.0f}, {35.0f, 5.0f}});
    Environment environment(config, 1);
    environment.reset(dungeon);
    const StepResult first = environment.step(Right);

    check(first.pellets_eaten == 2, "swept movement did not collect every crossed pellet");
    check(first.score == 2 && first.pellets_remaining == 1, "pickup counters are incorrect");
    check(environment.state().ticks_since_food == 0, "pickup did not reset the food timer");

    const StepResult second = environment.step(None);
    check(second.pellets_eaten == 0 && second.score == 2, "eaten pellet was collected more than once");
}

// verifies starvation occurs on its exact fixed-tick deadline
void test_starvation_exact_tick()
{
    using enum TerminationReason;
    constexpr Action IDLE = Action::None;

    EnvironmentConfig config = make_test_config();
    config.fixed_time_step = 0.25f;
    config.starvation_seconds = 1.0;
    const Dungeon dungeon = make_test_dungeon({0.0f, 0.0f, 20.0f, 10.0f}, {2.0f, 5.0f}, {{18.0f, 5.0f}});
    Environment environment(config, 1);
    environment.reset(dungeon);

    for (int index = 0; index < 3; ++index)
    {
        environment.step(IDLE);
        check(environment.state().termination == None, "starvation occurred too early");
    }

    environment.step(IDLE);
    check(environment.state().termination == Starved, "starvation did not occur on its deadline");
    check_near(static_cast<float>(environment.seconds_since_food()), 1.0f, 0.0001f, "food stopwatch is incorrect");
}

// verifies a deadline-tick pickup clears before starvation or timeout
void test_boundary_pickup_wins_over_deadlines()
{
    using enum Action;

    EnvironmentConfig config = make_test_config();
    config.fixed_time_step = 0.25f;
    config.player_size = 0.2f;
    config.acceleration = 4.0f;
    config.max_speed = 1.0f;
    config.starvation_seconds = 1.0;
    config.time_limit_seconds = 1.0;
    const Dungeon dungeon = make_test_dungeon({0.0f, 0.0f, 10.0f, 10.0f}, {2.0f, 5.0f}, {{3.0f, 5.0f}});
    Environment environment(config, 1);
    environment.reset(dungeon);

    for (int index = 0; index < 4; ++index)
    {
        environment.step(Right);
    }

    check(environment.state().termination == TerminationReason::Cleared,
        "clear did not take precedence over starvation and time limit");
    check(environment.state().score == 1, "deadline pickup did not update score");
    check(environment.state().ticks_since_food == 0, "deadline pickup did not reset hunger");
}

// verifies the episode time limit occurs on its exact fixed tick
void test_time_limit_exact_tick()
{
    using enum TerminationReason;
    constexpr Action IDLE = Action::None;

    EnvironmentConfig config = make_test_config();
    config.fixed_time_step = 0.25f;
    config.time_limit_seconds = 1.0;
    const Dungeon dungeon = make_test_dungeon({0.0f, 0.0f, 20.0f, 10.0f}, {2.0f, 5.0f}, {{18.0f, 5.0f}});
    Environment environment(config, 1);
    environment.reset(dungeon);

    for (int index = 0; index < 3; ++index)
    {
        environment.step(IDLE);
        check(environment.state().termination == None, "time limit occurred too early");
    }

    environment.step(IDLE);
    check(environment.state().termination == TimeLimit,
        "time limit did not occur on its deadline");
}

// verifies terminal state remains immutable until an explicit reset
void test_terminal_requires_explicit_reset()
{
    using enum TerminationReason;
    constexpr Action IDLE = Action::None;

    EnvironmentConfig config = make_test_config();
    config.fixed_time_step = 0.25f;
    config.time_limit_seconds = 0.25;
    const Dungeon dungeon = make_test_dungeon({0.0f, 0.0f, 20.0f, 10.0f}, {2.0f, 5.0f}, {{18.0f, 5.0f}});
    Environment environment(config, 1);
    environment.reset(dungeon);
    environment.step(IDLE);

    const EnvironmentState terminal_state = environment.state();
    bool threw = false;

    try
    {
        environment.step(IDLE);
    }
    catch (const logic_error&)
    {
        threw = true;
    }

    check(threw, "stepping a terminal environment did not fail");
    check(environment.state().position == terminal_state.position
        && environment.state().elapsed_ticks == terminal_state.elapsed_ticks
        && environment.state().score == terminal_state.score,
        "terminal step attempt mutated environment state");

    environment.reset(dungeon);
    check(environment.state().termination == None
        && environment.state().elapsed_ticks == 0
        && environment.state().score == 0,
        "explicit reset did not clear episode state");
}

// verifies invalid movement geometry is rejected during construction
void test_invalid_configuration_fails_loudly()
{
    EnvironmentConfig config = make_test_config();
    config.dungeon.min_corridor_width = config.player_size;
    config.dungeon.max_corridor_width = config.player_size;
    bool threw = false;

    try
    {
        Environment environment(config, 1);
    }
    catch (const invalid_argument&)
    {
        threw = true;
    }

    check(threw, "invalid player and corridor clearance was accepted");
}

}

// runs every environment test and reports failures through the exit code
int main()
{
    const vector<pair<const char*, function<void()>>> tests{
        {"generation_is_deterministic_and_varied", test_generation_is_deterministic_and_varied},
        {"acceleration_speed_cap_and_braking", test_acceleration_speed_cap_and_braking},
        {"wall_collision_clamps_without_tunneling", test_wall_collision_clamps_without_tunneling},
        {"swept_pickup_and_no_double_eat", test_swept_pickup_and_no_double_eat},
        {"starvation_exact_tick", test_starvation_exact_tick},
        {"boundary_pickup_wins_over_deadlines", test_boundary_pickup_wins_over_deadlines},
        {"time_limit_exact_tick", test_time_limit_exact_tick},
        {"terminal_requires_explicit_reset", test_terminal_requires_explicit_reset},
        {"invalid_configuration_fails_loudly", test_invalid_configuration_fails_loudly}
    };
    size_t failures = 0;

    for (const auto& [name, test] : tests)
    {
        try
        {
            test();
            cout << "pass " << name << '\n';
        }
        catch (const exception& error)
        {
            ++failures;
            cerr << "fail " << name << ": " << error.what() << '\n';
        }
    }

    if (failures > 0)
    {
        cerr << failures << " test(s) failed\n";
        return 1;
    }

    cout << tests.size() << " tests passed\n";
    return 0;
}
