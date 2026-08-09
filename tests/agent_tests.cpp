#include "neurosim/brain_agent.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace neurosim;

namespace
{

// fails an agent test when a boolean condition is false
void check(bool condition, const string& message)
{
    if (!condition)
    {
        throw runtime_error(message);
    }
}

// fails an agent test when two floats differ beyond tolerance
void check_near(float actual, float expected, float tolerance, const string& message)
{
    if (!isfinite(actual) || !isfinite(expected) || abs(actual - expected) > tolerance)
    {
        throw runtime_error(message + ": expected " + to_string(expected)
            + ", received " + to_string(actual));
    }
}

// creates a valid one-room dungeon with requested pellets
Dungeon make_test_dungeon(Rect floor, Vec2 spawn, vector<Vec2> pellets)
{
    Dungeon dungeon;
    dungeon.seed = 91;
    dungeon.rooms.push_back({floor.center(), floor, {floor}});
    dungeon.floor_rects.push_back(floor);
    dungeon.spawn = spawn;
    dungeon.bounds = floor.expanded(1.0f);

    for (const Vec2 pellet : pellets)
    {
        dungeon.pellets.push_back({pellet});
    }

    return dungeon;
}

// returns a compact environment configuration for agent tests
EnvironmentConfig make_environment_config(uint32_t pellet_count = 3U)
{
    EnvironmentConfig config;
    config.dungeon.min_rooms = 2;
    config.dungeon.max_rooms = 2;
    config.dungeon.min_wings = 0;
    config.dungeon.max_wings = 0;
    config.dungeon.min_pellets_per_room = pellet_count;
    config.dungeon.max_pellets_per_room = pellet_count;
    config.fixed_time_step = 0.1f;
    config.vision_radius = 10.0f;
    config.player_size = 1.0f;
    config.acceleration = 10.0f;
    config.braking = 20.0f;
    config.max_speed = 4.0f;
    config.starvation_seconds = 10.0;
    config.time_limit_seconds = 100.0;
    return config;
}

// returns a compact generic controller configuration for deterministic tests
AdaptiveControllerConfig make_controller_config()
{
    AdaptiveControllerConfig config;
    config.brain_shape = {7, 7, 7};
    config.brain_radius = 3.0f;
    config.brain_time_step_seconds = 0.02f;
    config.minimum_bias_drive = 0.0f;
    config.maximum_bias_drive = 0.0f;
    config.exploration_interval_seconds = 0.5f;
    config.exploration_duration_seconds = 0.1f;
    return config;
}

// returns a small dungeon agent configuration that fits every test input
BrainAgentConfig make_agent_config()
{
    BrainAgentConfig config;
    config.wall_ray_count = 4;
    config.controller = make_controller_config();
    return config;
}

// verifies every visible pellet receives exact bounded relative signals
void test_visible_pellets_are_complete_and_ordered()
{
    const EnvironmentConfig config = make_environment_config();
    Environment environment(config, 1);
    environment.reset(make_test_dungeon(
        {-12.0f, -8.0f, 12.0f, 8.0f},
        {0.0f, 0.0f},
        {{3.0f, -4.0f}, {-6.0f, 0.0f}, {11.0f, 0.0f}}));
    const AgentObservation observation = observe_environment(environment, 3, 4);

    check(observation.visible_pellets == 2U, "observation included a pellet beyond vision");
    check_near(observation.pellets[0].present, 1.0f, 0.0f,
        "nearest visible pellet is absent");
    check_near(observation.pellets[0].right, 0.3f, 0.00001f,
        "nearest pellet horizontal position is incorrect");
    check_near(observation.pellets[0].up, 0.4f, 0.00001f,
        "nearest pellet vertical position is incorrect");
    check_near(observation.pellets[1].left, 0.6f, 0.00001f,
        "second pellet relative position is incorrect");
    check_near(observation.pellets[2].present, 0.0f, 0.0f,
        "unused pellet slot was not silent");

    bool threw = false;

    try
    {
        observe_environment(environment, 1, 4);
    }
    catch (const runtime_error&)
    {
        threw = true;
    }

    check(threw, "observation silently truncated visible pellets");
}

// verifies cardinal wall rays measure the player-center navigable boundary
void test_wall_rays_use_navigable_geometry()
{
    const EnvironmentConfig config = make_environment_config();
    Environment environment(config, 1);
    environment.reset(make_test_dungeon(
        {-12.0f, -8.0f, 12.0f, 8.0f},
        {0.0f, 0.0f},
        {{3.0f, -4.0f}, {-6.0f, 0.0f}, {11.0f, 0.0f}}));
    const AgentObservation observation = observe_environment(environment, 3, 4);

    check_near(observation.wall_proximity[0], 0.25001f, 0.0001f,
        "upward wall ray ignored player clearance");
    check_near(observation.wall_proximity[1], 0.0f, 0.0001f,
        "right wall beyond vision produced proximity");
    check_near(observation.wall_proximity[2], 0.25001f, 0.0001f,
        "downward wall ray ignored player clearance");
    check_near(observation.wall_proximity[3], 0.0f, 0.0001f,
        "left wall beyond vision produced proximity");
}

// verifies the default brain leaves substantial capacity beyond its task interface
void test_default_brain_capacity()
{
    const EnvironmentConfig environment_config;
    const BrainAgent agent(environment_config, {}, 17);
    const BrainModel& model = agent.model();
    check(model.positions.size() == 5'041U, "default brain neuron count is incorrect");
    check(model.input_neurons.size() == 841U, "default brain interface count is incorrect");
    check(model.output_neurons.size() == 4U, "default brain output count is incorrect");
    check(model.positions.size() - model.input_neurons.size() == 4'200U,
        "default brain internal neuron capacity is incorrect");
}

// verifies generic seeded input and output selection is distinct and reproducible
void test_seeded_controller_interfaces_are_reproducible()
{
    constexpr uint32_t INPUT_COUNT = 17;
    constexpr uint32_t OUTPUT_COUNT = 5;
    constexpr float CONTROL_TIME_STEP = 0.1f;
    constexpr float SLOW_CONTROL_TIME_STEP = 0.2f;
    const AdaptiveControllerConfig config = make_controller_config();
    AdaptiveController first(INPUT_COUNT, OUTPUT_COUNT, CONTROL_TIME_STEP, config, 42);
    const AdaptiveController repeated(INPUT_COUNT, OUTPUT_COUNT, CONTROL_TIME_STEP, config, 42);
    const AdaptiveController different(INPUT_COUNT, OUTPUT_COUNT, CONTROL_TIME_STEP, config, 43);
    AdaptiveController slower_control(
        INPUT_COUNT,
        OUTPUT_COUNT,
        SLOW_CONTROL_TIME_STEP,
        config,
        42);
    AdaptiveControllerConfig alternate_config = config;
    alternate_config.exploration_patterns = {
        {1.0f, 1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 1.0f, 1.0f}
    };
    const AdaptiveController alternate_task(
        INPUT_COUNT,
        OUTPUT_COUNT,
        CONTROL_TIME_STEP,
        alternate_config,
        42);
    const BrainModel& model = first.model();

    check(model.input_neurons == repeated.model().input_neurons
        && model.output_neurons == repeated.model().output_neurons
        && model.initial_connection_strengths
            == repeated.model().initial_connection_strengths,
        "same controller seed did not reproduce brain construction");
    check(model.input_neurons == alternate_task.model().input_neurons
            && model.output_neurons == alternate_task.model().output_neurons
            && model.initial_connection_strengths
                == alternate_task.model().initial_connection_strengths,
        "task action patterns changed generic brain construction");
    check(model.input_neurons != different.model().input_neurons
            || model.output_neurons != different.model().output_neurons,
        "different controller seed did not change its interfaces");
    check(model.input_neurons.size() == INPUT_COUNT + OUTPUT_COUNT,
        "controller did not create the requested sensory and exploration inputs");
    check(model.output_neurons.size() == OUTPUT_COUNT,
        "controller did not create the requested outputs");

    vector<uint32_t> sensory_inputs(
        model.input_neurons.begin(),
        model.input_neurons.begin() + INPUT_COUNT);
    vector<uint32_t> unique_sensory_inputs = sensory_inputs;
    sort(unique_sensory_inputs.begin(), unique_sensory_inputs.end());
    check(adjacent_find(unique_sensory_inputs.begin(), unique_sensory_inputs.end())
            == unique_sensory_inputs.end(),
        "controller selected a sensory neuron more than once");

    vector<uint32_t> unique_outputs = model.output_neurons;
    sort(unique_outputs.begin(), unique_outputs.end());
    check(adjacent_find(unique_outputs.begin(), unique_outputs.end()) == unique_outputs.end(),
        "controller selected an output neuron more than once");

    for (const uint32_t sensory_input : sensory_inputs)
    {
        check(!binary_search(unique_outputs.begin(), unique_outputs.end(), sensory_input),
            "controller reused an output neuron as a sensory input");
    }

    for (uint32_t output = 0; output < OUTPUT_COUNT; ++output)
    {
        check(model.input_neurons[INPUT_COUNT + output] == model.output_neurons[output],
            "controller exploration input does not target its anonymous output");
    }

    check_near(
        model.config.dynamics.fixed_time_step_seconds,
        config.brain_time_step_seconds,
        0.000001f,
        "controller model did not preserve its configured neural time step");
    check_near(
        slower_control.model().config.dynamics.fixed_time_step_seconds,
        config.brain_time_step_seconds,
        0.000001f,
        "slower control timing changed the controller model time step");
    check_near(
        slower_control.model().config.dynamics.fixed_time_step_seconds,
        model.config.dynamics.fixed_time_step_seconds,
        0.0f,
        "compatible control rates produced different neural time steps");

    first.set_exploration_enabled(false);
    slower_control.set_exploration_enabled(false);
    first.reset_episode(7);
    slower_control.reset_episode(7);
    const vector<float> inputs(INPUT_COUNT, 0.4f);
    first.step(inputs);
    first.step(inputs);
    slower_control.step(inputs);
    const BrainState fast_state = first.copy_state();
    const BrainState slow_state = slower_control.copy_state();
    check(fast_state.tick == slow_state.tick
            && fast_state.membrane_potentials == slow_state.membrane_potentials
            && fast_state.activity_traces == slow_state.activity_traces
            && fast_state.connection_strengths == slow_state.connection_strengths
            && fast_state.connection_eligibilities == slow_state.connection_eligibilities
            && fast_state.spikes == slow_state.spikes,
        "host control rate changed equal-time neural evolution");
}

// verifies observations and outcome modulation cannot steer exploration scheduling
void test_exploration_is_task_blind()
{
    constexpr float CONTROL_TIME_STEP = 0.1f;
    AdaptiveControllerConfig config = make_controller_config();
    config.exploration_interval_seconds = 0.25f;
    config.exploration_duration_seconds = 0.1f;
    config.exploration_patterns = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 1.0f}
    };
    AdaptiveController quiet(4, 3, CONTROL_TIME_STEP, config, 81);
    AdaptiveController stimulated(4, 3, CONTROL_TIME_STEP, config, 81);
    quiet.reset_episode(19);
    stimulated.reset_episode(19);
    const vector<float> quiet_inputs(4, 0.0f);
    const vector<float> changing_inputs{1.0f, 0.25f, 0.75f, 0.5f};

    for (uint32_t tick = 0; tick < 12U; ++tick)
    {
        const AdaptiveControllerStatus& quiet_status = quiet.step(quiet_inputs);
        const AdaptiveControllerStatus& stimulated_status = stimulated.step(changing_inputs);
        const bool expected_exploration = tick % 3U == 0U;

        check(quiet_status.exploring == expected_exploration,
            "controller exploration did not follow its fixed schedule");
        check(stimulated_status.exploring == quiet_status.exploring,
            "observation values changed the exploration schedule");
        check(stimulated_status.exploration_outputs == quiet_status.exploration_outputs,
            "observation values changed the exploration pattern");
        stimulated.apply_modulation(tick % 2U == 0U ? 0.6f : -0.2f);
    }
}

// verifies a frozen controller rejects every direct modulation update
void test_controller_freeze_blocks_modulation()
{
    AdaptiveControllerConfig config = make_controller_config();
    config.inhibitory_fraction = 0.0f;
    config.minimum_bias_drive = 60.0f;
    config.maximum_bias_drive = 60.0f;
    AdaptiveController controller(4, 3, 0.1f, config, 23);
    const vector<float> inputs(4, 0.0f);
    controller.step(inputs);
    const vector<float> enabled_strengths = controller.copy_connection_strengths();
    controller.apply_modulation(1.0f);
    check(controller.copy_connection_strengths() != enabled_strengths,
        "enabled controller ignored eligible direct modulation");

    controller.step(inputs);
    controller.set_learning_enabled(false);
    check(!controller.learning_enabled(), "controller did not enter frozen mode");
    const vector<float> frozen_strengths = controller.copy_connection_strengths();
    controller.apply_modulation(1.0f);
    check(controller.copy_connection_strengths() == frozen_strengths,
        "frozen controller applied direct modulation");
}

// verifies food changes eligible strengths while frozen validation cannot
void test_food_modulation_respects_learning_mode()
{
    using enum Action;

    const EnvironmentConfig environment_config = make_environment_config();
    const BrainAgentConfig agent_config = make_agent_config();
    Environment environment(environment_config, 1);
    environment.reset(make_test_dungeon(
        {-12.0f, -8.0f, 12.0f, 8.0f},
        {0.0f, 0.0f},
        {{3.0f, -4.0f}, {-6.0f, 0.0f}, {11.0f, 0.0f}}));
    BrainAgent agent(environment_config, agent_config, 12);
    environment.step(None);
    agent.decide(environment);
    const vector<float> before = agent.copy_connection_strengths();
    StepResult food;
    food.pellets_eaten = 1U;
    agent.observe_transition(food);
    check(agent.copy_connection_strengths() != before,
        "food did not modulate eligible strengths");
    agent.set_learning_enabled(false);
    agent.reset_episode(2);
    agent.decide(environment);
    const vector<float> frozen = agent.copy_connection_strengths();
    agent.observe_transition(food);
    check(agent.copy_connection_strengths() == frozen,
        "frozen validation changed learned strengths");
}

// verifies dungeon outcomes map to signed rate-limited modulation
void test_dungeon_feedback_is_signed_and_rate_limited()
{
    const EnvironmentConfig environment_config = make_environment_config();
    BrainAgentConfig agent_config = make_agent_config();
    agent_config.food_modulator = 0.75f;
    agent_config.wall_modulator = -0.2f;
    agent_config.wall_feedback_interval_seconds = 0.25f;
    Environment environment(environment_config, 1);
    environment.reset(make_test_dungeon(
        {-12.0f, -8.0f, 12.0f, 8.0f},
        {0.0f, 0.0f},
        {{3.0f, -4.0f}, {-6.0f, 0.0f}, {11.0f, 0.0f}}));
    BrainAgent agent(environment_config, agent_config, 77);
    agent.decide(environment);

    StepResult food;
    food.pellets_eaten = 2U;
    agent.observe_transition(food);
    check_near(agent.status().applied_modulator, 1.5f, 0.000001f,
        "food did not produce positive proportional modulation");

    StepResult collision;
    collision.hit_wall = true;
    agent.decide(environment);
    agent.observe_transition(collision);
    check_near(agent.status().applied_modulator, -0.2f, 0.000001f,
        "wall collision did not produce negative modulation");

    for (uint32_t blocked_tick = 0; blocked_tick < 2U; ++blocked_tick)
    {
        agent.decide(environment);
        agent.observe_transition(collision);
        check_near(agent.status().applied_modulator, 0.0f, 0.0f,
            "wall feedback ignored its rate limit");
    }

    agent.decide(environment);
    agent.observe_transition(collision);
    check_near(agent.status().applied_modulator, -0.2f, 0.000001f,
        "wall feedback did not resume after its rate limit");
}

// verifies environment transitions report attempted movement into a wall
void test_environment_reports_wall_hits()
{
    using enum Action;

    EnvironmentConfig config = make_environment_config(1);
    Environment environment(config, 1);
    environment.reset(make_test_dungeon(
        {-2.0f, -2.0f, 2.0f, 2.0f},
        {0.0f, 0.0f},
        {{-1.0f, 1.0f}}));
    bool hit_wall = false;

    for (uint32_t step = 0; step < 20U; ++step)
    {
        hit_wall = environment.step(Right).hit_wall || hit_wall;
    }

    check(hit_wall, "environment did not report a wall collision");
}

// verifies resolved wall and corner positions remain observable across procedural dungeons
void test_observation_accepts_resolved_boundaries()
{
    using enum Action;

    constexpr array<Action, 8> ACTIONS{
        Up,
        Up | Right,
        Right,
        Down | Right,
        Down,
        Down | Left,
        Left,
        Up | Left
    };
    EnvironmentConfig config;
    const uint32_t pellet_slots = config.dungeon.max_rooms
        * config.dungeon.max_pellets_per_room;

    for (uint64_t seed = 1; seed <= 250U; ++seed)
    {
        Environment environment(config, seed);
        uint64_t step = 0;

        while (environment.state().termination == TerminationReason::None)
        {
            observe_environment(environment, pellet_slots, 4);
            const size_t action = static_cast<size_t>((step / 37U + seed) % ACTIONS.size());
            environment.step(ACTIONS[action]);
            ++step;
        }
    }
}

// verifies collision rejects sub-epsilon gaps between player-center regions
void test_resolver_rejects_sub_epsilon_corner_gap()
{
    using enum Action;

    EnvironmentConfig config = make_environment_config(1);
    config.player_size = 1.0f;
    const float inset = config.player_size * 0.5f + NAVIGATION_EPSILON;
    const vector<Rect> centers{
        {-2.0f, -2.0f, 0.0f, 0.0f},
        {NAVIGATION_EPSILON * 0.5f, 0.0f, 2.0f, 2.0f},
        {-2.0f, -1.0f, -1.0f, 2.0f},
        {-2.0f, 1.0f, 1.0f, 2.0f}
    };
    Dungeon dungeon;
    dungeon.seed = 31;

    for (const Rect center : centers)
    {
        dungeon.floor_rects.push_back(center.expanded(inset));
    }

    dungeon.spawn = {-0.5f, -0.5f};
    dungeon.pellets.push_back({{-1.5f, -1.5f}});
    dungeon.bounds = joined(dungeon.floor_rects[0], dungeon.floor_rects[1]);
    dungeon.bounds = joined(dungeon.bounds, dungeon.floor_rects[2]);
    dungeon.bounds = joined(dungeon.bounds, dungeon.floor_rects[3]).expanded(1.0f);
    dungeon.rooms.push_back({dungeon.spawn, dungeon.bounds, dungeon.floor_rects});
    Environment environment(config, 1);
    environment.reset(move(dungeon));

    for (uint32_t step = 0; step < 8U; ++step)
    {
        environment.step(Right);
        observe_environment(environment, 2, 4);
    }

    bool hit_wall = false;

    for (uint32_t step = 0; step < 8U; ++step)
    {
        const StepResult transition = environment.step(Down);
        hit_wall = hit_wall || transition.hit_wall;
        check(is_navigable(
                environment.dungeon(),
                environment.state().position,
                environment.config().player_size),
            "resolver placed the player inside a sub-epsilon floor gap");
        observe_environment(environment, 2, 4);
    }

    check(hit_wall, "sub-epsilon floor gap was not reported as a wall");
}

}

// runs every brain-agent test and reports failures through the exit code
int main()
{
    const vector<pair<const char*, function<void()>>> tests{
        {"visible_pellets_are_complete_and_ordered", test_visible_pellets_are_complete_and_ordered},
        {"wall_rays_use_navigable_geometry", test_wall_rays_use_navigable_geometry},
        {"default_brain_capacity", test_default_brain_capacity},
        {"seeded_controller_interfaces_are_reproducible",
            test_seeded_controller_interfaces_are_reproducible},
        {"exploration_is_task_blind", test_exploration_is_task_blind},
        {"controller_freeze_blocks_modulation", test_controller_freeze_blocks_modulation},
        {"food_modulation_respects_learning_mode", test_food_modulation_respects_learning_mode},
        {"dungeon_feedback_is_signed_and_rate_limited",
            test_dungeon_feedback_is_signed_and_rate_limited},
        {"environment_reports_wall_hits", test_environment_reports_wall_hits},
        {"observation_accepts_resolved_boundaries",
            test_observation_accepts_resolved_boundaries},
        {"resolver_rejects_sub_epsilon_corner_gap",
            test_resolver_rejects_sub_epsilon_corner_gap}
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

    if (failures > 0U)
    {
        cerr << failures << " agent test(s) failed\n";
        return 1;
    }

    cout << tests.size() << " agent tests passed\n";
    return 0;
}
