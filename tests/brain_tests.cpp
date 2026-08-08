#include "neurosim/brain.h"
#include "neurosim/cpu_brain.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
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
    if (!isfinite(actual) || !isfinite(expected) || abs(actual - expected) > tolerance)
    {
        throw runtime_error(message + ": expected " + to_string(expected)
            + ", received " + to_string(actual));
    }
}

// returns deterministic dynamics with plasticity disabled by default
BrainConfig make_test_config();

// verifies compiled models and restored states reject internal corruption
void test_compiled_model_and_state_validation()
{
    BrainLayout layout;
    layout.shape = {3, 1, 1};
    layout.neurons = {
        {{0, 0, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{1, 0, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{2, 0, 0}, NeuronPolarity::Excitatory, 0.0f}
    };
    BrainConfig config = make_test_config();
    config.dynamics.refractory_period_seconds = 0.002f;
    const BrainModel valid = build_brain_model(layout, config, 1);
    const uint32_t target = neuron_index_at(valid, {1, 0, 0});
    BrainModel duplicate_incoming = valid;
    const uint32_t incoming_begin = duplicate_incoming.incoming_offsets[target];
    duplicate_incoming.incoming_edges[incoming_begin + 1U]
        = duplicate_incoming.incoming_edges[incoming_begin];
    bool incoming_threw = false;

    try
    {
        validate_brain_model(duplicate_incoming);
    }
    catch (const invalid_argument&)
    {
        incoming_threw = true;
    }

    check(incoming_threw, "duplicate incoming edge was accepted");

    BrainModel wrong_decay = valid;
    wrong_decay.trace_decay *= 0.5f;
    bool decay_threw = false;

    try
    {
        validate_brain_model(wrong_decay);
    }
    catch (const invalid_argument&)
    {
        decay_threw = true;
    }

    check(decay_threw, "inconsistent derived trace decay was accepted");

    BrainState impossible = make_brain_state(valid);
    impossible.refractory_ticks[0] = valid.refractory_ticks + 1U;
    bool state_threw = false;

    try
    {
        validate_brain_state(valid, impossible);
    }
    catch (const invalid_argument&)
    {
        state_threw = true;
    }

    check(state_threw, "impossible refractory state was accepted");

    BrainLayout single;
    single.shape = {1, 1, 1};
    single.neurons = {{{0, 0, 0}, NeuronPolarity::Excitatory, 0.0f}};
    BrainConfig large_radius = make_test_config();
    large_radius.connection_radius_steps = numeric_limits<uint32_t>::max();
    const BrainModel bounded = build_brain_model(single, large_radius, 1);
    check(bounded.outgoing_targets.empty(),
        "large connection radius created an edge outside a one-cell grid");
}

// advances a test simulator from a short stable input list
void step_with(CpuBrainSimulator& simulator, initializer_list<float> input_drives)
{
    simulator.step(span<const float>(input_drives.begin(), input_drives.size()));
}

// returns deterministic dynamics with plasticity disabled by default
BrainConfig make_test_config()
{
    BrainConfig config;
    config.connection_radius_steps = 2;
    config.dynamics.fixed_time_step_seconds = 0.001f;
    config.dynamics.membrane_time_constant_seconds = 0.020f;
    config.dynamics.resting_potential = 0.0f;
    config.dynamics.reset_potential = 0.0f;
    config.dynamics.firing_threshold = 1.0f;
    config.dynamics.refractory_period_seconds = 0.0f;
    config.plasticity.activity_trace_time_constant_seconds = 0.020f;
    config.plasticity.learning_rate = 0.0f;
    config.plasticity.weakening_rate_per_second = 0.0f;
    config.plasticity.outgoing_strength_budget = 1.0f;
    config.plasticity.initial_budget_fraction = 0.0f;
    config.plasticity.initial_connection_density = 0.0f;
    return config;
}

// creates a two-neuron directed-signal fixture with ordered channels
BrainLayout make_pair_layout(NeuronPolarity source_polarity = NeuronPolarity::Excitatory)
{
    BrainLayout layout;
    layout.shape = {3, 1, 1};
    layout.neurons = {
        {{0, 0, 0}, source_polarity, 0.0f},
        {{1, 0, 0}, NeuronPolarity::Excitatory, 0.0f}
    };
    layout.input_positions = {{0, 0, 0}};
    layout.output_positions = {{1, 0, 0}, {0, 0, 0}};
    return layout;
}

// verifies sparse occupancy, exact local range, direction, and channel order
void test_sparse_topology_and_interfaces()
{
    BrainLayout layout;
    layout.shape = {5, 5, 5};
    layout.neurons = {
        {{3, 3, 3}, NeuronPolarity::Excitatory, 0.0f},
        {{2, 2, 2}, NeuronPolarity::Excitatory, 0.0f},
        {{4, 2, 2}, NeuronPolarity::Inhibitory, 0.0f},
        {{3, 3, 2}, NeuronPolarity::Excitatory, 0.0f},
        {{0, 0, 0}, NeuronPolarity::Excitatory, 0.0f}
    };
    layout.input_positions = {{4, 2, 2}, {2, 2, 2}};
    layout.output_positions = {{3, 3, 3}, {2, 2, 2}};
    const BrainModel model = build_brain_model(layout, make_test_config(), 17);
    const uint32_t center = neuron_index_at(model, {2, 2, 2});
    const uint32_t axial_two = neuron_index_at(model, {4, 2, 2});
    const uint32_t diagonal_two = neuron_index_at(model, {3, 3, 2});
    const uint32_t distance_three = neuron_index_at(model, {3, 3, 3});

    check(neuron_index_at(model, {1, 1, 1}) == INVALID_NEURON_INDEX,
        "empty brain cell mapped to a neuron");
    check(connection_index(model, center, axial_two) != INVALID_NEURON_INDEX,
        "two-step axial neighbor was not connectable");
    check(connection_index(model, center, diagonal_two) != INVALID_NEURON_INDEX,
        "two-step diagonal neighbor was not connectable");
    check(connection_index(model, center, distance_three) == INVALID_NEURON_INDEX,
        "three-step neighbor was connectable");

    const uint32_t forward = connection_index(model, center, axial_two);
    const uint32_t reverse = connection_index(model, axial_two, center);
    check(forward != INVALID_NEURON_INDEX && reverse != INVALID_NEURON_INDEX && forward != reverse,
        "reciprocal directions did not receive independent connection entries");
    check(model.input_neurons[0] == axial_two && model.input_neurons[1] == center,
        "input channel order changed during topology compilation");
    check(model.output_neurons[0] == distance_three && model.output_neurons[1] == center,
        "output channel order changed during topology compilation");

    for (uint32_t source = 0; source < model.positions.size(); ++source)
    {
        check(model.outgoing_offsets[source + 1U] - model.outgoing_offsets[source] <= 24U,
            "two-step Manhattan neighborhood exceeded 24 possible connections");
    }
}

// verifies spherical layouts leave cells empty and use the expected lattice points
void test_sphere_layout_is_sparse()
{
    const BrainLayout layout = make_sphere_brain_layout({5, 5, 5}, 2.0f);
    check(layout.neurons.size() == 33U, "radius-two sphere occupied an unexpected cell count");

    const BrainModel model = build_brain_model(layout, make_test_config(), 3);
    const uint32_t center = neuron_index_at(model, {2, 2, 2});
    check(neuron_index_at(model, {2, 2, 2}) != INVALID_NEURON_INDEX,
        "sphere omitted its center cell");
    check(neuron_index_at(model, {0, 0, 0}) == INVALID_NEURON_INDEX,
        "sphere occupied an outside corner cell");
    check(model.outgoing_offsets[center + 1U] - model.outgoing_offsets[center] == 24U,
        "filled two-step manhattan neighborhood did not contain 24 connections");
}

// verifies seeded initial strengths reproduce and vary without changing topology
void test_seeded_initial_connections()
{
    BrainLayout layout = make_sphere_brain_layout({5, 5, 5}, 2.0f);
    BrainConfig config = make_test_config();
    config.plasticity.initial_budget_fraction = 1.0f;
    config.plasticity.initial_connection_density = 0.35f;
    const BrainModel first = build_brain_model(layout, config, 1234);
    const BrainModel repeated = build_brain_model(layout, config, 1234);
    const BrainModel different = build_brain_model(layout, config, 1235);

    check(first.positions == repeated.positions
        && first.outgoing_offsets == repeated.outgoing_offsets
        && first.outgoing_targets == repeated.outgoing_targets
        && first.initial_connection_strengths == repeated.initial_connection_strengths,
        "same brain seed did not reproduce initial topology state");
    check(first.outgoing_offsets == different.outgoing_offsets
        && first.outgoing_targets == different.outgoing_targets,
        "brain seed changed geometric connection candidates");
    check(first.initial_connection_strengths != different.initial_connection_strengths,
        "different brain seeds did not vary initial strengths");

    for (uint32_t source = 0; source < first.positions.size(); ++source)
    {
        float sum = 0.0f;

        for (uint32_t edge = first.outgoing_offsets[source];
            edge < first.outgoing_offsets[source + 1U]; ++edge)
        {
            sum += first.initial_connection_strengths[edge];
        }

        check(sum <= config.plasticity.outgoing_strength_budget + 0.00001f,
            "initial connections exceeded their source budget");
    }
}

// verifies normalized euler integration and constant bias use documented units
void test_leaky_integration_and_bias()
{
    BrainLayout layout;
    layout.shape = {1, 1, 1};
    layout.neurons = {{{0, 0, 0}, NeuronPolarity::Excitatory, 20.0f}};
    BrainConfig config = make_test_config();
    config.dynamics.fixed_time_step_seconds = 0.005f;
    config.dynamics.membrane_time_constant_seconds = 0.020f;
    config.dynamics.firing_threshold = 10.0f;
    CpuBrainSimulator simulator(build_brain_model(layout, config, 1));
    step_with(simulator, {});
    check_near(simulator.state().membrane_potentials[0], 0.1f, 0.000001f,
        "first leaky integration step is incorrect");
    step_with(simulator, {});
    check_near(simulator.state().membrane_potentials[0], 0.175f, 0.000001f,
        "second leaky integration step is incorrect");
}

// verifies firing reset, refractory timing, and ordered output sampling
void test_threshold_reset_and_refractory()
{
    BrainLayout layout;
    layout.shape = {1, 1, 1};
    layout.neurons = {{{0, 0, 0}, NeuronPolarity::Excitatory, 0.0f}};
    layout.input_positions = {{0, 0, 0}};
    layout.output_positions = {{0, 0, 0}};
    BrainConfig config = make_test_config();
    config.dynamics.refractory_period_seconds = 0.002f;
    CpuBrainSimulator simulator(build_brain_model(layout, config, 1));

    step_with(simulator, {1000.0f});
    check(simulator.output_spike(0), "threshold crossing did not emit a spike");
    check_near(simulator.state().membrane_potentials[0], 0.0f, 0.0f,
        "spiking neuron did not reset its potential");
    check(simulator.state().refractory_ticks[0] == 2U,
        "spiking neuron started an incorrect refractory duration");

    step_with(simulator, {1000.0f});
    check(!simulator.output_spike(0) && simulator.state().refractory_ticks[0] == 1U,
        "first refractory tick was not suppressed");
    step_with(simulator, {1000.0f});
    check(!simulator.output_spike(0) && simulator.state().refractory_ticks[0] == 0U,
        "second refractory tick was not suppressed");
    step_with(simulator, {1000.0f});
    check(simulator.output_spike(0), "neuron did not fire after its refractory period");
}

// verifies recurrent signals arrive one tick later through a directed weight
void test_one_tick_signal_delay()
{
    const BrainModel model = build_brain_model(make_pair_layout(), make_test_config(), 1);
    CpuBrainSimulator simulator(model);
    BrainState state = simulator.state();
    const uint32_t source = neuron_index_at(model, {0, 0, 0});
    const uint32_t target = neuron_index_at(model, {1, 0, 0});
    state.connection_strengths[connection_index(model, source, target)] = 1.0f;
    simulator.reset(move(state));

    step_with(simulator, {1000.0f});
    check(simulator.output_spike(1), "input neuron did not fire on the first tick");
    check(!simulator.output_spike(0), "target received a same-tick recurrent signal");
    step_with(simulator, {0.0f});
    check(simulator.output_spike(0), "target did not receive the prior-tick recurrent signal");
}

// verifies inhibitory source polarity subtracts from target potential
void test_inhibitory_signal_sign()
{
    BrainConfig config = make_test_config();
    config.dynamics.membrane_time_constant_seconds = 1.0f;
    const BrainModel model = build_brain_model(
        make_pair_layout(NeuronPolarity::Inhibitory), config, 1);
    CpuBrainSimulator simulator(model);
    BrainState state = simulator.state();
    const uint32_t source = neuron_index_at(model, {0, 0, 0});
    const uint32_t target = neuron_index_at(model, {1, 0, 0});
    state.spikes[source] = 1U;
    state.membrane_potentials[target] = 0.5f;
    state.connection_strengths[connection_index(model, source, target)] = 0.25f;
    simulator.reset(move(state));
    step_with(simulator, {0.0f});
    check_near(simulator.state().membrane_potentials[target], 0.2495f, 0.000001f,
        "inhibitory connection did not subtract its strength");
}

// verifies joint activity strengthens one edge and consumes sibling budget
void test_hebbian_strengthening_and_budget_competition()
{
    BrainLayout layout;
    layout.shape = {3, 1, 1};
    layout.neurons = {
        {{0, 0, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{1, 0, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{2, 0, 0}, NeuronPolarity::Excitatory, 0.0f}
    };
    layout.input_positions = {{1, 0, 0}, {0, 0, 0}, {2, 0, 0}};
    BrainConfig config = make_test_config();
    config.plasticity.learning_rate = 0.1f;
    const BrainModel model = build_brain_model(layout, config, 1);
    CpuBrainSimulator simulator(model);
    BrainState state = simulator.state();
    const uint32_t source = neuron_index_at(model, {1, 0, 0});
    const uint32_t active_target = neuron_index_at(model, {0, 0, 0});
    const uint32_t quiet_target = neuron_index_at(model, {2, 0, 0});
    const uint32_t active_edge = connection_index(model, source, active_target);
    const uint32_t quiet_edge = connection_index(model, source, quiet_target);
    state.connection_strengths[active_edge] = 0.5f;
    state.connection_strengths[quiet_edge] = 0.5f;
    simulator.reset(move(state));
    step_with(simulator, {1000.0f, 1000.0f, 0.0f});

    const float strengthened = simulator.state().connection_strengths[active_edge];
    const float weakened_sibling = simulator.state().connection_strengths[quiet_edge];
    check(strengthened > 0.5f, "coactive connection did not strengthen");
    check(weakened_sibling < 0.5f, "sibling connection did not yield saturated budget");
    check_near(strengthened + weakened_sibling, 1.0f, 0.000001f,
        "outgoing budget projection produced an incorrect sum");

    const uint32_t reverse_edge = connection_index(model, active_target, source);
    check_near(simulator.state().connection_strengths[reverse_edge], 0.1f, 0.000001f,
        "reverse connection did not learn as an independent latent edge");
}

// verifies inactivity weakens weights and decays recent activity traces
void test_inactivity_decay()
{
    BrainConfig config = make_test_config();
    config.plasticity.weakening_rate_per_second = 2.0f;
    const BrainModel model = build_brain_model(make_pair_layout(), config, 1);
    CpuBrainSimulator simulator(model);
    BrainState state = simulator.state();
    const uint32_t source = neuron_index_at(model, {0, 0, 0});
    const uint32_t target = neuron_index_at(model, {1, 0, 0});
    const uint32_t edge = connection_index(model, source, target);
    state.connection_strengths[edge] = 0.5f;
    state.activity_traces[source] = 1.0f;
    simulator.reset(move(state));
    step_with(simulator, {0.0f});
    check_near(simulator.state().connection_strengths[edge], 0.5f * model.weight_decay, 0.000001f,
        "inactive connection did not use the configured weakening rate");
    check_near(simulator.state().activity_traces[source], model.trace_decay, 0.000001f,
        "inactive activity trace did not decay");
}

// verifies activity-only and full reset preserve different state intentionally
void test_reset_modes_and_replay()
{
    BrainConfig config = make_test_config();
    config.plasticity.learning_rate = 0.1f;
    BrainLayout layout = make_pair_layout();
    layout.input_positions.push_back({1, 0, 0});
    const BrainModel model = build_brain_model(layout, config, 1);
    CpuBrainSimulator simulator(model);
    step_with(simulator, {1000.0f, 1000.0f});
    step_with(simulator, {0.0f, 0.0f});
    const BrainState replay_state = simulator.state();
    const uint32_t edge = connection_index(model,
        neuron_index_at(model, {0, 0, 0}), neuron_index_at(model, {1, 0, 0}));
    const float learned_strength = replay_state.connection_strengths[edge];
    check(learned_strength > model.initial_connection_strengths[edge],
        "reset fixture did not learn a connection before reset");

    simulator.reset_activity();
    check_near(simulator.state().connection_strengths[edge], learned_strength, 0.0f,
        "activity reset discarded learned strength");
    check(simulator.state().tick == 0U
        && simulator.state().spikes[0] == 0U
        && simulator.state().activity_traces[0] == 0.0f,
        "activity reset retained transient neuron state");

    simulator.reset();
    check_near(simulator.state().connection_strengths[edge],
        model.initial_connection_strengths[edge], 0.0f,
        "full reset did not restore initial strength");
    step_with(simulator, {1000.0f, 1000.0f});
    step_with(simulator, {0.0f, 0.0f});
    check(simulator.state().membrane_potentials == replay_state.membrane_potentials
        && simulator.state().activity_traces == replay_state.activity_traces
        && simulator.state().connection_strengths == replay_state.connection_strengths
        && simulator.state().refractory_ticks == replay_state.refractory_ticks
        && simulator.state().spikes == replay_state.spikes
        && simulator.state().tick == replay_state.tick,
        "full reset did not reproduce the same CPU trajectory");
}

// verifies invalid layouts, configurations, and inputs fail before simulation
void test_invalid_boundaries_fail_loudly()
{
    BrainLayout duplicate = make_pair_layout();
    duplicate.neurons[1].position = duplicate.neurons[0].position;
    bool duplicate_threw = false;

    try
    {
        build_brain_model(duplicate, make_test_config(), 1);
    }
    catch (const invalid_argument&)
    {
        duplicate_threw = true;
    }

    check(duplicate_threw, "duplicate neuron position was accepted");

    BrainConfig invalid_config = make_test_config();
    invalid_config.dynamics.fixed_time_step_seconds = 0.1f;
    bool config_threw = false;

    try
    {
        build_brain_model(make_pair_layout(), invalid_config, 1);
    }
    catch (const invalid_argument&)
    {
        config_threw = true;
    }

    check(config_threw, "unstable brain time step was accepted");

    CpuBrainSimulator simulator(build_brain_model(make_pair_layout(), make_test_config(), 1));
    const BrainState before = simulator.state();
    bool input_threw = false;

    try
    {
        step_with(simulator, {numeric_limits<float>::quiet_NaN()});
    }
    catch (const invalid_argument&)
    {
        input_threw = true;
    }

    check(input_threw && simulator.state().tick == before.tick
        && simulator.state().membrane_potentials == before.membrane_potentials,
        "non-finite input did not fail before mutating brain state");

    bool count_threw = false;

    try
    {
        step_with(simulator, {});
    }
    catch (const invalid_argument&)
    {
        count_threw = true;
    }

    check(count_threw && simulator.state().tick == before.tick,
        "incorrect input count did not fail before mutating brain state");

    BrainLayout overflow_layout;
    overflow_layout.shape = {1, 1, 1};
    overflow_layout.neurons = {
        {{0, 0, 0}, NeuronPolarity::Excitatory, numeric_limits<float>::max()}
    };
    overflow_layout.input_positions = {{0, 0, 0}};
    CpuBrainSimulator overflow_simulator(build_brain_model(
        overflow_layout, make_test_config(), 1));
    bool overflow_threw = false;

    try
    {
        step_with(overflow_simulator, {numeric_limits<float>::max()});
    }
    catch (const runtime_error&)
    {
        overflow_threw = true;
    }

    bool reset_required = false;

    try
    {
        step_with(overflow_simulator, {0.0f});
    }
    catch (const logic_error&)
    {
        reset_required = true;
    }

    check(overflow_threw && reset_required,
        "numerical failure did not require an explicit simulator reset");
    overflow_simulator.reset();
}

}

// runs every brain model test and reports failures through the exit code
int main()
{
    const vector<pair<const char*, function<void()>>> tests{
        {"sparse_topology_and_interfaces", test_sparse_topology_and_interfaces},
        {"sphere_layout_is_sparse", test_sphere_layout_is_sparse},
        {"seeded_initial_connections", test_seeded_initial_connections},
        {"leaky_integration_and_bias", test_leaky_integration_and_bias},
        {"threshold_reset_and_refractory", test_threshold_reset_and_refractory},
        {"one_tick_signal_delay", test_one_tick_signal_delay},
        {"inhibitory_signal_sign", test_inhibitory_signal_sign},
        {"hebbian_strengthening_and_budget_competition", test_hebbian_strengthening_and_budget_competition},
        {"inactivity_decay", test_inactivity_decay},
        {"reset_modes_and_replay", test_reset_modes_and_replay},
        {"compiled_model_and_state_validation", test_compiled_model_and_state_validation},
        {"invalid_boundaries_fail_loudly", test_invalid_boundaries_fail_loudly}
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
        cerr << failures << " brain test(s) failed\n";
        return 1;
    }

    cout << tests.size() << " brain tests passed\n";
    return 0;
}
