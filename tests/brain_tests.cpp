#include "neurosim/brain.h"
#include "neurosim/cpu_brain.h"

#include <array>
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

// returns deterministic dynamics without weight changes by default
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

    BrainModel wrong_eligibility_decay = valid;
    wrong_eligibility_decay.eligibility_decay *= 0.5f;
    bool eligibility_decay_threw = false;

    try
    {
        validate_brain_model(wrong_eligibility_decay);
    }
    catch (const invalid_argument&)
    {
        eligibility_decay_threw = true;
    }

    check(eligibility_decay_threw, "inconsistent eligibility decay was accepted");

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

    BrainState excessive_eligibility = make_brain_state(valid);
    excessive_eligibility.connection_eligibilities[0]
        = valid.config.plasticity.maximum_eligibility + 0.1f;
    bool eligibility_threw = false;

    try
    {
        validate_brain_state(valid, excessive_eligibility);
    }
    catch (const invalid_argument&)
    {
        eligibility_threw = true;
    }

    check(eligibility_threw, "excessive connection eligibility was accepted");

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
void step_with(
    CpuBrainSimulator& simulator,
    initializer_list<float> input_drives,
    bool plasticity_enabled = true)
{
    simulator.step(
        span<const float>(input_drives.begin(), input_drives.size()),
        {},
        plasticity_enabled);
}

// returns deterministic dynamics without weight changes by default
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
    config.plasticity.eligibility_time_constant_seconds = 1.0f;
    config.plasticity.maximum_eligibility = 1.0f;
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

// verifies batched cpu execution matches repeated fixed-tick calls exactly
void test_batched_advance_matches_repeated_steps()
{
    BrainConfig config = make_test_config();
    config.plasticity.learning_rate = 0.1f;
    config.plasticity.weakening_rate_per_second = 0.02f;
    config.plasticity.initial_budget_fraction = 0.5f;
    config.plasticity.initial_connection_density = 1.0f;
    const BrainModel model = build_brain_model(make_pair_layout(), config, 31);
    CpuBrainSimulator repeated(model);
    CpuBrainSimulator batched(model);
    const array<float, 1> inputs{1000.0f};
    BrainAdvanceSummary expected;

    for (uint32_t tick = 0; tick < 4U; ++tick)
    {
        repeated.step(inputs);

        for (const uint8_t spike : repeated.state().spikes)
        {
            expected.neuron_spikes += spike;
        }

        for (const uint32_t output : model.output_neurons)
        {
            expected.output_spikes += repeated.state().spikes[output];
        }
    }

    vector<float> output_activity(model.output_neurons.size());
    const BrainAdvanceSummary actual = batched.advance(inputs, 4U, {}, true, output_activity);
    const BrainState& expected_state = repeated.state();
    const BrainState& actual_state = batched.state();
    check(actual.neuron_spikes == expected.neuron_spikes
            && actual.output_spikes == expected.output_spikes,
        "batched cpu execution reported incorrect spike totals");
    check(actual_state.membrane_potentials == expected_state.membrane_potentials
            && actual_state.activity_traces == expected_state.activity_traces
            && actual_state.connection_strengths == expected_state.connection_strengths
            && actual_state.connection_eligibilities == expected_state.connection_eligibilities
            && actual_state.refractory_ticks == expected_state.refractory_ticks
            && actual_state.spikes == expected_state.spikes
            && actual_state.tick == expected_state.tick,
        "batched cpu execution changed fixed-tick model state");

    for (uint32_t output = 0; output < output_activity.size(); ++output)
    {
        check_near(
            output_activity[output],
            expected_state.activity_traces[model.output_neurons[output]],
            0.0f,
            "batched cpu execution returned incorrect output activity");
    }
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

// verifies only earlier source activity marks the directed connection to a later target spike
void test_causal_eligibility_direction()
{
    BrainLayout layout = make_pair_layout();
    layout.input_positions.push_back({1, 0, 0});
    const BrainModel model = build_brain_model(layout, make_test_config(), 1);
    CpuBrainSimulator simulator(model);
    const uint32_t source = neuron_index_at(model, {0, 0, 0});
    const uint32_t target = neuron_index_at(model, {1, 0, 0});
    const uint32_t forward = connection_index(model, source, target);
    const uint32_t reverse = connection_index(model, target, source);

    step_with(simulator, {1000.0f, 0.0f});
    step_with(simulator, {0.0f, 0.0f});
    step_with(simulator, {0.0f, 1000.0f});

    check_near(simulator.state().connection_eligibilities[forward],
        model.trace_decay, 0.000001f,
        "causal source activity produced incorrect forward eligibility");
    check_near(simulator.state().connection_eligibilities[reverse], 0.0f, 0.0f,
        "causal activity incorrectly marked the reverse connection");
    check_near(simulator.state().connection_strengths[forward], 0.0f, 0.0f,
        "eligibility changed a connection before modulation");
}

// verifies inhibitory sources do not receive task-modulated eligibility
void test_inhibitory_plasticity_is_fixed()
{
    BrainLayout layout = make_pair_layout(NeuronPolarity::Inhibitory);
    layout.input_positions.push_back({1, 0, 0});
    const BrainModel model = build_brain_model(layout, make_test_config(), 1);
    CpuBrainSimulator simulator(model);
    const uint32_t source = neuron_index_at(model, {0, 0, 0});
    const uint32_t target = neuron_index_at(model, {1, 0, 0});
    const uint32_t edge = connection_index(model, source, target);
    step_with(simulator, {1000.0f, 0.0f});
    step_with(simulator, {0.0f, 0.0f});
    step_with(simulator, {0.0f, 1000.0f});
    check_near(simulator.state().connection_eligibilities[edge], 0.0f, 0.0f,
        "inhibitory source recorded task eligibility");
}

// verifies delayed positive and negative events change only eligible strengths
void test_delayed_signed_modulation()
{
    BrainLayout layout = make_pair_layout();
    layout.input_positions.push_back({1, 0, 0});
    BrainConfig config = make_test_config();
    config.plasticity.learning_rate = 0.2f;
    const BrainModel model = build_brain_model(layout, config, 1);
    const uint32_t source = neuron_index_at(model, {0, 0, 0});
    const uint32_t target = neuron_index_at(model, {1, 0, 0});
    const uint32_t edge = connection_index(model, source, target);
    CpuBrainSimulator rewarded(model);
    BrainState rewarded_state = rewarded.state();
    rewarded_state.connection_strengths[edge] = 0.4f;
    rewarded.reset(move(rewarded_state));
    step_with(rewarded, {1000.0f, 0.0f});
    step_with(rewarded, {0.0f, 1000.0f});
    step_with(rewarded, {0.0f, 0.0f});
    const float reward_eligibility = rewarded.state().connection_eligibilities[edge];
    rewarded.apply_plasticity_modulation(0.5f);
    check_near(rewarded.state().connection_strengths[edge],
        0.4f + config.plasticity.learning_rate * 0.5f * reward_eligibility,
        0.000001f,
        "delayed positive modulation produced an incorrect strength");

    CpuBrainSimulator punished(model);
    BrainState punished_state = punished.state();
    punished_state.connection_strengths[edge] = 0.4f;
    punished.reset(move(punished_state));
    step_with(punished, {1000.0f, 0.0f});
    step_with(punished, {0.0f, 1000.0f});
    step_with(punished, {0.0f, 0.0f});
    const float punishment_eligibility = punished.state().connection_eligibilities[edge];
    punished.apply_plasticity_modulation(-0.5f);
    check_near(punished.state().connection_strengths[edge],
        0.4f - config.plasticity.learning_rate * 0.5f * punishment_eligibility,
        0.000001f,
        "delayed negative modulation produced an incorrect strength");
}

// verifies positive modulation projects competing siblings onto one outgoing budget
void test_modulation_budget_competition()
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
    state.connection_eligibilities[active_edge] = 1.0f;
    simulator.reset(move(state));
    simulator.apply_plasticity_modulation(1.0f);

    const float strengthened = simulator.state().connection_strengths[active_edge];
    const float weakened_sibling = simulator.state().connection_strengths[quiet_edge];
    check(strengthened > 0.5f, "eligible connection did not strengthen");
    check(weakened_sibling < 0.5f, "sibling connection did not yield its saturated budget");
    check_near(strengthened + weakened_sibling, 1.0f, 0.000001f,
        "outgoing budget projection produced an incorrect sum");
}

// verifies eligibility is bounded and consumed only by a nonzero event
void test_bounded_and_consumed_eligibility()
{
    BrainConfig config = make_test_config();
    config.plasticity.learning_rate = 0.1f;
    config.plasticity.maximum_eligibility = 0.25f;
    BrainLayout layout = make_pair_layout();
    layout.input_positions.push_back({1, 0, 0});
    const BrainModel model = build_brain_model(layout, config, 1);
    CpuBrainSimulator simulator(model);
    BrainState state = simulator.state();
    const uint32_t source = neuron_index_at(model, {0, 0, 0});
    const uint32_t target = neuron_index_at(model, {1, 0, 0});
    const uint32_t edge = connection_index(model, source, target);
    state.activity_traces[source] = 10.0f;
    simulator.reset(move(state));
    step_with(simulator, {0.0f, 1000.0f});
    check_near(simulator.state().connection_eligibilities[edge], 0.25f, 0.0f,
        "causal eligibility exceeded its configured bound");
    simulator.apply_plasticity_modulation(0.0f);
    check_near(simulator.state().connection_eligibilities[edge], 0.25f, 0.0f,
        "zero modulation consumed pending eligibility");
    simulator.apply_plasticity_modulation(1.0f);

    for (const float eligibility : simulator.state().connection_eligibilities)
    {
        check_near(eligibility, 0.0f, 0.0f,
            "nonzero modulation did not consume all eligibility");
    }
}

// verifies disabled plasticity freezes strengths and eligibility while activity advances
void test_disabled_plasticity_freezes_local_state()
{
    BrainLayout layout = make_pair_layout();
    layout.input_positions.push_back({1, 0, 0});
    BrainConfig config = make_test_config();
    config.plasticity.weakening_rate_per_second = 1.0f;
    const BrainModel model = build_brain_model(layout, config, 1);
    CpuBrainSimulator simulator(model);
    BrainState state = simulator.state();
    const uint32_t source = neuron_index_at(model, {0, 0, 0});
    const uint32_t target = neuron_index_at(model, {1, 0, 0});
    const uint32_t edge = connection_index(model, source, target);
    state.connection_strengths[edge] = 0.5f;
    state.connection_eligibilities[edge] = 0.4f;
    state.activity_traces[source] = 1.0f;
    simulator.reset(move(state));
    step_with(simulator, {0.0f, 1000.0f}, false);
    check_near(simulator.state().connection_strengths[edge], 0.5f, 0.0f,
        "disabled plasticity decayed a connection strength");
    check_near(simulator.state().connection_eligibilities[edge], 0.4f, 0.0f,
        "disabled plasticity changed connection eligibility");
    check(simulator.state().tick == 1U && simulator.state().activity_traces[target] == 1.0f,
        "disabled plasticity also stopped neuron activity");
}

// verifies the default passive weakening preserves unused paths over short runs
void test_slow_passive_strength_decay()
{
    BrainConfig config;
    config.plasticity.initial_budget_fraction = 0.0f;
    config.plasticity.initial_connection_density = 0.0f;
    const BrainModel model = build_brain_model(make_pair_layout(), config, 1);
    CpuBrainSimulator simulator(model);
    BrainState state = simulator.state();
    const uint32_t source = neuron_index_at(model, {0, 0, 0});
    const uint32_t target = neuron_index_at(model, {1, 0, 0});
    const uint32_t edge = connection_index(model, source, target);
    state.connection_strengths[edge] = 0.5f;
    simulator.reset(move(state));

    for (uint32_t tick = 0; tick < 1000U; ++tick)
    {
        step_with(simulator, {0.0f});
    }

    const float expected = 0.5f * pow(model.weight_decay, 1000.0f);
    check_near(simulator.state().connection_strengths[edge], expected, 0.00001f,
        "passive connection weakening did not use the configured rate");
    check(simulator.state().connection_strengths[edge] > 0.499f,
        "default passive weakening erased an unused path too quickly");
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
    step_with(simulator, {1000.0f, 0.0f});
    step_with(simulator, {0.0f, 1000.0f});
    simulator.apply_plasticity_modulation(1.0f);
    const BrainState replay_state = simulator.state();
    const uint32_t edge = connection_index(model,
        neuron_index_at(model, {0, 0, 0}), neuron_index_at(model, {1, 0, 0}));
    const float learned_strength = replay_state.connection_strengths[edge];
    check(learned_strength > model.initial_connection_strengths[edge],
        "reset fixture did not learn a connection before reset");

    BrainState eligible_state = replay_state;
    eligible_state.connection_eligibilities[edge] = 0.25f;
    simulator.reset(move(eligible_state));

    simulator.reset_activity();
    check_near(simulator.state().connection_strengths[edge], learned_strength, 0.0f,
        "activity reset discarded learned strength");
    check(simulator.state().tick == 0U
        && simulator.state().spikes[0] == 0U
        && simulator.state().activity_traces[0] == 0.0f
        && simulator.state().connection_eligibilities[edge] == 0.0f,
        "activity reset retained transient neuron or eligibility state");

    simulator.reset();
    check_near(simulator.state().connection_strengths[edge],
        model.initial_connection_strengths[edge], 0.0f,
        "full reset did not restore initial strength");
    step_with(simulator, {1000.0f, 0.0f});
    step_with(simulator, {0.0f, 1000.0f});
    simulator.apply_plasticity_modulation(1.0f);
    check(simulator.state().membrane_potentials == replay_state.membrane_potentials
        && simulator.state().activity_traces == replay_state.activity_traces
        && simulator.state().connection_strengths == replay_state.connection_strengths
        && simulator.state().connection_eligibilities == replay_state.connection_eligibilities
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

    BrainConfig invalid_plasticity = make_test_config();
    invalid_plasticity.plasticity.eligibility_time_constant_seconds = 0.0f;
    bool plasticity_threw = false;

    try
    {
        build_brain_model(make_pair_layout(), invalid_plasticity, 1);
    }
    catch (const invalid_argument&)
    {
        plasticity_threw = true;
    }

    check(plasticity_threw, "invalid eligibility time constant was accepted");

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

    bool modulator_threw = false;

    try
    {
        simulator.apply_plasticity_modulation(numeric_limits<float>::quiet_NaN());
    }
    catch (const invalid_argument&)
    {
        modulator_threw = true;
    }

    check(modulator_threw
        && simulator.state().connection_strengths == before.connection_strengths
        && simulator.state().connection_eligibilities == before.connection_eligibilities,
        "non-finite modulation mutated brain plasticity state");

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

// verifies perturbation is reproducible and invalid values fail before mutation
void test_deterministic_global_perturbation()
{
    BrainLayout layout;
    layout.shape = {1, 1, 1};
    layout.neurons = {{{0, 0, 0}, NeuronPolarity::Excitatory, 0.0f}};
    BrainConfig config = make_test_config();
    config.dynamics.firing_threshold = 10.0f;
    const BrainModel model = build_brain_model(layout, config, 1);
    CpuBrainSimulator first(model);
    CpuBrainSimulator repeated(model);
    CpuBrainSimulator different(model);
    const BrainPerturbation perturbation{100.0f, 19};
    first.step({}, perturbation);
    repeated.step({}, perturbation);
    different.step({}, {100.0f, 20});
    check(first.state().membrane_potentials == repeated.state().membrane_potentials,
        "same perturbation seed did not reproduce neuron state");
    check(first.state().membrane_potentials != different.state().membrane_potentials,
        "different perturbation seeds produced identical neuron state");

    const BrainState before = first.state();
    bool threw = false;

    try
    {
        first.step({}, {numeric_limits<float>::quiet_NaN(), 1});
    }
    catch (const invalid_argument&)
    {
        threw = true;
    }

    check(threw && first.state().tick == before.tick
        && first.state().membrane_potentials == before.membrane_potentials,
        "invalid perturbation mutated brain state");
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
        {"batched_advance_matches_repeated_steps", test_batched_advance_matches_repeated_steps},
        {"inhibitory_signal_sign", test_inhibitory_signal_sign},
        {"causal_eligibility_direction", test_causal_eligibility_direction},
        {"inhibitory_plasticity_is_fixed", test_inhibitory_plasticity_is_fixed},
        {"delayed_signed_modulation", test_delayed_signed_modulation},
        {"modulation_budget_competition", test_modulation_budget_competition},
        {"bounded_and_consumed_eligibility", test_bounded_and_consumed_eligibility},
        {"disabled_plasticity_freezes_local_state", test_disabled_plasticity_freezes_local_state},
        {"slow_passive_strength_decay", test_slow_passive_strength_decay},
        {"reset_modes_and_replay", test_reset_modes_and_replay},
        {"compiled_model_and_state_validation", test_compiled_model_and_state_validation},
        {"invalid_boundaries_fail_loudly", test_invalid_boundaries_fail_loudly},
        {"deterministic_global_perturbation", test_deterministic_global_perturbation}
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
