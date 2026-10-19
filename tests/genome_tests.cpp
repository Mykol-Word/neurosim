#include "neurosim/brain_genome.h"

#include <exception>
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

// returns a small deterministic genome with explicit connection strengths
BrainGenome make_test_genome()
{
    BrainGenome genome;
    genome.seed = 47;
    genome.shape = {4, 2, 1};
    genome.config.connection_radius_steps = 2;
    genome.config.plasticity.outgoing_strength_budget = 1.0f;
    genome.config.plasticity.initial_budget_fraction = 1.0f;
    genome.config.plasticity.initial_connection_density = 1.0f;
    genome.neurons = {
        {{2, 0, 0}, NeuronPolarity::Inhibitory, 0.1f},
        {{0, 0, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{1, 1, 0}, NeuronPolarity::Excitatory, -0.1f},
        {{1, 0, 0}, NeuronPolarity::Excitatory, 0.2f}
    };
    genome.input_positions = {{1, 1, 0}, {0, 0, 0}};
    genome.output_positions = {{2, 0, 0}, {1, 0, 0}};
    genome.connections = {
        {{2, 0, 0}, {1, 0, 0}, 0.2f},
        {{1, 0, 0}, {0, 0, 0}, 0.4f},
        {{1, 1, 0}, {1, 0, 0}, 0.3f}
    };
    return genome;
}

// verifies two compiled models have identical topology and initial state
void check_models_equal(const BrainModel& left, const BrainModel& right)
{
    check(left.seed == right.seed, "round-trip changed the model seed");
    check(left.config == right.config, "round-trip changed the model configuration");
    check(left.shape == right.shape, "round-trip changed the lattice shape");
    check(left.positions == right.positions, "round-trip changed neuron positions");
    check(left.polarities == right.polarities, "round-trip changed neuron polarities");
    check(left.bias_drives == right.bias_drives, "round-trip changed neuron biases");
    check(left.cell_to_neuron == right.cell_to_neuron, "round-trip changed cell lookup entries");
    check(left.outgoing_offsets == right.outgoing_offsets, "round-trip changed outgoing offsets");
    check(left.outgoing_targets == right.outgoing_targets, "round-trip changed outgoing targets");
    check(left.edge_sources == right.edge_sources, "round-trip changed edge sources");
    check(left.incoming_offsets == right.incoming_offsets, "round-trip changed incoming offsets");
    check(left.incoming_edges == right.incoming_edges, "round-trip changed incoming edges");
    check(left.input_neurons == right.input_neurons, "round-trip changed input order");
    check(left.output_neurons == right.output_neurons, "round-trip changed output order");
    check(left.initial_connection_strengths == right.initial_connection_strengths,
        "round-trip changed connection strengths");
}

// requires genome compilation to reject an invalid definition
void expect_invalid(const BrainGenome& genome, const string& message)
{
    bool threw = false;

    try
    {
        static_cast<void>(build_brain_model(genome));
    }
    catch (const invalid_argument&)
    {
        threw = true;
    }

    check(threw, message);
}

// verifies model capture and compilation preserve every genome-owned value
void test_deterministic_round_trip()
{
    const BrainGenome source = make_test_genome();
    const BrainModel original = build_brain_model(source);
    const BrainGenome captured = make_brain_genome(
        original, original.initial_connection_strengths);
    const BrainModel rebuilt = build_brain_model(captured);
    const BrainGenome repeated = make_brain_genome(
        rebuilt, rebuilt.initial_connection_strengths);

    check(captured.connections.size() == 3U,
        "model capture retained zero-strength potential connections");
    check(captured == repeated, "repeated genome capture was not deterministic");
    check_models_equal(original, rebuilt);
    check(brain_genome_fingerprint(source) == brain_genome_fingerprint(captured),
        "semantic genome identity depended on definition order");
    check(brain_genome_fingerprint(captured) == brain_genome_fingerprint(repeated),
        "round-trip changed the genome fingerprint");
}

// verifies malformed connection genes fail before producing a model
void test_invalid_connection_genes()
{
    BrainGenome duplicate = make_test_genome();
    duplicate.connections.push_back(duplicate.connections.front());
    expect_invalid(duplicate, "duplicate connection gene was accepted");

    BrainGenome nonlocal = make_test_genome();
    nonlocal.neurons.push_back({{3, 1, 0}, NeuronPolarity::Excitatory, 0.0f});
    nonlocal.connections = {{{0, 0, 0}, {3, 1, 0}, 0.2f}};
    expect_invalid(nonlocal, "nonlocal connection gene was accepted");

    BrainGenome missing_endpoint = make_test_genome();
    missing_endpoint.connections = {{{0, 0, 0}, {3, 0, 0}, 0.2f}};
    expect_invalid(missing_endpoint, "connection gene with an empty endpoint was accepted");

    BrainGenome nonpositive = make_test_genome();
    nonpositive.connections.front().strength = 0.0f;
    expect_invalid(nonpositive, "zero-strength connection gene was accepted");

    BrainGenome over_budget = make_test_genome();
    over_budget.connections = {
        {{1, 0, 0}, {0, 0, 0}, 0.6f},
        {{1, 0, 0}, {2, 0, 0}, 0.6f}
    };
    expect_invalid(over_budget, "outgoing connection budget excess was accepted");
}

// verifies connection strength contributes to stable genome identity
void test_strength_changes_fingerprint()
{
    BrainGenome changed = make_test_genome();
    const uint64_t original = brain_genome_fingerprint(changed);
    changed.connections.front().strength += 0.05f;
    check(original != brain_genome_fingerprint(changed),
        "connection strength did not change the genome fingerprint");
}

// verifies coordinate endpoints survive neuron index renumbering
void test_position_keyed_connection_endpoints()
{
    BrainGenome genome;
    genome.seed = 5;
    genome.shape = {4, 1, 1};
    genome.config.connection_radius_steps = 2;
    genome.config.plasticity.initial_budget_fraction = 0.0f;
    genome.config.plasticity.initial_connection_density = 0.0f;
    genome.neurons = {
        {{3, 0, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{1, 0, 0}, NeuronPolarity::Excitatory, 0.0f}
    };
    genome.connections = {{{3, 0, 0}, {1, 0, 0}, 0.7f}};
    const BrainModel before = build_brain_model(genome);
    const uint32_t old_source = neuron_index_at(before, {3, 0, 0});
    const uint32_t old_target = neuron_index_at(before, {1, 0, 0});

    genome.neurons.insert(genome.neurons.begin(),
        {{0, 0, 0}, NeuronPolarity::Inhibitory, 0.0f});
    const BrainModel after = build_brain_model(genome);
    const uint32_t new_source = neuron_index_at(after, {3, 0, 0});
    const uint32_t new_target = neuron_index_at(after, {1, 0, 0});
    const uint32_t edge = connection_index(after, new_source, new_target);

    check(old_source != new_source && old_target != new_target,
        "fixture did not renumber connection endpoints");
    check(edge != INVALID_NEURON_INDEX
        && after.initial_connection_strengths[edge] == 0.7f,
        "coordinate connection endpoint changed after neuron renumbering");
}

}

// runs every brain genome test and reports failures through the exit code
int main()
{
    const vector<pair<const char*, void(*)()>> tests{
        {"deterministic_round_trip", test_deterministic_round_trip},
        {"invalid_connection_genes", test_invalid_connection_genes},
        {"strength_changes_fingerprint", test_strength_changes_fingerprint},
        {"position_keyed_connection_endpoints", test_position_keyed_connection_endpoints}
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
        cerr << failures << " genome test(s) failed\n";
        return 1;
    }

    cout << tests.size() << " genome tests passed\n";
    return 0;
}
