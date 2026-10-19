#include "neurosim/evolution.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using namespace std;
using namespace neurosim;

inline constexpr uint32_t SENSORY_INPUT_COUNT = 6U;

// throws one readable test failure when a condition is false
void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw runtime_error(message);
    }
}

// flattens one fixture position into a stable integer
uint32_t flatten(GridShape shape, GridPosition position)
{
    return position.x + shape.width * (position.y + shape.height * position.z);
}

// returns the manhattan distance between two fixture positions
uint32_t distance(GridPosition left, GridPosition right)
{
    const uint32_t x = left.x > right.x ? left.x - right.x : right.x - left.x;
    const uint32_t y = left.y > right.y ? left.y - right.y : right.y - left.y;
    const uint32_t z = left.z > right.z ? left.z - right.z : right.z - left.z;
    return x + y + z;
}

// returns every cell in the fixture's permitted evolution space
vector<GridPosition> make_allowed_positions()
{
    vector<GridPosition> positions;

    for (uint32_t y = 0; y < 9U; ++y)
    {
        for (uint32_t x = 0; x < 9U; ++x)
        {
            positions.push_back({x, y, 0U});
        }
    }

    return positions;
}

// creates one dense connected inherited genome with ordered interfaces
BrainGenome make_parent_genome()
{
    BrainLayout layout;
    layout.shape = {9U, 9U, 1U};

    for (uint32_t y = 1U; y <= 7U; ++y)
    {
        for (uint32_t x = 1U; x <= 7U; ++x)
        {
            layout.neurons.push_back({
                {x, y, 0U},
                (x + y) % 5U == 0U
                    ? NeuronPolarity::Inhibitory
                    : NeuronPolarity::Excitatory,
                static_cast<float>(x + y) * 0.01f
            });
        }
    }

    layout.input_positions = {
        {1U, 1U, 0U},
        {7U, 1U, 0U},
        {1U, 7U, 0U},
        {7U, 7U, 0U},
        {4U, 1U, 0U},
        {4U, 7U, 0U},
        {1U, 4U, 0U},
        {7U, 4U, 0U}
    };
    layout.output_positions = {{1U, 4U, 0U}, {7U, 4U, 0U}};
    BrainConfig config;
    config.connection_radius_steps = 1U;
    config.plasticity.initial_connection_density = 1.0f;
    config.plasticity.initial_budget_fraction = 0.65f;
    const BrainModel model = build_brain_model(layout, config, 41U);
    return make_brain_genome(model, model.initial_connection_strengths);
}

// creates a path whose only removable neurons are articulation points
BrainGenome make_protected_path_genome()
{
    BrainLayout layout;
    layout.shape = {7U, 1U, 1U};

    for (uint32_t x = 0U; x < layout.shape.width; ++x)
    {
        layout.neurons.push_back({{x, 0U, 0U}, NeuronPolarity::Excitatory, 0.0f});
    }

    layout.input_positions = {{0U, 0U, 0U}, {6U, 0U, 0U}};
    BrainConfig config;
    config.connection_radius_steps = 1U;
    const BrainModel model = build_brain_model(layout, config, 71U);
    return make_brain_genome(model, model.initial_connection_strengths);
}

// returns fixture mutation settings with reachable neuron bounds
EvolutionMutationConfig make_mutation_config()
{
    EvolutionMutationConfig config;
    config.minimum_neurons = 40U;
    config.maximum_neurons = 60U;
    config.neuron_count_sigma_fraction = 0.10f;
    config.input_position_mutation_fraction = 0.20f;
    config.output_position_mutation_fraction = 0.50f;
    config.connection_mutation_fraction = 0.10f;
    return config;
}

// returns the fixture controller's paired output-drive interface constraints
EvolutionInterfaceConstraints make_interface_constraints()
{
    return {{{SENSORY_INPUT_COUNT, 0U}, {SENSORY_INPUT_COUNT + 1U, 1U}}};
}

// returns whether all occupied cells form one local undirected graph
bool is_connected(const BrainGenome& genome)
{
    vector<uint8_t> visited(genome.neurons.size(), 0U);
    vector<size_t> pending{0U};
    visited[0] = 1U;
    size_t reached = 0U;

    while (!pending.empty())
    {
        const size_t source = pending.back();
        pending.pop_back();
        ++reached;

        for (size_t target = 0; target < genome.neurons.size(); ++target)
        {
            if (visited[target] == 0U
                && distance(genome.neurons[source].position, genome.neurons[target].position)
                    <= genome.config.connection_radius_steps)
            {
                visited[target] = 1U;
                pending.push_back(target);
            }
        }
    }

    return reached == genome.neurons.size();
}

// combines two endpoint cells into one fixture connection key
uint64_t connection_key(GridShape shape, GridPosition source, GridPosition target)
{
    return (static_cast<uint64_t>(flatten(shape, source)) << 32U)
        | flatten(shape, target);
}

// verifies equal seeds reproduce an exactly equal genome
void test_mutation_is_deterministic()
{
    const BrainGenome parent = make_parent_genome();
    const vector<GridPosition> allowed = make_allowed_positions();
    const EvolutionMutationConfig config = make_mutation_config();
    const BrainGenome first = mutate_brain_genome(
        parent, allowed, 12345U, config, make_interface_constraints());
    const BrainGenome repeated = mutate_brain_genome(
        parent, allowed, 12345U, config, make_interface_constraints());
    check(first == repeated, "equal mutation seeds produced different genomes");
}

// verifies mutations preserve topology, interface, and connection invariants
void test_mutation_bounds_and_invariants()
{
    const BrainGenome parent = make_parent_genome();
    const vector<GridPosition> allowed = make_allowed_positions();
    const EvolutionMutationConfig config = make_mutation_config();
    const BrainGenome child = mutate_brain_genome(
        parent, allowed, 77U, config, make_interface_constraints());
    validate_brain_genome(child);
    check(child.neurons.size() >= config.minimum_neurons
        && child.neurons.size() <= config.maximum_neurons,
        "mutated neuron count left its configured bounds");
    check(child.input_positions.size() == parent.input_positions.size()
        && child.output_positions.size() == parent.output_positions.size(),
        "mutation changed an ordered interface count");

    set<uint32_t> allowed_cells;
    set<uint32_t> neuron_cells;

    for (const GridPosition position : allowed)
    {
        allowed_cells.insert(flatten(child.shape, position));
    }

    for (const NeuronDefinition& neuron : child.neurons)
    {
        const uint32_t cell = flatten(child.shape, neuron.position);
        check(allowed_cells.contains(cell), "mutation occupied a disallowed cell");
        check(neuron_cells.insert(cell).second, "mutation duplicated a neuron position");
    }

    set<uint32_t> interface_cells;

    for (const GridPosition position : child.input_positions)
    {
        const uint32_t cell = flatten(child.shape, position);
        check(neuron_cells.contains(cell), "mutated input does not contain a neuron");
        check(interface_cells.insert(cell).second, "mutation duplicated an input position");
    }

    for (uint32_t output = 0; output < child.output_positions.size(); ++output)
    {
        check(child.output_positions[output]
                == child.input_positions[SENSORY_INPUT_COUNT + output],
            "mutated actuator input does not match its output");
    }

    check(is_connected(child), "mutation disconnected the occupied brain graph");

    for (const BrainConnectionGene& connection : child.connections)
    {
        check(distance(connection.source, connection.target)
                <= child.config.connection_radius_steps,
            "mutation created a nonlocal connection");
    }
}

// verifies four independent mutation seeds produce distinct inherited genomes
void test_four_mutations_are_distinct()
{
    const BrainGenome parent = make_parent_genome();
    const vector<GridPosition> allowed = make_allowed_positions();
    const EvolutionMutationConfig config = make_mutation_config();
    set<uint64_t> fingerprints;

    for (uint64_t seed = 1U; seed <= 4U; ++seed)
    {
        const BrainGenome child = mutate_brain_genome(
            parent, allowed, seed, config, make_interface_constraints());
        fingerprints.insert(brain_genome_fingerprint(child));
    }

    check(fingerprints.size() == 4U, "four mutation seeds did not produce four genomes");
}

// verifies count mutation remains meaningful when normal rounding reaches a bound
void test_neuron_count_changes_at_bound()
{
    const BrainGenome parent = make_parent_genome();
    const vector<GridPosition> allowed = make_allowed_positions();
    EvolutionMutationConfig config = make_mutation_config();
    config.minimum_neurons = 40U;
    config.maximum_neurons = static_cast<uint32_t>(parent.neurons.size());
    config.neuron_count_sigma_fraction = 0.000001f;
    config.input_position_mutation_fraction = 0.0f;
    config.output_position_mutation_fraction = 0.0f;
    config.connection_mutation_fraction = 0.0f;
    const BrainGenome child = mutate_brain_genome(
        parent, allowed, 19U, config, make_interface_constraints());
    check(child.neurons.size() == parent.neurons.size() - 1U,
        "bounded neuron mutation rounded back to the parent count");
}

// verifies interface fallback mutation stays on the nearest available neurons
void test_interface_mutation_is_local()
{
    const BrainGenome parent = make_parent_genome();
    const vector<GridPosition> allowed = make_allowed_positions();
    EvolutionMutationConfig config = make_mutation_config();
    config.minimum_neurons = static_cast<uint32_t>(parent.neurons.size());
    config.maximum_neurons = static_cast<uint32_t>(parent.neurons.size());
    config.input_position_mutation_fraction = 0.0f;
    config.output_position_mutation_fraction = 1.0f;
    config.interface_position_sigma_steps = 0.0f;
    config.connection_mutation_fraction = 0.0f;
    const BrainGenome child = mutate_brain_genome(
        parent, allowed, 51U, config, make_interface_constraints());

    for (uint32_t output = 0U; output < child.output_positions.size(); ++output)
    {
        check(distance(parent.output_positions[output], child.output_positions[output]) == 1U,
            "interface fallback skipped a nearer occupied neuron");
    }
}

// verifies a generic genome can mutate outputs independently from inputs
void test_independent_interfaces_need_no_controller_convention()
{
    const BrainGenome parent = make_parent_genome();
    const vector<GridPosition> allowed = make_allowed_positions();
    EvolutionMutationConfig config = make_mutation_config();
    config.minimum_neurons = static_cast<uint32_t>(parent.neurons.size());
    config.maximum_neurons = static_cast<uint32_t>(parent.neurons.size());
    config.input_position_mutation_fraction = 0.0f;
    config.output_position_mutation_fraction = 1.0f;
    config.interface_position_sigma_steps = 0.0f;
    config.connection_mutation_fraction = 0.0f;
    const BrainGenome child = mutate_brain_genome(parent, allowed, 52U, config);
    validate_brain_genome(child);
    check(child.input_positions == parent.input_positions,
        "independent output mutation changed an input channel");
    check(child.output_positions != parent.output_positions,
        "generic output positions remained coupled without a constraint");
}

// verifies an infeasible shrink leaves valid anatomy for other mutations
void test_infeasible_shrink_falls_back()
{
    const BrainGenome parent = make_protected_path_genome();
    vector<GridPosition> allowed;

    for (uint32_t x = 0U; x < parent.shape.width; ++x)
    {
        allowed.push_back({x, 0U, 0U});
    }

    EvolutionMutationConfig config;
    config.minimum_neurons = 5U;
    config.maximum_neurons = 7U;
    config.neuron_count_sigma_fraction = 0.000001f;
    config.input_position_mutation_fraction = 0.0f;
    config.output_position_mutation_fraction = 0.0f;
    config.connection_mutation_fraction = 0.0f;
    const BrainGenome child = mutate_brain_genome(parent, allowed, 53U, config);
    validate_brain_genome(child);
    check(child.neurons.size() == parent.neurons.size(),
        "infeasible shrink did not preserve the valid parent anatomy");
}

// verifies surviving endpoint pairs retain exact strengths across index changes
void test_retained_endpoint_strengths()
{
    const BrainGenome parent = make_parent_genome();
    const vector<GridPosition> allowed = make_allowed_positions();
    EvolutionMutationConfig config = make_mutation_config();
    config.minimum_neurons = 45U;
    config.maximum_neurons = 45U;
    config.input_position_mutation_fraction = 0.0f;
    config.output_position_mutation_fraction = 0.0f;
    config.connection_mutation_fraction = 0.0f;
    const BrainGenome child = mutate_brain_genome(
        parent, allowed, 992U, config, make_interface_constraints());
    unordered_map<uint64_t, float> parent_strengths;

    for (const BrainConnectionGene& connection : parent.connections)
    {
        parent_strengths.emplace(connection_key(
            parent.shape, connection.source, connection.target), connection.strength);
    }

    for (const BrainConnectionGene& connection : child.connections)
    {
        const auto found = parent_strengths.find(connection_key(
            child.shape, connection.source, connection.target));
        check(found != parent_strengths.end(),
            "neuron removal introduced a connection between retained neurons");
        check(connection.strength == found->second,
            "retained endpoint pair changed its learned strength");
    }
}

// verifies every mutated source remains within its outgoing strength budget
void test_mutation_preserves_outgoing_budgets()
{
    const BrainGenome parent = make_parent_genome();
    const vector<GridPosition> allowed = make_allowed_positions();
    EvolutionMutationConfig config = make_mutation_config();
    config.minimum_neurons = 54U;
    config.maximum_neurons = 54U;
    config.connection_mutation_fraction = 1.0f;
    config.connection_strength_sigma_fraction = 0.75f;
    const BrainGenome child = mutate_brain_genome(
        parent, allowed, 332U, config, make_interface_constraints());
    const BrainModel model = build_brain_model(child);
    validate_connection_strengths(model, model.initial_connection_strengths);
}

// verifies aggregate success outranks one exceptional episode and resolves all ties
void test_fitness_ordering()
{
    EvolutionFitness reliable;
    reliable.total_score = 10U;
    reliable.best_episode_score = 2U;
    reliable.genome_fingerprint = 9U;
    EvolutionFitness lucky;
    lucky.total_score = 9U;
    lucky.best_episode_score = 9U;
    lucky.genome_fingerprint = 1U;
    check(better_evolution_fitness(reliable, lucky),
        "one exceptional run outranked higher aggregate success");

    EvolutionFitness baseline;
    baseline.total_score = 10U;
    baseline.successful_episodes = 1U;
    baseline.best_episode_score = 4U;
    baseline.total_episode_seconds = 20.0;
    baseline.penalty_events = 5U;
    baseline.genome_fingerprint = 100U;
    EvolutionFitness contender = baseline;
    contender.successful_episodes = 2U;
    check(better_evolution_fitness(contender, baseline), "success count did not break a tie");
    contender = baseline;
    contender.best_episode_score = 5U;
    check(better_evolution_fitness(contender, baseline), "best episode did not break a tie");
    contender = baseline;
    contender.total_episode_seconds = 21.0;
    check(better_evolution_fitness(contender, baseline), "duration did not break a tie");
    contender = baseline;
    contender.penalty_events = 4U;
    check(better_evolution_fitness(contender, baseline), "penalty count did not break a tie");
    contender = baseline;
    contender.genome_fingerprint = 99U;
    check(better_evolution_fitness(contender, baseline), "fingerprint did not break a tie");
    check(!better_evolution_fitness(baseline, baseline),
        "equal fitness values did not compare equivalently");
}

// verifies invalid survival metrics fail before they enter ranking
void test_invalid_fitness_fails_loudly()
{
    EvolutionFitness invalid;
    invalid.total_episode_seconds = numeric_limits<double>::quiet_NaN();
    bool threw = false;

    try
    {
        better_evolution_fitness(invalid, {});
    }
    catch (const invalid_argument&)
    {
        threw = true;
    }

    check(threw, "non-finite evolution fitness was accepted");
}

// verifies invalid spatial mutation settings fail before changing a genome
void test_invalid_mutation_config_fails_loudly()
{
    EvolutionMutationConfig config = make_mutation_config();
    config.interface_position_sigma_steps = numeric_limits<float>::quiet_NaN();
    bool threw = false;

    try
    {
        mutate_brain_genome(
            make_parent_genome(),
            make_allowed_positions(),
            1U,
            config,
            make_interface_constraints());
    }
    catch (const invalid_argument&)
    {
        threw = true;
    }

    check(threw, "non-finite interface position sigma was accepted");
}

}

// runs every evolution test and reports failures through the exit code
int main()
{
    const vector<pair<const char*, function<void()>>> tests{
        {"mutation_is_deterministic", test_mutation_is_deterministic},
        {"mutation_bounds_and_invariants", test_mutation_bounds_and_invariants},
        {"four_mutations_are_distinct", test_four_mutations_are_distinct},
        {"neuron_count_changes_at_bound", test_neuron_count_changes_at_bound},
        {"interface_mutation_is_local", test_interface_mutation_is_local},
        {"independent_interfaces_need_no_controller_convention",
            test_independent_interfaces_need_no_controller_convention},
        {"infeasible_shrink_falls_back", test_infeasible_shrink_falls_back},
        {"retained_endpoint_strengths", test_retained_endpoint_strengths},
        {"mutation_preserves_outgoing_budgets", test_mutation_preserves_outgoing_budgets},
        {"fitness_ordering", test_fitness_ordering},
        {"invalid_fitness_fails_loudly", test_invalid_fitness_fails_loudly},
        {"invalid_mutation_config_fails_loudly", test_invalid_mutation_config_fails_loudly}
    };
    size_t failures = 0U;

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
        cerr << failures << " evolution test(s) failed\n";
        return 1;
    }

    cout << tests.size() << " evolution tests passed\n";
    return 0;
}
