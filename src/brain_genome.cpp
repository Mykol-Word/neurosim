#include "neurosim/brain_genome.h"

#include <bit>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace neurosim
{

using namespace std;

namespace
{

inline constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
inline constexpr uint64_t FNV_PRIME = 1099511628211ULL;

// throws when a genome invariant is false
void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw invalid_argument(message);
    }
}

// mixes one unsigned value into a stable genome fingerprint
template<typename T>
void hash_unsigned(uint64_t& hash, T value)
{
    static_assert(is_unsigned_v<T>);

    for (size_t byte = 0; byte < sizeof(T); ++byte)
    {
        hash ^= static_cast<uint8_t>(value >> (byte * 8U));
        hash *= FNV_PRIME;
    }
}

// mixes one float's exact representation into a stable genome fingerprint
void hash_float(uint64_t& hash, float value)
{
    hash_unsigned(hash, bit_cast<uint32_t>(value));
}

// compiles a genome while replacing generated strengths with explicit genes
BrainModel compile_brain_genome(const BrainGenome& genome)
{
    const BrainLayout layout{
        genome.shape,
        genome.neurons,
        genome.input_positions,
        genome.output_positions
    };
    BrainModel model = build_brain_model(layout, genome.config, genome.seed);
    model.initial_connection_strengths.assign(model.outgoing_targets.size(), 0.0f);
    vector<uint8_t> assigned(model.outgoing_targets.size(), 0U);

    for (const BrainConnectionGene& gene : genome.connections)
    {
        require(isfinite(gene.strength) && gene.strength > 0.0f,
            "brain connection gene strength must be finite and positive");
        const uint32_t source = neuron_index_at(model, gene.source);
        const uint32_t target = neuron_index_at(model, gene.target);
        require(source != INVALID_NEURON_INDEX && target != INVALID_NEURON_INDEX,
            "brain connection gene endpoint does not contain a neuron");
        const uint32_t edge = connection_index(model, source, target);
        require(edge != INVALID_NEURON_INDEX,
            "brain connection gene is not a valid local connection");
        require(assigned[edge] == 0U, "brain genome contains a duplicate connection gene");
        assigned[edge] = 1U;
        model.initial_connection_strengths[edge] = gene.strength;
    }

    validate_brain_model(model);
    return model;
}

// mixes every model parameter into a stable genome fingerprint
void hash_config(uint64_t& hash, const BrainConfig& config)
{
    hash_unsigned(hash, config.connection_radius_steps);
    const NeuronDynamicsConfig& dynamics = config.dynamics;
    hash_float(hash, dynamics.fixed_time_step_seconds);
    hash_float(hash, dynamics.membrane_time_constant_seconds);
    hash_float(hash, dynamics.resting_potential);
    hash_float(hash, dynamics.reset_potential);
    hash_float(hash, dynamics.firing_threshold);
    hash_float(hash, dynamics.refractory_period_seconds);
    const PlasticityConfig& plasticity = config.plasticity;
    hash_float(hash, plasticity.activity_trace_time_constant_seconds);
    hash_float(hash, plasticity.eligibility_time_constant_seconds);
    hash_float(hash, plasticity.maximum_eligibility);
    hash_float(hash, plasticity.learning_rate);
    hash_float(hash, plasticity.weakening_rate_per_second);
    hash_float(hash, plasticity.outgoing_strength_budget);
    hash_float(hash, plasticity.initial_budget_fraction);
    hash_float(hash, plasticity.initial_connection_density);
}

}

// compiles a validated genome into stable runtime arrays
BrainModel build_brain_model(const BrainGenome& genome)
{
    return compile_brain_genome(genome);
}

// captures one compiled topology and positive connection state as a genome
BrainGenome make_brain_genome(const BrainModel& model, span<const float> connection_strengths)
{
    validate_connection_strengths(model, connection_strengths);

    BrainGenome genome;
    genome.seed = model.seed;
    genome.config = model.config;
    genome.shape = model.shape;
    genome.neurons.reserve(model.positions.size());

    for (size_t neuron = 0; neuron < model.positions.size(); ++neuron)
    {
        genome.neurons.push_back({
            model.positions[neuron],
            model.polarities[neuron],
            model.bias_drives[neuron]
        });
    }

    genome.input_positions.reserve(model.input_neurons.size());

    for (const uint32_t neuron : model.input_neurons)
    {
        genome.input_positions.push_back(model.positions[neuron]);
    }

    genome.output_positions.reserve(model.output_neurons.size());

    for (const uint32_t neuron : model.output_neurons)
    {
        genome.output_positions.push_back(model.positions[neuron]);
    }

    for (uint32_t edge = 0; edge < connection_strengths.size(); ++edge)
    {
        if (connection_strengths[edge] <= 0.0f)
        {
            continue;
        }

        genome.connections.push_back({
            model.positions[model.edge_sources[edge]],
            model.positions[model.outgoing_targets[edge]],
            connection_strengths[edge]
        });
    }

    return genome;
}

// validates genome topology, interfaces, genes, and outgoing budgets
void validate_brain_genome(const BrainGenome& genome)
{
    static_cast<void>(compile_brain_genome(genome));
}

// returns a stable identity for the semantic contents of a brain genome
uint64_t brain_genome_fingerprint(const BrainGenome& genome)
{
    const BrainModel model = compile_brain_genome(genome);
    uint64_t hash = FNV_OFFSET;
    hash_unsigned(hash, model.seed);
    hash_unsigned(hash, model.shape.width);
    hash_unsigned(hash, model.shape.height);
    hash_unsigned(hash, model.shape.depth);
    hash_config(hash, model.config);
    hash_unsigned(hash, static_cast<uint64_t>(model.positions.size()));

    for (size_t neuron = 0; neuron < model.positions.size(); ++neuron)
    {
        hash_unsigned(hash, model.positions[neuron].x);
        hash_unsigned(hash, model.positions[neuron].y);
        hash_unsigned(hash, model.positions[neuron].z);
        hash_unsigned(hash, static_cast<uint8_t>(model.polarities[neuron]));
        hash_float(hash, model.bias_drives[neuron]);
    }

    hash_unsigned(hash, static_cast<uint64_t>(model.input_neurons.size()));

    for (const uint32_t neuron : model.input_neurons)
    {
        hash_unsigned(hash, neuron);
    }

    hash_unsigned(hash, static_cast<uint64_t>(model.output_neurons.size()));

    for (const uint32_t neuron : model.output_neurons)
    {
        hash_unsigned(hash, neuron);
    }

    hash_unsigned(hash, static_cast<uint64_t>(model.outgoing_targets.size()));

    for (uint32_t edge = 0; edge < model.outgoing_targets.size(); ++edge)
    {
        hash_unsigned(hash, model.edge_sources[edge]);
        hash_unsigned(hash, model.outgoing_targets[edge]);
        hash_float(hash, model.initial_connection_strengths[edge]);
    }

    return hash;
}

}
