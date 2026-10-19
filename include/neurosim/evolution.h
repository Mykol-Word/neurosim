#pragma once

#include "neurosim/brain_genome.h"

#include <cstdint>
#include <span>
#include <vector>

namespace neurosim
{

using namespace std;

// configures bounded inherited changes to one brain genome
struct EvolutionMutationConfig
{
    uint32_t minimum_neurons = 1024;
    uint32_t maximum_neurons = 5041;
    float neuron_count_sigma_fraction = 0.025f;
    float input_position_mutation_fraction = 0.02f;
    float output_position_mutation_fraction = 0.02f;
    float interface_position_sigma_steps = 2.0f;
    float connection_mutation_fraction = 0.02f;
    float connection_strength_sigma_fraction = 0.10f;
};

// couples one engineering input channel to one output neuron position
struct EvolutionInterfacePair
{
    uint32_t input = 0;
    uint32_t output = 0;

    // compares two interface pairs for equality
    bool operator==(const EvolutionInterfacePair&) const = default;
};

// supplies controller-specific interface relationships to generic mutation
struct EvolutionInterfaceConstraints
{
    vector<EvolutionInterfacePair> paired_positions;
};

// summarizes task evaluation outcomes used to rank one inherited genome
struct EvolutionFitness
{
    uint64_t total_score = 0;
    uint64_t successful_episodes = 0;
    uint64_t best_episode_score = 0;
    double total_episode_seconds = 0.0;
    uint64_t penalty_events = 0;
    uint64_t genome_fingerprint = 0;
};

// creates one deterministic connected mutation of an inherited brain genome
BrainGenome mutate_brain_genome(
    const BrainGenome& parent,
    span<const GridPosition> allowed_positions,
    uint64_t mutation_seed,
    const EvolutionMutationConfig& config = {},
    const EvolutionInterfaceConstraints& constraints = {});

// validates finite nonnegative fitness values before comparison
void validate_evolution_fitness(const EvolutionFitness& fitness);

// returns whether the left fitness ranks ahead of the right fitness
bool better_evolution_fitness(const EvolutionFitness& left, const EvolutionFitness& right);

}
