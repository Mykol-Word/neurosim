#pragma once

#include "neurosim/brain.h"

#include <cstdint>
#include <span>
#include <vector>

namespace neurosim
{

using namespace std;

// defines one positive directed initial connection by lattice position
struct BrainConnectionGene
{
    GridPosition source;
    GridPosition target;
    float strength = 0.0f;

    // compares two connection genes for equality
    bool operator==(const BrainConnectionGene&) const = default;
};

// describes a complete inheritable brain topology and its initial connections
struct BrainGenome
{
    uint64_t seed = 0;
    BrainConfig config;
    GridShape shape;
    vector<NeuronDefinition> neurons;
    vector<GridPosition> input_positions;
    vector<GridPosition> output_positions;
    vector<BrainConnectionGene> connections;

    // compares two brain genomes for equality
    bool operator==(const BrainGenome&) const = default;
};

// compiles a validated genome into stable runtime arrays
BrainModel build_brain_model(const BrainGenome& genome);

// captures one compiled topology and positive connection state as a genome
BrainGenome make_brain_genome(const BrainModel& model, span<const float> connection_strengths);

// validates genome topology, interfaces, genes, and outgoing budgets
void validate_brain_genome(const BrainGenome& genome);

// returns a stable identity for the semantic contents of a brain genome
uint64_t brain_genome_fingerprint(const BrainGenome& genome);

}
