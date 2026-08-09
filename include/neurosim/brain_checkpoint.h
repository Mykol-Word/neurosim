#pragma once

#include "neurosim/brain.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace neurosim
{

using namespace std;

// stores the persistent learned portion of one reproducible brain
struct BrainCheckpoint
{
    uint64_t brain_seed = 0;
    uint64_t model_fingerprint = 0;
    vector<float> connection_strengths;
};

// returns a stable identity for one compiled model and interface
uint64_t brain_model_fingerprint(const BrainModel& model);

// writes learned strengths with their model identity to one binary file
void save_brain_checkpoint(
    const filesystem::path& path,
    uint64_t brain_seed,
    const BrainModel& model,
    span<const float> connection_strengths);

// reads one versioned binary brain checkpoint
BrainCheckpoint load_brain_checkpoint(const filesystem::path& path);

// rejects a checkpoint that does not belong to the supplied model
void validate_brain_checkpoint(
    const BrainCheckpoint& checkpoint,
    uint64_t brain_seed,
    const BrainModel& model);

}
