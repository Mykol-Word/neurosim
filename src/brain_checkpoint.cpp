#include "neurosim/brain_checkpoint.h"

#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <type_traits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace neurosim
{

using namespace std;

namespace
{

inline constexpr array<char, 8> CHECKPOINT_MAGIC{'N', 'S', 'B', 'R', 'A', 'I', 'N', '1'};
inline constexpr uint32_t CHECKPOINT_VERSION = 1;
inline constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
inline constexpr uint64_t FNV_PRIME = 1099511628211ULL;

// mixes one unsigned value into a stable model fingerprint
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

// mixes one float's exact representation into a model fingerprint
void hash_float(uint64_t& hash, float value)
{
    hash_unsigned(hash, bit_cast<uint32_t>(value));
}

// writes one fixed-width value and reports stream failure
template<typename T>
void write_value(ofstream& stream, T value)
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));

    if (!stream)
    {
        throw runtime_error("failed to write brain checkpoint");
    }
}

// reads one fixed-width value and reports truncated input
template<typename T>
T read_value(ifstream& stream)
{
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));

    if (!stream)
    {
        throw runtime_error("brain checkpoint is truncated");
    }

    return value;
}

// atomically replaces a checkpoint after its temporary file is complete
void replace_checkpoint_file(
    const filesystem::path& temporary,
    const filesystem::path& destination)
{
#if defined(_WIN32)
    if (!MoveFileExW(
            temporary.c_str(),
            destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    {
        throw system_error(
            static_cast<int>(GetLastError()),
            system_category(),
            "failed to replace brain checkpoint");
    }
#else
    filesystem::rename(temporary, destination);
#endif
}

}

// returns a stable identity for one compiled model and interface
uint64_t brain_model_fingerprint(const BrainModel& model)
{
    uint64_t hash = FNV_OFFSET;
    hash_unsigned(hash, model.seed);
    hash_unsigned(hash, model.shape.width);
    hash_unsigned(hash, model.shape.height);
    hash_unsigned(hash, model.shape.depth);
    hash_unsigned(hash, model.config.connection_radius_steps);
    const NeuronDynamicsConfig& dynamics = model.config.dynamics;
    hash_float(hash, dynamics.fixed_time_step_seconds);
    hash_float(hash, dynamics.membrane_time_constant_seconds);
    hash_float(hash, dynamics.resting_potential);
    hash_float(hash, dynamics.reset_potential);
    hash_float(hash, dynamics.firing_threshold);
    hash_float(hash, dynamics.refractory_period_seconds);
    const PlasticityConfig& plasticity = model.config.plasticity;
    hash_float(hash, plasticity.activity_trace_time_constant_seconds);
    hash_float(hash, plasticity.eligibility_time_constant_seconds);
    hash_float(hash, plasticity.maximum_eligibility);
    hash_float(hash, plasticity.learning_rate);
    hash_float(hash, plasticity.weakening_rate_per_second);
    hash_float(hash, plasticity.outgoing_strength_budget);
    hash_float(hash, plasticity.initial_budget_fraction);
    hash_float(hash, plasticity.initial_connection_density);
    hash_unsigned(hash, static_cast<uint64_t>(model.positions.size()));

    for (size_t neuron = 0; neuron < model.positions.size(); ++neuron)
    {
        hash_unsigned(hash, model.positions[neuron].x);
        hash_unsigned(hash, model.positions[neuron].y);
        hash_unsigned(hash, model.positions[neuron].z);
        hash_unsigned(hash, static_cast<uint8_t>(model.polarities[neuron]));
        hash_float(hash, model.bias_drives[neuron]);
    }

    hash_unsigned(hash, static_cast<uint64_t>(model.outgoing_targets.size()));

    for (const uint32_t offset : model.outgoing_offsets)
    {
        hash_unsigned(hash, offset);
    }

    for (const uint32_t target : model.outgoing_targets)
    {
        hash_unsigned(hash, target);
    }

    for (const uint32_t input : model.input_neurons)
    {
        hash_unsigned(hash, input);
    }

    for (const uint32_t output : model.output_neurons)
    {
        hash_unsigned(hash, output);
    }

    return hash;
}

// writes learned strengths with their model identity to one binary file
void save_brain_checkpoint(
    const filesystem::path& path,
    uint64_t brain_seed,
    const BrainModel& model,
    span<const float> connection_strengths)
{
    validate_connection_strengths(model, connection_strengths);

    if (path.empty())
    {
        throw invalid_argument("brain checkpoint path must not be empty");
    }

    const filesystem::path parent = path.parent_path();

    if (!parent.empty())
    {
        filesystem::create_directories(parent);
    }

    filesystem::path temporary = path;
    temporary += ".tmp";

    try
    {
        ofstream stream(temporary, ios::binary | ios::trunc);

        if (!stream)
        {
            throw runtime_error("failed to open brain checkpoint for writing");
        }

        stream.write(CHECKPOINT_MAGIC.data(), CHECKPOINT_MAGIC.size());
        write_value(stream, CHECKPOINT_VERSION);
        write_value(stream, brain_seed);
        write_value(stream, brain_model_fingerprint(model));
        write_value(stream, static_cast<uint64_t>(connection_strengths.size()));

        for (const float strength : connection_strengths)
        {
            write_value(stream, strength);
        }

        stream.flush();

        if (!stream)
        {
            throw runtime_error("failed to flush brain checkpoint");
        }

        stream.close();

        if (!stream)
        {
            throw runtime_error("failed to close brain checkpoint");
        }

        replace_checkpoint_file(temporary, path);
    }
    catch (...)
    {
        error_code ignored;
        filesystem::remove(temporary, ignored);
        throw;
    }
}

// reads one versioned binary brain checkpoint
BrainCheckpoint load_brain_checkpoint(const filesystem::path& path)
{
    ifstream stream(path, ios::binary);

    if (!stream)
    {
        throw runtime_error("failed to open brain checkpoint");
    }

    array<char, CHECKPOINT_MAGIC.size()> magic{};
    stream.read(magic.data(), magic.size());

    if (!stream || magic != CHECKPOINT_MAGIC)
    {
        throw runtime_error("brain checkpoint has an invalid header");
    }

    if (read_value<uint32_t>(stream) != CHECKPOINT_VERSION)
    {
        throw runtime_error("brain checkpoint version is unsupported");
    }

    BrainCheckpoint checkpoint;
    checkpoint.brain_seed = read_value<uint64_t>(stream);
    checkpoint.model_fingerprint = read_value<uint64_t>(stream);
    const uint64_t strength_count = read_value<uint64_t>(stream);

    if (strength_count > static_cast<uint64_t>(numeric_limits<size_t>::max())
        || strength_count > static_cast<uint64_t>(numeric_limits<streamsize>::max()) / sizeof(float))
    {
        throw runtime_error("brain checkpoint strength count is too large");
    }

    checkpoint.connection_strengths.resize(static_cast<size_t>(strength_count));

    for (float& strength : checkpoint.connection_strengths)
    {
        strength = read_value<float>(stream);

        if (!isfinite(strength) || strength < 0.0f)
        {
            throw runtime_error("brain checkpoint contains an invalid strength");
        }
    }

    if (stream.peek() != char_traits<char>::eof())
    {
        throw runtime_error("brain checkpoint contains trailing data");
    }

    return checkpoint;
}

// rejects a checkpoint that does not belong to the supplied model
void validate_brain_checkpoint(
    const BrainCheckpoint& checkpoint,
    uint64_t brain_seed,
    const BrainModel& model)
{
    if (checkpoint.brain_seed != brain_seed)
    {
        throw invalid_argument("brain checkpoint seed does not match the agent");
    }

    if (checkpoint.model_fingerprint != brain_model_fingerprint(model))
    {
        throw invalid_argument("brain checkpoint model does not match the agent");
    }

    if (checkpoint.connection_strengths.size() != model.outgoing_targets.size())
    {
        throw invalid_argument("brain checkpoint strength count does not match the agent");
    }
}

}
