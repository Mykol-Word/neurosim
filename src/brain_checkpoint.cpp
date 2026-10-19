#include "neurosim/brain_checkpoint.h"

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
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
inline constexpr uint32_t CHECKPOINT_VERSION = 2;
inline constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
inline constexpr uint64_t FNV_PRIME = 1099511628211ULL;
inline constexpr uint64_t POSITION_BYTES = sizeof(uint32_t) * 3U;
inline constexpr uint64_t NEURON_BYTES = POSITION_BYTES + sizeof(int8_t) + sizeof(float);
inline constexpr uint64_t CONNECTION_BYTES = POSITION_BYTES * 2U + sizeof(float);

static_assert(sizeof(float) == sizeof(uint32_t));

// mixes raw serialized bytes into the checkpoint integrity hash
void hash_bytes(uint64_t& hash, const char* bytes, size_t count)
{
    for (size_t index = 0; index < count; ++index)
    {
        hash ^= static_cast<uint8_t>(bytes[index]);
        hash *= FNV_PRIME;
    }
}

// writes bytes while extending the checkpoint integrity hash
void write_bytes(ofstream& stream, uint64_t& hash, const char* bytes, size_t count)
{
    stream.write(bytes, static_cast<streamsize>(count));

    if (!stream)
    {
        throw runtime_error("failed to write brain checkpoint");
    }

    hash_bytes(hash, bytes, count);
}

// writes one fixed-width value while extending the checkpoint integrity hash
template<typename T>
void write_value(ofstream& stream, uint64_t& hash, T value)
{
    static_assert(is_integral_v<T> || is_same_v<T, float>);
    write_bytes(stream, hash, reinterpret_cast<const char*>(&value), sizeof(value));
}

// writes one fixed-width value without extending the integrity hash
template<typename T>
void write_footer_value(ofstream& stream, T value)
{
    static_assert(is_integral_v<T>);
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));

    if (!stream)
    {
        throw runtime_error("failed to write brain checkpoint");
    }
}

// reads bytes while extending the checkpoint integrity hash
void read_bytes(ifstream& stream, uint64_t& hash, char* bytes, size_t count)
{
    stream.read(bytes, static_cast<streamsize>(count));

    if (!stream)
    {
        throw runtime_error("brain checkpoint is truncated");
    }

    hash_bytes(hash, bytes, count);
}

// reads one fixed-width value while extending the checkpoint integrity hash
template<typename T>
T read_value(ifstream& stream, uint64_t& hash)
{
    static_assert(is_integral_v<T> || is_same_v<T, float>);
    T value{};
    read_bytes(stream, hash, reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

// reads one fixed-width value without extending the integrity hash
template<typename T>
T read_footer_value(ifstream& stream)
{
    static_assert(is_integral_v<T>);
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));

    if (!stream)
    {
        throw runtime_error("brain checkpoint is truncated");
    }

    return value;
}

// returns the unread byte count in one checkpoint stream
uint64_t remaining_bytes(ifstream& stream, uint64_t file_size)
{
    const streampos position = stream.tellg();

    if (position < 0 || static_cast<uint64_t>(position) > file_size)
    {
        throw runtime_error("brain checkpoint stream position is invalid");
    }

    return file_size - static_cast<uint64_t>(position);
}

// rejects a serialized collection count before memory is allocated
void validate_count(
    uint64_t count,
    uint64_t maximum,
    uint64_t bytes_per_item,
    uint64_t bytes_remaining,
    const char* message)
{
    if (count > maximum || count > bytes_remaining / bytes_per_item
        || count > static_cast<uint64_t>(numeric_limits<size_t>::max()))
    {
        throw runtime_error(message);
    }
}

// returns the representable cell count for a serialized grid shape
uint64_t checkpoint_cell_count(GridShape shape)
{
    if (shape.width == 0U || shape.height == 0U || shape.depth == 0U)
    {
        throw runtime_error("brain checkpoint contains an invalid grid shape");
    }

    const uint64_t plane = static_cast<uint64_t>(shape.width) * shape.height;

    if (plane > (static_cast<uint64_t>(INVALID_NEURON_INDEX) - 1U) / shape.depth)
    {
        throw runtime_error("brain checkpoint grid shape is too large");
    }

    return plane * shape.depth;
}

// returns the maximum directed connection count for a neuron count
uint64_t maximum_connection_count(uint64_t neuron_count)
{
    if (neuron_count < 2U)
    {
        return 0U;
    }

    return min(
        neuron_count * (neuron_count - 1U),
        static_cast<uint64_t>(INVALID_NEURON_INDEX) - 1U);
}

// writes one lattice position
void write_position(ofstream& stream, uint64_t& hash, GridPosition position)
{
    write_value(stream, hash, position.x);
    write_value(stream, hash, position.y);
    write_value(stream, hash, position.z);
}

// reads one lattice position
GridPosition read_position(ifstream& stream, uint64_t& hash)
{
    GridPosition position;
    position.x = read_value<uint32_t>(stream, hash);
    position.y = read_value<uint32_t>(stream, hash);
    position.z = read_value<uint32_t>(stream, hash);
    return position;
}

// writes every configurable brain parameter
void write_config(ofstream& stream, uint64_t& hash, const BrainConfig& config)
{
    write_value(stream, hash, config.connection_radius_steps);
    const NeuronDynamicsConfig& dynamics = config.dynamics;
    write_value(stream, hash, dynamics.fixed_time_step_seconds);
    write_value(stream, hash, dynamics.membrane_time_constant_seconds);
    write_value(stream, hash, dynamics.resting_potential);
    write_value(stream, hash, dynamics.reset_potential);
    write_value(stream, hash, dynamics.firing_threshold);
    write_value(stream, hash, dynamics.refractory_period_seconds);
    const PlasticityConfig& plasticity = config.plasticity;
    write_value(stream, hash, plasticity.activity_trace_time_constant_seconds);
    write_value(stream, hash, plasticity.eligibility_time_constant_seconds);
    write_value(stream, hash, plasticity.maximum_eligibility);
    write_value(stream, hash, plasticity.learning_rate);
    write_value(stream, hash, plasticity.weakening_rate_per_second);
    write_value(stream, hash, plasticity.outgoing_strength_budget);
    write_value(stream, hash, plasticity.initial_budget_fraction);
    write_value(stream, hash, plasticity.initial_connection_density);
}

// reads every configurable brain parameter
BrainConfig read_config(ifstream& stream, uint64_t& hash)
{
    BrainConfig config;
    config.connection_radius_steps = read_value<uint32_t>(stream, hash);
    NeuronDynamicsConfig& dynamics = config.dynamics;
    dynamics.fixed_time_step_seconds = read_value<float>(stream, hash);
    dynamics.membrane_time_constant_seconds = read_value<float>(stream, hash);
    dynamics.resting_potential = read_value<float>(stream, hash);
    dynamics.reset_potential = read_value<float>(stream, hash);
    dynamics.firing_threshold = read_value<float>(stream, hash);
    dynamics.refractory_period_seconds = read_value<float>(stream, hash);
    PlasticityConfig& plasticity = config.plasticity;
    plasticity.activity_trace_time_constant_seconds = read_value<float>(stream, hash);
    plasticity.eligibility_time_constant_seconds = read_value<float>(stream, hash);
    plasticity.maximum_eligibility = read_value<float>(stream, hash);
    plasticity.learning_rate = read_value<float>(stream, hash);
    plasticity.weakening_rate_per_second = read_value<float>(stream, hash);
    plasticity.outgoing_strength_budget = read_value<float>(stream, hash);
    plasticity.initial_budget_fraction = read_value<float>(stream, hash);
    plasticity.initial_connection_density = read_value<float>(stream, hash);
    return config;
}

// writes one inherited neuron definition
void write_neuron(ofstream& stream, uint64_t& hash, const NeuronDefinition& neuron)
{
    write_position(stream, hash, neuron.position);
    write_value(stream, hash, static_cast<int8_t>(neuron.polarity));
    write_value(stream, hash, neuron.bias_drive);
}

// reads one inherited neuron definition
NeuronDefinition read_neuron(ifstream& stream, uint64_t& hash)
{
    NeuronDefinition neuron;
    neuron.position = read_position(stream, hash);
    neuron.polarity = static_cast<NeuronPolarity>(read_value<int8_t>(stream, hash));
    neuron.bias_drive = read_value<float>(stream, hash);
    return neuron;
}

// writes one directed inherited connection
void write_connection(ofstream& stream, uint64_t& hash, const BrainConnectionGene& connection)
{
    write_position(stream, hash, connection.source);
    write_position(stream, hash, connection.target);
    write_value(stream, hash, connection.strength);
}

// reads one directed inherited connection
BrainConnectionGene read_connection(ifstream& stream, uint64_t& hash)
{
    BrainConnectionGene connection;
    connection.source = read_position(stream, hash);
    connection.target = read_position(stream, hash);
    connection.strength = read_value<float>(stream, hash);
    return connection;
}

// writes one complete immutable brain genome
void write_genome(ofstream& stream, uint64_t& hash, const BrainGenome& genome)
{
    write_value(stream, hash, genome.seed);
    write_config(stream, hash, genome.config);
    write_value(stream, hash, genome.shape.width);
    write_value(stream, hash, genome.shape.height);
    write_value(stream, hash, genome.shape.depth);
    write_value(stream, hash, static_cast<uint64_t>(genome.neurons.size()));

    for (const NeuronDefinition& neuron : genome.neurons)
    {
        write_neuron(stream, hash, neuron);
    }

    write_value(stream, hash, static_cast<uint64_t>(genome.input_positions.size()));

    for (const GridPosition position : genome.input_positions)
    {
        write_position(stream, hash, position);
    }

    write_value(stream, hash, static_cast<uint64_t>(genome.output_positions.size()));

    for (const GridPosition position : genome.output_positions)
    {
        write_position(stream, hash, position);
    }

    write_value(stream, hash, static_cast<uint64_t>(genome.connections.size()));

    for (const BrainConnectionGene& connection : genome.connections)
    {
        write_connection(stream, hash, connection);
    }
}

// reads one complete immutable brain genome with bounded allocations
BrainGenome read_genome(ifstream& stream, uint64_t& hash, uint64_t file_size)
{
    BrainGenome genome;
    genome.seed = read_value<uint64_t>(stream, hash);
    genome.config = read_config(stream, hash);
    genome.shape.width = read_value<uint32_t>(stream, hash);
    genome.shape.height = read_value<uint32_t>(stream, hash);
    genome.shape.depth = read_value<uint32_t>(stream, hash);
    const uint64_t cell_count = checkpoint_cell_count(genome.shape);
    const uint64_t neuron_count = read_value<uint64_t>(stream, hash);
    validate_count(
        neuron_count,
        cell_count,
        NEURON_BYTES,
        remaining_bytes(stream, file_size),
        "brain checkpoint neuron count is invalid");

    if (neuron_count == 0U)
    {
        throw runtime_error("brain checkpoint contains no neurons");
    }

    genome.neurons.resize(static_cast<size_t>(neuron_count));

    for (NeuronDefinition& neuron : genome.neurons)
    {
        neuron = read_neuron(stream, hash);
    }

    const uint64_t input_count = read_value<uint64_t>(stream, hash);
    validate_count(
        input_count,
        neuron_count,
        POSITION_BYTES,
        remaining_bytes(stream, file_size),
        "brain checkpoint input count is invalid");
    genome.input_positions.resize(static_cast<size_t>(input_count));

    for (GridPosition& position : genome.input_positions)
    {
        position = read_position(stream, hash);
    }

    const uint64_t output_count = read_value<uint64_t>(stream, hash);
    validate_count(
        output_count,
        neuron_count,
        POSITION_BYTES,
        remaining_bytes(stream, file_size),
        "brain checkpoint output count is invalid");
    genome.output_positions.resize(static_cast<size_t>(output_count));

    for (GridPosition& position : genome.output_positions)
    {
        position = read_position(stream, hash);
    }

    const uint64_t connection_count = read_value<uint64_t>(stream, hash);
    validate_count(
        connection_count,
        maximum_connection_count(neuron_count),
        CONNECTION_BYTES,
        remaining_bytes(stream, file_size),
        "brain checkpoint connection count is invalid");
    genome.connections.resize(static_cast<size_t>(connection_count));

    for (BrainConnectionGene& connection : genome.connections)
    {
        connection = read_connection(stream, hash);
    }

    return genome;
}

// converts invalid genome data into a checkpoint corruption error
BrainModel validate_checkpoint_genome(const BrainGenome& genome)
{
    try
    {
        validate_brain_genome(genome);
        return build_brain_model(genome);
    }
    catch (const invalid_argument& error)
    {
        throw runtime_error(string("brain checkpoint contains an invalid genome: ") + error.what());
    }
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
    return brain_genome_fingerprint(make_brain_genome(model, model.initial_connection_strengths));
}

// writes learned strengths with their model identity to one binary file
void save_brain_checkpoint(
    const filesystem::path& path,
    uint64_t brain_seed,
    const BrainModel& model,
    span<const float> connection_strengths)
{
    save_brain_checkpoint(
        path,
        brain_seed,
        make_brain_genome(model, model.initial_connection_strengths),
        connection_strengths);
}

// writes one genome and its learned strengths to one binary file
void save_brain_checkpoint(
    const filesystem::path& path,
    uint64_t brain_seed,
    const BrainGenome& genome,
    span<const float> connection_strengths)
{
    validate_brain_genome(genome);
    const BrainModel model = build_brain_model(genome);
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

        uint64_t hash = FNV_OFFSET;
        write_bytes(stream, hash, CHECKPOINT_MAGIC.data(), CHECKPOINT_MAGIC.size());
        write_value(stream, hash, CHECKPOINT_VERSION);
        write_value(stream, hash, brain_seed);
        write_value(stream, hash, brain_genome_fingerprint(genome));
        write_genome(stream, hash, genome);
        write_value(stream, hash, static_cast<uint64_t>(connection_strengths.size()));

        for (const float strength : connection_strengths)
        {
            write_value(stream, hash, strength);
        }

        write_footer_value(stream, hash);
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

    stream.seekg(0, ios::end);
    const streampos end = stream.tellg();

    if (end < 0)
    {
        throw runtime_error("failed to determine brain checkpoint size");
    }

    const uint64_t file_size = static_cast<uint64_t>(end);
    stream.seekg(0, ios::beg);

    if (!stream)
    {
        throw runtime_error("failed to seek brain checkpoint");
    }

    uint64_t hash = FNV_OFFSET;
    array<char, CHECKPOINT_MAGIC.size()> magic{};
    read_bytes(stream, hash, magic.data(), magic.size());

    if (magic != CHECKPOINT_MAGIC)
    {
        throw runtime_error("brain checkpoint has an invalid header");
    }

    const uint32_t version = read_value<uint32_t>(stream, hash);

    if (version == 1U)
    {
        throw runtime_error("brain checkpoint version 1 is unsupported");
    }

    if (version != CHECKPOINT_VERSION)
    {
        throw runtime_error("brain checkpoint version is unsupported");
    }

    BrainCheckpoint checkpoint;
    checkpoint.brain_seed = read_value<uint64_t>(stream, hash);
    checkpoint.model_fingerprint = read_value<uint64_t>(stream, hash);
    checkpoint.genome = read_genome(stream, hash, file_size);
    const uint64_t strength_count = read_value<uint64_t>(stream, hash);
    const uint64_t maximum_strength_count = maximum_connection_count(checkpoint.genome.neurons.size());
    validate_count(
        strength_count,
        maximum_strength_count,
        sizeof(float),
        remaining_bytes(stream, file_size),
        "brain checkpoint strength count is invalid");
    checkpoint.connection_strengths.resize(static_cast<size_t>(strength_count));

    for (float& strength : checkpoint.connection_strengths)
    {
        strength = read_value<float>(stream, hash);

        if (!isfinite(strength) || strength < 0.0f)
        {
            throw runtime_error("brain checkpoint contains an invalid strength");
        }
    }

    const uint64_t stored_hash = read_footer_value<uint64_t>(stream);

    if (stored_hash != hash)
    {
        throw runtime_error("brain checkpoint integrity check failed");
    }

    if (stream.peek() != char_traits<char>::eof())
    {
        throw runtime_error("brain checkpoint contains trailing data");
    }

    if (checkpoint.model_fingerprint != brain_genome_fingerprint(checkpoint.genome))
    {
        throw runtime_error("brain checkpoint genome fingerprint is invalid");
    }

    const BrainModel model = validate_checkpoint_genome(checkpoint.genome);

    try
    {
        validate_connection_strengths(model, checkpoint.connection_strengths);
    }
    catch (const invalid_argument& error)
    {
        throw runtime_error(string("brain checkpoint contains invalid learned strengths: ") + error.what());
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

    if (checkpoint.model_fingerprint != brain_genome_fingerprint(checkpoint.genome))
    {
        throw invalid_argument("brain checkpoint genome fingerprint is invalid");
    }

    validate_connection_strengths(model, checkpoint.connection_strengths);
}

}
