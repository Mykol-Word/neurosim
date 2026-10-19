#include "neurosim/brain_checkpoint.h"

#include <array>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;
using namespace neurosim;

namespace
{

// owns one disposable checkpoint path for a test
class TemporaryCheckpoint
{
public:
    // creates a unique path under the operating system temporary directory
    TemporaryCheckpoint()
        : path_(filesystem::temp_directory_path()
            / ("neurosim-checkpoint-" + to_string(
                chrono::steady_clock::now().time_since_epoch().count()) + ".bin"))
    {
    }

    // removes the disposable checkpoint when the test ends
    ~TemporaryCheckpoint()
    {
        error_code error;
        filesystem::remove(path_, error);
        filesystem::path temporary = path_;
        temporary += ".tmp";
        filesystem::remove(temporary, error);
    }

    // returns the disposable checkpoint path
    const filesystem::path& path() const
    {
        return path_;
    }

private:
    filesystem::path path_;
};

// fails the test when one condition is false
void check(bool condition, const string& message)
{
    if (!condition)
    {
        throw runtime_error(message);
    }
}

// creates a nondefault inherited brain for checkpoint tests
BrainGenome make_genome(uint64_t seed)
{
    BrainGenome genome;
    genome.seed = seed;
    genome.shape = {4, 2, 1};
    genome.config.connection_radius_steps = 2;
    genome.config.dynamics.fixed_time_step_seconds = 0.002f;
    genome.config.dynamics.membrane_time_constant_seconds = 0.030f;
    genome.config.dynamics.firing_threshold = 1.25f;
    genome.config.plasticity.learning_rate = 0.025f;
    genome.config.plasticity.outgoing_strength_budget = 1.5f;
    genome.neurons = {
        {{0, 0, 0}, NeuronPolarity::Excitatory, 0.10f},
        {{1, 0, 0}, NeuronPolarity::Inhibitory, -0.05f},
        {{2, 0, 0}, NeuronPolarity::Excitatory, 0.02f},
        {{2, 1, 0}, NeuronPolarity::Excitatory, 0.00f}
    };
    genome.input_positions = {{0, 0, 0}, {1, 0, 0}};
    genome.output_positions = {{2, 1, 0}};
    genome.connections = {
        {{0, 0, 0}, {1, 0, 0}, 0.40f},
        {{1, 0, 0}, {2, 0, 0}, 0.70f},
        {{2, 0, 0}, {2, 1, 0}, 0.50f}
    };
    validate_brain_genome(genome);
    return genome;
}

// checks that two compiled models have identical persistent behavior
void check_model_parity(const BrainModel& left, const BrainModel& right)
{
    check(brain_model_fingerprint(left) == brain_model_fingerprint(right),
        "rebuilt model fingerprint changed");
    check(left.seed == right.seed, "rebuilt model seed changed");
    check(left.shape.width == right.shape.width
        && left.shape.height == right.shape.height
        && left.shape.depth == right.shape.depth,
        "rebuilt model shape changed");
    check(left.positions == right.positions, "rebuilt neuron positions changed");
    check(left.polarities == right.polarities, "rebuilt neuron polarities changed");
    check(left.bias_drives == right.bias_drives, "rebuilt neuron biases changed");
    check(left.outgoing_offsets == right.outgoing_offsets, "rebuilt connection offsets changed");
    check(left.outgoing_targets == right.outgoing_targets, "rebuilt connection targets changed");
    check(left.input_neurons == right.input_neurons, "rebuilt input channels changed");
    check(left.output_neurons == right.output_neurons, "rebuilt output channels changed");
    check(left.initial_connection_strengths == right.initial_connection_strengths,
        "rebuilt initial strengths changed");
}

// requires checkpoint loading to fail with one diagnostic fragment
void expect_load_failure(const filesystem::path& path, const string& expected)
{
    try
    {
        static_cast<void>(load_brain_checkpoint(path));
    }
    catch (const exception& error)
    {
        check(string(error.what()).find(expected) != string::npos,
            "checkpoint failure did not identify " + expected);
        return;
    }

    throw runtime_error("malformed checkpoint was accepted");
}

// writes the minimal header for one obsolete version
void write_version_one_header(const filesystem::path& path)
{
    constexpr array<char, 8> MAGIC{'N', 'S', 'B', 'R', 'A', 'I', 'N', '1'};
    constexpr uint32_t VERSION = 1;
    ofstream stream(path, ios::binary | ios::trunc);
    stream.write(MAGIC.data(), static_cast<streamsize>(MAGIC.size()));
    stream.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
}

// flips one integrity byte without changing the checkpoint length
void corrupt_checkpoint_footer(const filesystem::path& path)
{
    fstream stream(path, ios::binary | ios::in | ios::out);
    stream.seekg(-1, ios::end);
    char value = 0;
    stream.read(&value, 1);
    value ^= 0x5a;
    stream.seekp(-1, ios::end);
    stream.write(&value, 1);
}

// verifies a saved genome can reconstruct a nondefault compiled model
void test_checkpoint_round_trip_rebuilds_model()
{
    constexpr uint64_t BRAIN_SEED = 42;
    const BrainGenome genome = make_genome(91);
    const BrainModel original = build_brain_model(genome);
    const BrainState state = make_brain_state(original);
    const TemporaryCheckpoint file;
    save_brain_checkpoint(file.path(), BRAIN_SEED, genome, state.connection_strengths);
    const BrainCheckpoint checkpoint = load_brain_checkpoint(file.path());
    const BrainModel rebuilt = build_brain_model(checkpoint.genome);
    validate_brain_checkpoint(checkpoint, BRAIN_SEED, rebuilt);
    check(checkpoint.model_fingerprint == brain_genome_fingerprint(genome),
        "checkpoint changed the genome fingerprint");
    check_model_parity(original, rebuilt);
}

// verifies learned strengths remain separate from inherited initial strengths
void test_checkpoint_preserves_learned_strengths()
{
    constexpr uint64_t BRAIN_SEED = 17;
    const BrainGenome genome = make_genome(8);
    const BrainModel model = build_brain_model(genome);
    BrainState state = make_brain_state(model);

    for (float& strength : state.connection_strengths)
    {
        strength *= 0.5f;
    }

    const TemporaryCheckpoint file;
    save_brain_checkpoint(file.path(), BRAIN_SEED, model, state.connection_strengths);
    const BrainCheckpoint checkpoint = load_brain_checkpoint(file.path());
    const BrainModel rebuilt = build_brain_model(checkpoint.genome);
    check(checkpoint.connection_strengths == state.connection_strengths,
        "checkpoint changed learned strengths");
    check(rebuilt.initial_connection_strengths == model.initial_connection_strengths,
        "checkpoint replaced inherited strengths with learned strengths");
}

// verifies a checkpoint cannot be applied to another inherited brain
void test_checkpoint_rejects_model_mismatch()
{
    constexpr uint64_t BRAIN_SEED = 7;
    const BrainGenome genome = make_genome(11);
    const BrainModel model = build_brain_model(genome);
    const BrainState state = make_brain_state(model);
    const TemporaryCheckpoint file;
    save_brain_checkpoint(file.path(), BRAIN_SEED, model, state.connection_strengths);
    const BrainCheckpoint checkpoint = load_brain_checkpoint(file.path());
    BrainGenome different_genome = genome;
    different_genome.connections.front().strength = 0.35f;
    const BrainModel different_model = build_brain_model(different_genome);
    bool threw = false;

    try
    {
        validate_brain_checkpoint(checkpoint, BRAIN_SEED, different_model);
    }
    catch (const invalid_argument&)
    {
        threw = true;
    }

    check(threw, "checkpoint accepted a different model");
}

// verifies saving again safely replaces the previous complete checkpoint
void test_checkpoint_replaces_existing_file()
{
    constexpr uint64_t BRAIN_SEED = 18;
    const BrainGenome genome = make_genome(23);
    const BrainModel model = build_brain_model(genome);
    BrainState state = make_brain_state(model);
    const TemporaryCheckpoint file;
    save_brain_checkpoint(file.path(), BRAIN_SEED, genome, state.connection_strengths);

    for (float& strength : state.connection_strengths)
    {
        strength *= 0.25f;
    }

    save_brain_checkpoint(file.path(), BRAIN_SEED, genome, state.connection_strengths);
    const BrainCheckpoint checkpoint = load_brain_checkpoint(file.path());
    check(checkpoint.connection_strengths == state.connection_strengths,
        "replacement checkpoint retained stale strengths");
    filesystem::path temporary = file.path();
    temporary += ".tmp";
    check(!filesystem::exists(temporary), "successful checkpoint left its temporary file");
}

// verifies malformed, corrupted, and trailing bytes are rejected
void test_checkpoint_rejects_malformed_data()
{
    const BrainGenome genome = make_genome(31);
    const BrainModel model = build_brain_model(genome);
    const BrainState state = make_brain_state(model);
    const TemporaryCheckpoint file;
    save_brain_checkpoint(file.path(), 9, genome, state.connection_strengths);
    corrupt_checkpoint_footer(file.path());
    expect_load_failure(file.path(), "integrity");

    save_brain_checkpoint(file.path(), 9, genome, state.connection_strengths);
    ofstream trailing(file.path(), ios::binary | ios::app);
    trailing.put('x');
    trailing.close();
    expect_load_failure(file.path(), "trailing");

    write_version_one_header(file.path());
    expect_load_failure(file.path(), "version 1");
}

// verifies every truncated form of a valid checkpoint is rejected
void test_checkpoint_rejects_truncation()
{
    const BrainGenome genome = make_genome(41);
    const BrainModel model = build_brain_model(genome);
    const BrainState state = make_brain_state(model);
    const TemporaryCheckpoint file;
    save_brain_checkpoint(file.path(), 12, genome, state.connection_strengths);
    const uintmax_t complete_size = filesystem::file_size(file.path());
    filesystem::resize_file(file.path(), complete_size - 1U);
    expect_load_failure(file.path(), "truncated");
}

}

// runs every checkpoint test and reports failures through the exit code
int main()
{
    try
    {
        test_checkpoint_round_trip_rebuilds_model();
        test_checkpoint_preserves_learned_strengths();
        test_checkpoint_rejects_model_mismatch();
        test_checkpoint_replaces_existing_file();
        test_checkpoint_rejects_malformed_data();
        test_checkpoint_rejects_truncation();
        cout << "6 checkpoint tests passed\n";
        return 0;
    }
    catch (const exception& error)
    {
        cerr << error.what() << '\n';
        return 1;
    }
}
