#include "neurosim/brain_checkpoint.h"

#include <chrono>
#include <exception>
#include <filesystem>
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

// creates a tiny reproducible model for checkpoint tests
BrainModel make_model(uint64_t seed)
{
    BrainLayout layout;
    layout.shape = {3, 1, 1};
    layout.neurons = {
        {{0, 0, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{1, 0, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{2, 0, 0}, NeuronPolarity::Inhibitory, 0.0f}
    };
    layout.input_positions = {{0, 0, 0}};
    layout.output_positions = {{2, 0, 0}};
    return build_brain_model(layout, {}, seed);
}

// verifies learned strengths survive a versioned checkpoint round trip
void test_checkpoint_round_trip()
{
    constexpr uint64_t BRAIN_SEED = 42;
    const BrainModel model = make_model(91);
    BrainState state = make_brain_state(model);

    for (float& strength : state.connection_strengths)
    {
        strength *= 0.5f;
    }

    const TemporaryCheckpoint file;
    save_brain_checkpoint(file.path(), BRAIN_SEED, model, state.connection_strengths);
    const BrainCheckpoint checkpoint = load_brain_checkpoint(file.path());
    validate_brain_checkpoint(checkpoint, BRAIN_SEED, model);
    check(checkpoint.connection_strengths == state.connection_strengths,
        "checkpoint changed learned strengths");
}

// verifies a checkpoint cannot be applied to another seeded topology
void test_checkpoint_rejects_model_mismatch()
{
    constexpr uint64_t BRAIN_SEED = 7;
    const BrainModel model = make_model(11);
    const BrainState state = make_brain_state(model);
    const TemporaryCheckpoint file;
    save_brain_checkpoint(file.path(), BRAIN_SEED, model, state.connection_strengths);
    const BrainCheckpoint checkpoint = load_brain_checkpoint(file.path());
    bool threw = false;

    try
    {
        validate_brain_checkpoint(checkpoint, BRAIN_SEED, make_model(12));
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
    const BrainModel model = make_model(23);
    BrainState state = make_brain_state(model);
    const TemporaryCheckpoint file;
    save_brain_checkpoint(file.path(), BRAIN_SEED, model, state.connection_strengths);

    for (float& strength : state.connection_strengths)
    {
        strength *= 0.25f;
    }

    save_brain_checkpoint(file.path(), BRAIN_SEED, model, state.connection_strengths);
    const BrainCheckpoint checkpoint = load_brain_checkpoint(file.path());
    check(checkpoint.connection_strengths == state.connection_strengths,
        "replacement checkpoint retained stale strengths");
    filesystem::path temporary = file.path();
    temporary += ".tmp";
    check(!filesystem::exists(temporary), "successful checkpoint left its temporary file");
}

}

// runs every checkpoint test and reports failures through the exit code
int main()
{
    try
    {
        test_checkpoint_round_trip();
        test_checkpoint_rejects_model_mismatch();
        test_checkpoint_replaces_existing_file();
        cout << "3 checkpoint tests passed\n";
        return 0;
    }
    catch (const exception& error)
    {
        cerr << error.what() << '\n';
        return 1;
    }
}
