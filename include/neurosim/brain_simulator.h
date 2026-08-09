#pragma once

#include "neurosim/brain.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace neurosim
{

using namespace std;

// identifies one explicit brain execution backend
enum class BrainBackend : uint8_t
{
    Cpu,
    Cuda
};

// selects one backend and its device without changing model semantics
struct BrainExecutionConfig
{
    BrainBackend backend = BrainBackend::Cpu;
    int32_t cuda_device = 0;
};

// summarizes spikes produced across one batched brain advance
struct BrainAdvanceSummary
{
    uint64_t neuron_spikes = 0;
    uint64_t output_spikes = 0;
};

// advances one compiled brain behind a backend-neutral execution boundary
class BrainSimulator
{
public:
    // releases backend-owned execution resources
    virtual ~BrainSimulator() = default;

    // advances fixed ticks and returns compact activity without full-state transfer
    virtual BrainAdvanceSummary advance(
        span<const float> input_drives,
        uint32_t tick_count,
        BrainPerturbation perturbation,
        bool plasticity_enabled,
        span<float> output_activity) = 0;

    // applies one signed global event to eligible local connections
    virtual void apply_plasticity_modulation(float modulator) = 0;

    // clears transient activity while preserving learned connection strengths
    virtual void reset_activity() = 0;

    // replaces learned strengths after validating them against the model
    virtual void load_connection_strengths(span<const float> strengths) = 0;

    // returns a host copy of the learned connection strengths
    virtual vector<float> copy_connection_strengths() const = 0;

    // returns a full host snapshot for diagnostics outside the hot path
    virtual BrainState copy_state() const = 0;

    // returns the immutable compiled model
    virtual const BrainModel& model() const = 0;

    // returns the explicit execution backend
    virtual BrainBackend backend() const = 0;
};

using BrainSimulatorFactory = unique_ptr<BrainSimulator> (*)(
    BrainModel,
    const BrainExecutionConfig&);

// returns the stable lowercase name of one execution backend
const char* brain_backend_name(BrainBackend backend);

}
