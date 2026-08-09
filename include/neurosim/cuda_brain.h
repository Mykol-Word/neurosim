#pragma once

#include "neurosim/brain_simulator.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace neurosim
{

using namespace std;

// reports whether a numbered cuda device is currently usable
bool cuda_device_available(int32_t device = 0);

// describes the selected cuda hardware and software stack for run metadata
struct CudaDeviceInfo
{
    string name;
    int32_t compute_major = 0;
    int32_t compute_minor = 0;
    int32_t driver_version = 0;
    int32_t runtime_version = 0;
    int32_t toolkit_version = 0;
};

// returns reproducibility metadata for one usable cuda device
CudaDeviceInfo cuda_device_info(int32_t device = 0);

// advances one brain with explicit cuda-owned state and no cpu fallback
class CudaBrainSimulator final : public BrainSimulator
{
public:
    // uploads a compiled model and fresh state to one cuda device
    explicit CudaBrainSimulator(BrainModel model, int32_t device = 0);

    // releases all cuda resources owned by the simulator
    ~CudaBrainSimulator();

    // prevents duplicate ownership of cuda allocations
    CudaBrainSimulator(const CudaBrainSimulator&) = delete;

    // prevents duplicate ownership of cuda allocations
    CudaBrainSimulator& operator=(const CudaBrainSimulator&) = delete;

    // transfers ownership from another cuda simulator
    CudaBrainSimulator(CudaBrainSimulator&& other) noexcept;

    // replaces this simulator with another simulator's cuda resources
    CudaBrainSimulator& operator=(CudaBrainSimulator&& other) noexcept;

    // restores activity and learned weights to their initial values
    void reset();

    // clears transient activity while preserving learned connection strengths
    void reset_activity();

    // replaces the complete device state after validating it on the host
    void reset(const BrainState& state);

    // advances every neuron and optionally records plasticity by one synchronized fixed tick
    void step(
        span<const float> input_drives,
        BrainPerturbation perturbation = {},
        bool plasticity_enabled = true);

    // advances multiple fixed ticks with one host-device synchronization
    BrainAdvanceSummary advance(
        span<const float> input_drives,
        uint32_t tick_count,
        BrainPerturbation perturbation,
        bool plasticity_enabled,
        span<float> output_activity) override;

    // applies one signed global event to eligible local connections
    void apply_plasticity_modulation(float modulator) override;

    // replaces learned strengths and clears transient activity
    void load_connection_strengths(span<const float> strengths) override;

    // returns a host copy of the learned connection strengths
    vector<float> copy_connection_strengths() const override;

    // returns a full host snapshot for diagnostics outside the hot path
    BrainState copy_state() const override;

    // returns the immutable host copy of the compiled model
    const BrainModel& model() const override;

    // returns the cuda execution backend
    BrainBackend backend() const override;

    // downloads the complete current device state for inspection or parity tests
    BrainState download_state() const;

    // downloads whether one ordered output neuron fired this tick
    bool output_spike(uint32_t output_channel) const;

    // downloads the recent activity trace of one ordered output neuron
    float output_activity(uint32_t output_channel) const;

private:
    class Impl;
    unique_ptr<Impl> implementation_;
};

// creates one cuda simulator for the requested cuda execution configuration
unique_ptr<BrainSimulator> make_cuda_brain_simulator(
    BrainModel model,
    const BrainExecutionConfig& execution);

}
