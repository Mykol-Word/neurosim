#pragma once

#include "neurosim/brain.h"

#include <cstdint>
#include <memory>
#include <span>

namespace neurosim
{

using namespace std;

// reports whether a numbered cuda device is currently usable
bool cuda_device_available(int32_t device = 0);

// advances one brain with explicit cuda-owned state and no cpu fallback
class CudaBrainSimulator
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

    // applies one signed global event to eligible local connections
    void apply_plasticity_modulation(float modulator);

    // returns the immutable host copy of the compiled model
    const BrainModel& model() const;

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

}
