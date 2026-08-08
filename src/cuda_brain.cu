#include "neurosim/cuda_brain.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace neurosim
{

using namespace std;

namespace
{

inline constexpr uint32_t CUDA_THREADS_PER_BLOCK = 256U;

// throws a contextual error when a cuda runtime call fails
void check_cuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
    {
        throw runtime_error(string(operation) + ": " + cudaGetErrorString(result));
    }
}

template<typename T>
// owns one reusable typed allocation in cuda device memory
class DeviceBuffer
{
public:
    // creates an empty device buffer
    DeviceBuffer() = default;

    // allocates a requested number of device elements
    explicit DeviceBuffer(size_t size)
    {
        allocate(size);
    }

    // releases the device allocation without throwing
    ~DeviceBuffer()
    {
        release();
    }

    // prevents duplicate ownership of a device allocation
    DeviceBuffer(const DeviceBuffer&) = delete;

    // prevents duplicate ownership of a device allocation
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // transfers one device allocation into this buffer
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(exchange(other.pointer_, nullptr)),
          size_(exchange(other.size_, 0U))
    {
    }

    // replaces this allocation with another buffer's allocation
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            pointer_ = exchange(other.pointer_, nullptr);
            size_ = exchange(other.size_, 0U);
        }

        return *this;
    }

    // allocates the buffer exactly once
    void allocate(size_t size)
    {
        if (pointer_ != nullptr || size_ != 0U)
        {
            throw logic_error("device buffer is already allocated");
        }

        size_ = size;

        if (size_ > 0U)
        {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_), sizeof(T) * size_),
                "allocate CUDA brain buffer");
        }
    }

    // uploads a complete host span into the buffer
    void upload(span<const T> values)
    {
        if (values.size() != size_)
        {
            throw invalid_argument("CUDA brain upload size does not match device buffer");
        }

        if (size_ > 0U)
        {
            check_cuda(cudaMemcpy(pointer_, values.data(), sizeof(T) * size_, cudaMemcpyHostToDevice),
                "upload CUDA brain buffer");
        }
    }

    // downloads the complete buffer into a host span
    void download(span<T> values) const
    {
        if (values.size() != size_)
        {
            throw invalid_argument("CUDA brain download size does not match device buffer");
        }

        if (size_ > 0U)
        {
            check_cuda(cudaMemcpy(values.data(), pointer_, sizeof(T) * size_, cudaMemcpyDeviceToHost),
                "download CUDA brain buffer");
        }
    }

    // fills the device allocation with zero bytes
    void clear()
    {
        if (size_ > 0U)
        {
            check_cuda(cudaMemset(pointer_, 0, sizeof(T) * size_), "clear CUDA brain buffer");
        }
    }

    // returns the mutable device pointer
    T* data()
    {
        return pointer_;
    }

    // returns the read-only device pointer
    const T* data() const
    {
        return pointer_;
    }

    // returns the number of allocated elements
    size_t size() const
    {
        return size_;
    }

private:
    // releases the current allocation without surfacing destructor failures
    void release() noexcept
    {
        if (pointer_ != nullptr)
        {
            if (cudaFree(pointer_) != cudaSuccess)
            {
                terminate();
            }

            pointer_ = nullptr;
        }

        size_ = 0U;
    }

    T* pointer_ = nullptr;
    size_t size_ = 0U;
};

// stores scalar dynamics needed by both cuda kernels
struct DeviceBrainConfig
{
    float fixed_time_step_seconds;
    float membrane_time_constant_seconds;
    float resting_potential;
    float reset_potential;
    float firing_threshold;
    float trace_decay;
    float weight_decay;
    float learning_rate;
    float outgoing_strength_budget;
    uint32_t refractory_ticks;
};

// advances independent neuron state from prior spikes and current strengths
__global__ void advance_neurons_kernel(
    uint32_t neuron_count,
    const uint32_t* incoming_offsets,
    const uint32_t* incoming_edges,
    const uint32_t* edge_sources,
    const int8_t* polarities,
    const float* bias_drives,
    const int32_t* input_slots,
    const float* input_drives,
    const float* connection_strengths,
    const uint8_t* previous_spikes,
    uint8_t* next_spikes,
    float* membrane_potentials,
    float* activity_traces,
    uint32_t* refractory_ticks,
    int32_t* error_flag,
    DeviceBrainConfig config)
{
    const uint32_t neuron = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron >= neuron_count)
    {
        return;
    }

    uint8_t spike = 0U;

    if (refractory_ticks[neuron] > 0U)
    {
        --refractory_ticks[neuron];
        membrane_potentials[neuron] = config.reset_potential;
    }
    else
    {
        float synaptic_drive = 0.0f;

        for (uint32_t incoming = incoming_offsets[neuron];
            incoming < incoming_offsets[neuron + 1U]; ++incoming)
        {
            const uint32_t edge = incoming_edges[incoming];
            const uint32_t source = edge_sources[edge];
            synaptic_drive += static_cast<float>(polarities[source])
                * connection_strengths[edge]
                * static_cast<float>(previous_spikes[source]);
        }

        float external_drive = 0.0f;
        const int32_t input_slot = input_slots[neuron];

        if (input_slot >= 0)
        {
            external_drive = input_drives[input_slot];
        }

        const float potential = membrane_potentials[neuron];
        const float continuous_drive = (config.resting_potential - potential)
                / config.membrane_time_constant_seconds
            + bias_drives[neuron]
            + external_drive;
        const float next_potential = potential
            + config.fixed_time_step_seconds * continuous_drive
            + synaptic_drive;

        if (!isfinite(next_potential))
        {
            atomicCAS(error_flag, 0, 1);
        }
        else if (next_potential >= config.firing_threshold)
        {
            spike = 1U;
            membrane_potentials[neuron] = config.reset_potential;
            refractory_ticks[neuron] = config.refractory_ticks;
        }
        else
        {
            membrane_potentials[neuron] = next_potential;
        }
    }

    next_spikes[neuron] = spike;
    const float trace = activity_traces[neuron] * config.trace_decay + static_cast<float>(spike);
    activity_traces[neuron] = trace;

    if (!isfinite(trace))
    {
        atomicCAS(error_flag, 0, 2);
    }
}

// updates one source-owned outgoing range and projects it onto its budget
__global__ void update_connections_kernel(
    uint32_t neuron_count,
    const uint32_t* outgoing_offsets,
    const uint32_t* outgoing_targets,
    const uint8_t* spikes,
    const float* activity_traces,
    float* connection_strengths,
    int32_t* error_flag,
    DeviceBrainConfig config)
{
    const uint32_t source = blockIdx.x * blockDim.x + threadIdx.x;

    if (source >= neuron_count)
    {
        return;
    }

    const float source_spike = static_cast<float>(spikes[source]);
    const float source_trace = activity_traces[source];
    float strength_sum = 0.0f;

    for (uint32_t edge = outgoing_offsets[source]; edge < outgoing_offsets[source + 1U]; ++edge)
    {
        const uint32_t target = outgoing_targets[edge];
        const float target_spike = static_cast<float>(spikes[target]);
        const float coactivity = source_spike * activity_traces[target]
            + target_spike * source_trace
            - source_spike * target_spike;
        const float strength = connection_strengths[edge] * config.weight_decay
            + config.learning_rate * coactivity;
        connection_strengths[edge] = strength;
        strength_sum += strength;

        if (!isfinite(strength) || strength < 0.0f)
        {
            atomicCAS(error_flag, 0, 3);
        }
    }

    if (!isfinite(strength_sum))
    {
        atomicCAS(error_flag, 0, 4);
        return;
    }

    if (strength_sum > config.outgoing_strength_budget)
    {
        const float scale = config.outgoing_strength_budget / strength_sum;

        for (uint32_t edge = outgoing_offsets[source]; edge < outgoing_offsets[source + 1U]; ++edge)
        {
            connection_strengths[edge] *= scale;
        }
    }
}

// converts a host model into the scalar cuda kernel configuration
DeviceBrainConfig make_device_config(const BrainModel& model)
{
    return {
        model.config.dynamics.fixed_time_step_seconds,
        model.config.dynamics.membrane_time_constant_seconds,
        model.config.dynamics.resting_potential,
        model.config.dynamics.reset_potential,
        model.config.dynamics.firing_threshold,
        model.trace_decay,
        model.weight_decay,
        model.config.plasticity.learning_rate,
        model.config.plasticity.outgoing_strength_budget,
        model.refractory_ticks
    };
}

}

// owns one compiled brain and all corresponding cuda buffers
class CudaBrainSimulator::Impl
{
public:
    // validates, allocates, and uploads one cuda brain
    Impl(BrainModel model, int32_t device)
        : model_(move(model)),
          device_(device),
          config_(make_device_config(model_))
    {
        validate_brain_model(model_);

        if (device_ < 0)
        {
            throw invalid_argument("CUDA device index must be nonnegative");
        }

        int32_t device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "query CUDA device count");

        if (device_ >= device_count)
        {
            throw runtime_error("requested CUDA brain device is unavailable");
        }

        activate_device();
        outgoing_offsets_.allocate(model_.outgoing_offsets.size());
        outgoing_targets_.allocate(model_.outgoing_targets.size());
        edge_sources_.allocate(model_.edge_sources.size());
        incoming_offsets_.allocate(model_.incoming_offsets.size());
        incoming_edges_.allocate(model_.incoming_edges.size());
        polarities_.allocate(model_.polarities.size());
        bias_drives_.allocate(model_.bias_drives.size());
        input_slots_.allocate(model_.input_slots_by_neuron.size());
        input_drives_.allocate(model_.input_neurons.size());
        membrane_potentials_.allocate(model_.positions.size());
        activity_traces_.allocate(model_.positions.size());
        connection_strengths_.allocate(model_.outgoing_targets.size());
        refractory_ticks_.allocate(model_.positions.size());
        previous_spikes_.allocate(model_.positions.size());
        next_spikes_.allocate(model_.positions.size());
        error_flag_.allocate(1U);
        outgoing_offsets_.upload(model_.outgoing_offsets);
        outgoing_targets_.upload(model_.outgoing_targets);
        edge_sources_.upload(model_.edge_sources);
        incoming_offsets_.upload(model_.incoming_offsets);
        incoming_edges_.upload(model_.incoming_edges);
        vector<int8_t> polarities(model_.polarities.size());

        for (size_t index = 0; index < model_.polarities.size(); ++index)
        {
            polarities[index] = static_cast<int8_t>(model_.polarities[index]);
        }

        polarities_.upload(polarities);
        bias_drives_.upload(model_.bias_drives);
        input_slots_.upload(model_.input_slots_by_neuron);
        reset();
    }

    // restores activity and learned weights to their initial values
    void reset()
    {
        upload_state(make_brain_state(model_), true);
    }

    // clears transient activity while preserving learned connection strengths
    void reset_activity()
    {
        if (requires_reset_)
        {
            throw logic_error("CUDA brain numerical failure requires a full state reset");
        }

        upload_state(make_brain_state(model_), false);
    }

    // replaces the complete device state after validating it on the host
    void reset(const BrainState& state)
    {
        validate_brain_state(model_, state);
        upload_state(state, true);
    }

    // advances the device state through both synchronized model phases
    void step(span<const float> input_drives)
    {
        if (requires_reset_)
        {
            throw logic_error("CUDA brain simulator must be reset after a failed device step");
        }

        if (input_drives.size() != model_.input_neurons.size())
        {
            throw invalid_argument("brain input count does not match the configured input channels");
        }

        for (const float input : input_drives)
        {
            if (!isfinite(input))
            {
                throw invalid_argument("brain input drives must be finite");
            }
        }

        if (tick_ == numeric_limits<uint64_t>::max())
        {
            throw overflow_error("brain tick counter overflow");
        }

        try
        {
            activate_device();
            input_drives_.upload(input_drives);
            error_flag_.clear();
            const uint32_t neuron_count = static_cast<uint32_t>(model_.positions.size());
            const uint32_t blocks = (neuron_count + CUDA_THREADS_PER_BLOCK - 1U)
                / CUDA_THREADS_PER_BLOCK;
            advance_neurons_kernel<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
                neuron_count,
                incoming_offsets_.data(),
                incoming_edges_.data(),
                edge_sources_.data(),
                polarities_.data(),
                bias_drives_.data(),
                input_slots_.data(),
                input_drives_.data(),
                connection_strengths_.data(),
                previous_spikes_.data(),
                next_spikes_.data(),
                membrane_potentials_.data(),
                activity_traces_.data(),
                refractory_ticks_.data(),
                error_flag_.data(),
                config_);
            check_cuda(cudaGetLastError(), "launch CUDA neuron update");
            update_connections_kernel<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
                neuron_count,
                outgoing_offsets_.data(),
                outgoing_targets_.data(),
                next_spikes_.data(),
                activity_traces_.data(),
                connection_strengths_.data(),
                error_flag_.data(),
                config_);
            check_cuda(cudaGetLastError(), "launch CUDA plasticity update");
            check_cuda(cudaDeviceSynchronize(), "synchronize CUDA brain step");
            int32_t error = 0;
            error_flag_.download(span<int32_t>(&error, 1U));

            if (error != 0)
            {
                throw runtime_error("CUDA brain step produced invalid numerical state");
            }
        }
        catch (...)
        {
            requires_reset_ = true;
            throw;
        }

        swap(previous_spikes_, next_spikes_);
        ++tick_;
    }

    // returns the immutable host copy of the compiled model
    const BrainModel& model() const
    {
        return model_;
    }

    // downloads the complete device state
    BrainState download_state() const
    {
        activate_device();
        BrainState state;
        state.membrane_potentials.resize(model_.positions.size());
        state.activity_traces.resize(model_.positions.size());
        state.connection_strengths.resize(model_.outgoing_targets.size());
        state.refractory_ticks.resize(model_.positions.size());
        state.spikes.resize(model_.positions.size());
        state.tick = tick_;
        membrane_potentials_.download(state.membrane_potentials);
        activity_traces_.download(state.activity_traces);
        connection_strengths_.download(state.connection_strengths);
        refractory_ticks_.download(state.refractory_ticks);
        previous_spikes_.download(state.spikes);
        validate_brain_state(model_, state);
        return state;
    }

    // downloads whether one ordered output neuron fired this tick
    bool output_spike(uint32_t output_channel) const
    {
        if (output_channel >= model_.output_neurons.size())
        {
            throw out_of_range("brain output channel is out of range");
        }

        activate_device();
        uint8_t spike = 0U;
        const uint32_t neuron = model_.output_neurons[output_channel];
        check_cuda(cudaMemcpy(&spike, previous_spikes_.data() + neuron,
            sizeof(spike), cudaMemcpyDeviceToHost), "download CUDA brain output spike");
        return spike != 0U;
    }

    // downloads the recent activity trace of one ordered output neuron
    float output_activity(uint32_t output_channel) const
    {
        if (output_channel >= model_.output_neurons.size())
        {
            throw out_of_range("brain output channel is out of range");
        }

        activate_device();
        float activity = 0.0f;
        const uint32_t neuron = model_.output_neurons[output_channel];
        check_cuda(cudaMemcpy(&activity, activity_traces_.data() + neuron,
            sizeof(activity), cudaMemcpyDeviceToHost), "download CUDA brain output activity");
        return activity;
    }

private:
    // makes this simulator's cuda device current for the calling thread
    void activate_device() const
    {
        check_cuda(cudaSetDevice(device_), "select CUDA brain device");
    }

    // uploads dynamic state and optionally replaces learned strengths
    void upload_state(const BrainState& state, bool include_strengths)
    {
        validate_brain_state(model_, state);
        try
        {
            activate_device();
            membrane_potentials_.upload(state.membrane_potentials);
            activity_traces_.upload(state.activity_traces);
            refractory_ticks_.upload(state.refractory_ticks);
            previous_spikes_.upload(state.spikes);
            next_spikes_.clear();

            if (include_strengths)
            {
                connection_strengths_.upload(state.connection_strengths);
            }
        }
        catch (...)
        {
            requires_reset_ = true;
            throw;
        }

        tick_ = state.tick;
        requires_reset_ = false;
    }

    BrainModel model_;
    int32_t device_ = 0;
    DeviceBrainConfig config_;
    DeviceBuffer<uint32_t> outgoing_offsets_;
    DeviceBuffer<uint32_t> outgoing_targets_;
    DeviceBuffer<uint32_t> edge_sources_;
    DeviceBuffer<uint32_t> incoming_offsets_;
    DeviceBuffer<uint32_t> incoming_edges_;
    DeviceBuffer<int8_t> polarities_;
    DeviceBuffer<float> bias_drives_;
    DeviceBuffer<int32_t> input_slots_;
    DeviceBuffer<float> input_drives_;
    DeviceBuffer<float> membrane_potentials_;
    DeviceBuffer<float> activity_traces_;
    DeviceBuffer<float> connection_strengths_;
    DeviceBuffer<uint32_t> refractory_ticks_;
    DeviceBuffer<uint8_t> previous_spikes_;
    DeviceBuffer<uint8_t> next_spikes_;
    DeviceBuffer<int32_t> error_flag_;
    uint64_t tick_ = 0;
    bool requires_reset_ = false;
};

// reports whether a numbered cuda device is currently usable
bool cuda_device_available(int32_t device)
{
    if (device < 0)
    {
        return false;
    }

    int32_t device_count = 0;
    const cudaError_t result = cudaGetDeviceCount(&device_count);

    if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver)
    {
        const cudaError_t cleared = cudaGetLastError();

        if (cleared != cudaSuccess && cleared != result)
        {
            check_cuda(cleared, "clear unavailable CUDA device error");
        }

        return false;
    }

    check_cuda(result, "query CUDA device count");
    return device < device_count;
}

// uploads a compiled model and fresh state to one cuda device
CudaBrainSimulator::CudaBrainSimulator(BrainModel model, int32_t device)
    : implementation_(make_unique<Impl>(move(model), device))
{
}

// releases all cuda resources owned by the simulator
CudaBrainSimulator::~CudaBrainSimulator() = default;

// transfers ownership from another cuda simulator
CudaBrainSimulator::CudaBrainSimulator(CudaBrainSimulator&& other) noexcept = default;

// replaces this simulator with another simulator's cuda resources
CudaBrainSimulator& CudaBrainSimulator::operator=(CudaBrainSimulator&& other) noexcept = default;

// restores activity and learned weights to their initial values
void CudaBrainSimulator::reset()
{
    implementation_->reset();
}

// clears transient activity while preserving learned connection strengths
void CudaBrainSimulator::reset_activity()
{
    implementation_->reset_activity();
}

// replaces the complete device state after validating it on the host
void CudaBrainSimulator::reset(const BrainState& state)
{
    implementation_->reset(state);
}

// advances every neuron and connection by one synchronized fixed tick
void CudaBrainSimulator::step(span<const float> input_drives)
{
    implementation_->step(input_drives);
}

// returns the immutable host copy of the compiled model
const BrainModel& CudaBrainSimulator::model() const
{
    return implementation_->model();
}

// downloads the complete current device state for inspection or parity tests
BrainState CudaBrainSimulator::download_state() const
{
    return implementation_->download_state();
}

// downloads whether one ordered output neuron fired this tick
bool CudaBrainSimulator::output_spike(uint32_t output_channel) const
{
    return implementation_->output_spike(output_channel);
}

// downloads the recent activity trace of one ordered output neuron
float CudaBrainSimulator::output_activity(uint32_t output_channel) const
{
    return implementation_->output_activity(output_channel);
}

}
