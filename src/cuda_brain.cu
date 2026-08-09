#include "neurosim/cuda_brain.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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

inline constexpr uint32_t CUDA_THREADS_PER_BLOCK = 128U;

// mixes a neuron, tick, and perturbation seed into a stable random value
__device__ uint64_t mix_noise(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

// returns one deterministic signed perturbation drive
__device__ float noise_drive(float amplitude, uint64_t seed, uint64_t tick, uint32_t neuron)
{
    if (amplitude == 0.0f)
    {
        return 0.0f;
    }

    const uint64_t keyed = seed
        ^ ((tick + 1U) * 0xd1b54a32d192ed03ULL)
        ^ ((static_cast<uint64_t>(neuron) + 1U) * 0x94d049bb133111ebULL);
    const float unit = static_cast<float>(mix_noise(keyed) >> 40U) * (1.0f / 16777216.0f);
    return (unit * 2.0f - 1.0f) * amplitude;
}

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

    // queues one complete host upload on the supplied cuda stream
    void upload_async(span<const T> values, cudaStream_t stream)
    {
        if (values.size() != size_)
        {
            throw invalid_argument("CUDA brain upload size does not match device buffer");
        }

        if (size_ > 0U)
        {
            check_cuda(cudaMemcpyAsync(
                pointer_,
                values.data(),
                sizeof(T) * size_,
                cudaMemcpyHostToDevice,
                stream),
                "queue CUDA brain upload");
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

    // queues a device prefix download on the supplied cuda stream
    void download_prefix_async(span<T> values, cudaStream_t stream) const
    {
        if (values.size() > size_)
        {
            throw invalid_argument("CUDA brain download exceeds device buffer");
        }

        if (!values.empty())
        {
            check_cuda(cudaMemcpyAsync(
                values.data(),
                pointer_,
                sizeof(T) * values.size(),
                cudaMemcpyDeviceToHost,
                stream),
                "queue CUDA brain download");
        }
    }

    // queues a zero fill on the supplied cuda stream
    void clear_async(cudaStream_t stream)
    {
        if (size_ > 0U)
        {
            check_cuda(cudaMemsetAsync(pointer_, 0, sizeof(T) * size_, stream),
                "queue CUDA brain buffer clear");
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
    // releases the current allocation while preserving any active exception
    void release() noexcept
    {
        if (pointer_ != nullptr)
        {
            static_cast<void>(cudaFree(pointer_));
            pointer_ = nullptr;
        }

        size_ = 0U;
    }

    T* pointer_ = nullptr;
    size_t size_ = 0U;
};

// owns one nonblocking cuda stream and restores its device before cleanup
class CudaStream
{
public:
    // creates an empty stream owner
    CudaStream() = default;

    // releases the stream without allowing destructor failures to escape
    ~CudaStream()
    {
        release();
    }

    // prevents duplicate ownership of one cuda stream
    CudaStream(const CudaStream&) = delete;

    // prevents duplicate ownership of one cuda stream
    CudaStream& operator=(const CudaStream&) = delete;

    // creates one nonblocking stream on the requested device
    void create(int32_t device)
    {
        if (stream_ != nullptr)
        {
            throw logic_error("CUDA brain stream is already created");
        }

        check_cuda(cudaSetDevice(device), "select CUDA brain device");
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "create CUDA brain stream");
        device_ = device;
    }

    // returns the owned cuda stream handle
    cudaStream_t get() const
    {
        return stream_;
    }

private:
    // destroys the stream while preserving any active exception
    void release() noexcept
    {
        if (stream_ == nullptr)
        {
            return;
        }

        if (cudaSetDevice(device_) == cudaSuccess)
        {
            static_cast<void>(cudaStreamDestroy(stream_));
        }

        stream_ = nullptr;
    }

    cudaStream_t stream_ = nullptr;
    int32_t device_ = 0;
};

template<typename T>
// owns one reusable page-locked host allocation for asynchronous transfers
class PinnedBuffer
{
public:
    // creates an empty pinned host buffer
    PinnedBuffer() = default;

    // releases the pinned host allocation without throwing
    ~PinnedBuffer()
    {
        if (pointer_ != nullptr)
        {
            static_cast<void>(cudaFreeHost(pointer_));
        }
    }

    // prevents duplicate ownership of pinned host memory
    PinnedBuffer(const PinnedBuffer&) = delete;

    // prevents duplicate ownership of pinned host memory
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // allocates the pinned buffer exactly once
    void allocate(size_t size)
    {
        if (pointer_ != nullptr || size_ != 0U)
        {
            throw logic_error("pinned buffer is already allocated");
        }

        size_ = size;

        if (size_ > 0U)
        {
            check_cuda(cudaMallocHost(reinterpret_cast<void**>(&pointer_), sizeof(T) * size_),
                "allocate CUDA brain result buffer");
        }
    }

    // returns the mutable pinned host pointer
    T* data()
    {
        return pointer_;
    }

private:
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
    float eligibility_decay;
    float weight_decay;
    float learning_rate;
    float maximum_eligibility;
    float outgoing_strength_budget;
    uint32_t refractory_ticks;
};

// stores one batched advance result and its numerical error code
struct DeviceAdvanceReport
{
    unsigned long long neuron_spikes;
    unsigned long long output_spikes;
    int32_t error;
};

static_assert(sizeof(DeviceAdvanceReport) % alignof(float) == 0U);

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
    uint32_t* refractory_ticks,
    DeviceAdvanceReport* report,
    uint64_t tick,
    uint64_t noise_seed,
    float noise_amplitude,
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
            + external_drive
            + noise_drive(noise_amplitude, noise_seed, tick, neuron);
        const float next_potential = potential
            + config.fixed_time_step_seconds * continuous_drive
            + synaptic_drive;

        if (!isfinite(next_potential))
        {
            atomicCAS(&report->error, 0, 1);
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
}

// updates one source-owned eligibility range and its source activity trace
__global__ void update_plasticity_and_traces_kernel(
    uint32_t neuron_count,
    const uint32_t* outgoing_offsets,
    const uint32_t* outgoing_targets,
    const int8_t* polarities,
    const uint8_t* spikes,
    uint32_t* spike_counts,
    float* activity_traces,
    float* connection_strengths,
    float* connection_eligibilities,
    DeviceAdvanceReport* report,
    bool plasticity_enabled,
    DeviceBrainConfig config)
{
    const uint32_t source = blockIdx.x * blockDim.x + threadIdx.x;

    if (source >= neuron_count)
    {
        return;
    }

    const float old_source_activity = activity_traces[source];
    const float excitatory = polarities[source] > 0 ? 1.0f : 0.0f;

    if (plasticity_enabled)
    {
        for (uint32_t edge = outgoing_offsets[source];
            edge < outgoing_offsets[source + 1U]; ++edge)
        {
            const uint32_t target = outgoing_targets[edge];
            const float strength = connection_strengths[edge] * config.weight_decay;
            const float eligibility = connection_eligibilities[edge] * config.eligibility_decay
                + excitatory * static_cast<float>(spikes[target]) * old_source_activity;

            if (!isfinite(strength) || strength < 0.0f
                || !isfinite(eligibility) || eligibility < 0.0f)
            {
                atomicCAS(&report->error, 0, 2);
            }

            connection_strengths[edge] = strength;
            connection_eligibilities[edge] = fminf(config.maximum_eligibility, eligibility);
        }
    }

    const float trace = old_source_activity * config.trace_decay
        + static_cast<float>(spikes[source]);
    activity_traces[source] = trace;
    spike_counts[source] += static_cast<uint32_t>(spikes[source]);

    if (!isfinite(trace))
    {
        atomicCAS(&report->error, 0, 3);
    }
}

// reduces per-neuron spike totals into one compact batch report
__global__ void summarize_spikes_kernel(
    uint32_t neuron_count,
    const uint32_t* spike_counts,
    const uint8_t* output_flags,
    DeviceAdvanceReport* report)
{
    __shared__ unsigned long long neuron_spikes[CUDA_THREADS_PER_BLOCK];
    __shared__ unsigned long long output_spikes[CUDA_THREADS_PER_BLOCK];

    const uint32_t neuron = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long spikes = neuron < neuron_count
        ? static_cast<unsigned long long>(spike_counts[neuron])
        : 0ULL;
    neuron_spikes[threadIdx.x] = spikes;
    output_spikes[threadIdx.x] = neuron < neuron_count && output_flags[neuron] != 0U
        ? spikes
        : 0ULL;
    __syncthreads();

    for (uint32_t offset = CUDA_THREADS_PER_BLOCK / 2U; offset > 0U; offset /= 2U)
    {
        if (threadIdx.x < offset)
        {
            neuron_spikes[threadIdx.x] += neuron_spikes[threadIdx.x + offset];
            output_spikes[threadIdx.x] += output_spikes[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0U)
    {
        atomicAdd(&report->neuron_spikes, neuron_spikes[0]);
        atomicAdd(&report->output_spikes, output_spikes[0]);
    }
}

// gathers ordered output traces into one contiguous transfer buffer
__global__ void gather_output_activity_kernel(
    uint32_t output_count,
    const uint32_t* output_neurons,
    const float* activity_traces,
    float* output_activity)
{
    const uint32_t output = blockIdx.x * blockDim.x + threadIdx.x;

    if (output < output_count)
    {
        output_activity[output] = activity_traces[output_neurons[output]];
    }
}

// applies signed modulation and outgoing budget competition for one source
__global__ void modulate_connections_kernel(
    uint32_t neuron_count,
    const uint32_t* outgoing_offsets,
    float* connection_strengths,
    float* connection_eligibilities,
    DeviceAdvanceReport* report,
    float modulator,
    DeviceBrainConfig config)
{
    const uint32_t source = blockIdx.x * blockDim.x + threadIdx.x;

    if (source >= neuron_count)
    {
        return;
    }

    float strength_sum = 0.0f;

    for (uint32_t edge = outgoing_offsets[source]; edge < outgoing_offsets[source + 1U]; ++edge)
    {
        const float delta = config.learning_rate * modulator * connection_eligibilities[edge];
        const float raw_strength = connection_strengths[edge] + delta;

        if (!isfinite(delta) || !isfinite(raw_strength))
        {
            atomicCAS(&report->error, 0, 4);
        }

        const float strength = fmaxf(0.0f, raw_strength);
        connection_strengths[edge] = strength;
        connection_eligibilities[edge] = 0.0f;
        strength_sum += strength;
    }

    if (!isfinite(strength_sum))
    {
        atomicCAS(&report->error, 0, 5);
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
        model.eligibility_decay,
        model.weight_decay,
        model.config.plasticity.learning_rate,
        model.config.plasticity.maximum_eligibility,
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
        stream_.create(device_);
        outgoing_offsets_.allocate(model_.outgoing_offsets.size());
        outgoing_targets_.allocate(model_.outgoing_targets.size());
        edge_sources_.allocate(model_.edge_sources.size());
        incoming_offsets_.allocate(model_.incoming_offsets.size());
        incoming_edges_.allocate(model_.incoming_edges.size());
        polarities_.allocate(model_.polarities.size());
        bias_drives_.allocate(model_.bias_drives.size());
        input_slots_.allocate(model_.input_slots_by_neuron.size());
        input_drives_.allocate(model_.input_neurons.size());
        output_flags_.allocate(model_.positions.size());
        output_neurons_.allocate(model_.output_neurons.size());
        membrane_potentials_.allocate(model_.positions.size());
        activity_traces_.allocate(model_.positions.size());
        connection_strengths_.allocate(model_.outgoing_targets.size());
        connection_eligibilities_.allocate(model_.outgoing_targets.size());
        refractory_ticks_.allocate(model_.positions.size());
        previous_spikes_.allocate(model_.positions.size());
        next_spikes_.allocate(model_.positions.size());
        spike_counts_.allocate(model_.positions.size());
        const size_t result_bytes = sizeof(DeviceAdvanceReport)
            + sizeof(float) * model_.output_neurons.size();
        advance_result_.allocate(result_bytes);
        host_advance_result_.allocate(result_bytes);
        outgoing_offsets_.upload_async(model_.outgoing_offsets, stream_.get());
        outgoing_targets_.upload_async(model_.outgoing_targets, stream_.get());
        edge_sources_.upload_async(model_.edge_sources, stream_.get());
        incoming_offsets_.upload_async(model_.incoming_offsets, stream_.get());
        incoming_edges_.upload_async(model_.incoming_edges, stream_.get());
        vector<int8_t> polarities(model_.polarities.size());

        for (size_t index = 0; index < model_.polarities.size(); ++index)
        {
            polarities[index] = static_cast<int8_t>(model_.polarities[index]);
        }

        polarities_.upload_async(polarities, stream_.get());
        bias_drives_.upload_async(model_.bias_drives, stream_.get());
        input_slots_.upload_async(model_.input_slots_by_neuron, stream_.get());
        vector<uint8_t> output_flags(model_.positions.size(), 0U);

        for (const uint32_t output_neuron : model_.output_neurons)
        {
            output_flags[output_neuron] = 1U;
        }

        output_flags_.upload_async(output_flags, stream_.get());
        output_neurons_.upload_async(model_.output_neurons, stream_.get());
        check_cuda(cudaStreamSynchronize(stream_.get()), "upload CUDA brain model");
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

    // advances fixed ticks while synchronizing with the host only once
    BrainAdvanceSummary advance(
        span<const float> input_drives,
        uint32_t tick_count,
        BrainPerturbation perturbation,
        bool plasticity_enabled,
        span<float> output_activity)
    {
        if (requires_reset_)
        {
            throw logic_error("CUDA brain simulator must be reset after a failed device step");
        }

        if (input_drives.size() != model_.input_neurons.size())
        {
            throw invalid_argument("brain input count does not match the configured input channels");
        }

        if (tick_count == 0U)
        {
            throw invalid_argument("brain advance tick count must be positive");
        }

        if (!output_activity.empty() && output_activity.size() != model_.output_neurons.size())
        {
            throw invalid_argument("brain output activity count does not match the configured outputs");
        }

        for (const float input : input_drives)
        {
            if (!isfinite(input))
            {
                throw invalid_argument("brain input drives must be finite");
            }
        }

        if (!isfinite(perturbation.noise_drive_amplitude)
            || perturbation.noise_drive_amplitude < 0.0f)
        {
            throw invalid_argument("brain perturbation amplitude must be finite and nonnegative");
        }

        if (tick_count > numeric_limits<uint64_t>::max() - tick_)
        {
            throw overflow_error("brain tick counter overflow");
        }

        try
        {
            activate_device();
            input_drives_.upload_async(input_drives, stream_.get());
            spike_counts_.clear_async(stream_.get());
            advance_result_.clear_async(stream_.get());
            DeviceAdvanceReport* report = reinterpret_cast<DeviceAdvanceReport*>(
                advance_result_.data());
            const uint32_t neuron_count = static_cast<uint32_t>(model_.positions.size());
            const uint32_t blocks = (neuron_count + CUDA_THREADS_PER_BLOCK - 1U)
                / CUDA_THREADS_PER_BLOCK;

            for (uint32_t tick = 0; tick < tick_count; ++tick)
            {
                advance_neurons_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream_.get()>>>(
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
                    refractory_ticks_.data(),
                    report,
                    tick_ + tick,
                    perturbation.noise_seed,
                    perturbation.noise_drive_amplitude,
                    config_);
                check_cuda(cudaGetLastError(), "launch CUDA neuron update");
                update_plasticity_and_traces_kernel<<<
                    blocks,
                    CUDA_THREADS_PER_BLOCK,
                    0,
                    stream_.get()>>>(
                    neuron_count,
                    outgoing_offsets_.data(),
                    outgoing_targets_.data(),
                    polarities_.data(),
                    next_spikes_.data(),
                    spike_counts_.data(),
                    activity_traces_.data(),
                    connection_strengths_.data(),
                    connection_eligibilities_.data(),
                    report,
                    plasticity_enabled,
                    config_);
                check_cuda(cudaGetLastError(), "launch CUDA plasticity and trace update");
                swap(previous_spikes_, next_spikes_);
            }

            summarize_spikes_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream_.get()>>>(
                neuron_count,
                spike_counts_.data(),
                output_flags_.data(),
                report);
            check_cuda(cudaGetLastError(), "launch CUDA spike summary");

            if (!output_activity.empty())
            {
                const uint32_t output_count = static_cast<uint32_t>(model_.output_neurons.size());
                const uint32_t output_blocks = (output_count + CUDA_THREADS_PER_BLOCK - 1U)
                    / CUDA_THREADS_PER_BLOCK;
                float* device_output_activity = reinterpret_cast<float*>(
                    advance_result_.data() + sizeof(DeviceAdvanceReport));
                gather_output_activity_kernel<<<
                    output_blocks,
                    CUDA_THREADS_PER_BLOCK,
                    0,
                    stream_.get()>>>(
                    output_count,
                    output_neurons_.data(),
                    activity_traces_.data(),
                    device_output_activity);
                check_cuda(cudaGetLastError(), "launch CUDA output activity gather");
            }

            const size_t result_bytes = sizeof(DeviceAdvanceReport)
                + sizeof(float) * output_activity.size();
            advance_result_.download_prefix_async(
                span<uint8_t>(host_advance_result_.data(), result_bytes),
                stream_.get());
            check_cuda(cudaStreamSynchronize(stream_.get()), "synchronize CUDA brain advance");
            DeviceAdvanceReport host_report{};
            memcpy(&host_report, host_advance_result_.data(), sizeof(host_report));

            if (host_report.error != 0)
            {
                throw runtime_error("CUDA brain advance produced invalid numerical state");
            }

            if (!output_activity.empty())
            {
                memcpy(
                    output_activity.data(),
                    host_advance_result_.data() + sizeof(DeviceAdvanceReport),
                    output_activity.size_bytes());
            }

            tick_ += tick_count;
            return {host_report.neuron_spikes, host_report.output_spikes};
        }
        catch (...)
        {
            requires_reset_ = true;
            throw;
        }
    }

    // applies one signed global event to eligible local connections
    void apply_plasticity_modulation(float modulator)
    {
        if (requires_reset_)
        {
            throw logic_error("CUDA brain simulator must be reset after a failed device step");
        }

        if (!isfinite(modulator))
        {
            throw invalid_argument("brain plasticity modulator must be finite");
        }

        if (modulator == 0.0f)
        {
            return;
        }

        try
        {
            activate_device();
            advance_result_.clear_async(stream_.get());
            DeviceAdvanceReport* report = reinterpret_cast<DeviceAdvanceReport*>(
                advance_result_.data());
            const uint32_t neuron_count = static_cast<uint32_t>(model_.positions.size());
            const uint32_t blocks = (neuron_count + CUDA_THREADS_PER_BLOCK - 1U)
                / CUDA_THREADS_PER_BLOCK;
            modulate_connections_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream_.get()>>>(
                neuron_count,
                outgoing_offsets_.data(),
                connection_strengths_.data(),
                connection_eligibilities_.data(),
                report,
                modulator,
                config_);
            check_cuda(cudaGetLastError(), "launch CUDA plasticity modulation");
            advance_result_.download_prefix_async(
                span<uint8_t>(host_advance_result_.data(), sizeof(DeviceAdvanceReport)),
                stream_.get());
            check_cuda(cudaStreamSynchronize(stream_.get()),
                "synchronize CUDA plasticity modulation");
            DeviceAdvanceReport host_report{};
            memcpy(&host_report, host_advance_result_.data(), sizeof(host_report));

            if (host_report.error != 0)
            {
                throw runtime_error("CUDA brain modulation produced invalid numerical state");
            }
        }
        catch (...)
        {
            requires_reset_ = true;
            throw;
        }
    }

    // replaces learned strengths and clears transient activity
    void load_connection_strengths(span<const float> strengths)
    {
        validate_connection_strengths(model_, strengths);
        BrainState state = make_brain_state(model_);
        state.connection_strengths.assign(strengths.begin(), strengths.end());
        upload_state(state, true);
    }

    // returns a host copy of the learned connection strengths
    vector<float> copy_connection_strengths() const
    {
        activate_device();
        vector<float> strengths(model_.outgoing_targets.size());
        connection_strengths_.download(strengths);
        return strengths;
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
        state.connection_eligibilities.resize(model_.outgoing_targets.size());
        state.refractory_ticks.resize(model_.positions.size());
        state.spikes.resize(model_.positions.size());
        state.tick = tick_;
        membrane_potentials_.download(state.membrane_potentials);
        activity_traces_.download(state.activity_traces);
        connection_strengths_.download(state.connection_strengths);
        connection_eligibilities_.download(state.connection_eligibilities);
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
            membrane_potentials_.upload_async(state.membrane_potentials, stream_.get());
            activity_traces_.upload_async(state.activity_traces, stream_.get());
            refractory_ticks_.upload_async(state.refractory_ticks, stream_.get());
            previous_spikes_.upload_async(state.spikes, stream_.get());
            next_spikes_.clear_async(stream_.get());
            connection_eligibilities_.upload_async(
                state.connection_eligibilities,
                stream_.get());

            if (include_strengths)
            {
                connection_strengths_.upload_async(state.connection_strengths, stream_.get());
            }

            check_cuda(cudaStreamSynchronize(stream_.get()), "upload CUDA brain state");
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
    DeviceBuffer<uint8_t> output_flags_;
    DeviceBuffer<uint32_t> output_neurons_;
    DeviceBuffer<float> membrane_potentials_;
    DeviceBuffer<float> activity_traces_;
    DeviceBuffer<float> connection_strengths_;
    DeviceBuffer<float> connection_eligibilities_;
    DeviceBuffer<uint32_t> refractory_ticks_;
    DeviceBuffer<uint8_t> previous_spikes_;
    DeviceBuffer<uint8_t> next_spikes_;
    DeviceBuffer<uint32_t> spike_counts_;
    DeviceBuffer<uint8_t> advance_result_;
    PinnedBuffer<uint8_t> host_advance_result_;
    uint64_t tick_ = 0;
    bool requires_reset_ = false;
    CudaStream stream_;
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

// returns reproducibility metadata for one usable cuda device
CudaDeviceInfo cuda_device_info(int32_t device)
{
    if (!cuda_device_available(device))
    {
        throw runtime_error("requested CUDA brain device is unavailable");
    }

    cudaDeviceProp properties{};
    int32_t driver_version = 0;
    int32_t runtime_version = 0;
    check_cuda(cudaGetDeviceProperties(&properties, device), "query CUDA device properties");
    check_cuda(cudaDriverGetVersion(&driver_version), "query CUDA driver version");
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "query CUDA runtime version");
    return {
        properties.name,
        properties.major,
        properties.minor,
        driver_version,
        runtime_version,
        CUDART_VERSION
    };
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

// advances every neuron and optionally records plasticity by one fixed tick
void CudaBrainSimulator::step(
    span<const float> input_drives,
    BrainPerturbation perturbation,
    bool plasticity_enabled)
{
    advance(input_drives, 1U, perturbation, plasticity_enabled, {});
}

// advances multiple fixed ticks with one host-device synchronization
BrainAdvanceSummary CudaBrainSimulator::advance(
    span<const float> input_drives,
    uint32_t tick_count,
    BrainPerturbation perturbation,
    bool plasticity_enabled,
    span<float> output_activity)
{
    return implementation_->advance(
        input_drives,
        tick_count,
        perturbation,
        plasticity_enabled,
        output_activity);
}

// applies one signed global event to eligible local connections
void CudaBrainSimulator::apply_plasticity_modulation(float modulator)
{
    implementation_->apply_plasticity_modulation(modulator);
}

// replaces learned strengths and clears transient activity
void CudaBrainSimulator::load_connection_strengths(span<const float> strengths)
{
    implementation_->load_connection_strengths(strengths);
}

// returns a host copy of the learned connection strengths
vector<float> CudaBrainSimulator::copy_connection_strengths() const
{
    return implementation_->copy_connection_strengths();
}

// returns a full host snapshot for diagnostics outside the hot path
BrainState CudaBrainSimulator::copy_state() const
{
    return implementation_->download_state();
}

// returns the immutable host copy of the compiled model
const BrainModel& CudaBrainSimulator::model() const
{
    return implementation_->model();
}

// returns the cuda execution backend
BrainBackend CudaBrainSimulator::backend() const
{
    using enum BrainBackend;
    return Cuda;
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

// creates one cuda simulator for the requested cuda execution configuration
unique_ptr<BrainSimulator> make_cuda_brain_simulator(
    BrainModel model,
    const BrainExecutionConfig& execution)
{
    using enum BrainBackend;

    if (execution.backend != Cuda)
    {
        throw invalid_argument("cuda brain factory received a non-cuda execution request");
    }

    if (execution.cuda_device < 0)
    {
        throw invalid_argument("CUDA device index must be nonnegative");
    }

    if (!cuda_device_available(execution.cuda_device))
    {
        throw runtime_error("requested CUDA brain device is unavailable");
    }

    return make_unique<CudaBrainSimulator>(move(model), execution.cuda_device);
}

}
