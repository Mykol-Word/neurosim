#include "neurosim/brain.h"
#include "neurosim/cpu_brain.h"
#include "neurosim/cuda_brain.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace neurosim;

namespace
{

inline constexpr float ABSOLUTE_TOLERANCE = 0.00002f;
inline constexpr float RELATIVE_TOLERANCE = 0.00002f;

// fails a cuda parity check when a boolean condition is false
void check(bool condition, const string& message)
{
    if (!condition)
    {
        throw runtime_error(message);
    }
}

// fails a cuda parity check when two floats exceed combined tolerances
void check_near(float actual, float expected, const string& message)
{
    const float tolerance = ABSOLUTE_TOLERANCE
        + RELATIVE_TOLERANCE * max(abs(actual), abs(expected));

    if (!isfinite(actual) || !isfinite(expected) || abs(actual - expected) > tolerance)
    {
        throw runtime_error(message + ": expected " + to_string(expected)
            + ", received " + to_string(actual));
    }
}

// verifies complete cpu and cuda mutable state within stated float tolerances
void check_state_parity(const BrainState& cpu, const BrainState& cuda)
{
    check(cpu.tick == cuda.tick, "CPU and CUDA tick counters differ");
    check(cpu.spikes == cuda.spikes, "CPU and CUDA spikes differ");
    check(cpu.refractory_ticks == cuda.refractory_ticks,
        "CPU and CUDA refractory counters differ");
    check(cpu.membrane_potentials.size() == cuda.membrane_potentials.size()
        && cpu.activity_traces.size() == cuda.activity_traces.size()
        && cpu.connection_strengths.size() == cuda.connection_strengths.size(),
        "CPU and CUDA state array sizes differ");

    for (size_t index = 0; index < cpu.membrane_potentials.size(); ++index)
    {
        check_near(cuda.membrane_potentials[index], cpu.membrane_potentials[index],
            "CPU and CUDA membrane potentials differ");
        check_near(cuda.activity_traces[index], cpu.activity_traces[index],
            "CPU and CUDA activity traces differ");
    }

    for (size_t edge = 0; edge < cpu.connection_strengths.size(); ++edge)
    {
        check_near(cuda.connection_strengths[edge], cpu.connection_strengths[edge],
            "CPU and CUDA connection strengths differ");
    }
}

// creates a mixed-polarity sparse model for backend parity tests
BrainModel make_parity_model()
{
    BrainLayout layout;
    layout.shape = {5, 3, 1};
    layout.neurons = {
        {{0, 1, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{1, 1, 0}, NeuronPolarity::Excitatory, 3.0f},
        {{2, 1, 0}, NeuronPolarity::Inhibitory, 0.0f},
        {{3, 1, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{4, 1, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{2, 0, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{2, 2, 0}, NeuronPolarity::Excitatory, 0.0f}
    };
    layout.input_positions = {{0, 1, 0}, {4, 1, 0}};
    layout.output_positions = {{3, 1, 0}, {1, 1, 0}};

    BrainConfig config;
    config.connection_radius_steps = 2;
    config.dynamics.fixed_time_step_seconds = 0.001f;
    config.dynamics.membrane_time_constant_seconds = 0.020f;
    config.dynamics.refractory_period_seconds = 0.002f;
    config.plasticity.activity_trace_time_constant_seconds = 0.025f;
    config.plasticity.learning_rate = 0.015f;
    config.plasticity.weakening_rate_per_second = 0.02f;
    config.plasticity.outgoing_strength_budget = 1.0f;
    config.plasticity.initial_budget_fraction = 0.8f;
    config.plasticity.initial_connection_density = 0.5f;
    return build_brain_model(layout, config, 9917);
}

// verifies dynamics, plasticity, outputs, and both reset modes across backends
void test_cpu_cuda_trajectory_parity()
{
    const BrainModel model = make_parity_model();
    CpuBrainSimulator cpu(model);
    CudaBrainSimulator cuda(model);
    const vector<array<float, 2>> inputs{
        {700.0f, 0.0f},
        {700.0f, 0.0f},
        {0.0f, 0.0f},
        {0.0f, 700.0f},
        {0.0f, 700.0f},
        {400.0f, 400.0f},
        {0.0f, 0.0f},
        {700.0f, 700.0f},
        {0.0f, 0.0f}
    };

    for (const array<float, 2>& input : inputs)
    {
        cpu.step(input);
        cuda.step(input);
        check_state_parity(cpu.state(), cuda.download_state());
    }

    check(cuda.output_spike(0) == cpu.output_spike(0),
        "CPU and CUDA output spike samples differ");
    check_near(cuda.output_activity(1), cpu.output_activity(1),
        "CPU and CUDA output activity samples differ");
    const BrainState learned = cuda.download_state();
    cuda.reset_activity();
    const BrainState activity_reset = cuda.download_state();
    check(activity_reset.connection_strengths == learned.connection_strengths,
        "CUDA activity reset discarded learned strengths");
    check(activity_reset.tick == 0U,
        "CUDA activity reset retained the tick counter");
    cuda.reset();
    check_state_parity(make_brain_state(model), cuda.download_state());
}

// verifies a neuron with no possible edges runs without zero-size allocation errors
void test_cuda_isolated_neuron()
{
    BrainLayout layout;
    layout.shape = {1, 1, 1};
    layout.neurons = {{{0, 0, 0}, NeuronPolarity::Excitatory, 0.0f}};
    layout.input_positions = {{0, 0, 0}};
    layout.output_positions = {{0, 0, 0}};
    BrainConfig config;
    config.plasticity.initial_connection_density = 0.0f;
    config.plasticity.initial_budget_fraction = 0.0f;
    const BrainModel model = build_brain_model(layout, config, 1);
    CpuBrainSimulator cpu(model);
    CudaBrainSimulator cuda(model);
    const array<float, 1> input{1200.0f};
    cpu.step(input);
    cuda.step(input);
    check_state_parity(cpu.state(), cuda.download_state());
}

// verifies kernel bounds immediately around block and multi-block sizes
void test_cuda_kernel_boundaries()
{
    constexpr array<uint32_t, 4> NEURON_COUNTS{31U, 32U, 33U, 257U};

    for (const uint32_t neuron_count : NEURON_COUNTS)
    {
        BrainLayout layout;
        layout.shape = {neuron_count, 1, 1};
        layout.neurons.reserve(neuron_count);

        for (uint32_t x = 0; x < neuron_count; ++x)
        {
            layout.neurons.push_back({{x, 0, 0}, NeuronPolarity::Excitatory, 0.0f});
        }

        layout.input_positions = {{0, 0, 0}};
        BrainConfig config;
        config.plasticity.initial_budget_fraction = 0.5f;
        const BrainModel model = build_brain_model(layout, config, neuron_count);
        CpuBrainSimulator cpu(model);
        CudaBrainSimulator cuda(model);
        const array<float, 1> input{700.0f};
        cpu.step(input);
        cuda.step(input);
        check_state_parity(cpu.state(), cuda.download_state());
    }
}

}

// runs cuda parity tests or reports a clear ctest skip without a device
int main()
{
    try
    {
        if (!cuda_device_available())
        {
            cout << "skip no usable CUDA device\n";
            return 77;
        }

        test_cpu_cuda_trajectory_parity();
        test_cuda_isolated_neuron();
        test_cuda_kernel_boundaries();
        cout << "3 CUDA brain tests passed\n";
        return 0;
    }
    catch (const exception& error)
    {
        cerr << "CUDA brain tests failed: " << error.what() << '\n';
        return 1;
    }
}
