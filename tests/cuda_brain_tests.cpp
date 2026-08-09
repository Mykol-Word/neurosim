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
        && cpu.connection_strengths.size() == cuda.connection_strengths.size()
        && cpu.connection_eligibilities.size() == cuda.connection_eligibilities.size(),
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
        check_near(cuda.connection_eligibilities[edge], cpu.connection_eligibilities[edge],
            "CPU and CUDA connection eligibilities differ");
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
    config.plasticity.eligibility_time_constant_seconds = 0.400f;
    config.plasticity.maximum_eligibility = 2.0f;
    config.plasticity.learning_rate = 0.015f;
    config.plasticity.weakening_rate_per_second = 0.02f;
    config.plasticity.outgoing_strength_budget = 1.0f;
    config.plasticity.initial_budget_fraction = 0.8f;
    config.plasticity.initial_connection_density = 0.5f;
    return build_brain_model(layout, config, 9917);
}

// creates a two-neuron model with one hand-checkable eligible connection
BrainModel make_modulation_model()
{
    BrainLayout layout;
    layout.shape = {2, 1, 1};
    layout.neurons = {
        {{0, 0, 0}, NeuronPolarity::Excitatory, 0.0f},
        {{1, 0, 0}, NeuronPolarity::Excitatory, 0.0f}
    };
    layout.input_positions = {{0, 0, 0}, {1, 0, 0}};
    layout.output_positions = {{0, 0, 0}, {1, 0, 0}};

    BrainConfig config;
    config.connection_radius_steps = 1;
    config.plasticity.activity_trace_time_constant_seconds = 0.020f;
    config.plasticity.eligibility_time_constant_seconds = 0.500f;
    config.plasticity.maximum_eligibility = 2.0f;
    config.plasticity.learning_rate = 0.250f;
    config.plasticity.weakening_rate_per_second = 0.1f;
    config.plasticity.outgoing_strength_budget = 1.0f;
    config.plasticity.initial_budget_fraction = 0.8f;
    config.plasticity.initial_connection_density = 1.0f;
    return build_brain_model(layout, config, 17);
}

// verifies one cpu batch exactly matches repeated reference steps
void test_cpu_batched_advance()
{
    const BrainModel model = make_parity_model();
    CpuBrainSimulator repeated(model);
    CpuBrainSimulator batched(model);
    const array<float, 2> input{520.0f, 180.0f};
    const BrainPerturbation perturbation{210.0f, 1872};
    constexpr uint32_t TICK_COUNT = 23U;
    BrainAdvanceSummary repeated_summary;

    for (uint32_t tick = 0; tick < TICK_COUNT; ++tick)
    {
        repeated.step(input, perturbation);

        for (const uint8_t spike : repeated.state().spikes)
        {
            repeated_summary.neuron_spikes += spike;
        }

        for (const uint32_t output_neuron : model.output_neurons)
        {
            repeated_summary.output_spikes += repeated.state().spikes[output_neuron];
        }
    }

    array<float, 2> output_activity{};
    const BrainAdvanceSummary batched_summary = batched.advance(
        input,
        TICK_COUNT,
        perturbation,
        true,
        output_activity);
    check(batched_summary.neuron_spikes == repeated_summary.neuron_spikes,
        "repeated and batched CPU neuron spike totals differ");
    check(batched_summary.output_spikes == repeated_summary.output_spikes,
        "repeated and batched CPU output spike totals differ");
    check_state_parity(repeated.state(), batched.state());

    for (uint32_t output = 0; output < output_activity.size(); ++output)
    {
        check_near(output_activity[output], repeated.output_activity(output),
            "batched CPU output activity differs from repeated steps");
    }
}

// verifies one cuda batch matches cpu summaries, outputs, and complete state
void test_cpu_cuda_batched_advance()
{
    const BrainModel model = make_parity_model();
    CpuBrainSimulator cpu(model);
    CudaBrainSimulator cuda(model);
    const array<float, 2> input{610.0f, 240.0f};
    const BrainPerturbation perturbation{160.0f, 8119};
    constexpr uint32_t TICK_COUNT = 31U;
    array<float, 2> cpu_activity{};
    array<float, 2> cuda_activity{};
    const BrainAdvanceSummary cpu_summary = cpu.advance(
        input,
        TICK_COUNT,
        perturbation,
        true,
        cpu_activity);
    const BrainAdvanceSummary cuda_summary = cuda.advance(
        input,
        TICK_COUNT,
        perturbation,
        true,
        cuda_activity);
    check(cuda_summary.neuron_spikes == cpu_summary.neuron_spikes,
        "CPU and CUDA batched neuron spike totals differ");
    check(cuda_summary.output_spikes == cpu_summary.output_spikes,
        "CPU and CUDA batched output spike totals differ");
    check_state_parity(cpu.state(), cuda.download_state());

    for (uint32_t output = 0; output < cuda_activity.size(); ++output)
    {
        check_near(cuda_activity[output], cpu_activity[output],
            "CPU and CUDA batched output activities differ");
    }
}

// verifies cuda batch validation leaves valid state usable
void test_cuda_advance_validation()
{
    const BrainModel model = make_parity_model();
    CpuBrainSimulator cpu(model);
    CudaBrainSimulator cuda(model);
    const array<float, 2> input{0.0f, 0.0f};
    array<float, 1> wrong_output{};
    bool rejected_zero_ticks = false;

    try
    {
        cuda.advance(input, 0U, {}, true, {});
    }
    catch (const invalid_argument&)
    {
        rejected_zero_ticks = true;
    }

    check(rejected_zero_ticks, "CUDA advance accepted zero ticks");
    bool cpu_rejected_zero_ticks = false;

    try
    {
        cpu.advance(input, 0U, {}, true, {});
    }
    catch (const invalid_argument&)
    {
        cpu_rejected_zero_ticks = true;
    }

    check(cpu_rejected_zero_ticks, "CPU advance accepted zero ticks");
    bool rejected_output_size = false;

    try
    {
        cuda.advance(input, 1U, {}, true, wrong_output);
    }
    catch (const invalid_argument&)
    {
        rejected_output_size = true;
    }

    check(rejected_output_size, "CUDA advance accepted the wrong output activity size");
    bool cpu_rejected_output_size = false;

    try
    {
        cpu.advance(input, 1U, {}, true, wrong_output);
    }
    catch (const invalid_argument&)
    {
        cpu_rejected_output_size = true;
    }

    check(cpu_rejected_output_size, "CPU advance accepted the wrong output activity size");
    const BrainAdvanceSummary summary = cuda.advance(input, 1U, {}, true, {});
    check(summary.neuron_spikes <= model.positions.size(),
        "CUDA advance returned an impossible neuron spike count");
    check(cuda.download_state().tick == 1U,
        "CUDA validation failure changed the simulator tick");
}

// verifies private-stream kernels never race model or state uploads
void test_cuda_state_upload_ordering()
{
    const BrainModel model = make_parity_model();
    BrainState initial = make_brain_state(model);

    for (float& strength : initial.connection_strengths)
    {
        strength *= 0.625f;
    }

    const array<float, 2> input{640.0f, 215.0f};

    for (uint32_t iteration = 0; iteration < 32U; ++iteration)
    {
        CpuBrainSimulator cpu(model);
        CudaBrainSimulator cuda(model);
        cpu.reset(initial);
        cuda.reset(initial);
        cpu.advance(input, 3U, {}, true, {});
        cuda.advance(input, 3U, {}, true, {});
        check_state_parity(cpu.state(), cuda.download_state());
    }
}

// verifies backend construction and learned strength transfers stay explicit
void test_cuda_backend_interface()
{
    using enum BrainBackend;

    const BrainModel model = make_modulation_model();
    const BrainExecutionConfig execution{Cuda, 0};
    unique_ptr<BrainSimulator> simulator = make_cuda_brain_simulator(model, execution);
    check(simulator->backend() == Cuda, "CUDA factory returned the wrong backend");
    vector<float> strengths = model.initial_connection_strengths;

    for (float& strength : strengths)
    {
        strength *= 0.5f;
    }

    simulator->load_connection_strengths(strengths);
    check(simulator->copy_connection_strengths() == strengths,
        "CUDA learned strength transfer changed values");
    const CudaDeviceInfo device = cuda_device_info();
    check(!device.name.empty(), "CUDA device metadata omitted the device name");
    check(device.compute_major > 0, "CUDA device metadata has invalid compute capability");
    check(device.driver_version > 0 && device.runtime_version > 0
        && device.toolkit_version > 0,
        "CUDA device metadata omitted software versions");
    bool rejected_cpu_request = false;

    try
    {
        make_cuda_brain_simulator(model, {Cpu, 0});
    }
    catch (const invalid_argument&)
    {
        rejected_cpu_request = true;
    }

    check(rejected_cpu_request, "CUDA factory accepted a CPU execution request");
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

    const BrainPerturbation perturbation{300.0f, 4821};

    for (uint32_t step = 0; step < 5U; ++step)
    {
        const array<float, 2> input{0.0f, 0.0f};
        cpu.step(input, perturbation);
        cuda.step(input, perturbation);
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

// verifies eligibility, signed modulation, disabled learning, and reset parity
void test_cuda_modulated_plasticity()
{
    const BrainModel model = make_modulation_model();
    CpuBrainSimulator cpu(model);
    CudaBrainSimulator cuda(model);
    const uint32_t source = neuron_index_at(model, {0, 0, 0});
    const uint32_t target = neuron_index_at(model, {1, 0, 0});
    const uint32_t edge = connection_index(model, source, target);
    const array<float, 2> source_spike{1200.0f, 0.0f};
    const array<float, 2> target_spike{0.0f, 1200.0f};

    cpu.step(source_spike);
    cuda.step(source_spike);
    cpu.step(target_spike);
    cuda.step(target_spike);
    check_state_parity(cpu.state(), cuda.download_state());
    const BrainState eligible_state = cpu.state();
    check(eligible_state.connection_eligibilities[edge] > 0.0f,
        "directed spike sequence did not create eligibility");

    cpu.apply_plasticity_modulation(1.0f);
    cuda.apply_plasticity_modulation(1.0f);
    check_state_parity(cpu.state(), cuda.download_state());
    const BrainState rewarded_state = cpu.state();
    check(rewarded_state.connection_strengths[edge]
            > eligible_state.connection_strengths[edge],
        "positive modulation did not strengthen an eligible connection");
    check(rewarded_state.connection_strengths[edge]
            <= model.config.plasticity.outgoing_strength_budget,
        "positive modulation exceeded the outgoing strength budget");

    for (const float eligibility : rewarded_state.connection_eligibilities)
    {
        check(eligibility == 0.0f, "positive modulation did not consume eligibility");
    }

    cpu.reset(eligible_state);
    cuda.reset(eligible_state);
    cpu.apply_plasticity_modulation(-10.0f);
    cuda.apply_plasticity_modulation(-10.0f);
    check_state_parity(cpu.state(), cuda.download_state());
    check(cpu.state().connection_strengths[edge] == 0.0f,
        "negative modulation did not clamp an eligible connection at zero");

    for (const float eligibility : cpu.state().connection_eligibilities)
    {
        check(eligibility == 0.0f, "negative modulation did not consume eligibility");
    }

    cpu.reset(eligible_state);
    cuda.reset(eligible_state);
    const vector<float> frozen_strengths = eligible_state.connection_strengths;
    const vector<float> frozen_eligibilities = eligible_state.connection_eligibilities;

    for (uint32_t step = 0; step < 3U; ++step)
    {
        cpu.step(target_spike, {}, false);
        cuda.step(target_spike, {}, false);
        check_state_parity(cpu.state(), cuda.download_state());
    }

    check(cpu.state().connection_strengths == frozen_strengths,
        "learning-disabled CPU steps changed connection strengths");
    check(cpu.state().connection_eligibilities == frozen_eligibilities,
        "learning-disabled CPU steps changed connection eligibilities");
    const BrainState before_activity_reset = cpu.state();
    cpu.reset_activity();
    cuda.reset_activity();
    check_state_parity(cpu.state(), cuda.download_state());
    check(cpu.state().connection_strengths == before_activity_reset.connection_strengths,
        "activity reset discarded learned connection strengths");

    for (const float eligibility : cpu.state().connection_eligibilities)
    {
        check(eligibility == 0.0f, "activity reset retained connection eligibility");
    }

    cpu.reset();
    cuda.reset();
    check_state_parity(cpu.state(), cuda.download_state());
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
    constexpr array<uint32_t, 4> NEURON_COUNTS{127U, 128U, 129U, 257U};

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

        test_cpu_batched_advance();
        test_cpu_cuda_batched_advance();
        test_cuda_advance_validation();
        test_cuda_state_upload_ordering();
        test_cuda_backend_interface();
        test_cpu_cuda_trajectory_parity();
        test_cuda_modulated_plasticity();
        test_cuda_isolated_neuron();
        test_cuda_kernel_boundaries();
        cout << "9 CUDA brain tests passed\n";
        return 0;
    }
    catch (const exception& error)
    {
        cerr << "CUDA brain tests failed: " << error.what() << '\n';
        return 1;
    }
}
