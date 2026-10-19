#include "neurosim/adaptive_controller.h"
#include "neurosim/cpu_brain.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace neurosim
{

using namespace std;

namespace
{

inline constexpr uint32_t MAX_BRAIN_STEPS_PER_CONTROL_STEP = 100'000;

// mixes one seed into a stable random value
uint64_t mix_seed(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

// returns one reproducible unit sample with 24 random bits
float unit_sample(uint64_t value)
{
    return static_cast<float>(mix_seed(value) >> 40U) * (1.0f / 16777216.0f);
}

// converts a positive duration to the first control tick meeting it
uint64_t duration_ticks(double seconds, double fixed_time_step)
{
    const double ticks = ceil(seconds / fixed_time_step - 1.0e-12);

    if (!isfinite(ticks) || ticks < 1.0
        || ticks > static_cast<double>(numeric_limits<uint64_t>::max()))
    {
        throw invalid_argument("controller duration cannot be represented as fixed ticks");
    }

    return static_cast<uint64_t>(ticks);
}

// creates one-hot exploration patterns when a task supplies no codebook
vector<vector<float>> make_default_patterns(uint32_t output_count)
{
    vector<vector<float>> patterns(output_count, vector<float>(output_count, 0.0f));

    for (uint32_t output = 0; output < output_count; ++output)
    {
        patterns[output][output] = 1.0f;
    }

    return patterns;
}

// validates and completes one task-neutral controller configuration
AdaptiveControllerConfig prepare_config(
    AdaptiveControllerConfig config,
    uint32_t input_count,
    uint32_t output_count,
    float control_time_step_seconds,
    const BrainGenome* genome)
{
    if (genome != nullptr)
    {
        config.brain = genome->config;
        config.brain_time_step_seconds = genome->config.dynamics.fixed_time_step_seconds;
    }

    if (input_count == 0U || output_count == 0U)
    {
        throw invalid_argument("controller input and output counts must be positive");
    }

    const array<float, 8> finite_values{
        config.brain_radius,
        config.brain_time_step_seconds,
        config.sensory_drive,
        config.exploration_drive,
        config.minimum_bias_drive,
        config.maximum_bias_drive,
        config.exploration_interval_seconds,
        config.exploration_duration_seconds
    };

    for (const float value : finite_values)
    {
        if (!isfinite(value))
        {
            throw invalid_argument("controller configuration values must be finite");
        }
    }

    if (config.brain_radius < 0.0f
        || config.brain_time_step_seconds <= 0.0f
        || config.sensory_drive <= 0.0f
        || config.exploration_drive <= 0.0f
        || config.minimum_bias_drive < 0.0f
        || config.maximum_bias_drive < config.minimum_bias_drive
        || config.exploration_interval_seconds <= 0.0f
        || config.exploration_duration_seconds <= 0.0f
        || config.exploration_duration_seconds > config.exploration_interval_seconds)
    {
        throw invalid_argument("controller configuration ranges are invalid");
    }

    if (!isfinite(config.inhibitory_fraction)
        || config.inhibitory_fraction < 0.0f
        || config.inhibitory_fraction > 1.0f)
    {
        throw invalid_argument("controller inhibitory fraction must be between zero and one");
    }

    if (!isfinite(control_time_step_seconds) || control_time_step_seconds <= 0.0f)
    {
        throw invalid_argument("controller time step must be finite and positive");
    }

    if (config.brain_time_step_seconds > control_time_step_seconds
        || ceil(control_time_step_seconds / config.brain_time_step_seconds) + 1.0f
            > static_cast<float>(MAX_BRAIN_STEPS_PER_CONTROL_STEP))
    {
        throw invalid_argument("controller neural timing is incompatible with its control step");
    }

    if (config.exploration_patterns.empty())
    {
        config.exploration_patterns = make_default_patterns(output_count);
    }

    for (const vector<float>& pattern : config.exploration_patterns)
    {
        if (pattern.size() != output_count)
        {
            throw invalid_argument("controller exploration pattern has the wrong output count");
        }

        for (const float value : pattern)
        {
            if (!isfinite(value) || value < 0.0f || value > 1.0f)
            {
                throw invalid_argument("controller exploration patterns must be normalized");
            }
        }
    }

    config.brain.dynamics.fixed_time_step_seconds = config.brain_time_step_seconds;
    return config;
}

// creates random task-neutral sensor and output locations in one sparse brain
BrainLayout make_controller_layout(
    const AdaptiveControllerConfig& config,
    uint32_t input_count,
    uint32_t output_count,
    uint64_t seed)
{
    BrainLayout layout = make_sphere_brain_layout(config.brain_shape, config.brain_radius);
    const uint64_t required = static_cast<uint64_t>(input_count) + output_count;

    if (layout.neurons.size() < required)
    {
        throw invalid_argument("controller brain does not contain enough interface neurons");
    }

    for (uint32_t neuron = 0; neuron < layout.neurons.size(); ++neuron)
    {
        NeuronDefinition& definition = layout.neurons[neuron];
        definition.polarity = unit_sample(seed ^ (0x8cb92baa3f3d8dd7ULL + neuron))
                < config.inhibitory_fraction
            ? NeuronPolarity::Inhibitory
            : NeuronPolarity::Excitatory;
        definition.bias_drive = config.minimum_bias_drive
            + (config.maximum_bias_drive - config.minimum_bias_drive)
                * unit_sample(seed ^ (0xdb4f0b9175ae2165ULL + neuron));
    }

    vector<uint32_t> order(layout.neurons.size());
    iota(order.begin(), order.end(), 0U);
    sort(order.begin(), order.end(), [seed](uint32_t left, uint32_t right)
    {
        const uint64_t left_key = mix_seed(seed ^ (0xa24baed4963ee407ULL + left));
        const uint64_t right_key = mix_seed(seed ^ (0xa24baed4963ee407ULL + right));
        return left_key == right_key ? left < right : left_key < right_key;
    });

    layout.input_positions.reserve(input_count + output_count);
    layout.output_positions.reserve(output_count);

    for (uint32_t input = 0; input < input_count; ++input)
    {
        layout.input_positions.push_back(layout.neurons[order[input]].position);
    }

    for (uint32_t output = 0; output < output_count; ++output)
    {
        const GridPosition position = layout.neurons[order[input_count + output]].position;
        layout.input_positions.push_back(position);
        layout.output_positions.push_back(position);
    }

    return layout;
}

// builds one reproducible task-neutral plastic brain
BrainModel make_controller_model(
    const AdaptiveControllerConfig& config,
    uint32_t input_count,
    uint32_t output_count,
    uint64_t seed,
    const BrainGenome* genome)
{
    if (genome != nullptr)
    {
        if (genome->input_positions.size()
                != static_cast<size_t>(input_count) + output_count
            || genome->output_positions.size() != output_count)
        {
            throw invalid_argument("controller genome interface count is incompatible");
        }

        for (uint32_t output = 0; output < output_count; ++output)
        {
            if (genome->input_positions[input_count + output]
                != genome->output_positions[output])
            {
                throw invalid_argument("controller genome output drive mapping is incompatible");
            }
        }

        return build_brain_model(*genome);
    }

    return build_brain_model(
        make_controller_layout(config, input_count, output_count, seed),
        config.brain,
        seed ^ 0x9e3779b97f4a7c15ULL);
}

// creates the requested backend after compiling one shared brain model
unique_ptr<BrainSimulator> make_controller_simulator(
    const AdaptiveControllerConfig& config,
    uint32_t input_count,
    uint32_t output_count,
    uint64_t seed,
    const BrainExecutionConfig& execution,
    BrainSimulatorFactory simulator_factory,
    const BrainGenome* genome)
{
    if (execution.cuda_device < 0)
    {
        throw invalid_argument("brain execution device must be nonnegative");
    }

    if (simulator_factory == nullptr)
    {
        if (execution.backend != BrainBackend::Cpu)
        {
            throw invalid_argument("requested brain backend has no simulator factory");
        }

        simulator_factory = make_cpu_brain_simulator;
    }

    unique_ptr<BrainSimulator> simulator = simulator_factory(
        make_controller_model(config, input_count, output_count, seed, genome),
        execution);

    if (simulator == nullptr || simulator->backend() != execution.backend)
    {
        throw logic_error("brain simulator factory returned the wrong backend");
    }

    return simulator;
}

}

// constructs a reproducible seeded or explicit task-neutral brain
AdaptiveController::AdaptiveController(
    uint32_t input_count,
    uint32_t output_count,
    float control_time_step_seconds,
    AdaptiveControllerConfig config,
    uint64_t seed,
    BrainExecutionConfig execution,
    BrainSimulatorFactory simulator_factory,
    const BrainGenome* genome)
    : config_(prepare_config(
          move(config),
          input_count,
          output_count,
          control_time_step_seconds,
          genome)),
      execution_(execution),
      seed_(seed),
      input_count_(input_count),
      output_count_(output_count),
      control_time_step_seconds_(control_time_step_seconds),
      simulator_(make_controller_simulator(
          config_,
          input_count_,
          output_count_,
          seed_,
          execution_,
          simulator_factory,
          genome))
{
    exploration_interval_ticks_ = duration_ticks(
        config_.exploration_interval_seconds,
        control_time_step_seconds);
    exploration_duration_ticks_ = duration_ticks(
        config_.exploration_duration_seconds,
        control_time_step_seconds);
    brain_inputs_.resize(simulator_->model().input_neurons.size());
    status_.output_activity.resize(output_count_);
    status_.exploration_outputs.resize(output_count_);
    reset_episode(0U);
}

// advances one control step from normalized task-owned inputs
const AdaptiveControllerStatus& AdaptiveController::step(span<const float> normalized_inputs)
{
    if (normalized_inputs.size() != input_count_)
    {
        throw invalid_argument("controller input count does not match its interface");
    }

    if (control_tick_ == numeric_limits<uint64_t>::max())
    {
        throw overflow_error("controller tick counter overflow");
    }

    if (exploration_enabled_ && control_tick_ >= next_exploration_tick_)
    {
        exploration_ticks_remaining_ = exploration_duration_ticks_;
        const uint32_t pattern_count = static_cast<uint32_t>(config_.exploration_patterns.size());
        const uint32_t selected = static_cast<uint32_t>(
            mix_seed(exploration_seed_ ^ ++exploration_event_) % pattern_count);

        exploration_pattern_ = selected;

        do
        {
            next_exploration_tick_ += exploration_interval_ticks_;
        }
        while (next_exploration_tick_ <= control_tick_);
    }

    status_.brain_spikes = 0U;
    status_.output_spikes = 0U;
    status_.exploring = exploration_enabled_ && exploration_ticks_remaining_ > 0U;
    fill(status_.exploration_outputs.begin(), status_.exploration_outputs.end(), 0.0f);

    for (uint32_t input = 0; input < input_count_; ++input)
    {
        const float value = normalized_inputs[input];

        if (!isfinite(value) || value < 0.0f || value > 1.0f)
        {
            throw invalid_argument("controller inputs must be finite and normalized");
        }

        brain_inputs_[input] = value * config_.sensory_drive;
    }

    for (uint32_t output = 0; output < output_count_; ++output)
    {
        const float exploration = status_.exploring
            ? config_.exploration_patterns[exploration_pattern_][output]
            : 0.0f;
        status_.exploration_outputs[output] = exploration;
        brain_inputs_[input_count_ + output] = exploration * config_.exploration_drive;
    }

    pending_brain_time_seconds_ += control_time_step_seconds_;
    const double brain_time_step = static_cast<double>(config_.brain_time_step_seconds);
    const double timing_tolerance = brain_time_step * 1.0e-6;
    uint32_t brain_steps = 0U;

    while (pending_brain_time_seconds_ + timing_tolerance >= brain_time_step)
    {
        if (brain_steps == MAX_BRAIN_STEPS_PER_CONTROL_STEP)
        {
            throw logic_error("controller exceeded its validated neural step bound");
        }

        pending_brain_time_seconds_ -= brain_time_step;
        pending_brain_time_seconds_ = max(0.0, pending_brain_time_seconds_);
        ++brain_steps;
    }

    if (brain_steps == 0U)
    {
        throw logic_error("controller produced no neural steps for one control step");
    }

    const BrainAdvanceSummary summary = simulator_->advance(
        brain_inputs_,
        brain_steps,
        {},
        learning_enabled_,
        status_.output_activity);
    status_.brain_spikes = summary.neuron_spikes;
    status_.output_spikes = summary.output_spikes;

    if (status_.exploring)
    {
        --exploration_ticks_remaining_;
    }

    ++control_tick_;
    return status_;
}

// applies one task-owned signed outcome to recent local eligibility
void AdaptiveController::apply_modulation(float value)
{
    if (learning_enabled_)
    {
        simulator_->apply_plasticity_modulation(value);
    }
}

// clears activity and restarts an independent exploration stream
void AdaptiveController::reset_episode(uint64_t exploration_seed)
{
    simulator_->reset_activity();
    exploration_seed_ = mix_seed(exploration_seed ^ 0x632be59bd9b4e019ULL);
    control_tick_ = 0U;
    next_exploration_tick_ = 0U;
    exploration_ticks_remaining_ = 0U;
    exploration_event_ = 0U;
    exploration_pattern_ = 0U;
    pending_brain_time_seconds_ = 0.0;
    fill(brain_inputs_.begin(), brain_inputs_.end(), 0.0f);
    fill(status_.output_activity.begin(), status_.output_activity.end(), 0.0f);
    fill(status_.exploration_outputs.begin(), status_.exploration_outputs.end(), 0.0f);
    status_.brain_spikes = 0U;
    status_.output_spikes = 0U;
    status_.exploring = false;
}

// enables or disables task-blind output exploration
void AdaptiveController::set_exploration_enabled(bool enabled)
{
    exploration_enabled_ = enabled;

    if (!enabled)
    {
        exploration_ticks_remaining_ = 0U;
        fill(status_.exploration_outputs.begin(), status_.exploration_outputs.end(), 0.0f);
        status_.exploring = false;
    }
}

// enables or freezes every connection-strength adaptation path
void AdaptiveController::set_learning_enabled(bool enabled)
{
    learning_enabled_ = enabled;
}

// replaces learned strengths after validating them against this controller
void AdaptiveController::load_connection_strengths(span<const float> strengths)
{
    simulator_->load_connection_strengths(strengths);
}

// returns a host copy of the learned connection strengths
vector<float> AdaptiveController::copy_connection_strengths() const
{
    return simulator_->copy_connection_strengths();
}

// returns a full host snapshot for diagnostics outside the hot path
BrainState AdaptiveController::copy_state() const
{
    return simulator_->copy_state();
}

// returns the fixed construction seed
uint64_t AdaptiveController::seed() const
{
    return seed_;
}

// returns the normalized sensory channel count
uint32_t AdaptiveController::input_count() const
{
    return input_count_;
}

// returns the anonymous output channel count
uint32_t AdaptiveController::output_count() const
{
    return output_count_;
}

// returns the immutable generic controller configuration
const AdaptiveControllerConfig& AdaptiveController::config() const
{
    return config_;
}

// returns the most recent neural and exploration state
const AdaptiveControllerStatus& AdaptiveController::status() const
{
    return status_;
}

// returns whether task-blind output exploration is enabled
bool AdaptiveController::exploration_enabled() const
{
    return exploration_enabled_;
}

// returns whether connection-strength adaptation is enabled
bool AdaptiveController::learning_enabled() const
{
    return learning_enabled_;
}

// returns the immutable compiled brain model
const BrainModel& AdaptiveController::model() const
{
    return simulator_->model();
}

// returns the explicit execution backend
BrainBackend AdaptiveController::backend() const
{
    return simulator_->backend();
}

}
