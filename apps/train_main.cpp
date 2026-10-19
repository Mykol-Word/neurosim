#include "neurosim/brain_agent.h"
#include "neurosim/brain_checkpoint.h"

#if defined(NEUROSIM_HAS_CUDA)
#include "neurosim/cuda_brain.h"
#endif

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace std;
using namespace neurosim;

namespace
{

inline constexpr uint64_t DEFAULT_VALIDATION_SEED = 1'000'001;
inline constexpr string_view SOFTWARE_VERSION = NEUROSIM_VERSION;

// stores command-line settings for one learning run
struct Options
{
    uint64_t first_environment_seed = 1;
    uint64_t brain_seed = 1;
    uint64_t episodes = 1;
    uint64_t validation_interval = 25;
    uint64_t validation_episodes = 8;
    uint64_t validation_seed = DEFAULT_VALIDATION_SEED;
    BrainBackend backend = BrainBackend::Cpu;
    int32_t cuda_device = 0;
    filesystem::path load_path;
    filesystem::path save_path;
    bool brain_seed_explicit = false;
    bool cuda_device_explicit = false;
};

// accumulates behavior and neural activity for one episode
struct EpisodeMetrics
{
    uint64_t wall_hits = 0;
    uint64_t exploration_steps = 0;
    uint64_t moving_steps = 0;
    uint64_t none_steps = 0;
    uint64_t direction_changes = 0;
    uint64_t visible_food_moving_steps = 0;
    uint64_t toward_food_steps = 0;
    uint64_t brain_spikes = 0;
    uint64_t output_spikes = 0;
    float distance_traveled = 0.0f;
};

// summarizes learned connection strength distribution
struct WeightMetrics
{
    float mean = 0.0f;
    float maximum = 0.0f;
    float zero_fraction = 0.0f;
    float mean_outgoing_entropy = 0.0f;
};

// summarizes one frozen checkpoint-selection validation round
struct ValidationScore
{
    uint64_t pellets = 0;
    uint64_t clears = 0;
    double elapsed_seconds = 0.0;
};

// fingerprints one exact learned-strength state for run reproducibility
uint64_t strength_fingerprint(span<const float> strengths)
{
    uint64_t hash = 1469598103934665603ULL;

    for (const float strength : strengths)
    {
        const uint32_t bits = bit_cast<uint32_t>(strength);

        for (uint32_t shift = 0U; shift < 32U; shift += 8U)
        {
            hash ^= static_cast<uint8_t>(bits >> shift);
            hash *= 1099511628211ULL;
        }
    }

    return hash;
}

// emits the complete configuration needed to reproduce one learning run
void print_run_metadata(
    const Options& options,
    const EnvironmentConfig& environment,
    const BrainAgent& agent,
    uint64_t starting_strength_fingerprint,
    bool resumed_from_checkpoint)
{
    const DungeonConfig& dungeon = environment.dungeon;
    const BrainAgentConfig& dungeon_agent = agent.config();
    const AdaptiveControllerConfig& controller = agent.controller().config();
    const BrainConfig& brain = agent.model().config;
    const NeuronDynamicsConfig& dynamics = brain.dynamics;
    const PlasticityConfig& plasticity = brain.plasticity;
    cout
        << "{\"phase\":\"run_metadata\""
        << ",\"schema_version\":1"
        << ",\"software_version\":\"" << SOFTWARE_VERSION << "\""
        << ",\"backend\":\"" << brain_backend_name(agent.backend()) << "\""
        << ",\"precision\":\"float32\"";

#if defined(NEUROSIM_HAS_CUDA)
    if (agent.backend() == BrainBackend::Cuda)
    {
        const CudaDeviceInfo device = cuda_device_info(options.cuda_device);
        cout
            << ",\"cuda\":{"
            << "\"device\":" << options.cuda_device
            << ",\"name\":\"" << device.name << "\""
            << ",\"compute_capability\":\"" << device.compute_major
            << '.' << device.compute_minor << "\""
            << ",\"driver_version\":" << device.driver_version
            << ",\"runtime_version\":" << device.runtime_version
            << ",\"toolkit_version\":" << device.toolkit_version
            << '}';
    }
#endif

    cout
        << ",\"first_environment_seed\":" << options.first_environment_seed
        << ",\"brain_seed\":" << agent.seed()
        << ",\"exploration_random_stream\":\"episode_seed_only\""
        << ",\"episodes\":" << options.episodes
        << ",\"validation_interval\":" << options.validation_interval
        << ",\"validation_episodes\":" << options.validation_episodes
        << ",\"validation_seed\":" << options.validation_seed
        << ",\"resumed_from_checkpoint\":"
        << (resumed_from_checkpoint ? "true" : "false")
        << ",\"starting_strength_fingerprint\":" << starting_strength_fingerprint
        << ",\"environment\":{"
        << "\"fixed_time_step\":" << environment.fixed_time_step
        << ",\"vision_radius\":" << environment.vision_radius
        << ",\"player_size\":" << environment.player_size
        << ",\"acceleration\":" << environment.acceleration
        << ",\"braking\":" << environment.braking
        << ",\"max_speed\":" << environment.max_speed
        << ",\"starvation_seconds\":" << environment.starvation_seconds
        << ",\"time_limit_seconds\":" << environment.time_limit_seconds
        << "}"
        << ",\"dungeon\":{"
        << "\"min_rooms\":" << dungeon.min_rooms
        << ",\"max_rooms\":" << dungeon.max_rooms
        << ",\"min_hub_width\":" << dungeon.min_hub_width
        << ",\"max_hub_width\":" << dungeon.max_hub_width
        << ",\"min_hub_height\":" << dungeon.min_hub_height
        << ",\"max_hub_height\":" << dungeon.max_hub_height
        << ",\"min_wings\":" << dungeon.min_wings
        << ",\"max_wings\":" << dungeon.max_wings
        << ",\"min_wing_depth\":" << dungeon.min_wing_depth
        << ",\"max_wing_depth\":" << dungeon.max_wing_depth
        << ",\"min_wing_span_ratio\":" << dungeon.min_wing_span_ratio
        << ",\"max_wing_span_ratio\":" << dungeon.max_wing_span_ratio
        << ",\"min_room_gap\":" << dungeon.min_room_gap
        << ",\"max_room_gap\":" << dungeon.max_room_gap
        << ",\"room_spacing\":" << dungeon.room_spacing
        << ",\"min_corridor_width\":" << dungeon.min_corridor_width
        << ",\"max_corridor_width\":" << dungeon.max_corridor_width
        << ",\"extra_connection_ratio\":" << dungeon.extra_connection_ratio
        << ",\"two_bend_chance\":" << dungeon.two_bend_chance
        << ",\"min_pellets_per_room\":" << dungeon.min_pellets_per_room
        << ",\"max_pellets_per_room\":" << dungeon.max_pellets_per_room
        << ",\"pellet_min_spacing\":" << dungeon.pellet_min_spacing
        << ",\"spawn_clearance\":" << dungeon.spawn_clearance
        << ",\"placement_attempts_per_room\":" << dungeon.placement_attempts_per_room
        << ",\"pellet_placement_attempts\":" << dungeon.pellet_placement_attempts
        << "}"
        << ",\"dungeon_agent\":{"
        << "\"wall_ray_count\":" << dungeon_agent.wall_ray_count
        << ",\"output_activity_threshold\":"
        << dungeon_agent.output_activity_threshold
        << ",\"wall_feedback_interval_seconds\":"
        << dungeon_agent.wall_feedback_interval_seconds
        << ",\"food_modulator\":" << dungeon_agent.food_modulator
        << ",\"wall_modulator\":" << dungeon_agent.wall_modulator
        << "}"
        << ",\"controller\":{"
        << "\"brain_shape\":[" << controller.brain_shape.width
        << ',' << controller.brain_shape.height << ',' << controller.brain_shape.depth << ']'
        << ",\"brain_radius\":" << controller.brain_radius
        << ",\"brain_time_step_seconds\":"
        << controller.brain_time_step_seconds
        << ",\"sensory_drive\":" << controller.sensory_drive
        << ",\"exploration_drive\":" << controller.exploration_drive
        << ",\"inhibitory_fraction\":" << controller.inhibitory_fraction
        << ",\"minimum_bias_drive\":" << controller.minimum_bias_drive
        << ",\"maximum_bias_drive\":" << controller.maximum_bias_drive
        << ",\"exploration_interval_seconds\":"
        << controller.exploration_interval_seconds
        << ",\"exploration_duration_seconds\":"
        << controller.exploration_duration_seconds
        << ",\"exploration_pattern_count\":"
        << controller.exploration_patterns.size()
        << "}"
        << ",\"brain\":{"
        << "\"model_fingerprint\":" << brain_model_fingerprint(agent.model())
        << ",\"connection_radius_steps\":" << brain.connection_radius_steps
        << ",\"fixed_time_step_seconds\":" << dynamics.fixed_time_step_seconds
        << ",\"membrane_time_constant_seconds\":"
        << dynamics.membrane_time_constant_seconds
        << ",\"resting_potential\":" << dynamics.resting_potential
        << ",\"reset_potential\":" << dynamics.reset_potential
        << ",\"firing_threshold\":" << dynamics.firing_threshold
        << ",\"refractory_period_seconds\":" << dynamics.refractory_period_seconds
        << ",\"activity_trace_time_constant_seconds\":"
        << plasticity.activity_trace_time_constant_seconds
        << ",\"eligibility_time_constant_seconds\":"
        << plasticity.eligibility_time_constant_seconds
        << ",\"maximum_eligibility\":" << plasticity.maximum_eligibility
        << ",\"learning_rate\":" << plasticity.learning_rate
        << ",\"weakening_rate_per_second\":" << plasticity.weakening_rate_per_second
        << ",\"outgoing_strength_budget\":" << plasticity.outgoing_strength_budget
        << ",\"initial_budget_fraction\":" << plasticity.initial_budget_fraction
        << ",\"initial_connection_density\":" << plasticity.initial_connection_density
        << "}}\n";
}

// parses an unsigned command-line value with full validation
uint64_t parse_unsigned(string_view text, const char* name)
{
    uint64_t value = 0;
    const auto result = from_chars(text.data(), text.data() + text.size(), value);

    if (result.ec != errc{} || result.ptr != text.data() + text.size())
    {
        throw invalid_argument(name);
    }

    return value;
}

// parses one explicit brain execution backend
BrainBackend parse_backend(string_view text)
{
    using enum BrainBackend;

    if (text == "cpu")
    {
        return Cpu;
    }

    if (text == "cuda")
    {
        return Cuda;
    }

    throw invalid_argument("invalid --backend value; expected cpu or cuda");
}

// returns the factory compiled for one explicitly selected backend
BrainSimulatorFactory backend_factory(BrainBackend backend)
{
    using enum BrainBackend;

    if (backend == Cpu)
    {
        return nullptr;
    }

#if defined(NEUROSIM_HAS_CUDA)
    return make_cuda_brain_simulator;
#else
    throw runtime_error("cuda backend is unavailable; rebuild with NEUROSIM_BUILD_CUDA=ON");
#endif
}

// parses all supported learning-run options
Options parse_options(int argc, char** argv)
{
    Options options;

    for (int index = 1; index < argc; ++index)
    {
        const string_view argument = argv[index];

        if (argument == "--seed" && index + 1 < argc)
        {
            options.first_environment_seed = parse_unsigned(
                argv[++index], "invalid --seed value");
        }
        else if (argument == "--brain-seed" && index + 1 < argc)
        {
            options.brain_seed = parse_unsigned(argv[++index], "invalid --brain-seed value");
            options.brain_seed_explicit = true;
        }
        else if (argument == "--episodes" && index + 1 < argc)
        {
            options.episodes = parse_unsigned(argv[++index], "invalid --episodes value");
        }
        else if (argument == "--validation-interval" && index + 1 < argc)
        {
            options.validation_interval = parse_unsigned(
                argv[++index], "invalid --validation-interval value");
        }
        else if (argument == "--validation-episodes" && index + 1 < argc)
        {
            options.validation_episodes = parse_unsigned(
                argv[++index], "invalid --validation-episodes value");
        }
        else if (argument == "--validation-seed" && index + 1 < argc)
        {
            options.validation_seed = parse_unsigned(
                argv[++index], "invalid --validation-seed value");
        }
        else if (argument == "--backend" && index + 1 < argc)
        {
            options.backend = parse_backend(argv[++index]);
        }
        else if (argument == "--cuda-device" && index + 1 < argc)
        {
            const uint64_t device = parse_unsigned(argv[++index], "invalid --cuda-device value");

            if (device > static_cast<uint64_t>(numeric_limits<int32_t>::max()))
            {
                throw invalid_argument("cuda device index is too large");
            }

            options.cuda_device = static_cast<int32_t>(device);
            options.cuda_device_explicit = true;
        }
        else if (argument == "--load-brain" && index + 1 < argc)
        {
            options.load_path = argv[++index];
        }
        else if (argument == "--save-brain" && index + 1 < argc)
        {
            options.save_path = argv[++index];
        }
        else
        {
            throw invalid_argument(
                "usage: neurosim_train [--seed N] [--brain-seed N] [--episodes N] "
                "[--validation-interval N] [--validation-episodes N] [--validation-seed N] "
                "[--backend cpu|cuda] [--cuda-device N] "
                "[--load-brain PATH] [--save-brain PATH]");
        }
    }

    if (options.episodes == 0U)
    {
        throw invalid_argument("episode count must be positive");
    }

    if (options.episodes > static_cast<uint64_t>(numeric_limits<int64_t>::max()))
    {
        throw invalid_argument("episode count is too large");
    }

    if (options.validation_interval > 0U && options.validation_episodes == 0U)
    {
        throw invalid_argument("validation episode count must be positive when validation is enabled");
    }

    if (options.cuda_device_explicit && options.backend != BrainBackend::Cuda)
    {
        throw invalid_argument("--cuda-device requires --backend cuda");
    }

    return options;
}

// maps one simultaneous movement action to a unit direction
Vec2 action_direction(Action action)
{
    using enum Action;
    const auto has = [action](Action direction)
    {
        return (static_cast<uint8_t>(action) & static_cast<uint8_t>(direction)) != 0U;
    };
    Vec2 direction{
        static_cast<float>(has(Right)) - static_cast<float>(has(Left)),
        static_cast<float>(has(Down)) - static_cast<float>(has(Up))
    };
    const float magnitude = length(direction);
    return magnitude == 0.0f ? Vec2{} : direction / magnitude;
}

// measures whether one action points toward the nearest visible pellet
void measure_food_alignment(
    const BrainAgent& agent,
    Action action,
    EpisodeMetrics& metrics)
{
    using enum Action;

    if (action == None || agent.observation().pellets.empty())
    {
        return;
    }

    const PelletObservation& pellet = agent.observation().pellets.front();

    if (pellet.present == 0.0f)
    {
        return;
    }

    ++metrics.visible_food_moving_steps;
    const Vec2 direction = action_direction(action);
    const Vec2 offset{pellet.right - pellet.left, pellet.down - pellet.up};

    if (direction.x * offset.x + direction.y * offset.y > 0.0f)
    {
        ++metrics.toward_food_steps;
    }
}

// advances one complete episode and records behavior and firing
EpisodeMetrics run_episode(Environment& environment, BrainAgent& agent)
{
    using enum Action;

    EpisodeMetrics metrics;
    Action previous_action = None;

    while (environment.state().termination == TerminationReason::None)
    {
        const Action action = agent.decide(environment);
        const BrainAgentStatus& status = agent.status();
        metrics.brain_spikes += status.brain_spikes;
        metrics.output_spikes += status.output_spikes;
        metrics.exploration_steps += static_cast<uint64_t>(status.exploring);

        if (action == None)
        {
            ++metrics.none_steps;
        }
        else
        {
            ++metrics.moving_steps;
        }

        if (previous_action != None && action != None && action != previous_action)
        {
            ++metrics.direction_changes;
        }

        measure_food_alignment(agent, action, metrics);
        const Vec2 previous_position = environment.state().position;
        const StepResult transition = environment.step(action);
        metrics.distance_traveled += length(environment.state().position - previous_position);
        metrics.wall_hits += static_cast<uint64_t>(transition.hit_wall);
        agent.observe_transition(transition);
        previous_action = action;
    }

    return metrics;
}

// summarizes current learned strengths and outgoing competition
WeightMetrics measure_weights(const BrainAgent& agent)
{
    const BrainModel& model = agent.model();
    const vector<float> strengths = agent.copy_connection_strengths();
    WeightMetrics metrics;
    uint64_t zero_count = 0;
    uint64_t entropy_sources = 0;

    for (const float strength : strengths)
    {
        metrics.mean += strength;
        metrics.maximum = max(metrics.maximum, strength);
        zero_count += static_cast<uint64_t>(strength <= 1.0e-6f);
    }

    if (!strengths.empty())
    {
        metrics.mean /= static_cast<float>(strengths.size());
        metrics.zero_fraction = static_cast<float>(zero_count)
            / static_cast<float>(strengths.size());
    }

    for (uint32_t source = 0; source < model.positions.size(); ++source)
    {
        const uint32_t begin = model.outgoing_offsets[source];
        const uint32_t end = model.outgoing_offsets[source + 1U];
        float sum = 0.0f;

        for (uint32_t edge = begin; edge < end; ++edge)
        {
            sum += strengths[edge];
        }

        if (sum <= 0.0f || end - begin < 2U)
        {
            continue;
        }

        float entropy = 0.0f;

        for (uint32_t edge = begin; edge < end; ++edge)
        {
            const float probability = strengths[edge] / sum;

            if (probability > 0.0f)
            {
                entropy -= probability * log(probability);
            }
        }

        metrics.mean_outgoing_entropy += entropy / log(static_cast<float>(end - begin));
        ++entropy_sources;
    }

    if (entropy_sources > 0U)
    {
        metrics.mean_outgoing_entropy /= static_cast<float>(entropy_sources);
    }

    return metrics;
}

// emits one machine-readable episode and model summary
void print_episode(
    string_view phase,
    int64_t episode,
    uint64_t environment_seed,
    const Environment& environment,
    const BrainAgent& agent,
    const EpisodeMetrics& episode_metrics)
{
    const WeightMetrics weights = measure_weights(agent);
    const double elapsed = environment.elapsed_seconds();
    const double neuron_seconds = elapsed
        * static_cast<double>(agent.model().positions.size());
    const double output_seconds = elapsed
        * static_cast<double>(agent.model().output_neurons.size());
    const double alignment = episode_metrics.visible_food_moving_steps == 0U
        ? 0.0
        : static_cast<double>(episode_metrics.toward_food_steps)
            / static_cast<double>(episode_metrics.visible_food_moving_steps);

    cout
        << "{\"phase\":\"" << phase << "\""
        << ",\"episode\":" << episode
        << ",\"environment_seed\":" << environment_seed
        << ",\"brain_seed\":" << agent.seed()
        << ",\"result\":\"" << termination_name(environment.state().termination) << "\""
        << ",\"score\":" << environment.state().score
        << ",\"pellets_remaining\":" << environment.state().pellets_remaining
        << ",\"wall_hits\":" << episode_metrics.wall_hits
        << ",\"exploration_steps\":" << episode_metrics.exploration_steps
        << ",\"moving_steps\":" << episode_metrics.moving_steps
        << ",\"none_steps\":" << episode_metrics.none_steps
        << ",\"direction_changes\":" << episode_metrics.direction_changes
        << ",\"toward_food_fraction\":" << alignment
        << ",\"distance_traveled\":" << episode_metrics.distance_traveled
        << ",\"mean_firing_hz\":"
        << (neuron_seconds == 0.0 ? 0.0 : episode_metrics.brain_spikes / neuron_seconds)
        << ",\"output_firing_hz\":"
        << (output_seconds == 0.0 ? 0.0 : episode_metrics.output_spikes / output_seconds)
        << ",\"mean_weight\":" << weights.mean
        << ",\"max_weight\":" << weights.maximum
        << ",\"zero_weight_fraction\":" << weights.zero_fraction
        << ",\"mean_outgoing_entropy\":" << weights.mean_outgoing_entropy
        << ",\"elapsed_seconds\":" << elapsed
        << "}\n";
}

// runs frozen validation episodes used only for checkpoint selection
ValidationScore run_validation(
    Environment& environment,
    BrainAgent& agent,
    const Options& options,
    int64_t training_episode)
{
    agent.set_learning_enabled(false);
    agent.set_exploration_enabled(false);
    ValidationScore score;

    for (uint64_t validation = 0; validation < options.validation_episodes; ++validation)
    {
        const uint64_t seed = options.validation_seed + validation;
        environment.reset(seed);
        agent.reset_episode(seed);
        const EpisodeMetrics metrics = run_episode(environment, agent);
        print_episode("validation", training_episode, seed, environment, agent, metrics);
        score.pellets += environment.state().score;
        score.clears += static_cast<uint64_t>(
            environment.state().termination == TerminationReason::Cleared);
        score.elapsed_seconds += environment.elapsed_seconds();
    }

    agent.set_learning_enabled(true);
    agent.set_exploration_enabled(true);
    cout
        << "{\"phase\":\"validation_summary\""
        << ",\"episode\":" << training_episode
        << ",\"pellets\":" << score.pellets
        << ",\"clears\":" << score.clears
        << ",\"elapsed_seconds\":" << score.elapsed_seconds
        << "}\n";
    return score;
}

// returns whether one frozen validation is a better checkpoint candidate
bool is_better_validation(const ValidationScore& candidate, const ValidationScore& current)
{
    if (candidate.pellets != current.pellets)
    {
        return candidate.pellets > current.pellets;
    }

    if (candidate.clears != current.clears)
    {
        return candidate.clears > current.clears;
    }

    return candidate.elapsed_seconds >= current.elapsed_seconds;
}

}

// runs repeated procedural learning episodes with validation and checkpointing
int main(int argc, char** argv)
{
    try
    {
        Options options = parse_options(argc, argv);
        optional<BrainCheckpoint> checkpoint;

        if (!options.load_path.empty())
        {
            checkpoint = load_brain_checkpoint(options.load_path);

            if (options.brain_seed_explicit && options.brain_seed != checkpoint->brain_seed)
            {
                throw invalid_argument("--brain-seed does not match the loaded checkpoint");
            }

            options.brain_seed = checkpoint->brain_seed;
        }

        if (options.save_path.empty())
        {
            options.save_path = filesystem::path("runs")
                / ("brain-" + to_string(options.brain_seed) + ".nsbrain");
        }

        Environment environment({}, options.first_environment_seed);
        const BrainExecutionConfig execution{options.backend, options.cuda_device};
        BrainAgent agent(
            environment.config(),
            {},
            options.brain_seed,
            execution,
            backend_factory(options.backend),
            checkpoint ? &checkpoint->genome : nullptr);

        if (checkpoint)
        {
            validate_brain_checkpoint(*checkpoint, agent.seed(), agent.model());
            agent.load_connection_strengths(checkpoint->connection_strengths);
        }

        const vector<float> starting_strengths = agent.copy_connection_strengths();

        print_run_metadata(
            options,
            environment.config(),
            agent,
            strength_fingerprint(starting_strengths),
            checkpoint.has_value());

        vector<float> best_strengths = starting_strengths;
        ValidationScore best_validation;
        int64_t best_episode = -1;
        bool has_validation = false;

        if (options.validation_interval > 0U)
        {
            best_validation = run_validation(environment, agent, options, -1);
            has_validation = true;
        }

        for (uint64_t episode = 0; episode < options.episodes; ++episode)
        {
            const uint64_t environment_seed = options.first_environment_seed + episode;
            environment.reset(environment_seed);
            agent.reset_episode(environment_seed);
            const EpisodeMetrics metrics = run_episode(environment, agent);
            print_episode(
                "training",
                static_cast<int64_t>(episode),
                environment_seed,
                environment,
                agent,
                metrics);

            if (options.validation_interval > 0U
                && (episode + 1U) % options.validation_interval == 0U)
            {
                const ValidationScore validation = run_validation(
                    environment,
                    agent,
                    options,
                    static_cast<int64_t>(episode));

                if (!has_validation || is_better_validation(validation, best_validation))
                {
                    best_validation = validation;
                    best_strengths = agent.copy_connection_strengths();
                    best_episode = static_cast<int64_t>(episode);
                    has_validation = true;
                }
            }
        }

        if (options.validation_interval > 0U
            && options.episodes % options.validation_interval != 0U)
        {
            const int64_t final_episode = static_cast<int64_t>(options.episodes - 1U);
            const ValidationScore validation = run_validation(
                environment,
                agent,
                options,
                final_episode);

            if (!has_validation || is_better_validation(validation, best_validation))
            {
                best_validation = validation;
                best_strengths = agent.copy_connection_strengths();
                best_episode = final_episode;
                has_validation = true;
            }
        }

        if (has_validation)
        {
            agent.load_connection_strengths(best_strengths);
        }

        const vector<float> saved_strengths = agent.copy_connection_strengths();
        save_brain_checkpoint(options.save_path, agent.seed(), agent.model(), saved_strengths);
        cout << "{\"phase\":\"checkpoint\",\"path\":\""
            << options.save_path.generic_string() << "\""
            << ",\"selected_episode\":" << best_episode
            << ",\"validation_pellets\":" << best_validation.pellets
            << ",\"validation_clears\":" << best_validation.clears
            << "}\n";
        return 0;
    }
    catch (const exception& error)
    {
        cerr << error.what() << '\n';
        return 1;
    }
}
