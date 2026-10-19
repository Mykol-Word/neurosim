#include "neurosim/brain_agent.h"
#include "neurosim/brain_checkpoint.h"
#include "neurosim/brain_genome.h"
#include "neurosim/evolution.h"

#if defined(NEUROSIM_HAS_CUDA)
#include "neurosim/cuda_brain.h"
#endif

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace std;
using namespace neurosim;

namespace
{

using enum BrainBackend;

inline constexpr uint32_t MUTANTS_PER_GENERATION = 4U;
inline constexpr string_view SOFTWARE_VERSION = NEUROSIM_VERSION;

// stores validated evolution command-line options
struct Options
{
    string name;
    uint32_t initial_population = 10U;
    uint32_t episodes_per_candidate = 50U;
    uint32_t generations = 20U;
    uint32_t evaluation_episodes = 8U;
    uint64_t seed = 1U;
    BrainBackend backend = Cpu;
    int32_t cuda_device = 0;
    bool cuda_device_explicit = false;
};

// describes one genome entering a generation
struct Candidate
{
    uint64_t brain_seed = 0U;
    optional<BrainGenome> genome;
    uint32_t parent_generation = 0U;
    uint32_t mutation = 0U;
};

// summarizes one completed environment episode
struct EpisodeSummary
{
    uint32_t score = 0U;
    uint64_t wall_hits = 0U;
    double elapsed_seconds = 0.0;
    bool cleared = false;
};

// stores one trained candidate and its comparable fitness
struct CandidateResult
{
    uint32_t candidate = 0U;
    uint64_t brain_seed = 0U;
    uint64_t training_pellets = 0U;
    uint64_t training_clears = 0U;
    BrainGenome learned_genome;
    EvolutionFitness fitness;
};

// stores one generation winner for checkpoint archives
struct Champion
{
    uint32_t generation = 0U;
    uint64_t brain_seed = 0U;
    BrainGenome genome;
    EvolutionFitness selection_fitness;
    EvolutionFitness archive_fitness;
};

using EvaluationSeedFunction = uint64_t (*)(const Options&, uint32_t, uint32_t);

// mixes one experiment seed into a reproducible independent stream
uint64_t mix_seed(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

// parses one nonnegative integer with full input validation
uint64_t parse_unsigned(string_view text, const char* message)
{
    uint64_t value = 0U;
    const auto result = from_chars(text.data(), text.data() + text.size(), value);

    if (result.ec != errc{} || result.ptr != text.data() + text.size())
    {
        throw invalid_argument(message);
    }

    return value;
}

// narrows one positive command-line count to its stored representation
uint32_t parse_count(string_view text, const char* message)
{
    const uint64_t value = parse_unsigned(text, message);

    if (value == 0U || value >= numeric_limits<uint32_t>::max())
    {
        throw invalid_argument(message);
    }

    return static_cast<uint32_t>(value);
}

// parses one explicit brain execution backend
BrainBackend parse_backend(string_view text)
{
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

// validates a run name as one portable directory component
void validate_run_name(string_view name)
{
    if (name.empty() || name == "." || name == "..")
    {
        throw invalid_argument("--name must be a nonempty run name");
    }

    for (const unsigned char character : name)
    {
        if (!isalnum(character) && character != '-' && character != '_')
        {
            throw invalid_argument("--name may contain only letters, numbers, '-' and '_'");
        }
    }
}

// parses and validates every evolution command-line option
Options parse_options(int argc, char** argv)
{
    Options options;

    for (int index = 1; index < argc; ++index)
    {
        const string_view argument = argv[index];

        if (argument == "--name" && index + 1 < argc)
        {
            options.name = argv[++index];
        }
        else if (argument == "--population" && index + 1 < argc)
        {
            options.initial_population = parse_count(
                argv[++index], "invalid --population value");
        }
        else if (argument == "--episodes" && index + 1 < argc)
        {
            options.episodes_per_candidate = parse_count(
                argv[++index], "invalid --episodes value");
        }
        else if (argument == "--generations" && index + 1 < argc)
        {
            options.generations = parse_count(
                argv[++index], "invalid --generations value");
        }
        else if (argument == "--evaluation-episodes" && index + 1 < argc)
        {
            options.evaluation_episodes = parse_count(
                argv[++index], "invalid --evaluation-episodes value");
        }
        else if (argument == "--seed" && index + 1 < argc)
        {
            options.seed = parse_unsigned(argv[++index], "invalid --seed value");
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
        else
        {
            throw invalid_argument(
                "usage: neurosim_evolve --name NAME [--population N] [--episodes N] "
                "[--generations N] [--evaluation-episodes N] [--seed N] "
                "[--backend cpu|cuda] [--cuda-device N]");
        }
    }

    validate_run_name(options.name);

    if (options.cuda_device_explicit && options.backend != Cuda)
    {
        throw invalid_argument("--cuda-device requires --backend cuda");
    }

    return options;
}

// creates a new empty run directory without mixing prior experiment files
filesystem::path prepare_run_directory(const string& name)
{
    const filesystem::path directory = filesystem::path("runs") / name;

    if (filesystem::exists(directory))
    {
        if (!filesystem::is_directory(directory)
            || filesystem::directory_iterator(directory) != filesystem::directory_iterator{})
        {
            throw runtime_error("evolution run directory already exists and is not empty");
        }
    }
    else
    {
        filesystem::create_directories(directory);
    }

    return directory;
}

// flushes one experiment stream and rejects output failures
void flush_output(ostream& stream)
{
    stream.flush();

    if (!stream)
    {
        throw runtime_error("failed to write evolution output");
    }
}

// advances one episode while collecting task fitness data
EpisodeSummary run_episode(Environment& environment, BrainAgent& agent)
{
    EpisodeSummary summary;

    while (environment.state().termination == TerminationReason::None)
    {
        const Action action = agent.decide(environment);
        const StepResult transition = environment.step(action);
        summary.wall_hits += static_cast<uint64_t>(transition.hit_wall);
        agent.observe_transition(transition);
    }

    summary.score = environment.state().score;
    summary.elapsed_seconds = environment.elapsed_seconds();
    summary.cleared = environment.state().termination == TerminationReason::Cleared;
    return summary;
}

// returns one shared generation-specific training environment seed
uint64_t training_seed(const Options& options, uint32_t generation, uint32_t episode)
{
    return mix_seed(options.seed
        ^ 0x56e8feb86659fd93ULL
        ^ (static_cast<uint64_t>(generation) << 32U)
        ^ episode);
}

// returns one selection seed shared within a generation and rotated afterward
uint64_t selection_seed(const Options& options, uint32_t generation, uint32_t episode)
{
    return mix_seed(options.seed
        ^ 0xa0761d6478bd642fULL
        ^ (static_cast<uint64_t>(generation) << 32U)
        ^ episode);
}

// returns one fixed archive benchmark seed shared across generations
uint64_t archive_seed(const Options& options, uint32_t generation, uint32_t episode)
{
    static_cast<void>(generation);
    return mix_seed(options.seed ^ 0xe0d5e91e6b92c963ULL ^ episode);
}

// returns one reporting seed untouched until evolution completes
uint64_t final_test_seed(const Options& options, uint32_t generation, uint32_t episode)
{
    static_cast<void>(generation);
    return mix_seed(options.seed ^ 0xc6bc279692b5c323ULL ^ episode);
}

// adds one completed episode to aggregate evolution fitness
void record_fitness(EvolutionFitness& fitness, const EpisodeSummary& summary)
{
    fitness.total_score += summary.score;
    fitness.successful_episodes += static_cast<uint64_t>(summary.cleared);
    fitness.best_episode_score = max<uint64_t>(fitness.best_episode_score, summary.score);
    fitness.total_episode_seconds += summary.elapsed_seconds;
    fitness.penalty_events += summary.wall_hits;
}

// evaluates one frozen policy on a caller-selected matched environment suite
EvolutionFitness evaluate_frozen_agent(
    const Options& options,
    uint32_t generation,
    Environment& environment,
    BrainAgent& agent,
    EvaluationSeedFunction environment_seed)
{
    agent.set_learning_enabled(false);
    agent.set_exploration_enabled(false);
    EvolutionFitness fitness;

    for (uint32_t episode = 0; episode < options.evaluation_episodes; ++episode)
    {
        const uint64_t seed = environment_seed(options, generation, episode);
        environment.reset(seed);
        agent.reset_episode(seed);
        record_fitness(fitness, run_episode(environment, agent));
    }

    return fitness;
}

// reconstructs and evaluates one learned genome without changing it
EvolutionFitness evaluate_frozen_genome(
    const Options& options,
    uint32_t generation,
    uint64_t brain_seed,
    const BrainGenome& genome,
    EvaluationSeedFunction environment_seed)
{
    const BrainExecutionConfig execution{options.backend, options.cuda_device};
    Environment environment({}, environment_seed(options, generation, 0U));
    BrainAgent agent(
        environment.config(),
        {},
        brain_seed,
        execution,
        backend_factory(options.backend),
        &genome);
    EvolutionFitness fitness = evaluate_frozen_agent(
        options,
        generation,
        environment,
        agent,
        environment_seed);
    fitness.genome_fingerprint = brain_genome_fingerprint(genome);
    return fitness;
}

// trains and scores one candidate without retaining backend resources
CandidateResult evaluate_candidate(
    const Options& options,
    uint32_t generation,
    uint32_t candidate_index,
    const Candidate& candidate)
{
    const BrainExecutionConfig execution{options.backend, options.cuda_device};
    Environment environment({}, training_seed(options, generation, 0U));
    const BrainGenome* genome = candidate.genome ? &*candidate.genome : nullptr;
    BrainAgent agent(
        environment.config(),
        {},
        candidate.brain_seed,
        execution,
        backend_factory(options.backend),
        genome);
    CandidateResult result;
    result.candidate = candidate_index;
    result.brain_seed = candidate.brain_seed;

    for (uint32_t episode = 0; episode < options.episodes_per_candidate; ++episode)
    {
        const uint64_t seed = training_seed(options, generation, episode);
        environment.reset(seed);
        agent.reset_episode(seed);
        const EpisodeSummary summary = run_episode(environment, agent);
        result.training_pellets += summary.score;
        result.training_clears += static_cast<uint64_t>(summary.cleared);
    }

    result.fitness = evaluate_frozen_agent(
        options,
        generation,
        environment,
        agent,
        selection_seed);

    const vector<float> learned_strengths = agent.copy_connection_strengths();
    result.learned_genome = make_brain_genome(agent.model(), learned_strengths);
    result.fitness.genome_fingerprint = brain_genome_fingerprint(result.learned_genome);
    return result;
}

// writes complete evolution configuration before any candidate runs
void write_metadata(
    ostream& stream,
    const Options& options,
    const EnvironmentConfig& environment,
    const BrainAgent& representative_agent,
    const EvolutionMutationConfig& mutation)
{
    const DungeonConfig& dungeon = environment.dungeon;
    const BrainAgentConfig& dungeon_agent = representative_agent.config();
    const AdaptiveControllerConfig& controller = representative_agent.controller().config();
    const BrainConfig& brain = representative_agent.model().config;
    const NeuronDynamicsConfig& dynamics = brain.dynamics;
    const PlasticityConfig& plasticity = brain.plasticity;
    stream
        << setprecision(numeric_limits<double>::max_digits10)
        << "{\"phase\":\"run_metadata\""
        << ",\"schema_version\":1"
        << ",\"software_version\":\"" << SOFTWARE_VERSION << "\""
        << ",\"backend\":\"" << brain_backend_name(options.backend) << "\""
        << ",\"precision\":\"float32\""
        << ",\"name\":\"" << options.name << "\""
        << ",\"seed\":" << options.seed
        << ",\"initial_population\":" << options.initial_population
        << ",\"initial_population_mutated\":true"
        << ",\"later_population\":" << MUTANTS_PER_GENERATION + 1U
        << ",\"episodes_per_candidate\":" << options.episodes_per_candidate
        << ",\"evaluation_episodes\":" << options.evaluation_episodes
        << ",\"generations\":" << options.generations
        << ",\"selection\":\"generation_matched_evaluation\""
        << ",\"archive_ranking\":\"fixed_benchmark_evaluation\""
        << ",\"final_test\":\"reporting_only_unseen_evaluation\""
        << ",\"inheritance\":\"learned_strengths\""
        << ",\"exploration_random_stream\":\"episode_seed_only\""
        << ",\"evaluation_exploration_enabled\":false"
        << ",\"mutation\":{"
        << "\"minimum_neurons\":" << mutation.minimum_neurons
        << ",\"maximum_neurons\":" << mutation.maximum_neurons
        << ",\"neuron_count_sigma_fraction\":"
        << mutation.neuron_count_sigma_fraction
        << ",\"input_position_mutation_fraction\":"
        << mutation.input_position_mutation_fraction
        << ",\"output_position_mutation_fraction\":"
        << mutation.output_position_mutation_fraction
        << ",\"interface_position_sigma_steps\":"
        << mutation.interface_position_sigma_steps
        << ",\"connection_mutation_fraction\":"
        << mutation.connection_mutation_fraction
        << ",\"connection_strength_sigma_fraction\":"
        << mutation.connection_strength_sigma_fraction
        << '}'
        << ",\"environment\":{"
        << "\"fixed_time_step\":" << environment.fixed_time_step
        << ",\"vision_radius\":" << environment.vision_radius
        << ",\"player_size\":" << environment.player_size
        << ",\"acceleration\":" << environment.acceleration
        << ",\"braking\":" << environment.braking
        << ",\"max_speed\":" << environment.max_speed
        << ",\"starvation_seconds\":" << environment.starvation_seconds
        << ",\"time_limit_seconds\":" << environment.time_limit_seconds
        << '}'
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
        << ",\"placement_attempts_per_room\":"
        << dungeon.placement_attempts_per_room
        << ",\"pellet_placement_attempts\":" << dungeon.pellet_placement_attempts
        << '}'
        << ",\"dungeon_agent\":{"
        << "\"wall_ray_count\":" << dungeon_agent.wall_ray_count
        << ",\"output_activity_threshold\":"
        << dungeon_agent.output_activity_threshold
        << ",\"wall_feedback_interval_seconds\":"
        << dungeon_agent.wall_feedback_interval_seconds
        << ",\"food_modulator\":" << dungeon_agent.food_modulator
        << ",\"wall_modulator\":" << dungeon_agent.wall_modulator
        << '}'
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
        << '}'
        << ",\"brain\":{"
        << "\"connection_radius_steps\":" << brain.connection_radius_steps
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
        << ",\"weakening_rate_per_second\":"
        << plasticity.weakening_rate_per_second
        << ",\"outgoing_strength_budget\":"
        << plasticity.outgoing_strength_budget
        << ",\"initial_budget_fraction\":" << plasticity.initial_budget_fraction
        << ",\"initial_connection_density\":"
        << plasticity.initial_connection_density
        << '}';

#if defined(NEUROSIM_HAS_CUDA)
    if (options.backend == Cuda)
    {
        const CudaDeviceInfo device = cuda_device_info(options.cuda_device);
        stream
            << ",\"cuda\":{\"device\":" << options.cuda_device
            << ",\"name\":\"" << device.name << "\""
            << ",\"compute_capability\":\"" << device.compute_major
            << '.' << device.compute_minor << "\""
            << ",\"driver_version\":" << device.driver_version
            << ",\"runtime_version\":" << device.runtime_version
            << ",\"toolkit_version\":" << device.toolkit_version
            << '}';
    }
#endif

    stream << "}\n";
}

// writes one candidate summary as machine-readable json lines
void write_candidate(
    ostream& stream,
    uint32_t generation,
    const Candidate& candidate,
    const CandidateResult& result)
{
    stream
        << "{\"phase\":\"candidate\""
        << ",\"generation\":" << generation
        << ",\"candidate\":" << result.candidate
        << ",\"parent_generation\":" << candidate.parent_generation
        << ",\"mutation\":" << candidate.mutation
        << ",\"brain_seed\":" << result.brain_seed
        << ",\"neurons\":" << result.learned_genome.neurons.size()
        << ",\"learned_connections\":" << result.learned_genome.connections.size()
        << ",\"training_pellets\":" << result.training_pellets
        << ",\"training_clears\":" << result.training_clears
        << ",\"selection_pellets\":" << result.fitness.total_score
        << ",\"selection_clears\":" << result.fitness.successful_episodes
        << ",\"selection_best_episode_score\":"
        << result.fitness.best_episode_score
        << ",\"selection_total_survival_seconds\":"
        << result.fitness.total_episode_seconds
        << ",\"selection_wall_hits\":" << result.fitness.penalty_events
        << ",\"genome_fingerprint\":" << result.fitness.genome_fingerprint
        << "}\n";
    flush_output(stream);
}

// writes one generation winner summary as a json line
void write_generation_summary(ostream& stream, const Champion& champion)
{
    stream
        << "{\"phase\":\"generation_summary\""
        << ",\"generation\":" << champion.generation
        << ",\"selection_pellets\":"
        << champion.selection_fitness.total_score
        << ",\"selection_clears\":"
        << champion.selection_fitness.successful_episodes
        << ",\"selection_best_episode_score\":"
        << champion.selection_fitness.best_episode_score
        << ",\"selection_total_survival_seconds\":"
        << champion.selection_fitness.total_episode_seconds
        << ",\"selection_wall_hits\":"
        << champion.selection_fitness.penalty_events
        << ",\"archive_pellets\":"
        << champion.archive_fitness.total_score
        << ",\"archive_clears\":" << champion.archive_fitness.successful_episodes
        << ",\"archive_best_episode_score\":"
        << champion.archive_fitness.best_episode_score
        << ",\"archive_total_survival_seconds\":"
        << champion.archive_fitness.total_episode_seconds
        << ",\"archive_wall_hits\":" << champion.archive_fitness.penalty_events
        << ",\"neurons\":" << champion.genome.neurons.size()
        << ",\"genome_fingerprint\":"
        << champion.archive_fitness.genome_fingerprint
        << "}\n";
    flush_output(stream);
}

// returns the requested generation checkpoint path
filesystem::path generation_path(
    const filesystem::path& directory,
    const string& name,
    uint32_t generation)
{
    return directory / (name + "-g" + to_string(generation) + ".nsbrain");
}

// returns the requested all-time rank checkpoint path
filesystem::path rank_path(
    const filesystem::path& directory,
    const string& name,
    uint32_t rank)
{
    return directory / (name + "-n" + to_string(rank) + ".nsbrain");
}

// writes the final best checkpoint path as a json line
void write_completion(
    ostream& stream,
    const filesystem::path& directory,
    const string& name,
    const EvolutionFitness& final_test)
{
    stream
        << "{\"phase\":\"complete\""
        << ",\"best_checkpoint\":\""
        << rank_path(directory, name, 1U).generic_string()
        << "\""
        << ",\"final_test_pellets\":"
        << final_test.total_score
        << ",\"final_test_clears\":" << final_test.successful_episodes
        << ",\"final_test_best_episode_score\":"
        << final_test.best_episode_score
        << ",\"final_test_total_survival_seconds\":"
        << final_test.total_episode_seconds
        << ",\"final_test_wall_hits\":" << final_test.penalty_events
        << "}\n";
    flush_output(stream);
}

// saves one learned genome as a directly playable checkpoint
void save_champion(const filesystem::path& path, const Champion& champion)
{
    const BrainModel model = build_brain_model(champion.genome);
    save_brain_checkpoint(
        path,
        champion.brain_seed,
        model,
        model.initial_connection_strengths);
}

// retains only the ten most recent generation winners
void save_generation_archive(
    const filesystem::path& directory,
    const string& name,
    const Champion& champion)
{
    save_champion(generation_path(directory, name, champion.generation), champion);

    if (champion.generation > 10U)
    {
        const filesystem::path expired = generation_path(
            directory,
            name,
            champion.generation - 10U);
        error_code error;
        filesystem::remove(expired, error);

        if (error)
        {
            throw filesystem::filesystem_error(
                "failed to remove expired generation checkpoint",
                expired,
                error);
        }
    }
}

// orders champions by fitness and stable generation identity
bool is_better_champion(const Champion& candidate, const Champion& incumbent)
{
    if (better_evolution_fitness(candidate.archive_fitness, incumbent.archive_fitness))
    {
        return true;
    }

    if (better_evolution_fitness(incumbent.archive_fitness, candidate.archive_fitness))
    {
        return false;
    }

    return candidate.generation < incumbent.generation;
}

// updates and rewrites the five strongest generation winners
void update_best_archive(
    const filesystem::path& directory,
    const string& name,
    Champion champion,
    vector<Champion>& best)
{
    best.push_back(move(champion));
    stable_sort(best.begin(), best.end(), is_better_champion);

    if (best.size() > 5U)
    {
        best.resize(5U);
    }

    for (uint32_t rank = 0; rank < best.size(); ++rank)
    {
        save_champion(rank_path(directory, name, rank + 1U), best[rank]);
    }
}

// describes the controller's anonymous output-drive input pairings
EvolutionInterfaceConstraints controller_interface_constraints(const BrainGenome& genome)
{
    EvolutionInterfaceConstraints constraints;
    const uint32_t output_count = static_cast<uint32_t>(genome.output_positions.size());

    if (genome.input_positions.size() < output_count)
    {
        throw invalid_argument("controller genome does not contain its output-drive inputs");
    }

    const uint32_t output_input_offset = static_cast<uint32_t>(
        genome.input_positions.size() - output_count);
    constraints.paired_positions.reserve(output_count);

    for (uint32_t output = 0U; output < output_count; ++output)
    {
        constraints.paired_positions.push_back({output_input_offset + output, output});
    }

    return constraints;
}

// creates independently seeded and anatomically varied initial genomes
vector<Candidate> make_initial_candidates(
    const Options& options,
    const EnvironmentConfig& environment_config,
    span<const GridPosition> allowed_positions,
    const EvolutionMutationConfig& mutation_config)
{
    vector<Candidate> candidates(options.initial_population);

    for (uint32_t candidate = 0; candidate < candidates.size(); ++candidate)
    {
        const uint64_t genome_seed = mix_seed(
            options.seed ^ 0xe7037ed1a0b428dbULL ^ candidate);
        const BrainAgent seeded_agent(environment_config, {}, genome_seed);
        const BrainGenome seeded_genome = make_brain_genome(
            seeded_agent.model(),
            seeded_agent.copy_connection_strengths());
        const uint64_t mutation_seed = mix_seed(
            options.seed ^ 0x589965cc75374cc3ULL ^ candidate);
        candidates[candidate].brain_seed = mix_seed(
            options.seed ^ 0x1d8e4e27c47d124fULL ^ candidate);
        candidates[candidate].genome = mutate_brain_genome(
            seeded_genome,
            allowed_positions,
            mutation_seed,
            mutation_config,
            controller_interface_constraints(seeded_genome));
    }

    return candidates;
}

// creates one learned elite and four deterministic nearby mutations
vector<Candidate> make_next_candidates(
    const Options& options,
    const Champion& parent,
    span<const GridPosition> allowed_positions,
    const EvolutionMutationConfig& mutation_config)
{
    vector<Candidate> candidates;
    candidates.reserve(MUTANTS_PER_GENERATION + 1U);
    candidates.push_back({parent.brain_seed, parent.genome, parent.generation, 0U});
    const EvolutionInterfaceConstraints constraints = controller_interface_constraints(
        parent.genome);

    for (uint32_t mutation = 1U; mutation <= MUTANTS_PER_GENERATION; ++mutation)
    {
        const uint64_t mutation_seed = mix_seed(
            options.seed
            ^ (static_cast<uint64_t>(parent.generation + 1U) << 32U)
            ^ mutation);
        Candidate child;
        child.brain_seed = mix_seed(mutation_seed ^ 0x8ebc6af09c88c6e3ULL);
        child.genome = mutate_brain_genome(
            parent.genome,
            allowed_positions,
            mutation_seed,
            mutation_config,
            constraints);
        child.parent_generation = parent.generation;
        child.mutation = mutation;
        candidates.push_back(move(child));
    }

    return candidates;
}

// returns every cell in the current maximum evolvable sphere
vector<GridPosition> make_allowed_positions()
{
    const AdaptiveControllerConfig config;
    const BrainLayout layout = make_sphere_brain_layout(config.brain_shape, config.brain_radius);
    vector<GridPosition> positions;
    positions.reserve(layout.neurons.size());

    for (const NeuronDefinition& neuron : layout.neurons)
    {
        positions.push_back(neuron.position);
    }

    return positions;
}

}

// runs seeded evolution and writes rolling playable checkpoints
int main(int argc, char** argv)
{
    try
    {
        const Options options = parse_options(argc, argv);
        const vector<GridPosition> allowed_positions = make_allowed_positions();
        const EnvironmentConfig environment_config;
        EvolutionMutationConfig mutation_config;
        mutation_config.maximum_neurons = static_cast<uint32_t>(allowed_positions.size());
        vector<Candidate> candidates = make_initial_candidates(
            options,
            environment_config,
            allowed_positions,
            mutation_config);
        ostringstream metadata_stream;

        {
            const BrainExecutionConfig execution{options.backend, options.cuda_device};
            const BrainAgent representative_agent(
                environment_config,
                {},
                candidates.front().brain_seed,
                execution,
                backend_factory(options.backend),
                &candidates.front().genome.value());
            write_metadata(
                metadata_stream,
                options,
                environment_config,
                representative_agent,
                mutation_config);
        }

        const filesystem::path directory = prepare_run_directory(options.name);
        const filesystem::path results_path = directory / (options.name + "-results.jsonl");
        ofstream results_stream(results_path, ios::out | ios::trunc);

        if (!results_stream)
        {
            throw runtime_error("failed to open evolution results file");
        }

        const string metadata = metadata_stream.str();
        results_stream << metadata;
        flush_output(results_stream);
        cout << metadata;
        flush_output(cout);
        vector<Champion> best;

        for (uint32_t generation = 1U;; ++generation)
        {
            optional<CandidateResult> winner;

            for (uint32_t candidate = 0; candidate < candidates.size(); ++candidate)
            {
                CandidateResult result = evaluate_candidate(
                    options,
                    generation,
                    candidate,
                    candidates[candidate]);
                write_candidate(results_stream, generation, candidates[candidate], result);
                write_candidate(cout, generation, candidates[candidate], result);

                if (!winner
                    || better_evolution_fitness(result.fitness, winner->fitness))
                {
                    winner = move(result);
                }
            }

            const EvolutionFitness archive_fitness = evaluate_frozen_genome(
                options,
                generation,
                winner->brain_seed,
                winner->learned_genome,
                archive_seed);
            Champion champion{
                generation,
                winner->brain_seed,
                move(winner->learned_genome),
                winner->fitness,
                archive_fitness
            };
            save_generation_archive(directory, options.name, champion);
            update_best_archive(directory, options.name, champion, best);
            write_generation_summary(results_stream, champion);
            write_generation_summary(cout, champion);

            if (generation == options.generations)
            {
                break;
            }

            candidates = make_next_candidates(
                options,
                champion,
                allowed_positions,
                mutation_config);
        }

        const Champion& selected = best.front();
        const EvolutionFitness final_test = evaluate_frozen_genome(
            options,
            options.generations,
            selected.brain_seed,
            selected.genome,
            final_test_seed);
        write_completion(results_stream, directory, options.name, final_test);
        write_completion(cout, directory, options.name, final_test);
        return 0;
    }
    catch (const exception& error)
    {
        cerr << error.what() << '\n';
        return 1;
    }
}
