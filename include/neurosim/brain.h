#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace neurosim
{

using namespace std;

inline constexpr uint32_t INVALID_NEURON_INDEX = UINT32_MAX;

// identifies one cell in a discrete three-dimensional brain lattice
struct GridPosition
{
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;

    // compares two lattice positions for equality
    bool operator==(const GridPosition&) const = default;
};

// defines the outer dimensions of a brain lattice
struct GridShape
{
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;

    // compares two lattice shapes for equality
    bool operator==(const GridShape&) const = default;
};

// fixes the sign of every signal sent by one neuron
enum class NeuronPolarity : int8_t
{
    Inhibitory = -1,
    Excitatory = 1
};

// defines one occupied lattice cell and its constant drive
struct NeuronDefinition
{
    GridPosition position;
    NeuronPolarity polarity = NeuronPolarity::Excitatory;
    float bias_drive = 0.0f;

    // compares two neuron definitions for equality
    bool operator==(const NeuronDefinition&) const = default;
};

// describes occupied cells and ordered external input and output channels
struct BrainLayout
{
    GridShape shape;
    vector<NeuronDefinition> neurons;
    vector<GridPosition> input_positions;
    vector<GridPosition> output_positions;
};

// configures normalized leaky integrate-and-fire dynamics
struct NeuronDynamicsConfig
{
    float fixed_time_step_seconds = 0.001f;
    float membrane_time_constant_seconds = 0.020f;
    float resting_potential = 0.0f;
    float reset_potential = 0.0f;
    float firing_threshold = 1.0f;
    float refractory_period_seconds = 0.002f;

    // compares two neuron dynamics configurations for equality
    bool operator==(const NeuronDynamicsConfig&) const = default;
};

// configures local activity-trace learning and outgoing weight competition
struct PlasticityConfig
{
    float activity_trace_time_constant_seconds = 0.020f;
    float eligibility_time_constant_seconds = 1.0f;
    float maximum_eligibility = 1.0f;
    float learning_rate = 0.050f;
    float weakening_rate_per_second = 0.0001f;
    float outgoing_strength_budget = 1.0f;
    float initial_budget_fraction = 1.0f;
    float initial_connection_density = 0.25f;

    // compares two plasticity configurations for equality
    bool operator==(const PlasticityConfig&) const = default;
};

// configures topology range, neuron dynamics, and plasticity
struct BrainConfig
{
    uint32_t connection_radius_steps = 2;
    NeuronDynamicsConfig dynamics;
    PlasticityConfig plasticity;

    // compares two brain configurations for equality
    bool operator==(const BrainConfig&) const = default;
};

// configures deterministic signed noise applied independently to every neuron
struct BrainPerturbation
{
    float noise_drive_amplitude = 0.0f;
    uint64_t noise_seed = 0;
};

// stores immutable compiled topology and initial connection strengths
struct BrainModel
{
    uint64_t seed = 0;
    BrainConfig config;
    GridShape shape;
    vector<GridPosition> positions;
    vector<NeuronPolarity> polarities;
    vector<float> bias_drives;
    vector<uint32_t> cell_to_neuron;
    vector<uint32_t> outgoing_offsets;
    vector<uint32_t> outgoing_targets;
    vector<uint32_t> edge_sources;
    vector<uint32_t> incoming_offsets;
    vector<uint32_t> incoming_edges;
    vector<uint32_t> input_neurons;
    vector<int32_t> input_slots_by_neuron;
    vector<uint32_t> output_neurons;
    vector<float> initial_connection_strengths;
    float trace_decay = 0.0f;
    float eligibility_decay = 0.0f;
    float weight_decay = 0.0f;
    uint32_t refractory_ticks = 0;
};

// stores all mutable neuron and connection state for one brain
struct BrainState
{
    vector<float> membrane_potentials;
    vector<float> activity_traces;
    vector<float> connection_strengths;
    vector<float> connection_eligibilities;
    vector<uint32_t> refractory_ticks;
    vector<uint8_t> spikes;
    uint64_t tick = 0;
};

// creates a sparse spherical occupancy pattern centered in a lattice
BrainLayout make_sphere_brain_layout(GridShape shape, float radius);

// validates and compiles a sparse layout into stable gpu-ready arrays
BrainModel build_brain_model(const BrainLayout& layout, const BrainConfig& config, uint64_t seed);

// creates resting mutable state from a compiled model
BrainState make_brain_state(const BrainModel& model);

// validates compiled topology, interfaces, and initial strengths
void validate_brain_model(const BrainModel& model);

// validates learned strengths and every source-owned outgoing budget
void validate_connection_strengths(const BrainModel& model, span<const float> strengths);

// validates mutable arrays and outgoing strength budgets
void validate_brain_state(const BrainModel& model, const BrainState& state);

// returns the neuron occupying a cell or the invalid neuron sentinel
uint32_t neuron_index_at(const BrainModel& model, GridPosition position);

// returns a directed connection index or the invalid index sentinel
uint32_t connection_index(const BrainModel& model, uint32_t source, uint32_t target);

}
