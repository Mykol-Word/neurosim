#include "neurosim/brain.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace neurosim
{

using namespace std;

namespace
{

// throws when a brain construction invariant is false
void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw invalid_argument(message);
    }
}

// returns whether a scalar is finite
bool finite(float value)
{
    return isfinite(value);
}

// returns the validated number of cells in a lattice
uint32_t grid_cell_count(GridShape shape)
{
    require(shape.width > 0 && shape.height > 0 && shape.depth > 0,
        "brain grid dimensions must be positive");

    const uint64_t plane = static_cast<uint64_t>(shape.width) * shape.height;
    const uint64_t count = plane * shape.depth;
    require(count < INVALID_NEURON_INDEX, "brain grid has too many cells for 32-bit indexing");
    return static_cast<uint32_t>(count);
}

// returns whether a lattice position lies inside a shape
bool is_in_bounds(GridShape shape, GridPosition position)
{
    return position.x < shape.width
        && position.y < shape.height
        && position.z < shape.depth;
}

// flattens one validated lattice position into x-major storage
uint32_t flatten_position(GridShape shape, GridPosition position)
{
    return position.x + shape.width * (position.y + shape.height * position.z);
}

// converts a container size to the model's index representation
uint32_t checked_index(size_t value, const char* message)
{
    if (value >= INVALID_NEURON_INDEX)
    {
        throw invalid_argument(message);
    }

    return static_cast<uint32_t>(value);
}

// mixes a seed into a stable uniformly distributed 64-bit value
uint64_t mix_seed(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

// returns a stable 24-bit unit sample from a mixed seed
float unit_sample(uint64_t value)
{
    return static_cast<float>(mix_seed(value) >> 40U) * (1.0f / 16777216.0f);
}

// returns the manhattan distance between two lattice positions
uint64_t manhattan_distance(GridPosition left, GridPosition right)
{
    const uint64_t x = left.x > right.x ? left.x - right.x : right.x - left.x;
    const uint64_t y = left.y > right.y ? left.y - right.y : right.y - left.y;
    const uint64_t z = left.z > right.z ? left.z - right.z : right.z - left.z;
    return x + y + z;
}

// validates all user-configurable neuron and plasticity parameters
void validate_brain_config(const BrainConfig& config)
{
    const NeuronDynamicsConfig& dynamics = config.dynamics;
    const PlasticityConfig& plasticity = config.plasticity;

    require(config.connection_radius_steps > 0, "connection radius must be positive");
    require(finite(dynamics.fixed_time_step_seconds) && dynamics.fixed_time_step_seconds > 0.0f,
        "brain time step must be finite and positive");
    require(finite(dynamics.membrane_time_constant_seconds)
        && dynamics.membrane_time_constant_seconds > 0.0f,
        "membrane time constant must be finite and positive");
    require(dynamics.fixed_time_step_seconds <= dynamics.membrane_time_constant_seconds,
        "brain time step must not exceed the membrane time constant");
    require(finite(dynamics.resting_potential), "resting potential must be finite");
    require(finite(dynamics.reset_potential), "reset potential must be finite");
    require(finite(dynamics.firing_threshold), "firing threshold must be finite");
    require(dynamics.firing_threshold > dynamics.resting_potential
        && dynamics.firing_threshold > dynamics.reset_potential,
        "firing threshold must exceed resting and reset potentials");
    require(finite(dynamics.refractory_period_seconds)
        && dynamics.refractory_period_seconds >= 0.0f,
        "refractory period must be finite and nonnegative");
    require(finite(plasticity.activity_trace_time_constant_seconds)
        && plasticity.activity_trace_time_constant_seconds > 0.0f,
        "activity trace time constant must be finite and positive");
    require(finite(plasticity.learning_rate) && plasticity.learning_rate >= 0.0f,
        "learning rate must be finite and nonnegative");
    require(finite(plasticity.weakening_rate_per_second)
        && plasticity.weakening_rate_per_second >= 0.0f,
        "weakening rate must be finite and nonnegative");
    require(finite(plasticity.outgoing_strength_budget)
        && plasticity.outgoing_strength_budget > 0.0f,
        "outgoing strength budget must be finite and positive");
    require(finite(plasticity.initial_budget_fraction)
        && plasticity.initial_budget_fraction >= 0.0f
        && plasticity.initial_budget_fraction <= 1.0f,
        "initial budget fraction must be within zero and one");
    require(finite(plasticity.initial_connection_density)
        && plasticity.initial_connection_density >= 0.0f
        && plasticity.initial_connection_density <= 1.0f,
        "initial connection density must be within zero and one");

    const double refractory_ticks = ceil(
        static_cast<double>(dynamics.refractory_period_seconds)
        / static_cast<double>(dynamics.fixed_time_step_seconds));
    require(refractory_ticks <= static_cast<double>(numeric_limits<uint32_t>::max()),
        "refractory period cannot be represented in fixed ticks");
}

// validates sparse occupancy and ordered external channel coordinates
void validate_brain_layout(const BrainLayout& layout)
{
    const uint32_t cell_count = grid_cell_count(layout.shape);
    require(!layout.neurons.empty(), "brain layout must contain at least one neuron");
    require(layout.neurons.size() < INVALID_NEURON_INDEX,
        "brain layout has too many neurons for 32-bit indexing");
    require(layout.input_positions.size() <= static_cast<size_t>(numeric_limits<int32_t>::max()),
        "brain layout has too many input channels for signed slot indexing");

    vector<uint8_t> occupied(cell_count, 0U);

    for (const NeuronDefinition& neuron : layout.neurons)
    {
        require(is_in_bounds(layout.shape, neuron.position), "neuron position is outside the brain grid");
        require(finite(neuron.bias_drive), "neuron bias drive must be finite");
        require(neuron.polarity == NeuronPolarity::Excitatory
            || neuron.polarity == NeuronPolarity::Inhibitory,
            "neuron polarity is invalid");

        const uint32_t cell = flatten_position(layout.shape, neuron.position);
        require(occupied[cell] == 0U, "brain layout contains duplicate neuron positions");
        occupied[cell] = 1U;
    }

    vector<uint8_t> input_cells(cell_count, 0U);

    for (const GridPosition position : layout.input_positions)
    {
        require(is_in_bounds(layout.shape, position), "input position is outside the brain grid");
        const uint32_t cell = flatten_position(layout.shape, position);
        require(occupied[cell] != 0U, "input position does not contain a neuron");
        require(input_cells[cell] == 0U, "brain layout contains duplicate input positions");
        input_cells[cell] = 1U;
    }

    vector<uint8_t> output_cells(cell_count, 0U);

    for (const GridPosition position : layout.output_positions)
    {
        require(is_in_bounds(layout.shape, position), "output position is outside the brain grid");
        const uint32_t cell = flatten_position(layout.shape, position);
        require(occupied[cell] != 0U, "output position does not contain a neuron");
        require(output_cells[cell] == 0U, "brain layout contains duplicate output positions");
        output_cells[cell] = 1U;
    }
}

// compiles ordered channel coordinates into stable neuron indices
vector<uint32_t> compile_channels(
    const BrainModel& model,
    const vector<GridPosition>& positions)
{
    vector<uint32_t> channels;
    channels.reserve(positions.size());

    for (const GridPosition position : positions)
    {
        channels.push_back(model.cell_to_neuron[flatten_position(model.shape, position)]);
    }

    return channels;
}

// initializes random latent and active strengths under each source budget
void initialize_connection_strengths(BrainModel& model)
{
    const PlasticityConfig& config = model.config.plasticity;
    model.initial_connection_strengths.assign(model.outgoing_targets.size(), 0.0f);

    for (uint32_t source = 0; source < model.positions.size(); ++source)
    {
        const uint32_t begin = model.outgoing_offsets[source];
        const uint32_t end = model.outgoing_offsets[source + 1U];

        if (begin == end || config.initial_budget_fraction == 0.0f
            || config.initial_connection_density == 0.0f)
        {
            continue;
        }

        float raw_sum = 0.0f;

        for (uint32_t edge = begin; edge < end; ++edge)
        {
            const uint64_t edge_seed = model.seed
                ^ (static_cast<uint64_t>(source) << 32U)
                ^ static_cast<uint64_t>(model.outgoing_targets[edge]);

            if (unit_sample(edge_seed) >= config.initial_connection_density)
            {
                continue;
            }

            const float strength = 0.5f + 0.5f * unit_sample(edge_seed ^ 0xd1b54a32d192ed03ULL);
            model.initial_connection_strengths[edge] = strength;
            raw_sum += strength;
        }

        if (raw_sum == 0.0f)
        {
            const uint32_t edge_count = end - begin;
            const uint32_t selected = begin + static_cast<uint32_t>(
                mix_seed(model.seed ^ source) % edge_count);
            model.initial_connection_strengths[selected] = 1.0f;
            raw_sum = 1.0f;
        }

        const float target_sum = config.outgoing_strength_budget * config.initial_budget_fraction;
        const float scale = target_sum / raw_sum;

        for (uint32_t edge = begin; edge < end; ++edge)
        {
            model.initial_connection_strengths[edge] *= scale;
        }
    }
}

// returns the fixed number of suppressed ticks after one spike
uint32_t calculate_refractory_ticks(const NeuronDynamicsConfig& config)
{
    return static_cast<uint32_t>(ceil(
        static_cast<double>(config.refractory_period_seconds)
        / static_cast<double>(config.fixed_time_step_seconds)));
}

}

// creates a sparse spherical occupancy pattern centered in a lattice
BrainLayout make_sphere_brain_layout(GridShape shape, float radius)
{
    grid_cell_count(shape);
    require(finite(radius) && radius >= 0.0f, "sphere radius must be finite and nonnegative");

    BrainLayout layout;
    layout.shape = shape;
    const float center_x = static_cast<float>(shape.width - 1U) * 0.5f;
    const float center_y = static_cast<float>(shape.height - 1U) * 0.5f;
    const float center_z = static_cast<float>(shape.depth - 1U) * 0.5f;
    const float radius_squared = radius * radius;
    require(finite(radius_squared), "sphere radius is too large for 32-bit geometry");

    for (uint32_t z = 0; z < shape.depth; ++z)
    {
        for (uint32_t y = 0; y < shape.height; ++y)
        {
            for (uint32_t x = 0; x < shape.width; ++x)
            {
                const float offset_x = static_cast<float>(x) - center_x;
                const float offset_y = static_cast<float>(y) - center_y;
                const float offset_z = static_cast<float>(z) - center_z;
                const float distance_squared = offset_x * offset_x
                    + offset_y * offset_y
                    + offset_z * offset_z;

                if (distance_squared <= radius_squared)
                {
                    layout.neurons.push_back({{x, y, z}, NeuronPolarity::Excitatory, 0.0f});
                }
            }
        }
    }

    require(!layout.neurons.empty(), "sphere radius does not occupy any brain cells");
    return layout;
}

// validates and compiles a sparse layout into stable gpu-ready arrays
BrainModel build_brain_model(const BrainLayout& layout, const BrainConfig& config, uint64_t seed)
{
    validate_brain_config(config);
    validate_brain_layout(layout);

    BrainModel model;
    model.seed = seed;
    model.config = config;
    model.shape = layout.shape;
    model.cell_to_neuron.assign(grid_cell_count(layout.shape), INVALID_NEURON_INDEX);

    vector<uint32_t> order(layout.neurons.size());
    iota(order.begin(), order.end(), 0U);
    sort(order.begin(), order.end(), [&layout](uint32_t left, uint32_t right)
    {
        return flatten_position(layout.shape, layout.neurons[left].position)
            < flatten_position(layout.shape, layout.neurons[right].position);
    });

    model.positions.reserve(order.size());
    model.polarities.reserve(order.size());
    model.bias_drives.reserve(order.size());

    for (const uint32_t original_index : order)
    {
        const NeuronDefinition& neuron = layout.neurons[original_index];
        const uint32_t neuron_index = checked_index(model.positions.size(), "too many neurons");
        model.positions.push_back(neuron.position);
        model.polarities.push_back(neuron.polarity);
        model.bias_drives.push_back(neuron.bias_drive);
        model.cell_to_neuron[flatten_position(model.shape, neuron.position)] = neuron_index;
    }

    model.input_neurons = compile_channels(model, layout.input_positions);
    model.output_neurons = compile_channels(model, layout.output_positions);
    model.input_slots_by_neuron.assign(model.positions.size(), -1);

    for (uint32_t slot = 0; slot < model.input_neurons.size(); ++slot)
    {
        model.input_slots_by_neuron[model.input_neurons[slot]] = static_cast<int32_t>(slot);
    }

    model.outgoing_offsets.reserve(model.positions.size() + 1U);
    model.outgoing_offsets.push_back(0U);
    const int64_t radius = static_cast<int64_t>(config.connection_radius_steps);

    for (uint32_t source = 0; source < model.positions.size(); ++source)
    {
        const GridPosition source_position = model.positions[source];
        vector<uint32_t> targets;
        const int64_t minimum_z = -min(radius, static_cast<int64_t>(source_position.z));
        const int64_t maximum_z = min(radius,
            static_cast<int64_t>(model.shape.depth - 1U - source_position.z));
        const int64_t minimum_y = -min(radius, static_cast<int64_t>(source_position.y));
        const int64_t maximum_y = min(radius,
            static_cast<int64_t>(model.shape.height - 1U - source_position.y));
        const int64_t minimum_x = -min(radius, static_cast<int64_t>(source_position.x));
        const int64_t maximum_x = min(radius,
            static_cast<int64_t>(model.shape.width - 1U - source_position.x));

        for (int64_t z_offset = minimum_z; z_offset <= maximum_z; ++z_offset)
        {
            for (int64_t y_offset = minimum_y; y_offset <= maximum_y; ++y_offset)
            {
                for (int64_t x_offset = minimum_x; x_offset <= maximum_x; ++x_offset)
                {
                    const uint64_t distance = static_cast<uint64_t>(abs(x_offset)
                        + abs(y_offset) + abs(z_offset));

                    if (distance == 0U || distance > config.connection_radius_steps)
                    {
                        continue;
                    }

                    const int64_t x = static_cast<int64_t>(source_position.x) + x_offset;
                    const int64_t y = static_cast<int64_t>(source_position.y) + y_offset;
                    const int64_t z = static_cast<int64_t>(source_position.z) + z_offset;

                    const GridPosition target_position{
                        static_cast<uint32_t>(x),
                        static_cast<uint32_t>(y),
                        static_cast<uint32_t>(z)
                    };
                    const uint32_t target = model.cell_to_neuron[
                        flatten_position(model.shape, target_position)];

                    if (target != INVALID_NEURON_INDEX)
                    {
                        targets.push_back(target);
                    }
                }
            }
        }

        sort(targets.begin(), targets.end());

        for (const uint32_t target : targets)
        {
            model.edge_sources.push_back(source);
            model.outgoing_targets.push_back(target);
        }

        model.outgoing_offsets.push_back(checked_index(
            model.outgoing_targets.size(), "brain model has too many potential connections"));
    }

    initialize_connection_strengths(model);
    model.incoming_offsets.assign(model.positions.size() + 1U, 0U);

    for (const uint32_t target : model.outgoing_targets)
    {
        ++model.incoming_offsets[target + 1U];
    }

    for (uint32_t index = 1; index < model.incoming_offsets.size(); ++index)
    {
        model.incoming_offsets[index] += model.incoming_offsets[index - 1U];
    }

    model.incoming_edges.resize(model.outgoing_targets.size());
    vector<uint32_t> incoming_cursors = model.incoming_offsets;

    for (uint32_t edge = 0; edge < model.outgoing_targets.size(); ++edge)
    {
        const uint32_t target = model.outgoing_targets[edge];
        model.incoming_edges[incoming_cursors[target]++] = edge;
    }

    model.trace_decay = exp(-config.dynamics.fixed_time_step_seconds
        / config.plasticity.activity_trace_time_constant_seconds);
    model.weight_decay = exp(-config.plasticity.weakening_rate_per_second
        * config.dynamics.fixed_time_step_seconds);
    model.refractory_ticks = calculate_refractory_ticks(config.dynamics);
    validate_brain_model(model);
    return model;
}

// creates resting mutable state from a compiled model
BrainState make_brain_state(const BrainModel& model)
{
    validate_brain_model(model);

    BrainState state;
    state.membrane_potentials.assign(model.positions.size(), model.config.dynamics.resting_potential);
    state.activity_traces.assign(model.positions.size(), 0.0f);
    state.connection_strengths = model.initial_connection_strengths;
    state.refractory_ticks.assign(model.positions.size(), 0U);
    state.spikes.assign(model.positions.size(), 0U);
    return state;
}

// validates compiled topology, interfaces, and initial strengths
void validate_brain_model(const BrainModel& model)
{
    validate_brain_config(model.config);
    const uint32_t cell_count = grid_cell_count(model.shape);
    const size_t neuron_count = model.positions.size();
    const size_t edge_count = model.outgoing_targets.size();
    require(neuron_count > 0 && neuron_count < INVALID_NEURON_INDEX,
        "brain model must contain a representable neuron count");
    require(model.polarities.size() == neuron_count, "brain polarity array size is invalid");
    require(model.bias_drives.size() == neuron_count, "brain bias array size is invalid");
    require(model.cell_to_neuron.size() == cell_count, "brain cell lookup size is invalid");
    require(model.outgoing_offsets.size() == neuron_count + 1U,
        "brain outgoing offset array size is invalid");
    require(model.edge_sources.size() == edge_count, "brain edge source array size is invalid");
    require(model.initial_connection_strengths.size() == edge_count,
        "brain initial strength array size is invalid");
    require(model.incoming_offsets.size() == neuron_count + 1U,
        "brain incoming offset array size is invalid");
    require(model.incoming_edges.size() == edge_count, "brain incoming edge array size is invalid");
    require(model.input_slots_by_neuron.size() == neuron_count,
        "brain input slot lookup size is invalid");
    require(model.outgoing_offsets.front() == 0U
        && model.outgoing_offsets.back() == edge_count,
        "brain outgoing offsets do not cover every edge");
    require(model.incoming_offsets.front() == 0U
        && model.incoming_offsets.back() == edge_count,
        "brain incoming offsets do not cover every edge");

    vector<uint8_t> occupied(cell_count, 0U);

    for (uint32_t neuron = 0; neuron < neuron_count; ++neuron)
    {
        require(is_in_bounds(model.shape, model.positions[neuron]),
            "brain model contains an out-of-bounds neuron");
        require(finite(model.bias_drives[neuron]), "brain model contains a non-finite bias drive");
        require(model.polarities[neuron] == NeuronPolarity::Excitatory
            || model.polarities[neuron] == NeuronPolarity::Inhibitory,
            "brain model contains an invalid neuron polarity");

        const uint32_t cell = flatten_position(model.shape, model.positions[neuron]);
        require(occupied[cell] == 0U, "brain model contains duplicate neuron positions");
        require(model.cell_to_neuron[cell] == neuron, "brain cell lookup does not match neuron positions");
        occupied[cell] = 1U;
    }

    for (uint32_t cell = 0; cell < cell_count; ++cell)
    {
        require((occupied[cell] == 0U && model.cell_to_neuron[cell] == INVALID_NEURON_INDEX)
            || (occupied[cell] != 0U && model.cell_to_neuron[cell] < neuron_count),
            "brain cell lookup contains an invalid entry");
    }

    for (uint32_t source = 0; source < neuron_count; ++source)
    {
        const uint32_t begin = model.outgoing_offsets[source];
        const uint32_t end = model.outgoing_offsets[source + 1U];
        require(begin <= end && end <= edge_count, "brain outgoing offsets are not monotonic");
        float strength_sum = 0.0f;
        uint32_t previous_target = INVALID_NEURON_INDEX;

        for (uint32_t edge = begin; edge < end; ++edge)
        {
            const uint32_t target = model.outgoing_targets[edge];
            const float strength = model.initial_connection_strengths[edge];
            require(model.edge_sources[edge] == source, "brain edge source does not match its range");
            require(target < neuron_count && target != source, "brain edge target is invalid");
            require(previous_target == INVALID_NEURON_INDEX || target > previous_target,
                "brain outgoing targets are duplicated or unsorted");
            require(manhattan_distance(model.positions[source], model.positions[target])
                <= model.config.connection_radius_steps,
                "brain edge exceeds the configured connection radius");
            require(finite(strength) && strength >= 0.0f,
                "brain initial connection strength must be finite and nonnegative");
            previous_target = target;
            strength_sum += strength;
        }

        const float tolerance = max(1.0f, model.config.plasticity.outgoing_strength_budget) * 1.0e-5f;
        require(finite(strength_sum)
            && strength_sum <= model.config.plasticity.outgoing_strength_budget + tolerance,
            "brain initial connection strengths exceed the outgoing budget");
    }

    vector<uint8_t> incoming_seen(edge_count, 0U);

    for (uint32_t target = 0; target < neuron_count; ++target)
    {
        const uint32_t begin = model.incoming_offsets[target];
        const uint32_t end = model.incoming_offsets[target + 1U];
        require(begin <= end && end <= edge_count, "brain incoming offsets are not monotonic");

        for (uint32_t incoming = begin; incoming < end; ++incoming)
        {
            const uint32_t edge = model.incoming_edges[incoming];
            require(edge < edge_count && model.outgoing_targets[edge] == target,
                "brain incoming edge lookup is invalid");
            require(incoming_seen[edge] == 0U, "brain incoming edge lookup contains a duplicate");
            incoming_seen[edge] = 1U;
        }
    }

    for (const uint8_t seen : incoming_seen)
    {
        require(seen != 0U, "brain incoming edge lookup omits a connection");
    }

    vector<uint8_t> input_seen(neuron_count, 0U);

    for (uint32_t slot = 0; slot < model.input_neurons.size(); ++slot)
    {
        const uint32_t neuron = model.input_neurons[slot];
        require(neuron < neuron_count && input_seen[neuron] == 0U,
            "brain input channel list is invalid");
        require(model.input_slots_by_neuron[neuron] == static_cast<int32_t>(slot),
            "brain input slot lookup is inconsistent");
        input_seen[neuron] = 1U;
    }

    for (uint32_t neuron = 0; neuron < neuron_count; ++neuron)
    {
        require((input_seen[neuron] == 0U && model.input_slots_by_neuron[neuron] == -1)
            || (input_seen[neuron] != 0U && model.input_slots_by_neuron[neuron] >= 0),
            "brain input slot lookup contains an invalid entry");
    }

    vector<uint8_t> output_seen(neuron_count, 0U);

    for (const uint32_t neuron : model.output_neurons)
    {
        require(neuron < neuron_count && output_seen[neuron] == 0U,
            "brain output channel list is invalid");
        output_seen[neuron] = 1U;
    }

    const float expected_trace_decay = exp(-model.config.dynamics.fixed_time_step_seconds
        / model.config.plasticity.activity_trace_time_constant_seconds);
    const float expected_weight_decay = exp(-model.config.plasticity.weakening_rate_per_second
        * model.config.dynamics.fixed_time_step_seconds);
    require(finite(model.trace_decay) && model.trace_decay == expected_trace_decay,
        "brain trace decay is invalid");
    require(finite(model.weight_decay) && model.weight_decay == expected_weight_decay,
        "brain weight decay is invalid");
    require(model.refractory_ticks == calculate_refractory_ticks(model.config.dynamics),
        "brain refractory tick count is inconsistent");
}

// validates mutable arrays and outgoing strength budgets
void validate_brain_state(const BrainModel& model, const BrainState& state)
{
    validate_brain_model(model);
    const size_t neuron_count = model.positions.size();
    const size_t edge_count = model.outgoing_targets.size();
    require(state.membrane_potentials.size() == neuron_count,
        "brain membrane state size is invalid");
    require(state.activity_traces.size() == neuron_count, "brain activity trace state size is invalid");
    require(state.connection_strengths.size() == edge_count,
        "brain connection strength state size is invalid");
    require(state.refractory_ticks.size() == neuron_count, "brain refractory state size is invalid");
    require(state.spikes.size() == neuron_count, "brain spike state size is invalid");

    for (uint32_t neuron = 0; neuron < neuron_count; ++neuron)
    {
        require(finite(state.membrane_potentials[neuron]),
            "brain state contains a non-finite membrane potential");
        require(state.membrane_potentials[neuron] < model.config.dynamics.firing_threshold,
            "brain state contains an unconsumed threshold crossing");
        require(finite(state.activity_traces[neuron]) && state.activity_traces[neuron] >= 0.0f,
            "brain state contains an invalid activity trace");
        require(state.spikes[neuron] <= 1U, "brain state contains an invalid spike value");
        require(state.refractory_ticks[neuron] <= model.refractory_ticks,
            "brain state contains an invalid refractory counter");

        if (state.refractory_ticks[neuron] > 0U || state.spikes[neuron] != 0U)
        {
            require(state.membrane_potentials[neuron] == model.config.dynamics.reset_potential,
                "brain refractory or spiking state is not at reset potential");
        }

        if (state.spikes[neuron] != 0U)
        {
            require(state.refractory_ticks[neuron] == model.refractory_ticks,
                "brain spike state does not start the configured refractory period");
        }
    }

    for (uint32_t source = 0; source < neuron_count; ++source)
    {
        float strength_sum = 0.0f;

        for (uint32_t edge = model.outgoing_offsets[source];
            edge < model.outgoing_offsets[source + 1U]; ++edge)
        {
            const float strength = state.connection_strengths[edge];
            require(finite(strength) && strength >= 0.0f,
                "brain state contains an invalid connection strength");
            strength_sum += strength;
        }

        const float tolerance = max(1.0f, model.config.plasticity.outgoing_strength_budget) * 1.0e-5f;
        require(finite(strength_sum)
            && strength_sum <= model.config.plasticity.outgoing_strength_budget + tolerance,
            "brain state exceeds an outgoing connection budget");
    }
}

// returns the neuron occupying a cell or the invalid neuron sentinel
uint32_t neuron_index_at(const BrainModel& model, GridPosition position)
{
    if (!is_in_bounds(model.shape, position))
    {
        return INVALID_NEURON_INDEX;
    }

    return model.cell_to_neuron[flatten_position(model.shape, position)];
}

// returns a directed connection index or the invalid index sentinel
uint32_t connection_index(const BrainModel& model, uint32_t source, uint32_t target)
{
    if (source >= model.positions.size() || target >= model.positions.size())
    {
        return INVALID_NEURON_INDEX;
    }

    const auto begin = model.outgoing_targets.begin() + model.outgoing_offsets[source];
    const auto end = model.outgoing_targets.begin() + model.outgoing_offsets[source + 1U];
    const auto found = lower_bound(begin, end, target);
    return found == end || *found != target
        ? INVALID_NEURON_INDEX
        : static_cast<uint32_t>(found - model.outgoing_targets.begin());
}

}
