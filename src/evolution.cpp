#include "neurosim/evolution.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace neurosim
{

using namespace std;

namespace
{

// stores one signed local lattice offset
struct GridOffset
{
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;
};

// provides deterministic portable sampling for inherited mutations
class EvolutionRandom
{
public:
    // seeds the generator into a stable initial state
    explicit EvolutionRandom(uint64_t seed)
    {
        state_ = 0;
        next_u32();
        state_ += seed;
        next_u32();
    }

    // returns the next uniformly distributed 32-bit value
    uint32_t next_u32()
    {
        const uint64_t old_state = state_;
        state_ = old_state * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint32_t shifted = static_cast<uint32_t>(((old_state >> 18U) ^ old_state) >> 27U);
        const uint32_t rotation = static_cast<uint32_t>(old_state >> 59U);
        return (shifted >> rotation) | (shifted << ((0U - rotation) & 31U));
    }

    // returns an unbiased integer within an inclusive range
    uint32_t range_u32(uint32_t minimum, uint32_t maximum)
    {
        if (minimum > maximum)
        {
            throw invalid_argument("invalid evolution random range");
        }

        const uint32_t bound = maximum - minimum + 1U;

        if (bound == 0U)
        {
            return next_u32();
        }

        const uint32_t threshold = static_cast<uint32_t>(0U - bound) % bound;

        for (;;)
        {
            const uint32_t value = next_u32();

            if (value >= threshold)
            {
                return minimum + value % bound;
            }
        }
    }

    // returns a stable unit sample in the half-open interval from zero to one
    float unit_sample()
    {
        return static_cast<float>(next_u32() >> 8U) * (1.0f / 16777216.0f);
    }

    // returns a portable approximately normal sample with unit variance
    float bell_sample()
    {
        float sample = -6.0f;

        for (uint32_t draw = 0; draw < 12U; ++draw)
        {
            sample += unit_sample();
        }

        return sample;
    }

private:
    uint64_t state_ = 0;
};

// throws when an evolution invariant is false
void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw invalid_argument(message);
    }
}

// returns the validated cell count of one lattice
uint32_t cell_count(GridShape shape)
{
    require(shape.width > 0U && shape.height > 0U && shape.depth > 0U,
        "evolution grid dimensions must be positive");
    const uint64_t plane = static_cast<uint64_t>(shape.width) * shape.height;
    const uint64_t count = plane * shape.depth;
    require(count < INVALID_NEURON_INDEX, "evolution grid is too large for 32-bit indexing");
    return static_cast<uint32_t>(count);
}

// returns whether one position lies inside a lattice
bool is_in_bounds(GridShape shape, GridPosition position)
{
    return position.x < shape.width
        && position.y < shape.height
        && position.z < shape.depth;
}

// flattens one in-bounds position into stable lattice storage
uint32_t flatten(GridShape shape, GridPosition position)
{
    return position.x + shape.width * (position.y + shape.height * position.z);
}

// expands one stable lattice index into a three-dimensional position
GridPosition expand(GridShape shape, uint32_t cell)
{
    const uint32_t plane = shape.width * shape.height;
    return {
        cell % shape.width,
        cell / shape.width % shape.height,
        cell / plane
    };
}

// builds every nonzero offset within the configured manhattan radius
vector<GridOffset> make_offsets(GridShape shape, uint32_t radius)
{
    vector<GridOffset> offsets;
    const int64_t x_radius = min<uint64_t>(radius, shape.width - 1U);
    const int64_t y_radius = min<uint64_t>(radius, shape.height - 1U);
    const int64_t z_radius = min<uint64_t>(radius, shape.depth - 1U);

    for (int64_t z = -z_radius; z <= z_radius; ++z)
    {
        for (int64_t y = -y_radius; y <= y_radius; ++y)
        {
            for (int64_t x = -x_radius; x <= x_radius; ++x)
            {
                const uint64_t distance = static_cast<uint64_t>(abs(x))
                    + static_cast<uint64_t>(abs(y))
                    + static_cast<uint64_t>(abs(z));

                if (distance > 0U && distance <= radius)
                {
                    offsets.push_back({x, y, z});
                }
            }
        }
    }

    return offsets;
}

// visits each valid local neighbor cell of one lattice cell
template<typename Visitor>
void visit_neighbors(
    GridShape shape,
    uint32_t cell,
    span<const GridOffset> offsets,
    Visitor&& visitor)
{
    const GridPosition position = expand(shape, cell);

    for (const GridOffset offset : offsets)
    {
        const int64_t x = static_cast<int64_t>(position.x) + offset.x;
        const int64_t y = static_cast<int64_t>(position.y) + offset.y;
        const int64_t z = static_cast<int64_t>(position.z) + offset.z;

        if (x < 0 || y < 0 || z < 0
            || x >= shape.width || y >= shape.height || z >= shape.depth)
        {
            continue;
        }

        visitor(flatten(shape, {
            static_cast<uint32_t>(x),
            static_cast<uint32_t>(y),
            static_cast<uint32_t>(z)
        }));
    }
}

// shuffles a vector with the project-owned deterministic generator
template<typename Value>
void shuffle_values(vector<Value>& values, EvolutionRandom& random)
{
    for (size_t remaining = values.size(); remaining > 1U; --remaining)
    {
        const uint32_t selected = random.range_u32(0U, static_cast<uint32_t>(remaining - 1U));
        swap(values[remaining - 1U], values[selected]);
    }
}

// returns whether occupied cells remain connected after excluding one cell
bool remains_connected(
    GridShape shape,
    span<const uint8_t> occupied,
    uint32_t occupied_count,
    uint32_t excluded,
    span<const GridOffset> offsets)
{
    const bool removes_cell = excluded < occupied.size() && occupied[excluded] != 0U;
    const uint32_t expected = occupied_count - static_cast<uint32_t>(removes_cell);

    if (expected <= 1U)
    {
        return true;
    }

    uint32_t start = INVALID_NEURON_INDEX;

    for (uint32_t cell = 0; cell < occupied.size(); ++cell)
    {
        if (cell != excluded && occupied[cell] != 0U)
        {
            start = cell;
            break;
        }
    }

    vector<uint8_t> visited(occupied.size(), 0U);
    vector<uint32_t> pending{start};
    visited[start] = 1U;
    uint32_t reached = 0U;

    while (!pending.empty())
    {
        const uint32_t cell = pending.back();
        pending.pop_back();
        ++reached;
        visit_neighbors(shape, cell, offsets, [&](uint32_t neighbor)
        {
            if (neighbor != excluded && occupied[neighbor] != 0U && visited[neighbor] == 0U)
            {
                visited[neighbor] = 1U;
                pending.push_back(neighbor);
            }
        });
    }

    return reached == expected;
}

// returns whether one unoccupied cell touches the occupied graph
bool touches_occupied(
    GridShape shape,
    uint32_t cell,
    span<const uint8_t> occupied,
    span<const GridOffset> offsets)
{
    bool touches = false;
    visit_neighbors(shape, cell, offsets, [&](uint32_t neighbor)
    {
        touches = touches || occupied[neighbor] != 0U;
    });
    return touches;
}

// chooses one occupied neighbor to inherit a new neuron's fixed properties
uint32_t choose_occupied_neighbor(
    GridShape shape,
    uint32_t cell,
    span<const uint8_t> occupied,
    span<const GridOffset> offsets,
    EvolutionRandom& random)
{
    vector<uint32_t> neighbors;
    visit_neighbors(shape, cell, offsets, [&](uint32_t neighbor)
    {
        if (occupied[neighbor] != 0U)
        {
            neighbors.push_back(neighbor);
        }
    });
    require(!neighbors.empty(), "new neuron does not touch the inherited brain");
    return neighbors[random.range_u32(0U, static_cast<uint32_t>(neighbors.size() - 1U))];
}

// returns a bounded mutation count for an ordered channel list
uint32_t mutation_count(size_t count, float fraction)
{
    if (count == 0U || fraction == 0.0f)
    {
        return 0U;
    }

    return min(static_cast<uint32_t>(count), max(1U,
        static_cast<uint32_t>(lround(static_cast<double>(count) * fraction))));
}

// chooses one nearby unused occupied cell for an interface mutation
uint32_t choose_local_interface_cell(
    GridShape shape,
    uint32_t previous,
    span<const uint8_t> occupied,
    span<const uint32_t> used,
    float sigma_steps,
    EvolutionRandom& random)
{
    const GridPosition origin = expand(shape, previous);

    for (uint32_t attempt = 0U; attempt < 32U; ++attempt)
    {
        const double x_offset = static_cast<double>(random.bell_sample()) * sigma_steps;
        const double y_offset = static_cast<double>(random.bell_sample()) * sigma_steps;
        const double z_offset = static_cast<double>(random.bell_sample()) * sigma_steps;

        if (abs(x_offset) > shape.width
            || abs(y_offset) > shape.height
            || abs(z_offset) > shape.depth)
        {
            continue;
        }

        const int64_t x = static_cast<int64_t>(origin.x) + llround(x_offset);
        const int64_t y = static_cast<int64_t>(origin.y) + llround(y_offset);
        const int64_t z = static_cast<int64_t>(origin.z) + llround(z_offset);

        if (x < 0 || y < 0 || z < 0
            || x >= shape.width || y >= shape.height || z >= shape.depth)
        {
            continue;
        }

        const uint32_t candidate = flatten(shape, {
            static_cast<uint32_t>(x),
            static_cast<uint32_t>(y),
            static_cast<uint32_t>(z)
        });

        if (candidate != previous && occupied[candidate] != 0U && used[candidate] == 0U)
        {
            return candidate;
        }
    }

    uint64_t nearest_distance = numeric_limits<uint64_t>::max();
    vector<uint32_t> nearest;

    for (uint32_t cell = 0U; cell < occupied.size(); ++cell)
    {
        if (cell == previous || occupied[cell] == 0U || used[cell] != 0U)
        {
            continue;
        }

        const GridPosition position = expand(shape, cell);
        const uint64_t distance = static_cast<uint64_t>(
            position.x > origin.x ? position.x - origin.x : origin.x - position.x)
            + static_cast<uint64_t>(
                position.y > origin.y ? position.y - origin.y : origin.y - position.y)
            + static_cast<uint64_t>(
                position.z > origin.z ? position.z - origin.z : origin.z - position.z);

        if (distance < nearest_distance)
        {
            nearest_distance = distance;
            nearest.clear();
        }

        if (distance == nearest_distance)
        {
            nearest.push_back(cell);
        }
    }

    if (nearest.empty())
    {
        return previous;
    }

    return nearest[random.range_u32(0U, static_cast<uint32_t>(nearest.size() - 1U))];
}

// moves one interface channel onto a nearby unused occupied neuron
void move_channel_position(
    GridPosition& position,
    GridShape shape,
    span<const uint8_t> occupied,
    vector<uint32_t>& used,
    float sigma_steps,
    EvolutionRandom& random)
{
    const uint32_t previous = flatten(shape, position);
    require(used[previous] > 0U, "interface usage accounting is invalid");
    --used[previous];
    const uint32_t selected = choose_local_interface_cell(
        shape, previous, occupied, used, sigma_steps, random);
    position = expand(shape, selected);
    ++used[selected];
}

// moves one paired input and output onto the same nearby neuron
void move_paired_position(
    GridPosition& input,
    GridPosition& output,
    GridShape shape,
    span<const uint8_t> occupied,
    vector<uint32_t>& used,
    float sigma_steps,
    EvolutionRandom& random)
{
    require(input == output, "paired interface positions do not match");
    const uint32_t previous = flatten(shape, input);
    require(used[previous] >= 2U, "paired interface usage accounting is invalid");
    used[previous] -= 2U;
    const uint32_t selected = choose_local_interface_cell(
        shape, previous, occupied, used, sigma_steps, random);
    input = expand(shape, selected);
    output = input;
    used[selected] += 2U;
}

// combines two endpoint cells into one stable directed connection key
uint64_t connection_key(uint32_t source, uint32_t target)
{
    return (static_cast<uint64_t>(source) << 32U) | target;
}

// validates configurable mutation ranges before changing a genome
void validate_mutation_config(const EvolutionMutationConfig& config)
{
    require(config.minimum_neurons > 0U, "minimum evolution neuron count must be positive");
    require(config.minimum_neurons <= config.maximum_neurons,
        "evolution neuron count bounds are reversed");
    require(isfinite(config.neuron_count_sigma_fraction)
        && config.neuron_count_sigma_fraction >= 0.0f,
        "neuron count mutation sigma must be finite and nonnegative");
    require(isfinite(config.input_position_mutation_fraction)
        && config.input_position_mutation_fraction >= 0.0f
        && config.input_position_mutation_fraction <= 1.0f,
        "input position mutation fraction must be within zero and one");
    require(isfinite(config.output_position_mutation_fraction)
        && config.output_position_mutation_fraction >= 0.0f
        && config.output_position_mutation_fraction <= 1.0f,
        "output position mutation fraction must be within zero and one");
    require(isfinite(config.interface_position_sigma_steps)
        && config.interface_position_sigma_steps >= 0.0f,
        "interface position mutation sigma must be finite and nonnegative");
    require(isfinite(config.connection_mutation_fraction)
        && config.connection_mutation_fraction >= 0.0f
        && config.connection_mutation_fraction <= 1.0f,
        "connection mutation fraction must be within zero and one");
    require(isfinite(config.connection_strength_sigma_fraction)
        && config.connection_strength_sigma_fraction >= 0.0f,
        "connection strength mutation sigma must be finite and nonnegative");
}

// returns a normally perturbed neuron count within reachable configured bounds
uint32_t choose_neuron_count(
    uint32_t parent_count,
    uint32_t minimum,
    uint32_t maximum,
    float sigma_fraction,
    EvolutionRandom& random)
{
    if (parent_count < minimum)
    {
        return minimum;
    }

    if (parent_count > maximum)
    {
        return maximum;
    }

    const double sigma = static_cast<double>(parent_count) * sigma_fraction;
    const bool can_decrease = parent_count > minimum;
    const bool can_increase = parent_count < maximum;

    if (sigma_fraction == 0.0f || (!can_decrease && !can_increase))
    {
        return clamp(parent_count, minimum, maximum);
    }

    for (uint32_t attempt = 0U; attempt < 8U; ++attempt)
    {
        const double sample = static_cast<double>(parent_count)
            + static_cast<double>(random.bell_sample()) * sigma;
        const double bounded = clamp(sample, static_cast<double>(minimum),
            static_cast<double>(maximum));
        const uint32_t candidate = static_cast<uint32_t>(llround(bounded));

        if (candidate != parent_count)
        {
            return candidate;
        }
    }

    const uint64_t maximum_delta = max<uint64_t>(
        static_cast<uint64_t>(parent_count) - minimum,
        static_cast<uint64_t>(maximum) - parent_count);
    const uint64_t fallback_delta = max<uint64_t>(1U, static_cast<uint64_t>(llround(
        min(sigma * 0.5, static_cast<double>(maximum_delta)))));
    bool increase = random.unit_sample() >= 0.5f;

    if (!can_decrease)
    {
        increase = true;
    }
    else if (!can_increase)
    {
        increase = false;
    }

    if (increase)
    {
        return static_cast<uint32_t>(min<uint64_t>(
            maximum, static_cast<uint64_t>(parent_count) + fallback_delta));
    }

    return static_cast<uint32_t>(max<int64_t>(minimum,
        static_cast<int64_t>(parent_count) - static_cast<int64_t>(fallback_delta)));
}

// mutates potential strengths while preserving every untouched inherited weight
void mutate_connection_strengths(
    const BrainModel& model,
    vector<float>& strengths,
    float mutation_fraction,
    float sigma_fraction,
    EvolutionRandom& random)
{
    const uint32_t edge_count = static_cast<uint32_t>(strengths.size());
    const uint32_t mutations = mutation_count(edge_count, mutation_fraction);

    if (mutations == 0U || sigma_fraction == 0.0f)
    {
        return;
    }

    vector<uint32_t> edges(edge_count);
    iota(edges.begin(), edges.end(), 0U);
    shuffle_values(edges, random);
    vector<uint8_t> mutated(edge_count, 0U);
    const float budget = model.config.plasticity.outgoing_strength_budget;

    for (uint32_t mutation = 0; mutation < mutations; ++mutation)
    {
        const uint32_t edge = edges[mutation];
        const uint32_t source = model.edge_sources[edge];
        const uint32_t degree = model.outgoing_offsets[source + 1U]
            - model.outgoing_offsets[source];
        const float local_scale = max(strengths[edge], budget / static_cast<float>(degree));
        const double changed = static_cast<double>(strengths[edge])
            + static_cast<double>(random.bell_sample()) * sigma_fraction * local_scale;
        strengths[edge] = static_cast<float>(clamp(
            changed, 0.0, static_cast<double>(budget)));
        mutated[edge] = 1U;
    }

    for (uint32_t source = 0; source < model.positions.size(); ++source)
    {
        const uint32_t begin = model.outgoing_offsets[source];
        const uint32_t end = model.outgoing_offsets[source + 1U];
        float fixed_sum = 0.0f;
        float mutable_sum = 0.0f;

        for (uint32_t edge = begin; edge < end; ++edge)
        {
            if (mutated[edge] != 0U)
            {
                mutable_sum += strengths[edge];
            }
            else
            {
                fixed_sum += strengths[edge];
            }
        }

        const float available = max(0.0f, budget - fixed_sum);

        if (mutable_sum <= available || mutable_sum == 0.0f)
        {
            continue;
        }

        const float scale = available / mutable_sum;

        for (uint32_t edge = begin; edge < end; ++edge)
        {
            if (mutated[edge] != 0U)
            {
                strengths[edge] *= scale;
            }
        }
    }
}

}

// creates one deterministic connected mutation of an inherited brain genome
BrainGenome mutate_brain_genome(
    const BrainGenome& parent,
    span<const GridPosition> allowed_positions,
    uint64_t mutation_seed,
    const EvolutionMutationConfig& config,
    const EvolutionInterfaceConstraints& constraints)
{
    validate_brain_genome(parent);
    validate_mutation_config(config);
    vector<uint8_t> paired_inputs(parent.input_positions.size(), 0U);
    vector<uint32_t> paired_input_by_output(
        parent.output_positions.size(), INVALID_NEURON_INDEX);

    for (const EvolutionInterfacePair pair : constraints.paired_positions)
    {
        require(pair.input < parent.input_positions.size()
                && pair.output < parent.output_positions.size(),
            "evolution interface pair index is out of range");
        require(paired_inputs[pair.input] == 0U
                && paired_input_by_output[pair.output] == INVALID_NEURON_INDEX,
            "evolution interface constraints reuse a channel");
        require(parent.input_positions[pair.input] == parent.output_positions[pair.output],
            "paired evolution interface positions do not match");
        paired_inputs[pair.input] = 1U;
        paired_input_by_output[pair.output] = pair.input;
    }

    const uint32_t total_cells = cell_count(parent.shape);
    vector<uint8_t> allowed(total_cells, 0U);

    for (const GridPosition position : allowed_positions)
    {
        require(is_in_bounds(parent.shape, position),
            "allowed evolution position is outside the brain grid");
        const uint32_t cell = flatten(parent.shape, position);
        require(allowed[cell] == 0U, "allowed evolution positions contain a duplicate");
        allowed[cell] = 1U;
    }

    require(!allowed_positions.empty(), "evolution requires at least one allowed position");
    vector<uint8_t> occupied(total_cells, 0U);
    vector<uint8_t> inherited(total_cells, 0U);
    vector<uint8_t> protected_cells(total_cells, 0U);
    vector<NeuronDefinition> definitions(total_cells);

    for (const NeuronDefinition& neuron : parent.neurons)
    {
        const uint32_t cell = flatten(parent.shape, neuron.position);
        require(allowed[cell] != 0U, "parent neuron is outside the allowed evolution space");
        occupied[cell] = 1U;
        inherited[cell] = 1U;
        definitions[cell] = neuron;
    }

    for (const GridPosition position : parent.input_positions)
    {
        protected_cells[flatten(parent.shape, position)] = 1U;
    }

    for (const GridPosition position : parent.output_positions)
    {
        protected_cells[flatten(parent.shape, position)] = 1U;
    }

    const uint32_t protected_count = static_cast<uint32_t>(count(
        protected_cells.begin(), protected_cells.end(), static_cast<uint8_t>(1U)));
    const uint32_t minimum = max(config.minimum_neurons, protected_count);
    const uint32_t maximum = min(config.maximum_neurons,
        static_cast<uint32_t>(allowed_positions.size()));
    require(minimum <= maximum, "evolution neuron bounds cannot fit the required interfaces");

    EvolutionRandom random(mutation_seed);
    uint32_t occupied_count = static_cast<uint32_t>(parent.neurons.size());
    const uint32_t target_count = choose_neuron_count(occupied_count, minimum, maximum,
        config.neuron_count_sigma_fraction, random);
    const vector<GridOffset> offsets = make_offsets(
        parent.shape, parent.config.connection_radius_steps);
    require(remains_connected(parent.shape, occupied, occupied_count,
            INVALID_NEURON_INDEX, offsets),
        "parent brain must be connected before mutation");

    while (occupied_count > target_count)
    {
        vector<uint32_t> candidates;

        for (uint32_t cell = 0; cell < total_cells; ++cell)
        {
            if (occupied[cell] != 0U && protected_cells[cell] == 0U)
            {
                candidates.push_back(cell);
            }
        }

        shuffle_values(candidates, random);
        uint32_t removed = INVALID_NEURON_INDEX;

        for (const uint32_t candidate : candidates)
        {
            if (remains_connected(parent.shape, occupied, occupied_count, candidate, offsets))
            {
                removed = candidate;
                break;
            }
        }

        if (removed == INVALID_NEURON_INDEX)
        {
            break;
        }

        occupied[removed] = 0U;
        --occupied_count;
    }

    while (occupied_count < target_count)
    {
        vector<uint32_t> candidates;

        for (uint32_t cell = 0; cell < total_cells; ++cell)
        {
            if (allowed[cell] != 0U && occupied[cell] == 0U
                && touches_occupied(parent.shape, cell, occupied, offsets))
            {
                candidates.push_back(cell);
            }
        }

        require(!candidates.empty(),
            "cannot expand the brain within the allowed connected space");
        const uint32_t added = candidates[random.range_u32(
            0U, static_cast<uint32_t>(candidates.size() - 1U))];
        const uint32_t source = choose_occupied_neighbor(
            parent.shape, added, occupied, offsets, random);
        definitions[added] = definitions[source];
        definitions[added].position = expand(parent.shape, added);
        occupied[added] = 1U;
        ++occupied_count;
    }

    BrainGenome child;
    child.seed = mutation_seed;
    child.config = parent.config;
    child.shape = parent.shape;
    child.input_positions = parent.input_positions;
    child.output_positions = parent.output_positions;
    child.neurons.reserve(occupied_count);

    for (uint32_t cell = 0; cell < total_cells; ++cell)
    {
        if (occupied[cell] != 0U)
        {
            child.neurons.push_back(definitions[cell]);
        }
    }

    vector<uint32_t> used(total_cells, 0U);

    for (const GridPosition position : child.input_positions)
    {
        ++used[flatten(child.shape, position)];
    }

    for (const GridPosition position : child.output_positions)
    {
        ++used[flatten(child.shape, position)];
    }

    vector<uint32_t> input_channels;

    for (uint32_t input = 0U; input < child.input_positions.size(); ++input)
    {
        if (paired_inputs[input] == 0U)
        {
            input_channels.push_back(input);
        }
    }

    shuffle_values(input_channels, random);
    const uint32_t input_mutations = mutation_count(
        input_channels.size(), config.input_position_mutation_fraction);

    for (uint32_t mutation = 0U; mutation < input_mutations; ++mutation)
    {
        move_channel_position(
            child.input_positions[input_channels[mutation]],
            child.shape,
            occupied,
            used,
            config.interface_position_sigma_steps,
            random);
    }

    vector<uint32_t> output_channels(child.output_positions.size());
    iota(output_channels.begin(), output_channels.end(), 0U);
    shuffle_values(output_channels, random);
    const uint32_t output_mutations = mutation_count(
        output_channels.size(), config.output_position_mutation_fraction);

    for (uint32_t mutation = 0U; mutation < output_mutations; ++mutation)
    {
        const uint32_t output = output_channels[mutation];
        const uint32_t input = paired_input_by_output[output];

        if (input == INVALID_NEURON_INDEX)
        {
            move_channel_position(
                child.output_positions[output],
                child.shape,
                occupied,
                used,
                config.interface_position_sigma_steps,
                random);
        }
        else
        {
            move_paired_position(
                child.input_positions[input],
                child.output_positions[output],
                child.shape,
                occupied,
                used,
                config.interface_position_sigma_steps,
                random);
        }
    }

    BrainLayout layout;
    layout.shape = child.shape;
    layout.neurons = child.neurons;
    layout.input_positions = child.input_positions;
    layout.output_positions = child.output_positions;
    const BrainModel child_model = build_brain_model(layout, child.config, child.seed);
    unordered_map<uint64_t, float> inherited_strengths;
    inherited_strengths.reserve(parent.connections.size());

    for (const BrainConnectionGene& connection : parent.connections)
    {
        inherited_strengths.emplace(connection_key(flatten(parent.shape, connection.source),
            flatten(parent.shape, connection.target)), connection.strength);
    }

    vector<float> strengths(child_model.outgoing_targets.size(), 0.0f);

    for (uint32_t edge = 0; edge < strengths.size(); ++edge)
    {
        const uint32_t source = child_model.edge_sources[edge];
        const uint32_t target = child_model.outgoing_targets[edge];
        const uint32_t source_cell = flatten(child.shape, child_model.positions[source]);
        const uint32_t target_cell = flatten(child.shape, child_model.positions[target]);
        const auto found = inherited_strengths.find(connection_key(source_cell, target_cell));

        if (found != inherited_strengths.end())
        {
            strengths[edge] = found->second;
        }
        else if (inherited[source_cell] == 0U)
        {
            strengths[edge] = child_model.initial_connection_strengths[edge];
        }
    }

    mutate_connection_strengths(child_model, strengths,
        config.connection_mutation_fraction,
        config.connection_strength_sigma_fraction,
        random);

    for (uint32_t edge = 0; edge < strengths.size(); ++edge)
    {
        if (strengths[edge] > 0.0f)
        {
            child.connections.push_back({
                child_model.positions[child_model.edge_sources[edge]],
                child_model.positions[child_model.outgoing_targets[edge]],
                strengths[edge]
            });
        }
    }

    validate_brain_genome(child);
    return child;
}

// validates finite nonnegative fitness values before comparison
void validate_evolution_fitness(const EvolutionFitness& fitness)
{
    require(isfinite(fitness.total_episode_seconds)
        && fitness.total_episode_seconds >= 0.0,
        "evolution episode duration must be finite and nonnegative");
}

// returns whether the left fitness ranks ahead of the right fitness
bool better_evolution_fitness(const EvolutionFitness& left, const EvolutionFitness& right)
{
    validate_evolution_fitness(left);
    validate_evolution_fitness(right);

    if (left.total_score != right.total_score)
    {
        return left.total_score > right.total_score;
    }

    if (left.successful_episodes != right.successful_episodes)
    {
        return left.successful_episodes > right.successful_episodes;
    }

    if (left.best_episode_score != right.best_episode_score)
    {
        return left.best_episode_score > right.best_episode_score;
    }

    if (left.total_episode_seconds != right.total_episode_seconds)
    {
        return left.total_episode_seconds > right.total_episode_seconds;
    }

    if (left.penalty_events != right.penalty_events)
    {
        return left.penalty_events < right.penalty_events;
    }

    return left.genome_fingerprint < right.genome_fingerprint;
}

}
