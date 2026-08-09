#include "neurosim/dungeon.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace neurosim
{

using namespace std;

namespace
{

// provides deterministic project-owned random sampling for dungeon generation
class PcgRandom
{
public:
    // seeds the generator into a stable initial state
    explicit PcgRandom(uint64_t seed)
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
            throw invalid_argument("invalid integer random range");
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

    // returns a stable 24-bit float sample within a half-open range
    float range_float(float minimum, float maximum)
    {
        const float unit = static_cast<float>(next_u32() >> 8U) * (1.0f / 16777216.0f);
        return minimum + (maximum - minimum) * unit;
    }

    // returns whether a probability trial succeeds
    bool chance(float probability)
    {
        return range_float(0.0f, 1.0f) < probability;
    }

private:
    uint64_t state_ = 0;
};

// throws when a dungeon invariant is false
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

// creates an axis-aligned rectangle around a center point
Rect centered_rect(Vec2 center, float width, float height)
{
    return {
        center.x - width * 0.5f,
        center.y - height * 0.5f,
        center.x + width * 0.5f,
        center.y + height * 0.5f
    };
}

// creates a varied compound room around the local origin
Room make_room_shape(PcgRandom& random, const DungeonConfig& config, float player_size)
{
    const float hub_width = random.range_float(config.min_hub_width, config.max_hub_width);
    const float hub_height = random.range_float(config.min_hub_height, config.max_hub_height);
    const Rect hub = centered_rect({}, hub_width, hub_height);

    Room room;
    room.center = {};
    room.bounds = hub;
    room.floor_rects.push_back(hub);

    const uint32_t wing_count = random.range_u32(config.min_wings, config.max_wings);
    const float seam_depth = player_size + 0.5f;

    for (uint32_t index = 0; index < wing_count; ++index)
    {
        const uint32_t side = random.range_u32(0, 3);
        const float depth = random.range_float(config.min_wing_depth, config.max_wing_depth);
        Rect wing;

        if (side <= 1)
        {
            const float ratio = random.range_float(config.min_wing_span_ratio, config.max_wing_span_ratio);
            const float span = min(hub_width, max(player_size * 2.5f, hub_width * ratio));
            const float offset_limit = (hub_width - span) * 0.5f;
            const float offset = random.range_float(-offset_limit, offset_limit);
            const float center_x = offset;

            if (side == 0)
            {
                wing = {
                    center_x - span * 0.5f,
                    hub.min_y - depth,
                    center_x + span * 0.5f,
                    hub.min_y + seam_depth
                };
            }
            else
            {
                wing = {
                    center_x - span * 0.5f,
                    hub.max_y - seam_depth,
                    center_x + span * 0.5f,
                    hub.max_y + depth
                };
            }
        }
        else
        {
            const float ratio = random.range_float(config.min_wing_span_ratio, config.max_wing_span_ratio);
            const float span = min(hub_height, max(player_size * 2.5f, hub_height * ratio));
            const float offset_limit = (hub_height - span) * 0.5f;
            const float offset = random.range_float(-offset_limit, offset_limit);
            const float center_y = offset;

            if (side == 2)
            {
                wing = {
                    hub.min_x - depth,
                    center_y - span * 0.5f,
                    hub.min_x + seam_depth,
                    center_y + span * 0.5f
                };
            }
            else
            {
                wing = {
                    hub.max_x - seam_depth,
                    center_y - span * 0.5f,
                    hub.max_x + depth,
                    center_y + span * 0.5f
                };
            }
        }

        room.floor_rects.push_back(wing);
        room.bounds = joined(room.bounds, wing);
    }

    return room;
}

// translates all geometry belonging to one room
void translate_room(Room& room, Vec2 offset)
{
    room.center = room.center + offset;
    room.bounds = room.bounds.translated(offset);

    for (Rect& rect : room.floor_rects)
    {
        rect = rect.translated(offset);
    }
}

// returns whether a padded room envelope overlaps an existing room
bool overlaps_existing_room(const Room& candidate, const vector<Room>& rooms, float spacing)
{
    const Rect padded = candidate.bounds.expanded(spacing);

    for (const Room& room : rooms)
    {
        if (padded.intersects(room.bounds))
        {
            return true;
        }
    }

    return false;
}

// creates a padded horizontal corridor segment between two points
Rect horizontal_segment(Vec2 start, Vec2 end, float width)
{
    const float half_width = width * 0.5f;
    return {
        min(start.x, end.x) - half_width,
        start.y - half_width,
        max(start.x, end.x) + half_width,
        start.y + half_width
    };
}

// creates a padded vertical corridor segment between two points
Rect vertical_segment(Vec2 start, Vec2 end, float width)
{
    const float half_width = width * 0.5f;
    return {
        start.x - half_width,
        min(start.y, end.y) - half_width,
        start.x + half_width,
        max(start.y, end.y) + half_width
    };
}

// appends corridor geometry to both render and navigation collections
void append_corridor(Dungeon& dungeon, const Rect& rect)
{
    dungeon.corridors.push_back(rect);
    dungeon.floor_rects.push_back(rect);
}

// connects two room centers with a randomized orthogonal corridor path
void connect_rooms(Dungeon& dungeon, size_t first, size_t second, const DungeonConfig& config, PcgRandom& random)
{
    const Vec2 start = dungeon.rooms[first].center;
    const Vec2 end = dungeon.rooms[second].center;
    const float width = random.range_float(config.min_corridor_width, config.max_corridor_width);
    const bool horizontal_first = random.chance(0.5f);
    const bool two_bends = random.chance(config.two_bend_chance);

    if (two_bends && horizontal_first)
    {
        const float blend = random.range_float(0.25f, 0.75f);
        const float middle_x = start.x + (end.x - start.x) * blend;
        const Vec2 first_turn{middle_x, start.y};
        const Vec2 second_turn{middle_x, end.y};
        append_corridor(dungeon, horizontal_segment(start, first_turn, width));
        append_corridor(dungeon, vertical_segment(first_turn, second_turn, width));
        append_corridor(dungeon, horizontal_segment(second_turn, end, width));
    }
    else if (two_bends)
    {
        const float blend = random.range_float(0.25f, 0.75f);
        const float middle_y = start.y + (end.y - start.y) * blend;
        const Vec2 first_turn{start.x, middle_y};
        const Vec2 second_turn{end.x, middle_y};
        append_corridor(dungeon, vertical_segment(start, first_turn, width));
        append_corridor(dungeon, horizontal_segment(first_turn, second_turn, width));
        append_corridor(dungeon, vertical_segment(second_turn, end, width));
    }
    else if (horizontal_first)
    {
        const Vec2 turn{end.x, start.y};
        append_corridor(dungeon, horizontal_segment(start, turn, width));
        append_corridor(dungeon, vertical_segment(turn, end, width));
    }
    else
    {
        const Vec2 turn{start.x, end.y};
        append_corridor(dungeon, vertical_segment(start, turn, width));
        append_corridor(dungeon, horizontal_segment(turn, end, width));
    }
}

// builds a spanning tree and optional loop connections between rooms
void add_connections(Dungeon& dungeon, const DungeonConfig& config, PcgRandom& random)
{
    const size_t room_count = dungeon.rooms.size();
    vector<bool> connected(room_count, false);
    set<pair<size_t, size_t>> edges;
    connected[0] = true;

    for (size_t connected_count = 1; connected_count < room_count; ++connected_count)
    {
        float best_distance = numeric_limits<float>::infinity();
        size_t best_first = 0;
        size_t best_second = 0;

        for (size_t first = 0; first < room_count; ++first)
        {
            if (!connected[first])
            {
                continue;
            }

            for (size_t second = 0; second < room_count; ++second)
            {
                if (connected[second])
                {
                    continue;
                }

                const float candidate_distance = distance_squared(dungeon.rooms[first].center, dungeon.rooms[second].center);

                if (candidate_distance < best_distance)
                {
                    best_distance = candidate_distance;
                    best_first = first;
                    best_second = second;
                }
            }
        }

        connected[best_second] = true;
        const auto edge = minmax(best_first, best_second);
        edges.emplace(edge.first, edge.second);
        connect_rooms(dungeon, best_first, best_second, config, random);
    }

    const size_t extra_count = static_cast<size_t>(lround(
        static_cast<double>(room_count - 1) * static_cast<double>(config.extra_connection_ratio)));
    const size_t max_attempts = room_count * room_count * 4;
    size_t added = 0;

    for (size_t attempt = 0; attempt < max_attempts && added < extra_count; ++attempt)
    {
        const size_t first = random.range_u32(0, static_cast<uint32_t>(room_count - 1));
        const size_t second = random.range_u32(0, static_cast<uint32_t>(room_count - 1));

        if (first == second)
        {
            continue;
        }

        const auto edge = minmax(first, second);

        if (!edges.emplace(edge.first, edge.second).second)
        {
            continue;
        }

        connect_rooms(dungeon, first, second, config, random);
        ++added;
    }
}

// selects a room floor rectangle with probability proportional to usable area
const Rect* choose_room_piece(const Room& room, float inset, PcgRandom& random)
{
    float total_area = 0.0f;

    for (const Rect& rect : room.floor_rects)
    {
        const Rect navigable = rect.inset(inset);

        if (navigable.width() > 0.0f && navigable.height() > 0.0f)
        {
            total_area += navigable.width() * navigable.height();
        }
    }

    if (total_area <= 0.0f)
    {
        return nullptr;
    }

    float selection = random.range_float(0.0f, total_area);

    for (const Rect& rect : room.floor_rects)
    {
        const Rect navigable = rect.inset(inset);
        const float area = max(0.0f, navigable.width()) * max(0.0f, navigable.height());

        if (selection <= area)
        {
            return &rect;
        }

        selection -= area;
    }

    return &room.floor_rects.front();
}

// scatters reachable and separated pellets throughout every room
void place_pellets(Dungeon& dungeon, const DungeonConfig& config, float player_size, PcgRandom& random)
{
    const float inset = player_size * 0.5f + NAVIGATION_EPSILON;
    const float minimum_spacing_squared = config.pellet_min_spacing * config.pellet_min_spacing;
    const float spawn_clearance_squared = config.spawn_clearance * config.spawn_clearance;

    for (size_t room_index = 0; room_index < dungeon.rooms.size(); ++room_index)
    {
        const Room& room = dungeon.rooms[room_index];
        const uint32_t pellet_count = random.range_u32(
            config.min_pellets_per_room,
            config.max_pellets_per_room);

        for (uint32_t pellet_index = 0; pellet_index < pellet_count; ++pellet_index)
        {
            bool found_candidate = false;
            Vec2 best_position;
            float best_clearance_squared = -1.0f;

            for (uint32_t attempt = 0; attempt < config.pellet_placement_attempts; ++attempt)
            {
                const Rect* piece = choose_room_piece(room, inset, random);

                if (piece == nullptr)
                {
                    break;
                }

                const Rect navigable = piece->inset(inset);
                const Vec2 position{
                    random.range_float(navigable.min_x, navigable.max_x),
                    random.range_float(navigable.min_y, navigable.max_y)
                };

                if (room_index == 0 && distance_squared(position, dungeon.spawn) < spawn_clearance_squared)
                {
                    continue;
                }

                float clearance_squared = numeric_limits<float>::infinity();

                for (const Pellet& pellet : dungeon.pellets)
                {
                    clearance_squared = min(
                        clearance_squared,
                        distance_squared(position, pellet.position));
                }

                if (!found_candidate || clearance_squared > best_clearance_squared)
                {
                    found_candidate = true;
                    best_position = position;
                    best_clearance_squared = clearance_squared;
                }
            }

            if (!found_candidate || best_clearance_squared < minimum_spacing_squared)
            {
                throw runtime_error("failed to place dungeon pellet with the requested clearances");
            }

            dungeon.pellets.push_back({best_position});
        }
    }
}

// returns floor rectangles inset to valid player-center regions
vector<Rect> navigable_rects(const Dungeon& dungeon, float player_size)
{
    vector<Rect> result;
    result.reserve(dungeon.floor_rects.size());
    const float inset = player_size * 0.5f + NAVIGATION_EPSILON;

    for (const Rect& rect : dungeon.floor_rects)
    {
        const Rect navigable = rect.inset(inset);

        if (navigable.width() <= 0.0f || navigable.height() <= 0.0f)
        {
            throw invalid_argument("floor rectangle is too small for the player");
        }

        result.push_back(navigable);
    }

    return result;
}

// returns whether two player-center regions overlap with usable area
bool has_traversable_overlap(const Rect& first, const Rect& second)
{
    const float overlap_x = min(first.max_x, second.max_x) - max(first.min_x, second.min_x);
    const float overlap_y = min(first.max_y, second.max_y) - max(first.min_y, second.min_y);
    return overlap_x > NAVIGATION_EPSILON && overlap_y > NAVIGATION_EPSILON;
}

}

// validates generation settings against the configured player size
void validate_dungeon_config(const DungeonConfig& config, float player_size)
{
    require(finite(player_size) && player_size > 0.0f, "player size must be finite and positive");
    require(config.min_rooms >= 2 && config.max_rooms >= config.min_rooms, "invalid room count range");
    require(config.max_rooms <= 10000, "room count is unreasonably large");
    require(finite(config.min_hub_width) && finite(config.max_hub_width)
        && config.min_hub_width > player_size && config.max_hub_width >= config.min_hub_width,
        "invalid hub width range");
    require(finite(config.min_hub_height) && finite(config.max_hub_height)
        && config.min_hub_height > player_size && config.max_hub_height >= config.min_hub_height,
        "invalid hub height range");
    require(config.max_wings >= config.min_wings, "invalid wing count range");
    require(finite(config.min_wing_depth) && finite(config.max_wing_depth)
        && config.min_wing_depth > 0.0f && config.max_wing_depth >= config.min_wing_depth,
        "invalid wing depth range");
    require(finite(config.min_wing_span_ratio) && finite(config.max_wing_span_ratio)
        && config.min_wing_span_ratio > 0.0f && config.max_wing_span_ratio >= config.min_wing_span_ratio
        && config.max_wing_span_ratio <= 1.0f,
        "invalid wing span ratio range");
    require(finite(config.min_room_gap) && finite(config.max_room_gap)
        && config.min_room_gap >= 0.0f && config.max_room_gap >= config.min_room_gap,
        "invalid room gap range");
    require(finite(config.room_spacing) && config.room_spacing >= 0.0f, "invalid room spacing");
    require(finite(config.min_corridor_width) && finite(config.max_corridor_width)
        && config.min_corridor_width > player_size + 2.0f * NAVIGATION_EPSILON
        && config.max_corridor_width >= config.min_corridor_width,
        "invalid corridor width range");
    require(finite(config.extra_connection_ratio) && config.extra_connection_ratio >= 0.0f
        && config.extra_connection_ratio <= 1.0f,
        "invalid extra connection ratio");
    require(finite(config.two_bend_chance) && config.two_bend_chance >= 0.0f
        && config.two_bend_chance <= 1.0f,
        "invalid two bend chance");
    require(config.min_pellets_per_room >= 1
        && config.max_pellets_per_room >= config.min_pellets_per_room,
        "invalid pellet count range");
    require(static_cast<uint64_t>(config.max_rooms) * config.max_pellets_per_room
        <= numeric_limits<uint32_t>::max(),
        "pellet capacity exceeds environment counters");
    require(finite(config.pellet_min_spacing) && config.pellet_min_spacing >= 0.0f,
        "invalid pellet spacing");
    require(finite(config.spawn_clearance) && config.spawn_clearance >= 0.0f,
        "invalid spawn clearance");
    require(config.placement_attempts_per_room > 0, "room placement attempts must be positive");
    require(config.pellet_placement_attempts > 0, "pellet placement attempts must be positive");
}

// generates a deterministic continuous-space dungeon from a seed
Dungeon generate_dungeon(const DungeonConfig& config, float player_size, uint64_t seed)
{
    validate_dungeon_config(config, player_size);
    PcgRandom random(seed);
    Dungeon dungeon;
    dungeon.seed = seed;

    const uint32_t room_count = random.range_u32(config.min_rooms, config.max_rooms);
    dungeon.rooms.reserve(room_count);
    dungeon.rooms.push_back(make_room_shape(random, config, player_size));

    for (uint32_t room_index = 1; room_index < room_count; ++room_index)
    {
        bool placed = false;

        for (uint32_t attempt = 0; attempt < config.placement_attempts_per_room; ++attempt)
        {
            Room candidate = make_room_shape(random, config, player_size);
            const Room& anchor = dungeon.rooms[random.range_u32(0, static_cast<uint32_t>(dungeon.rooms.size() - 1))];
            const uint32_t side = random.range_u32(0, 3);
            const float gap = random.range_float(config.min_room_gap, config.max_room_gap);
            Vec2 offset;

            if (side <= 1)
            {
                const float jitter_limit = (anchor.bounds.height() + candidate.bounds.height()) * 0.35f;
                const float desired_y = anchor.center.y + random.range_float(-jitter_limit, jitter_limit);
                offset.y = desired_y - candidate.center.y;

                if (side == 0)
                {
                    offset.x = anchor.bounds.min_x - gap - candidate.bounds.max_x;
                }
                else
                {
                    offset.x = anchor.bounds.max_x + gap - candidate.bounds.min_x;
                }
            }
            else
            {
                const float jitter_limit = (anchor.bounds.width() + candidate.bounds.width()) * 0.35f;
                const float desired_x = anchor.center.x + random.range_float(-jitter_limit, jitter_limit);
                offset.x = desired_x - candidate.center.x;

                if (side == 2)
                {
                    offset.y = anchor.bounds.min_y - gap - candidate.bounds.max_y;
                }
                else
                {
                    offset.y = anchor.bounds.max_y + gap - candidate.bounds.min_y;
                }
            }

            translate_room(candidate, offset);

            if (overlaps_existing_room(candidate, dungeon.rooms, config.room_spacing))
            {
                continue;
            }

            dungeon.rooms.push_back(move(candidate));
            placed = true;
            break;
        }

        if (!placed)
        {
            throw runtime_error("failed to place all requested dungeon rooms");
        }
    }

    for (const Room& room : dungeon.rooms)
    {
        dungeon.floor_rects.insert(dungeon.floor_rects.end(), room.floor_rects.begin(), room.floor_rects.end());
    }

    dungeon.spawn = dungeon.rooms.front().center;
    add_connections(dungeon, config, random);
    place_pellets(dungeon, config, player_size, random);

    dungeon.bounds = dungeon.floor_rects.front();

    for (const Rect& rect : dungeon.floor_rects)
    {
        dungeon.bounds = joined(dungeon.bounds, rect);
    }

    dungeon.bounds = dungeon.bounds.expanded(4.0f);
    validate_dungeon(dungeon, player_size);
    return dungeon;
}

// returns whether a player center fits within the dungeon floor
bool is_navigable(const Dungeon& dungeon, Vec2 position, float player_size)
{
    if (!is_finite(position) || !isfinite(player_size) || player_size <= 0.0f)
    {
        return false;
    }

    const float inset = player_size * 0.5f + NAVIGATION_EPSILON;

    for (const Rect& rect : dungeon.floor_rects)
    {
        const Rect navigable = rect.inset(inset);

        if (navigable.width() > 0.0f && navigable.height() > 0.0f && navigable.contains(position))
        {
            return true;
        }
    }

    return false;
}

// validates generated geometry, reachability, and pellet placement
void validate_dungeon(const Dungeon& dungeon, float player_size)
{
    require(!dungeon.rooms.empty(), "dungeon must contain rooms");
    require(!dungeon.floor_rects.empty(), "dungeon must contain floor geometry");
    require(!dungeon.pellets.empty(), "dungeon must contain pellets");
    require(is_finite(dungeon.spawn), "dungeon spawn must be finite");
    require(is_finite(dungeon.bounds) && dungeon.bounds.width() > 0.0f && dungeon.bounds.height() > 0.0f,
        "dungeon bounds are invalid");

    for (const Room& room : dungeon.rooms)
    {
        require(is_finite(room.center) && is_finite(room.bounds), "room geometry must be finite");
        require(!room.floor_rects.empty(), "room must contain floor geometry");
    }

    const vector<Rect> navigation = navigable_rects(dungeon, player_size);
    require(is_navigable(dungeon, dungeon.spawn, player_size), "dungeon spawn is not navigable");

    vector<bool> visited(navigation.size(), false);
    queue<size_t> pending;

    for (size_t index = 0; index < navigation.size(); ++index)
    {
        if (navigation[index].contains(dungeon.spawn))
        {
            visited[index] = true;
            pending.push(index);
        }
    }

    while (!pending.empty())
    {
        const size_t current = pending.front();
        pending.pop();

        for (size_t candidate = 0; candidate < navigation.size(); ++candidate)
        {
            if (visited[candidate] || !has_traversable_overlap(navigation[current], navigation[candidate]))
            {
                continue;
            }

            visited[candidate] = true;
            pending.push(candidate);
        }
    }

    require(all_of(visited.begin(), visited.end(), [](bool value) { return value; }),
        "dungeon floor geometry is disconnected for the configured player size");

    for (const Room& room : dungeon.rooms)
    {
        require(is_navigable(dungeon, room.center, player_size), "room center is not navigable");
    }

    for (const Pellet& pellet : dungeon.pellets)
    {
        require(is_finite(pellet.position), "pellet position must be finite");
        require(is_navigable(dungeon, pellet.position, player_size), "pellet is not reachable by the player center");
    }
}

}
