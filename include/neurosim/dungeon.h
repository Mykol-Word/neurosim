#pragma once

#include "neurosim/math.h"

#include <cstdint>
#include <vector>

namespace neurosim
{

using namespace std;

inline constexpr float NAVIGATION_EPSILON = 0.0001f;

// configures procedural room, corridor, and pellet generation
struct DungeonConfig
{
    uint32_t min_rooms = 10;
    uint32_t max_rooms = 20;
    float min_hub_width = 12.0f;
    float max_hub_width = 30.0f;
    float min_hub_height = 10.0f;
    float max_hub_height = 24.0f;
    uint32_t min_wings = 0;
    uint32_t max_wings = 4;
    float min_wing_depth = 4.0f;
    float max_wing_depth = 18.0f;
    float min_wing_span_ratio = 0.35f;
    float max_wing_span_ratio = 0.9f;
    float min_room_gap = 5.0f;
    float max_room_gap = 20.0f;
    float room_spacing = 2.0f;
    float min_corridor_width = 5.0f;
    float max_corridor_width = 8.0f;
    float extra_connection_ratio = 0.25f;
    float two_bend_chance = 0.35f;
    uint32_t min_pellets_per_room = 1;
    uint32_t max_pellets_per_room = 4;
    float pellet_min_spacing = 3.0f;
    float spawn_clearance = 5.0f;
    uint32_t placement_attempts_per_room = 300;
    uint32_t pellet_placement_attempts = 200;
};

// stores one compound room and its floor rectangles
struct Room
{
    Vec2 center;
    Rect bounds;
    vector<Rect> floor_rects;

    // compares complete room geometry for equality
    bool operator==(const Room&) const = default;
};

// stores one collectible pellet position
struct Pellet
{
    Vec2 position;

    // compares pellet positions for equality
    bool operator==(const Pellet&) const = default;
};

// stores immutable geometry and collectibles for one generated dungeon
struct Dungeon
{
    uint64_t seed = 0;
    vector<Room> rooms;
    vector<Rect> corridors;
    vector<Rect> floor_rects;
    vector<Pellet> pellets;
    Vec2 spawn;
    Rect bounds;

    // compares complete dungeon geometry and metadata for equality
    bool operator==(const Dungeon&) const = default;
};

// validates generation settings against the configured player size
void validate_dungeon_config(const DungeonConfig& config, float player_size);

// validates generated geometry, reachability, and pellet placement
void validate_dungeon(const Dungeon& dungeon, float player_size);

// generates a deterministic continuous-space dungeon from a seed
Dungeon generate_dungeon(const DungeonConfig& config, float player_size, uint64_t seed);

// returns whether a player center fits within the dungeon floor
bool is_navigable(const Dungeon& dungeon, Vec2 position, float player_size);

}

