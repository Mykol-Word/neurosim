#pragma once

#include <algorithm>
#include <cmath>

namespace neurosim
{

using namespace std;

// stores a two-dimensional continuous-space vector
struct Vec2
{
    float x = 0.0f;
    float y = 0.0f;

    // compares both vector components for equality
    bool operator==(const Vec2&) const = default;
};

// adds two vectors component by component
constexpr Vec2 operator+(Vec2 left, Vec2 right)
{
    return {left.x + right.x, left.y + right.y};
}

// subtracts one vector from another component by component
constexpr Vec2 operator-(Vec2 left, Vec2 right)
{
    return {left.x - right.x, left.y - right.y};
}

// scales a vector by a scalar value
constexpr Vec2 operator*(Vec2 value, float scalar)
{
    return {value.x * scalar, value.y * scalar};
}

// divides a vector by a scalar value
constexpr Vec2 operator/(Vec2 value, float scalar)
{
    return {value.x / scalar, value.y / scalar};
}

// returns the squared length of a vector
inline float length_squared(Vec2 value)
{
    return value.x * value.x + value.y * value.y;
}

// returns the euclidean length of a vector
inline float length(Vec2 value)
{
    return sqrt(length_squared(value));
}

// returns the squared distance between two points
inline float distance_squared(Vec2 left, Vec2 right)
{
    return length_squared(left - right);
}

// returns whether both vector components are finite
inline bool is_finite(Vec2 value)
{
    return isfinite(value.x) && isfinite(value.y);
}

// stores an axis-aligned rectangle in continuous coordinates
struct Rect
{
    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;

    // compares every rectangle boundary for equality
    bool operator==(const Rect&) const = default;

    // returns the rectangle width
    float width() const
    {
        return max_x - min_x;
    }

    // returns the rectangle height
    float height() const
    {
        return max_y - min_y;
    }

    // returns the rectangle center
    Vec2 center() const
    {
        return {(min_x + max_x) * 0.5f, (min_y + max_y) * 0.5f};
    }

    // returns whether the rectangle contains a point
    bool contains(Vec2 point, float epsilon = 0.0f) const
    {
        return point.x >= min_x - epsilon && point.x <= max_x + epsilon
            && point.y >= min_y - epsilon && point.y <= max_y + epsilon;
    }

    // returns whether two rectangles overlap or touch within an epsilon
    bool intersects(const Rect& other, float epsilon = 0.0f) const
    {
        return max_x >= other.min_x - epsilon && other.max_x >= min_x - epsilon
            && max_y >= other.min_y - epsilon && other.max_y >= min_y - epsilon;
    }

    // returns a rectangle expanded equally on every side
    Rect expanded(float amount) const
    {
        return {min_x - amount, min_y - amount, max_x + amount, max_y + amount};
    }

    // returns a rectangle inset equally on every side
    Rect inset(float amount) const
    {
        return {min_x + amount, min_y + amount, max_x - amount, max_y - amount};
    }

    // returns a translated copy of the rectangle
    Rect translated(Vec2 offset) const
    {
        return {min_x + offset.x, min_y + offset.y, max_x + offset.x, max_y + offset.y};
    }
};

// returns whether every rectangle boundary is finite
inline bool is_finite(const Rect& rect)
{
    return isfinite(rect.min_x) && isfinite(rect.min_y)
        && isfinite(rect.max_x) && isfinite(rect.max_y);
}

// returns the smallest rectangle containing both inputs
inline Rect joined(const Rect& left, const Rect& right)
{
    return {
        min(left.min_x, right.min_x),
        min(left.min_y, right.min_y),
        max(left.max_x, right.max_x),
        max(left.max_y, right.max_y)
    };
}

}

