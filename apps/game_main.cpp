#include "neurosim/brain_agent.h"
#include "neurosim/brain_checkpoint.h"

#include <raylib.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace std;
using namespace neurosim;

namespace
{

constexpr int INITIAL_WIDTH = 1280;
constexpr int INITIAL_HEIGHT = 800;
constexpr int MINIMUM_WIDTH = 800;
constexpr int MINIMUM_HEIGHT = 500;
constexpr int HUD_HEIGHT = 76;
constexpr float VISION_RADIUS_PIXELS = 200.0f;
constexpr int VISION_MASK_SEGMENTS = 128;
constexpr float RESET_DELAY_SECONDS = 2.0f;
constexpr double MAX_FRAME_SECONDS = 0.25;

constexpr Color BACKGROUND{15, 18, 22, 255};
constexpr Color CORRIDOR{48, 55, 62, 255};
constexpr Color ROOM{58, 66, 74, 255};
constexpr Color PELLET{232, 184, 82, 255};
constexpr Color PELLET_CORE{255, 224, 147, 255};
constexpr Color PLAYER_SHADOW{8, 10, 12, 150};
constexpr Color PLAYER{87, 205, 181, 255};
constexpr Color PLAYER_EDGE{184, 246, 230, 255};
constexpr Color PLAYER_EXPLORE{225, 105, 104, 255};
constexpr Color HUD_BACKGROUND{21, 25, 30, 255};
constexpr Color HUD_DIVIDER{53, 60, 67, 255};
constexpr Color LABEL_TEXT{143, 153, 162, 255};
constexpr Color VALUE_TEXT{231, 235, 238, 255};

// stores command-line settings for one viewer run
struct Options
{
    uint64_t seed = 1;
    uint64_t brain_seed = 1;
    string capture_path;
    filesystem::path load_brain_path;
    uint64_t exit_after_frames = 0;
    bool brain_control = false;
    bool brain_seed_explicit = false;
    bool learning_enabled = true;
    bool learning_explicit = false;
};

// maps one environment action to two keyboard keys
struct InputBinding
{
    Action action;
    KeyboardKey primary;
    KeyboardKey secondary;
};

// stores the fitted raylib camera and user zoom adjustment
struct CameraState
{
    Camera2D camera{};
    float zoom_multiplier = 1.0f;
};

// owns the raylib window lifetime
class RaylibWindow
{
public:
    // opens and configures the interactive window
    RaylibWindow()
    {
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(INITIAL_WIDTH, INITIAL_HEIGHT, "neurosim environment");

        if (!IsWindowReady())
        {
            throw runtime_error("failed to create raylib window");
        }

        SetWindowMinSize(MINIMUM_WIDTH, MINIMUM_HEIGHT);
        SetTargetFPS(144);
    }

    // closes the raylib window when ownership ends
    ~RaylibWindow()
    {
        if (IsWindowReady())
        {
            CloseWindow();
        }
    }

    // prevents duplicate ownership of the raylib window
    RaylibWindow(const RaylibWindow&) = delete;

    // prevents assigning ownership of the raylib window
    RaylibWindow& operator=(const RaylibWindow&) = delete;
};

// parses an unsigned command-line value with full validation
uint64_t parse_unsigned(string_view text, const char* message)
{
    uint64_t value = 0;
    const auto result = from_chars(text.data(), text.data() + text.size(), value);

    if (result.ec != errc{} || result.ptr != text.data() + text.size())
    {
        throw invalid_argument(message);
    }

    return value;
}

// parses viewer command-line options
Options parse_options(int argc, char** argv)
{
    Options options;

    for (int index = 1; index < argc; ++index)
    {
        const string_view argument = argv[index];

        if (argument == "--seed" && index + 1 < argc)
        {
            options.seed = parse_unsigned(argv[++index], "invalid --seed value");
        }
        else if (argument == "--brain")
        {
            options.brain_control = true;
        }
        else if (argument == "--brain-seed" && index + 1 < argc)
        {
            options.brain_seed = parse_unsigned(argv[++index], "invalid --brain-seed value");
            options.brain_seed_explicit = true;
        }
        else if (argument == "--load-brain" && index + 1 < argc)
        {
            options.load_brain_path = argv[++index];
            options.brain_control = true;
        }
        else if (argument == "--learn")
        {
            options.brain_control = true;
            options.learning_enabled = true;
            options.learning_explicit = true;
        }
        else if (argument == "--capture" && index + 1 < argc)
        {
            options.capture_path = argv[++index];

            if (options.capture_path.empty())
            {
                throw invalid_argument("capture path must not be empty");
            }
        }
        else if (argument == "--exit-after-frames" && index + 1 < argc)
        {
            options.exit_after_frames = parse_unsigned(
                argv[++index],
                "invalid --exit-after-frames value");

            if (options.exit_after_frames == 0)
            {
                throw invalid_argument("exit frame count must be positive");
            }
        }
        else
        {
            throw invalid_argument(
                "usage: neurosim_game [--seed N] [--brain] [--brain-seed N] "
                "[--load-brain PATH] [--learn] [--capture PATH] "
                "[--exit-after-frames N]");
        }
    }

    if (!options.load_brain_path.empty() && !options.learning_explicit)
    {
        options.learning_enabled = false;
    }

    return options;
}

// converts a simulation vector to a raylib vector
Vector2 to_raylib(Vec2 value)
{
    return {value.x, value.y};
}

// converts a simulation rectangle to a raylib rectangle
Rectangle to_raylib(const Rect& rect)
{
    return {rect.min_x, rect.min_y, rect.width(), rect.height()};
}

// returns whether either key for an action is held
bool is_down(const InputBinding& binding)
{
    return IsKeyDown(binding.primary) || IsKeyDown(binding.secondary);
}

// combines every currently held movement input
Action read_action()
{
    using enum Action;

    static constexpr array<InputBinding, 4> BINDINGS{
        InputBinding{Up, KEY_W, KEY_UP},
        InputBinding{Down, KEY_S, KEY_DOWN},
        InputBinding{Left, KEY_A, KEY_LEFT},
        InputBinding{Right, KEY_D, KEY_RIGHT}
    };

    Action action = None;

    for (const InputBinding& binding : BINDINGS)
    {
        if (is_down(binding))
        {
            action = action | binding.action;
        }
    }

    return action;
}

// follows the player while preserving the configured default vision size
void update_camera(CameraState& state, const Environment& environment)
{
    const int screen_width = GetScreenWidth();
    const int screen_height = GetScreenHeight();
    const float wheel = GetMouseWheelMove();

    if (wheel != 0.0f)
    {
        state.zoom_multiplier = clamp(
            state.zoom_multiplier * pow(1.15f, wheel),
            0.35f,
            5.0f);
    }

    state.camera.target = to_raylib(environment.state().position);
    state.camera.offset = {
        static_cast<float>(screen_width) * 0.5f,
        static_cast<float>(HUD_HEIGHT) + static_cast<float>(screen_height - HUD_HEIGHT) * 0.5f
    };
    state.camera.rotation = 0.0f;
    state.camera.zoom = VISION_RADIUS_PIXELS
        / environment.config().vision_radius
        * state.zoom_multiplier;
}

// formats seconds as a minute-second stopwatch value
string format_clock(double seconds)
{
    const uint64_t tenths = static_cast<uint64_t>(max(0.0, seconds) * 10.0 + 1.0e-6);
    const uint64_t minutes = tenths / 600;
    const uint64_t whole_seconds = (tenths / 10) % 60;
    const uint64_t fraction = tenths % 10;
    array<char, 32> buffer{};
    snprintf(
        buffer.data(),
        buffer.size(),
        "%02llu:%02llu.%llu",
        static_cast<unsigned long long>(minutes),
        static_cast<unsigned long long>(whole_seconds),
        static_cast<unsigned long long>(fraction));
    return buffer.data();
}

// formats hunger time against its configured deadline
string format_food_time(double seconds, double limit)
{
    array<char, 32> buffer{};
    snprintf(buffer.data(), buffer.size(), "%.1f / %.0f s", seconds, limit);
    return buffer.data();
}

// renders dungeon geometry, live pellets, and the player
void draw_world(const Environment& environment, bool exploring)
{
    const Dungeon& dungeon = environment.dungeon();

    for (const Rect& corridor : dungeon.corridors)
    {
        DrawRectangleRec(to_raylib(corridor), CORRIDOR);
    }

    for (const Room& room : dungeon.rooms)
    {
        for (const Rect& floor : room.floor_rects)
        {
            DrawRectangleRec(to_raylib(floor), ROOM);
        }
    }

    const EnvironmentState& state = environment.state();

    for (size_t index = 0; index < dungeon.pellets.size(); ++index)
    {
        if (state.pellet_alive[index] == 0U)
        {
            continue;
        }

        const Vector2 position = to_raylib(dungeon.pellets[index].position);
        DrawCircleV(position, 0.34f, PELLET);
        DrawCircleV(position, 0.12f, PELLET_CORE);
    }

    const float player_size = environment.config().player_size;
    const float half_size = player_size * 0.5f;
    const Rectangle player{
        state.position.x - half_size,
        state.position.y - half_size,
        player_size,
        player_size
    };
    Rectangle shadow = player;
    shadow.x += 0.12f;
    shadow.y += 0.14f;
    DrawRectangleRec(shadow, PLAYER_SHADOW);
    DrawRectangleRec(player, exploring ? PLAYER_EXPLORE : PLAYER);
    DrawRectangleLinesEx(player, 0.08f, PLAYER_EDGE);
}

// hides world pixels beyond the configured world-space vision radius
void draw_vision_mask(const Environment& environment, const Camera2D& camera)
{
    const Vector2 center = GetWorldToScreen2D(to_raylib(environment.state().position), camera);
    const float outer_radius = hypot(
        static_cast<float>(GetScreenWidth()),
        static_cast<float>(GetScreenHeight()));
    DrawRing(
        center,
        environment.config().vision_radius * camera.zoom,
        outer_radius,
        0.0f,
        360.0f,
        VISION_MASK_SEGMENTS,
        BACKGROUND);
}

// shrinks text only when required to fit its available width
float fitted_font_size(const string& text, float maximum_width, float preferred_size)
{
    const Font font = GetFontDefault();
    const float width = MeasureTextEx(font, text.c_str(), preferred_size, 0.0f).x;

    if (width <= maximum_width || width == 0.0f)
    {
        return preferred_size;
    }

    return max(11.0f, preferred_size * maximum_width / width);
}

// draws one line of text centered around a horizontal point
void draw_text_centered(const string& text, float center_x, float y, float size, Color color)
{
    const Font font = GetFontDefault();
    const Vector2 dimensions = MeasureTextEx(font, text.c_str(), size, 0.0f);
    DrawTextEx(font, text.c_str(), {center_x - dimensions.x * 0.5f, y}, size, 0.0f, color);
}

// renders the live score, stopwatches, and controller state
void draw_hud(const Environment& environment, uint64_t seed, const BrainAgent* agent)
{
    const int screen_width = GetScreenWidth();
    const float column_width = static_cast<float>(screen_width) / 6.0f;
    const EnvironmentState& state = environment.state();
    const array<const char*, 6> labels{
        "SCORE",
        "REMAINING",
        "RUN TIME",
        "SINCE FOOD",
        "CONTROL",
        "SEED"
    };
    const string control = agent == nullptr
        ? "PLAYER"
        : agent->status().exploring
            ? "EXPLORE"
            : agent->learning_enabled() ? "BRAIN" : "FROZEN";
    const array<string, 6> values{
        to_string(state.score),
        to_string(state.pellets_remaining),
        format_clock(environment.elapsed_seconds()),
        format_food_time(environment.seconds_since_food(), environment.config().starvation_seconds),
        control,
        to_string(seed)
    };

    DrawRectangle(0, 0, screen_width, HUD_HEIGHT, HUD_BACKGROUND);
    DrawLine(0, HUD_HEIGHT - 1, screen_width, HUD_HEIGHT - 1, HUD_DIVIDER);

    const Font font = GetFontDefault();

    for (size_t index = 0; index < values.size(); ++index)
    {
        const float left = column_width * static_cast<float>(index);
        const float center = left + column_width * 0.5f;

        if (index > 0)
        {
            const int divider_x = static_cast<int>(left);
            DrawLine(divider_x, 13, divider_x, HUD_HEIGHT - 13, HUD_DIVIDER);
        }

        const Vector2 label_size = MeasureTextEx(font, labels[index], 12.0f, 0.0f);
        DrawTextEx(
            font,
            labels[index],
            {center - label_size.x * 0.5f, 12.0f},
            12.0f,
            0.0f,
            LABEL_TEXT);

        const float value_size = fitted_font_size(values[index], column_width - 20.0f, 22.0f);
        draw_text_centered(values[index], center, 36.0f, value_size, VALUE_TEXT);
    }
}

// returns the display title for an episode termination reason
const char* terminal_title(TerminationReason reason)
{
    using enum TerminationReason;

    switch (reason)
    {
    case Cleared:
        return "DUNGEON CLEARED";
    case Starved:
        return "STARVED";
    case TimeLimit:
        return "TIME LIMIT REACHED";
    case None:
        return "";
    }

    return "RUN ENDED";
}

// returns the accent color for an episode termination reason
Color terminal_color(TerminationReason reason)
{
    using enum TerminationReason;

    if (reason == Cleared)
    {
        return PLAYER;
    }

    if (reason == Starved)
    {
        return {225, 105, 104, 255};
    }

    return PELLET;
}

// renders the terminal overlay and automatic-reset countdown
void draw_terminal(TerminationReason reason, double seconds_remaining)
{
    using enum TerminationReason;

    if (reason == None)
    {
        return;
    }

    const int screen_width = GetScreenWidth();
    const int screen_height = GetScreenHeight();
    const float content_center_y = static_cast<float>(HUD_HEIGHT)
        + static_cast<float>(screen_height - HUD_HEIGHT) * 0.5f;
    const float panel_width = min(420.0f, static_cast<float>(screen_width) - 48.0f);
    const Rectangle panel{
        (static_cast<float>(screen_width) - panel_width) * 0.5f,
        content_center_y - 56.0f,
        panel_width,
        112.0f
    };
    const Color panel_color{22, 27, 32, 244};
    const Color border_color{70, 79, 87, 255};

    DrawRectangle(0, HUD_HEIGHT, screen_width, screen_height - HUD_HEIGHT, Fade(BLACK, 0.38f));
    DrawRectangleRounded(panel, 0.06f, 6, panel_color);
    DrawRectangleRoundedLinesEx(panel, 0.06f, 6, 1.0f, border_color);
    draw_text_centered(
        terminal_title(reason),
        static_cast<float>(screen_width) * 0.5f,
        panel.y + 24.0f,
        24.0f,
        terminal_color(reason));

    array<char, 48> detail{};
    snprintf(
        detail.data(),
        detail.size(),
        "next seed in %.1f s",
        max(0.0, seconds_remaining));
    draw_text_centered(
        detail.data(),
        static_cast<float>(screen_width) * 0.5f,
        panel.y + 65.0f,
        15.0f,
        LABEL_TEXT);
}

// advances to a fresh seed and clears viewer timing state
void reset_environment(
    Environment& environment,
    BrainAgent* agent,
    uint64_t& seed,
    CameraState& camera,
    double& accumulator,
    double& terminal_started)
{
    ++seed;
    environment.reset(seed);

    if (agent != nullptr)
    {
        agent->reset_episode(seed);
    }

    camera.zoom_multiplier = 1.0f;
    accumulator = 0.0;
    terminal_started = 0.0;
}

// runs the interactive fixed-step viewer loop
int run(const Options& options)
{
    constexpr TerminationReason RUNNING = TerminationReason::None;

    optional<BrainCheckpoint> checkpoint;
    uint64_t brain_seed = options.brain_seed;

    if (!options.load_brain_path.empty())
    {
        checkpoint = load_brain_checkpoint(options.load_brain_path);

        if (options.brain_seed_explicit && brain_seed != checkpoint->brain_seed)
        {
            throw invalid_argument("--brain-seed does not match the loaded checkpoint");
        }

        brain_seed = checkpoint->brain_seed;
    }

    RaylibWindow window;
    Environment environment({}, options.seed);
    optional<BrainAgent> brain_agent;

    if (options.brain_control)
    {
        brain_agent.emplace(environment.config(), BrainAgentConfig{}, brain_seed);

        if (checkpoint)
        {
            validate_brain_checkpoint(*checkpoint, brain_agent->seed(), brain_agent->simulator().model());
            brain_agent->load_connection_strengths(checkpoint->connection_strengths);
        }

        brain_agent->set_learning_enabled(options.learning_enabled);
        brain_agent->set_exploration_enabled(options.learning_enabled);
        brain_agent->reset_episode(options.seed);
    }

    BrainAgent* agent = brain_agent ? &*brain_agent : nullptr;
    uint64_t seed = options.seed;
    CameraState camera;
    double accumulator = 0.0;
    double terminal_started = 0.0;
    uint64_t rendered_frames = 0;
    bool captured = false;

    while (!WindowShouldClose())
    {
        const double now = GetTime();

        if (IsKeyPressed(KEY_R))
        {
            reset_environment(environment, agent, seed, camera, accumulator, terminal_started);
        }

        const Action keyboard_action = read_action();

        if (environment.state().termination == RUNNING)
        {
            accumulator += min(static_cast<double>(GetFrameTime()), MAX_FRAME_SECONDS);
            const double fixed_time_step = static_cast<double>(environment.config().fixed_time_step);

            while (accumulator >= fixed_time_step)
            {
                const Action action = agent == nullptr
                    ? keyboard_action
                    : agent->decide(environment);
                const StepResult transition = environment.step(action);

                if (agent != nullptr)
                {
                    agent->observe_transition(transition);
                }

                accumulator -= fixed_time_step;

                if (environment.state().termination != RUNNING)
                {
                    terminal_started = now;
                    accumulator = 0.0;
                    break;
                }
            }
        }
        else if (now - terminal_started >= RESET_DELAY_SECONDS)
        {
            reset_environment(environment, agent, seed, camera, accumulator, terminal_started);
        }

        update_camera(camera, environment);
        const double terminal_remaining = RESET_DELAY_SECONDS - (now - terminal_started);
        const bool capture_this_frame = !captured
            && !options.capture_path.empty()
            && (options.exit_after_frames == 0 || rendered_frames + 1 >= options.exit_after_frames);

        BeginDrawing();
        ClearBackground(BACKGROUND);
        BeginScissorMode(0, HUD_HEIGHT, GetScreenWidth(), GetScreenHeight() - HUD_HEIGHT);
        BeginMode2D(camera.camera);
        draw_world(
            environment,
            agent != nullptr && agent->status().exploring);
        EndMode2D();
        draw_vision_mask(environment, camera.camera);
        EndScissorMode();
        draw_hud(environment, seed, agent);
        draw_terminal(environment.state().termination, terminal_remaining);

        EndDrawing();

        if (capture_this_frame)
        {
            TakeScreenshot(options.capture_path.c_str());
            captured = true;
        }

        ++rendered_frames;

        if (options.exit_after_frames > 0 && rendered_frames >= options.exit_after_frames)
        {
            break;
        }
    }

    return 0;
}

}

// reports viewer startup failures through the process exit code
int main(int argc, char** argv)
{
    try
    {
        return run(parse_options(argc, argv));
    }
    catch (const exception& error)
    {
        cerr << error.what() << '\n';
        return 1;
    }
}
