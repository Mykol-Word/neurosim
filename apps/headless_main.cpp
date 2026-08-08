#include "neurosim/environment.h"

#include <array>
#include <charconv>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace std;
using namespace neurosim;

namespace
{

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

// returns a deterministic smoke-test action for one simulation tick
Action scripted_action(uint64_t seed, uint64_t tick)
{
    using enum Action;

    static constexpr array<Action, 5> ACTIONS{
        Up,
        Right,
        Down,
        Left,
        None
    };
    const uint64_t phase = tick / 90U;
    const uint64_t mixed = seed + phase * 0x9E3779B97F4A7C15ULL;
    return ACTIONS[(mixed ^ (mixed >> 32U)) % ACTIONS.size()];
}

}

// runs deterministic headless episodes and emits json summaries
int main(int argc, char** argv)
{
    using enum TerminationReason;

    try
    {
        uint64_t first_seed = 1;
        uint64_t episodes = 1;

        for (int index = 1; index < argc; ++index)
        {
            const string_view argument = argv[index];

            if (argument == "--seed" && index + 1 < argc)
            {
                first_seed = parse_unsigned(argv[++index], "invalid --seed value");
            }
            else if (argument == "--episodes" && index + 1 < argc)
            {
                episodes = parse_unsigned(argv[++index], "invalid --episodes value");
            }
            else
            {
                throw invalid_argument("usage: neurosim_headless [--seed N] [--episodes N]");
            }
        }

        if (episodes == 0)
        {
            throw invalid_argument("episode count must be positive");
        }

        Environment environment({}, first_seed);

        for (uint64_t episode = 0; episode < episodes; ++episode)
        {
            const uint64_t seed = first_seed + episode;

            if (episode > 0)
            {
                environment.reset(seed);
            }

            while (environment.state().termination == None)
            {
                environment.step(scripted_action(seed, environment.state().elapsed_ticks));
            }

            cout
                << "{\"episode\":" << episode
                << ",\"seed\":" << seed
                << ",\"result\":\"" << termination_name(environment.state().termination) << "\""
                << ",\"score\":" << environment.state().score
                << ",\"pellets_remaining\":" << environment.state().pellets_remaining
                << ",\"elapsed_seconds\":" << environment.elapsed_seconds()
                << "}\n";
        }

        return 0;
    }
    catch (const exception& error)
    {
        cerr << error.what() << '\n';
        return 1;
    }
}
