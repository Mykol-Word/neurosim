#include "neurosim/brain_simulator.h"

#include <stdexcept>

namespace neurosim
{

using namespace std;

// returns the stable lowercase name of one execution backend
const char* brain_backend_name(BrainBackend backend)
{
    using enum BrainBackend;

    switch (backend)
    {
        case Cpu:
            return "cpu";
        case Cuda:
            return "cuda";
    }

    throw invalid_argument("brain backend is invalid");
}

}
