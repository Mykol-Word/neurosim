# neurosim

A from-scratch biological-neuron-inspired system and a continuous-space task environment for developing it.

The project currently has two independent pieces: a playable procedural dungeon and the first sparse three-dimensional spiking brain model. They are intentionally not connected yet, so model behavior can be tested without game or rendering concerns.

## Brain model

- Neurons occupy selected cells of a discrete 3D lattice; unoccupied cells stay empty.
- Every occupied cell within two Manhattan steps is a potential directed connection.
- Neurons use normalized leaky integrate-and-fire dynamics with one-tick signal delay.
- Excitatory and inhibitory neurons give outgoing signals a fixed sign.
- Local joint activity strengthens a directed connection, while inactivity weakens it.
- Each neuron has one outgoing strength budget, so saturated connections compete.
- Input and output channels are ordered lists of arbitrary neuron positions rather than fixed regions.
- Full reset restores seeded weights; activity reset preserves learning between episodes.

The exact equations, units, update order, assumptions, and biological limitations are recorded in [docs/brain-model.md](docs/brain-model.md). The model has no backpropagation, gradients, reward signal, action decoder, or game integration.

## Dungeon rules

- Actions combine `up`, `down`, `left`, and `right`, allowing diagonal movement.
- Movement accelerates toward one shared maximum speed in every direction and brakes toward rest on `none`.
- Eating a pellet removes it and increments the episode score.
- The episode ends as `starved` after 15 seconds without food.
- The episode ends as `cleared` when every pellet is eaten.
- The episode ends as `time_limit` after 5 minutes.
- Clearing takes precedence when food is eaten on the same tick as another deadline.

Simulation advances at a fixed 120 Hz. Timers are stored as integer ticks, and movement uses continuous coordinates rather than tile positions.

## Build

CMake fetches the pinned raylib 6.0 source when the viewer is enabled. On Windows, the project helper locates Visual Studio's C++ and CMake tools, builds the project, and runs the tests:

```powershell
.\tools\build.ps1
```

Use `-Configuration Debug` for a debug build or `-Headless` to avoid fetching and building the graphics dependency:

```powershell
.\tools\build.ps1 -Configuration Debug
.\tools\build.ps1 -Headless
```

The CUDA backend is explicit and uses a separate build directory:

```powershell
.\tools\build.ps1 -Headless -Cuda
```

`-Cuda` requires an installed CUDA Toolkit with `nvcc`; requesting it without a compiler fails configuration rather than falling back to the CPU. `NEUROSIM_CUDA_ARCHITECTURES` defaults to CMake's portable `all-major` set and can be overridden for a local build, such as `86` for an RTX 3060 Ti.

On other platforms, configure normally with CMake and set `NEUROSIM_BUILD_VIEWER=OFF` when raylib is not wanted.

## Run

```powershell
.\build-neurosim\Release\neurosim_game.exe
.\build-neurosim\Release\neurosim_game.exe --seed 42
.\build-neurosim\Release\neurosim_headless.exe --seed 42 --episodes 10
```

Use WASD or the arrow keys to move. Press `R` to generate a new episode immediately. The viewer automatically starts a new seed shortly after a terminal state.

The viewer only reveals the world within 200 pixels of the player. Unseen space uses the same background as the area outside the dungeon.

The headless executable emits one JSON object per completed episode. Its built-in policy is only a deterministic smoke test, not an agent.

## Architecture

- `neurosim_core` owns the sparse brain model, CPU reference backend, dungeon generation, collision, and game rules.
- `neurosim_cuda` is an optional explicit backend over the same compiled brain arrays and tick order.
- `neurosim_game` is a raylib client of the core environment.
- `neurosim_headless` demonstrates rendering-free episode execution.
- `neurosim_tests` covers environment generation, movement, collision, pickup, timing, and reset behavior.
- `neurosim_brain_tests` covers topology, dynamics, inhibition, learning, competition, decay, delay, and reset behavior.
- `neurosim_cuda_tests` compares every CUDA tick with the CPU reference when that backend is built.

CUDA remains absent from the cheap game simulation. The first GPU backend uses reusable contiguous buffers and separate neuron and plasticity kernels, but one small brain may still be slower than the CPU because of launch and synchronization overhead. Large networks and future batches of evolving brains are the intended acceleration case.
