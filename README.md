# neurosim

A continuous-space procedural dungeon environment for developing a biological-neuron-based agent from scratch.

The current slice is deliberately small: one square agent moves through connected rooms and hallways, eats pellets, and obeys fixed episode rules. The simulation is independent from rendering so the same C++ API can drive interactive play, deterministic replay, or a future neural controller.

## Rules

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

- `neurosim_core` owns dungeon generation, continuous collision, episode state, and game rules.
- `neurosim_game` is a raylib client of the core environment.
- `neurosim_headless` demonstrates rendering-free episode execution.
- `neurosim_tests` covers generation determinism, variation, movement, collision, pickup, timing, and reset behavior.

CUDA is intentionally absent from this environment slice. A single moving body is cheaper on the CPU; CUDA will be introduced for parallel neuron and synapse computation with C++ parity tests when that work begins.
