# neurosim

A from-scratch biological-neuron-inspired system and a continuous-space task environment for developing it.

The project has a playable procedural dungeon, a sparse three-dimensional spiking brain model, and a first brain-controlled learning loop. Model behavior remains independent from rendering.

## Brain model

- The default brain has 5,041 neurons occupying selected cells of a discrete 3D lattice; unoccupied cells stay empty.
- Every occupied cell within two Manhattan steps is a potential directed connection.
- Neurons use normalized leaky integrate-and-fire dynamics with one-tick signal delay.
- Excitatory and inhibitory neurons give outgoing signals a fixed sign.
- Local causal activity records eligibility; food and wall events modulate eligible connections.
- Each neuron has one outgoing strength budget, so saturated connections compete.
- Sensor and output channels use reproducible random neuron positions rather than fixed regions.
- Full reset restores seeded weights; activity reset preserves learning between episodes.

The exact equations, units, update order, assumptions, and biological limitations are recorded in [docs/brain-model.md](docs/brain-model.md). The generic controller boundary and dungeon-specific observation, action, modulation, and validation semantics are recorded in [docs/agent-learning.md](docs/agent-learning.md).

## Dungeon rules

- Actions combine `up`, `down`, `left`, and `right`, allowing diagonal movement.
- Movement accelerates toward one shared maximum speed in every direction and brakes toward rest on `none`.
- Eating a pellet removes it and increments the episode score.
- Procedural rooms contain four to eight pellets each so early random exploration produces frequent learning events.
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

The CUDA backend is explicit and uses a separate build directory. Build with the
viewer when headed CUDA playback is needed:

```powershell
.\tools\build.ps1 -Cuda
.\tools\build.ps1 -Headless -Cuda
```

`-Cuda` requires an installed CUDA Toolkit with `nvcc`; requesting it without a compiler fails configuration rather than falling back to the CPU. `NEUROSIM_CUDA_ARCHITECTURES` defaults to CMake's portable `all-major` set and can be overridden for a local build, such as `86` for an RTX 3060 Ti.

On other platforms, configure normally with CMake and set `NEUROSIM_BUILD_VIEWER=OFF` when raylib is not wanted.

## Run

```powershell
.\build-neurosim\Release\neurosim_game.exe
.\build-neurosim\Release\neurosim_game.exe --seed 42
.\build-neurosim\Release\neurosim_game.exe --brain --brain-seed 7
.\build-neurosim\Release\neurosim_headless.exe --seed 42 --episodes 10
.\build-neurosim\Release\neurosim_train.exe --seed 1 --brain-seed 2 --episodes 500
.\build-neurosim\Release\neurosim_game.exe --load-brain runs\brain-2.nsbrain
.\build-neurosim\Release\neurosim_evolve.exe --name test-evo --population 20 --episodes 100 --generations 20
.\build-neurosim-cuda\Release\neurosim_train.exe --seed 1 --brain-seed 2 --episodes 500 --backend cuda
.\build-neurosim-cuda\Release\neurosim_game.exe --load-brain runs\brain-2.nsbrain --backend cuda
.\build-neurosim-cuda\Release\neurosim_evolve.exe --name test-evo --population 20 --episodes 100 --generations 20 --backend cuda
```

The CPU backend remains the default in every build. Select CUDA explicitly with
`--backend cuda`; use `--cuda-device N` when the selected device is not device
zero. A request for an unavailable or unbuilt CUDA backend fails instead of
silently running on the CPU. Checkpoints are portable between the CPU and CUDA
backends because they store the complete genome and learned strengths, not
backend-specific state.

Use WASD or the arrow keys to move, or pass `--brain` to let the seeded brain control the player and retain learned strengths across automatic resets. Press `R` to generate a new episode immediately. The viewer automatically starts a new seed shortly after a terminal state.

The viewer and brain share a 24-world-unit vision radius. Unseen space uses the same background as the area outside the dungeon.

The headless executable emits one JSON object per completed episode. Its built-in policy is only a deterministic smoke test, not an agent.

The training executable emits JSON lines for procedural training, frozen checkpoint-selection validation, summaries, and the selected checkpoint. Food positively modulates recently eligible connections, while wall contact applies smaller rate-limited negative modulation. Exploration follows a fixed per-episode seed independent from brain construction and does not react to hunger, food, collisions, observations, or modulation. Training and validation use ordinary procedural dungeons from the first episode. The best frozen validation result is saved under `runs/`; use `--save-brain` to choose another path or `--load-brain` to continue training.

Loading a checkpoint in the viewer freezes learning and exploration so it shows the selected policy. Add `--learn` to continue adapting during headed playback; using `--learn` by itself starts a fresh brain-controlled run.

Version-2 checkpoints include the full genome, learned strengths, and model fingerprint, so evolved topologies are directly playable and portable between CPU and CUDA. Version-1 seed-only checkpoints are rejected because they cannot reconstruct evolved anatomy.

`neurosim_evolve` starts with the requested random population, learns each candidate on shared procedural episodes, and selects by cumulative score on a matched evaluation suite that rotates each generation. Later generations contain the learned winner and four mutations. A separate fixed benchmark ranks the archive, and an unseen suite reports final performance without affecting selection. See [brain evolution](docs/evolution.md) for mutation semantics and the rolling ten-generation/five-best checkpoint archive.

## Architecture

- `neurosim_core` owns the sparse brain model, generic adaptive controller, CPU reference backend, dungeon generation, collision, and game rules.
- `BrainGenome` stores evolvable anatomy, interfaces, polarity, bias, and positive initial connections independently of compiled indices.
- `AdaptiveController` owns seeded random interfaces, fixed neural timing, uniform input drive, task-blind output exploration, learning freeze state, and an explicitly selected brain backend.
- `BrainAgent` adapts dungeon observations, movement actions, and signed outcomes to the generic controller.
- `neurosim_cuda` is an optional explicit backend over the same compiled brain arrays and tick order.
- `neurosim_game` is a raylib client of the core environment.
- `neurosim_headless` demonstrates rendering-free episode execution.
- `neurosim_train` runs repeated learning episodes without rendering.
- `neurosim_evolve` trains, evaluates, mutates, and archives genome populations.
- `neurosim_tests` covers environment generation, movement, collision, pickup, timing, and reset behavior.
- `neurosim_brain_tests` covers topology, dynamics, inhibition, learning, competition, decay, delay, and reset behavior.
- `neurosim_agent_tests` covers sensing, generic interface placement, task-blind exploration, modulation, and learned-state preservation.
- `neurosim_checkpoint_tests` covers learned-state persistence and model identity validation.
- `neurosim_genome_tests` covers explicit anatomy compilation, capture, validation, and identity.
- `neurosim_evolution_tests` covers deterministic bounded mutation, inheritance, and fitness ordering.
- `neurosim_cuda_tests` compares every CUDA tick with the CPU reference when that backend is built.

CUDA remains absent from the cheap game simulation. The controller batches all
neural ticks for one 120 Hz task step, uploads sensory input once, and downloads
only compact spike totals and the four output activities. Device buffers are
reused across steps. The CPU backend uses the same model semantics and remains
the readable parity reference.
