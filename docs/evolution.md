# Brain evolution

Evolution searches over task-neutral brain anatomy while each candidate still learns through the existing local eligibility and signed modulation rules. It does not add gradients or task-specific connections.

## Genome

A genome stores the complete occupied neuron set inside the fixed three-dimensional lattice, each neuron's polarity and bias drive, ordered input and output positions, and every positive directed initial connection. Missing local connections begin at zero but remain available to plasticity. Position-based connection endpoints keep inheritance stable when adding or removing a neuron changes compiled indices.

The current maximum space is the existing 5,041-cell sphere. Mutations cannot exceed that space and default to at least 1,024 neurons, which leaves room beyond the 841 external interface neurons.

## Generations

Generation one contains the requested number of independently seeded random brains with mild initial mutations, including varied neuron counts and occupancy. Every candidate learns for the requested number of procedural episodes. Candidates within a generation receive the same training dungeon seeds and the same task-blind exploration sequence; exploration randomness is independent from genome construction randomness.

After learning, connection changes and task-blind exploration are disabled while every candidate runs the same selection suite for that generation. The suite rotates between generations, so descendants are not repeatedly selected against a small fixed set of layouts. This matches normal checkpoint playback and measures the learned policy rather than externally driven exploratory actions. Fitness ranks candidates by:

1. total evaluation pellets;
2. cleared evaluation episodes;
3. best evaluation episode score;
4. total survival time;
5. fewer wall hits;
6. stable genome fingerprint.

Total evaluation score is more reliable than selecting one lucky run. The generation winner is also measured on a separate fixed archive benchmark so winners from different generations can be ranked comparably as `n1` through `n5`. The archive benchmark does not choose descendants. After evolution, `n1` is measured once on a third unseen reporting-only suite; that final test is written to the completion record and cannot change the selected checkpoint.

The winner's learned strengths become its inherited initial connections. The next generation contains that learned winner unchanged and four deterministic mutations. This is Lamarckian inheritance used as an engineering mechanism so learning can accumulate between generations; it is not a claim about biological genetic inheritance.

Mutations make bounded changes to neuron count, occupied cells, input and output locations, and initial connection strengths. Occupancy stays connected, interfaces stay valid, connection range stays local, and every source remains within its outgoing strength budget. A requested shrink that cannot preserve connectivity and interfaces keeps the nearest feasible count while still applying other mutations. Generic inputs and outputs can move independently; the adaptive-controller adapter explicitly pairs each anonymous output with its direct exploration-drive input.

## Running

Build normally for CPU evolution or request the CUDA backend:

```powershell
.\tools\build.ps1 -Cuda
```

Run an experiment with a new archive name:

```powershell
.\build-neurosim-cuda\Release\neurosim_evolve.exe `
  --name test-evo `
  --population 20 `
  --episodes 100 `
  --generations 20 `
  --evaluation-episodes 8 `
  --seed 1 `
  --backend cuda
```

The output directory must be empty or absent. This prevents two experiments from being mixed accidentally.

## Output

`runs/test-evo/test-evo-results.jsonl` records run metadata, lineage, training totals, within-generation selection fitness, fixed archive fitness, final unseen-test results, neuron counts, and genome fingerprints.

The latest ten generation winners are retained as `test-evo-gN.nsbrain`. The five strongest generation winners across the full run are written as `test-evo-n1.nsbrain` through `test-evo-n5.nsbrain`, where `n1` is best. Before five generations complete, only the available ranks exist.

Every version-2 `.nsbrain` contains its complete genome and learned strengths. A ranked or generation checkpoint can therefore be played directly:

```powershell
.\build-neurosim-cuda\Release\neurosim_game.exe `
  --load-brain .\runs\test-evo\test-evo-n1.nsbrain `
  --backend cuda
```

Checkpoint version 1 is intentionally rejected because it contains only a seed and cannot reconstruct evolved anatomy.
