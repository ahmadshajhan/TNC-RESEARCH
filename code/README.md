# TNC Source Code

**License:** Apache 2.0  
All code is original — no third-party ML frameworks used.

---

## Files

### `tnc_core.c` — C Engine

A complete, self-contained C11 implementation of the TNC training algorithm.

**Dependencies:** Standard C library only (`math.h`, `string.h`, `sys/mman.h`)  
**Compile:**
```bash
gcc -O2 -o tnc_core tnc_core.c -lm
./tnc_core
```

**What it does:**
- Allocates a POSIX `mmap()` memory-mapped thermal buffer (physical RAM pages)
- Runs 5000 Langevin-TNC training steps on 256 weights
- Computes all 14 novel equations at each step
- Simulates L1/L2 cache hit/miss patterns as thermodynamic observables
- Exports full training history to `tnc_history.csv`
- Prints live metrics every 500 steps

**Key structs:**
```c
TNC_WeightBuffer  // weights + gradient arrays (∇U, ∇S, ∇F)
TNC_MemMap        // memory-mapped thermal buffer + MEM array
TNC_State         // all current metrics (T, F, U, S, PQ, CTI, NER, ...)
TNC_History       // full time-series (5000 × 10 metrics)
```

**Novel equation implementations:**
```c
tnc_temperature()               // EQ 3: Adaptive Resonant Cooling
tnc_internal_energy()           // U(θ)
tnc_mem_entropy()               // EQ 2: Memory-Entropy Map per param
tnc_entropy_field()             // EQ 4: Learnable Entropy Field S_φ
tnc_partition_function()        // EQ 5: Monte Carlo Z(t)
tnc_cache_thermal_index()       // EQ 9: CTI metric
tnc_neuromorphic_energy_ratio() // EQ 7: NER metric
tnc_detect_phase_transition()   // EQ 13: Phase criterion
tnc_langevin_update()           // EQ 8: 4-term update rule
tnc_mmap_init/free()            // Physical memory mapping
tnc_train()                     // Full training loop
tnc_export_csv()                // History export
```

---

### `tnc_research.py` — Python Research Suite

Complete Python implementation + all 15 research plots.

**Dependencies:**
```bash
pip install numpy scipy matplotlib
```

**Run:**
```bash
python3 tnc_research.py
```

**Classes:**

`TNCMathematics` — all 14 novel equations as static methods:
```python
adaptive_resonant_cooling()     # EQ 3
helmholtz_free_energy()         # EQ 1
memory_entropy_map()            # EQ 2
entropy_field()                 # EQ 4
partition_function()            # EQ 5
entropic_capacity()             # EQ 10
thermal_gradient_alignment()    # EQ 11
boltzmann_neural_weight()       # EQ 12
neuromorphic_energy_ratio()     # EQ 7
```

`TNCSimulator` — full training simulation:
```python
sim = TNCSimulator(n_weights=256, T0=80.0, Tf=0.25, n_steps=5000)
hist = sim.run()           # TNC training
hist_sgd = sim.run_baseline_sgd()  # SGD for comparison
```

`plot_01` through `plot_15` — individual plot functions, each self-contained.

---

## License Headers

Both files carry the Apache 2.0 header:

```
Copyright (c) 2026 TNC Research Group
Licensed under the Apache License, Version 2.0
```

You may use, modify, and distribute these files under the terms of
the Apache 2.0 License. See the root `LICENSE` file for full terms.
