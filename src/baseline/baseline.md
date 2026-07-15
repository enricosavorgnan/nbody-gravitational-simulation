# Serial C11 direct N-body baseline

This directory contains three stand-alone programs for the direct gravitational N-body exercise:

- `nbody_direct_serial.c`: serial softened direct solver using a DKD leapfrog
  step and a relative energy-drift verifier.
- `gen_plummer_sphere.c`: Plummer-sphere initial-condition generator.
- `gen_uniform_ball_maxwell.c`: uniform-ball generator with isotropic Maxwellian
  velocities.

The codes are intended as *almost complete* exam skeletons. The direct force kernel is deliberately correct but naive. It uses an O(N^2) all-pairs loop, scalar `sqrt`, one accumulator per component, and no Newton-third-law reuse. 
The comments in `compute_accelerations_naive` mark this as the kernel whose optimization is part of the assignment, along with the hybrid parallelization.

## Arithmetic type

All physical quantities in the solver and generators use the typedef `dtype`, defined in `nbody_common.h`.

Default build, double-precision arithmetic:

```sh
make
```

Single-precision arithmetic:

```sh
make clean
make PRECISION=float
```

The equivalent manual switches are:

```sh
-DNBODY_USE_DOUBLE
-DNBODY_USE_FLOAT
```

Only one of the two should be defined. If neither is defined, the header falls back to double precision.

## Binary file format

All programs use the same native-endian binary format. Particle data are stored in single precision, independently of the selected `dtype` used for arithmetic:

```text
byte 0..7       magic: "NBODYF1\0"
next 8 bytes    uint64_t particle count N
then N records  x y z vx vy vz, six float values per particle
```

The solver assigns one mass to every particle through `--mass`; mass is not stored per particle in the file. This keeps the initial-condition file compact and makes the equal-mass assumption explicit in the command line.

Because the format is deliberately minimal and native-endian, it is intended for same-machine teaching runs and benchmarks, not for long-term archival exchange between heterogeneous systems.

## Build

```sh
make
```

or explicitly:

```sh
cc -std=c11 -DNBODY_USE_DOUBLE -O2 -Wall -Wextra -Wpedantic nbody_direct_serial.c -lm -o nbody_direct_serial
cc -std=c11 -DNBODY_USE_DOUBLE -O2 -Wall -Wextra -Wpedantic gen_plummer_sphere.c -lm -o gen_plummer_sphere
cc -std=c11 -DNBODY_USE_DOUBLE -O2 -Wall -Wextra -Wpedantic gen_uniform_ball_maxwell.c -lm -o gen_uniform_ball_maxwell
```


Part of the assignment is to determine the best compiler’s flags and options, and the CPU bindings. List them in the final report.

## Example runs

Generate a small Plummer sphere and evolve it:

```sh
./gen_plummer_sphere --n 1000 --seed 123 --scale 1.0 --mass 1.0 --output plummer_1000.bin
./nbody_direct_serial --input plummer_1000.bin --nsteps 100 --dt 1e-4 --eps 0.05 --mass 1.0 --energy-every 10 --output final_state.bin
```

Generate a uniform ball with Maxwellian velocities. If `--sigma` is negative or omitted, the generator uses the uniform-sphere virial estimate
`sigma^2 = G M / (5 R)`.

```sh
./gen_uniform_ball_maxwell --n 1000 --seed 456 --radius 1.0 --mass 1.0 --output ball_1000.bin
./nbody_direct_serial --input ball_1000.bin --nsteps 100 --dt 1e-4 --eps 0.05 --mass 1.0 --energy-every 10
```

Run both smoke tests:

```sh
make run-smoke
```

## Solver notes

The implemented time integrator is Drift-Kick-Drift:

1. drift positions by `dt/2`;
2. compute accelerations at the half-step positions;
3. kick velocities by `dt`;
4. drift positions by `dt/2` with the updated velocities.

The energy check uses the same softened potential as the force law:

$$
U = -\sum_{i \lt j} G m^2 / \sqrt{(|r_i-r_j|^2 + \varepsilon^2)}
$$

The reported verification metric is

$$
\left | E(t) - E(0) \right | / \max(\left | E(0) \right |, \text{dtype_min_normal})
$$

A warning is printed if the maximum observed drift exceeds `--energy-tol` (default `1e-3`). This does not terminate the run, because large drift is often an intentional teaching signal: reduce `dt`, increase `eps`, or inspect the initial conditions.

## Intended optimisation path

The baseline is serial on purpose. Natural extensions are:

- **Pay attention to the data qualifiers, like `const`, `resatrict`, and so on, to let the compiler optimize the code**

- convert `compute_accelerations_naive` into an OpenMP loop without inner-loop
  atomics;
- compare Newton-third-law reuse against thread-private force buffers;  
  when is it convenient, against the price of using atomics for a non-local write?
- split accumulators to shorten the floating-point dependency chain;
- compare scalar `sqrt` with an approximate reciprocal-square-root path and
  verify that energy conservation remains meaningful;
- preserve the SoA layout when adding MPI ring-shift communication;
- can you measure the achieved FLOP/s before and after each change.
- Instrument your code so that you can tie every section and assess their scalability separately, instead of just the total run-time



## Files Structure:

#### `nbody_direct_serial.c`

Main Body

- `main`: 
  - Accepts several command-line arguments.
  - Initializes an empty particle SoA using `particles_init_empty`.
  - Reads the *input binary file* using `particles_read_binary` that operates some checks and allocates a SoA.
  - Computes the total energy using `total_energy` that internally calls `kinetic_energy` and `potential_energy_naive`.
  - Computes the integration over a `for` loop using `leapfrog_dkd_step`.
  - Writes an *output binary file* using `particles_write_binary`-
  - Frees the occupied memory using `particles_free`.
  - Meanwhile, lots of checks are run, as well as several messages are printed.

---

Integration Scheme

- `compute_acceleration_naive`:
  - This is the most delicate method. Computes accelerations (unsurprisingly)
  - for each particle `i`, for each particle `j`:
    - Compute distance $r_ij$.
    - Computes $\sqrt{r_ij}$ and invert it.
    - Elevates to the cube and multiplies by the mass.
    - Sums the result to a temporary variable 
  - Stores the resulting sum into the i-th component of the particles' SoA.

- `drift`:
  - Drift step for the DKD procedure. 
  - Drifts the particles' position by a given time interval `dt`:
    $$
  x_i = x_i + v_i * dt
    $$
  
- `kick`:
  - Kick step for the DKD procedure.
  - Kicks the particles' velocity by a given time interval `dt`:
    $$
  v_i = v_i + a_i * dt
    $$

- `leapfrog_dkd_step`:
  - Applies a Leapfrog step suing DKD procedure: Drift - Acceleration - Kick - Drift
  - Calls in sequence: `drift`, `compute_Acceleration_naive`, `kick`, `drift`

---

Type Checks

- `dtype_to_storage_float`:
  - Converts physical quantities to disk-types.

- `kinetic_energy`:
  - Defines the physical quantity of the kinetic energy via a `for` loop and exploiting long-double accumulators.

- `potential_energy_naive`:
  - Defines the physical quantity of potential energy via a nested `for` loop
  - Defined as:
  $$
    - g \cdot m_2 \cdot \sum_j \frac{1}{\sqrt{r_{\cdot, j}}}
  $$

- `total_energy`:
  - Simply computes `kinetic_energy` + `potential_energy_naive` 

---

Checks

- `die`:
  - Method launched in case of errors. Throws an `stderr` and terminates.

- `parse_size`:
  - Parses command line values' size. 
  - In case of errors, calls `die`.

- `parse_dtyp`:
  - Converts command line floats into `dtype` type, i.e., `float` or `double` depending on `NBODY_USE_FLOAT` or `NBODY_USE_DOUBLE`.

- `option_value`:
  - Returns the argument given the key.

- `checked_align_alloc`:
  - Allocates cache-line aligned blocks.
  - Is used by `particles_allocate` during particles' SoA allocation in memory. 

- `checked_fread`, `checked_fwrite`:
  - Respectively read and write exactly `nmemb` items from and to a binary stream.

---

Memory allocation

- `particles_init_empty`:
  - Simply defines a SoA in the form of `particles_s` struct having null pointers but for `p->mass = 1.0`.

- `particles_allocate`:
  - Initializes an empty SoA using `particles_init_empty`.
  - Ensures that each member of the SoA is correctly aligned in cache memory using `checked_aligned_alloc`.

- `particles_free`:
  - Frees the memory occupied by particles' SoA.

- `particles_read_binary`:
  - Reads a binary file whose path is in `path` variable.
  - Operates some checks during the read using `checked _fread`.
  - Allocates the particles' SoA using `particles_allocate`.
  - Iterates a `for` loop for storing the initial values for the particles' position and velocity. Acceleration is kept un-initialized since they will be computed in a second moment. Internally to the for loop, an expensive check is run: possibly change it.

- `particles_write_binaries`:
  - Opens a binary stream in the output file path `*path`.
  - Operates a check during writing operations using `checked_fwrite`.
  - Converts types using `dtype_to_storage_float`. 
  - Writes records into the stream file and closes.

---




#### `generate_ic.c`
- ``