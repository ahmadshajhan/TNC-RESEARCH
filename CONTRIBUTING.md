# Contributing to TNC Research

Thank you for your interest in contributing to Thermodynamic Neural Computation.

## How to Contribute

### Reporting Issues
- Open a GitHub issue describing the problem
- Include your OS, Python version, and GCC version
- Attach any error output

### Proposing Extensions
TNC is open to extensions in these areas:
1. **New energy functions** U(θ) for specific problem domains
2. **New entropy field parameterisations** S_φ(θ)
3. **Hardware implementations** (CUDA, OpenCL, neuromorphic chips)
4. **Application domains** (new materials, new ML architectures)

### Code Standards
- C code: C11 standard, compile cleanly with `gcc -Wall -Wextra`
- Python code: PEP8, type hints encouraged, docstrings required
- All new mathematical contributions must include derivation in comments

### Mathematical Contributions
Any new equations must:
- Be derived from first principles
- Not duplicate existing literature
- Include a proof sketch or reference
- Be implemented in both C and Python if computational

## License Agreement
By contributing, you agree that your contributions will be licensed
under the Apache 2.0 License.

## Attribution
All contributors will be listed in the AUTHORS file.
