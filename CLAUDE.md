# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repo is currently **vision/planning stage**: only `README.md` exists. There is no source code, no `requirements.txt`, no chapter directories yet. The README is in Chinese and describes the intended project. When implementing, scaffold what's described below rather than assuming files exist.

## Project intent

A Python-based project that re-teaches math by retracing its **historical discovery path** (Babylonian tablets → Newton's fluxions → Galois → Fourier → modern deep learning). Every visualization should connect a modern technique to the historical figure / real-world problem that produced it. This framing — "math as a 2000-year human story, not a buy-vegetables formula" — is the load-bearing design constraint, not a marketing tagline. Code, comments, and animations should make the historical lineage visible (e.g., `autograd_as_fluxion.py` deliberately maps `loss.backward()` back to Newton's notation).

## Planned architecture

Code is organized by **historical chapter**, not by technique. Each chapter is a self-contained directory with runnable scripts:

```
ch01_sequences/        Pythagoras, Fibonacci, Pascal — induction, golden spiral
ch02_calculus/         Archimedes, Newton, Leibniz — fluxions, autograd as fluxion
ch03_polynomials/      Cardano, Abel, Galois — quintic unsolvability, S_5 group
ch04_linear_algebra/   Liu Hui, Gauss, Cayley — Gaussian elimination → SVD
ch05_fourier/          Fourier, Maxwell — heat equation, FFT, spectrograms
ch06_probability/      Pascal/Fermat, Bayes, Gauss — MC sims, MLE/MAP
ch07_deep_learning/    Synthesis — MNIST from scratch, each line tagged with its historical origin
```

Chapter 7 is the keystone: it reuses primitives from chapters 1–6 to make the claim that deep learning is the convergence of three centuries of math. When adding to ch07, prefer demonstrating the lineage over introducing new techniques.

## Pedagogical style: code over formulas, always visualize, physics first

This project teaches math through **executable code and visualization**, not through formula transcription. Three hard rules apply to every script:

1. **Prefer code to formulas.** Express mathematical ideas as runnable PyTorch / NumPy / SymPy code rather than LaTeX or comment-block equations. A derivative is `torch.autograd.grad(...)`, not `∂f/∂x` written in a docstring. A Fourier series is a loop that sums tensors, not a `\sum_{n=0}^{\infty}` block. When a formula is unavoidable (e.g., to name a historical equation), keep it to one line and immediately follow it with the code that computes it. The reader should be able to delete every formula in the file and still understand the math from the code alone.
2. **Every script must produce a visualization.** No script is complete if it only prints numbers. Use Matplotlib / Seaborn for static plots, Manim for animated derivations, Streamlit / Jupyter widgets for interactive exploration. The visualization is the deliverable — the code exists to generate it. If a concept genuinely cannot be visualized (rare), justify that in a comment and produce a tensor-shape diagram or computation-graph render instead.
3. **Physics first, problem-driven, not tool-driven** (per Chapter 0.7 — the Tsinghua-professor / Musk teaching critique). Almost every important formula was originally invented to answer a concrete physical question. Strip the question away and the formula becomes an unintelligible incantation. So for any script, **lead with the real-world phenomenon** (falling body, pendulum, RC decay, heat-rod, planet orbit, resonance, engine PV cycle…), simulate it first, and let the math drop out as the explanation of what the simulation is doing. The order is **phenomenon → simulation → dissection → formula**, never the reverse. The brain only durably remembers things it can attach to reality; respect that constraint. "Disassemble the engine, look at every part, then reassemble" is the working metaphor — applied to formulas instead of pistons.

Lean on PyTorch even where NumPy would suffice: autograd makes the "compute gradient by running the program backwards" idea tangible, which is the whole point of framing modern math as Newton's fluxions modernized.

The three meta-chapters (Chapter 0 perspectives, Chapter 0.5 formula-idioms, Chapter 0.7 physics-first) are *how* to learn; Chapters 1–7 are *what* to learn. When adding new content anywhere, check that it embodies all three: a structural perspective, a recurring idiom (squared / inverse-square / log / exp), and a real-world phenomenon it descends from.

## Tech stack (from README)

- `PyTorch` — numerics + autograd (used pedagogically as "Newton's fluxions, modernized")
- `Matplotlib` / `Seaborn` — static plots
- `Manim Community Edition` — math animations (3Blue1Brown style)
- `Jupyter` / `Streamlit` — interactive exploration
- `SymPy` — symbolic verification of hand derivations

## Commands

No build system exists yet. The README proposes:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # requirements.txt does not yet exist
python ch01_sequences/fibonacci_spiral.py # representative entry point
```

When adding the first real code, also create `requirements.txt` pinning the stack above.
