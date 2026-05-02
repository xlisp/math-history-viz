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
