# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python course workspace centered on Jupyter notebooks. Root-level notebooks named `Sec N - ... .ipynb` contain lecture material by topic. Standalone examples live beside them, such as `main.py` and `Sec 3.1 - case test.py`. Homework submissions and exports are under `homework/`, including `HW1.ipynb`, `HW2.ipynb`, `HW3.ipynb`, `HW3.py`, and generated `.html`/`.pdf` files. Image and notebook support files are kept near the material that uses them, for example `girl.jpeg` and `homework/HW2_files/`.

## Build, Test, and Development Commands

Create and activate a local environment before running notebooks or scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run notebooks locally:

```bash
jupyter notebook
```

Run script-based homework or examples:

```bash
python homework/HW3.py
python "Sec 3.1 - case test.py"
```

Run type checks for Python files:

```bash
pyright
```

## Coding Style & Naming Conventions

Use clear, beginner-readable Python. Prefer 4-space indentation, descriptive variable names, f-strings, and `pathlib` for filesystem paths. Keep notebook section filenames consistent with the existing pattern: `Sec 4.2 - Package Intro - pandas.ipynb`. Keep homework names as `HW<number>.ipynb` or `HW<number>.py`. Avoid committing generated caches such as `__pycache__/`, `.DS_Store`, or temporary notebook checkpoints.

## Testing Guidelines

There is no formal pytest suite in this repository. Validate changes by running the affected `.py` file and executing modified notebook cells from top to bottom. For notebooks, restart the kernel before final export to catch hidden state errors. If adding reusable Python modules, add small `test_*.py` files and document the command used to run them.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style prefixes, including `chore:`, `docs:`, and `feat:`. Keep commit messages imperative and scoped to the change, for example `docs: add contributor guide` or `chore: update homework exports`. Pull requests should describe the changed notebooks/scripts, list validation commands run, and mention any regenerated `.html` or `.pdf` outputs. Include screenshots only when visual output or plots changed.

## Agent-Specific Instructions

Preserve course materials and generated exports unless the task explicitly updates them. Do not rename notebooks casually; external course references may depend on current filenames. When editing notebooks, keep code cells executable in order and avoid adding machine-specific absolute paths.
