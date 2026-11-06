# Coding Conventions

- Python 3.10+
- PyTorch
- Avoid unnecessary abstractions.
- Do not modify unrelated code.
- Preserve existing type hints.
- Prefer small focused classes/functions.

# Environment

PyTorch
Ubuntu Linux

Work happens in the `retargeting` conda environment:

conda activate retargeting

Install the package (editable) and its dependencies with:

pip install -e .

Tests are not routinely required. Make sure changed code runs, and run the test suite (pytest) only when explicitly requested.

# Working Guidelines

## Scope Control

- Do exactly what was requested and no more.
- Make the smallest change necessary to satisfy the request.
- Do not refactor, rename, reorganize, optimize, or clean up unrelated code.
- Do not change APIs, interfaces, tensor shapes, configuration structure, or behavior unless explicitly requested.
- Do not introduce new abstractions, helper classes, utilities, or dependencies unless they are necessary for the requested change.
- Do not modify nearby code merely because it could be improved.
- Preserve the existing coding style and architecture unless the task specifically asks to change them.

If you notice an unrelated bug, design issue, or possible improvement:
- Mention it separately.
- Do not fix it unless explicitly asked.

If the requested change appears to require a broader modification than expected:
- Explain why.
- Describe the minimum broader change required.
- Do not proceed with the broader change unless it is clearly necessary to complete the request.

## Before Editing

Before making significant changes:

1. Inspect the relevant implementation and its dependencies.
2. Determine the exact scope of the requested change.
3. Identify important tensor shapes, data flow, and behavioral assumptions.
4. Prefer modifying existing code over introducing new architecture.
5. Make the smallest necessary modification.

## Response Discipline

When asked to implement a specific change:

- Focus the response on that change.
- Do not propose multiple alternative architectures unless asked.
- Clearly identify any assumptions you had to make.