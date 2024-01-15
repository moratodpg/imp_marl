# Guidelines for contributing to the project.

## Linting:

Your code should be linted with [flake8](https://flake8.pycqa.org/en/latest/) that ensure PEP8.

By using pre-commit hooks, you can ensure that your code is linted before you commit it.

To install pre-commit, see [here](https://pre-commit.com/#install).

You can either run the pre-commit before every commit

```bash
pre-commit run --all-files
```

Or install it as a git hook so that every commit will run the pre-commit hooks for you.
To install the pre-commit hooks, run:

```bash
pre-commit install
```
Linting configuration, see [.pre-commit-config.yaml](.pre-commit-config.yaml) :
- pre-commit-hooks checks some of the simple things that can be checked before committing.
- µfmt is a safe, atomic code formatter for Python. It changes your file automatically to make it conform to the PEP 8 style guide.
- flake8 checks the style and quality of the python code.
- pydocstyle checks the style of the docstrings in the python code.
