# Contributing to UniPruning

Thank you for your interest in contributing to UniPruning! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

---

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or identity.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or discriminatory comments
- Personal attacks or insults
- Publishing others' private information without permission
- Any other conduct that would be inappropriate in a professional setting

---

## 🛠️ How Can I Contribute?

### Reporting Bugs

Before creating a bug report:
1. Check the [existing issues](https://github.com/your-repo/issues) to avoid duplicates
2. Collect relevant information (error messages, system info, reproduction steps)

When creating a bug report, include:
- **Clear title** describing the issue
- **Detailed description** of the problem
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **System information** (OS, Python version, PyTorch version, GPU model)
- **Code snippets** or logs (if applicable)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a **clear and descriptive title**
- Provide a **detailed description** of the proposed functionality
- Explain **why this enhancement would be useful**
- Include **examples or mockups** if applicable

### Code Contributions

We welcome code contributions! Areas where you can help:

1. **Bug Fixes**: Fix reported bugs or issues
2. **New Features**: Implement new pruning strategies or optimization techniques
3. **Documentation**: Improve existing docs or add new tutorials
4. **Tests**: Add or improve test coverage
5. **Performance**: Optimize existing code for better efficiency
6. **Examples**: Add new example scripts or notebooks

---

## 🔧 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/UniPruning.git
cd UniPruning

# Add upstream remote
git remote add upstream https://github.com/original-repo/UniPruning.git
```

### 2. Create Environment

```bash
conda create -n Uniprune-dev python=3.10
conda activate Uniprune-dev
```

### 3. Install Dependencies

```bash
# Install in editable mode with development dependencies
pip install -e .
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 4. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 80)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings (unless single quotes avoid escaping)

### Code Formatting

We use **Black** for code formatting:

```bash
# Format your code
black searching/

# Check formatting
black searching/ --check
```

### Import Ordering

We use **isort** for import sorting:

```bash
# Sort imports
isort searching/

# Check import sorting
isort searching/ --check
```

### Type Hints

Use type hints for function signatures:

```python
def prune_model(
    model: torch.nn.Module,
    sparsity: float,
    calibration_data: Optional[Dataset] = None
) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    """
    Prune a model to achieve target sparsity.

    Args:
        model: The model to prune
        sparsity: Target sparsity ratio (0.0 to 1.0)
        calibration_data: Optional calibration dataset

    Returns:
        Tuple of (pruned_model, pruning_masks)
    """
    pass
```

### Documentation

- **Module docstrings**: Describe the module's purpose
- **Class docstrings**: Describe the class and its attributes
- **Function docstrings**: Use Google style format
- **Inline comments**: Use sparingly, only when necessary

Example:

```python
def extract_mask(
    saliency_scores: torch.Tensor,
    sparsity: float,
    pattern: str = "unstructured"
) -> torch.Tensor:
    """
    Extract pruning mask from saliency scores.

    Args:
        saliency_scores: Learned saliency variable Γ
        sparsity: Target sparsity ratio
        pattern: Pruning pattern ('unstructured' or '2:4')

    Returns:
        Binary pruning mask

    Raises:
        ValueError: If pattern is not supported

    Example:
        >>> scores = torch.randn(1024, 1024)
        >>> mask = extract_mask(scores, sparsity=0.5)
        >>> assert mask.sum() / mask.numel() == 0.5
    """
    pass
```

---

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Use `pytest` for testing
- Name test files as `test_*.py`
- Name test functions as `test_*`

Example test:

```python
import pytest
import torch
from searching.extract_mask_mixed import extract_mask

def test_extract_mask_unstructured():
    """Test unstructured mask extraction."""
    scores = torch.randn(100, 100)
    mask = extract_mask(scores, sparsity=0.5, pattern="unstructured")

    assert mask.shape == scores.shape
    assert torch.isclose(mask.sum() / mask.numel(), torch.tensor(0.5), atol=0.01)

def test_extract_mask_2_4():
    """Test 2:4 semi-structured mask extraction."""
    scores = torch.randn(100, 100)
    mask = extract_mask(scores, sparsity=0.5, pattern="2:4")

    # Check 2:4 pattern
    mask_reshaped = mask.reshape(-1, 4)
    assert (mask_reshaped.sum(dim=1) == 2).all()
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_extract_mask.py

# Run with coverage
pytest --cov=searching tests/

# Run with verbose output
pytest -v tests/
```

---

## 🔄 Pull Request Process

### Before Submitting

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```

3. **Check code style**:
   ```bash
   black searching/ --check
   flake8 searching/
   isort searching/ --check
   ```

4. **Update documentation** if you changed APIs or added features

5. **Add/update tests** for new functionality

### Submitting the PR

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what changed and why
   - Reference to related issues (e.g., "Fixes #123")
   - Screenshots or examples (if applicable)

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement

   ## Testing
   - [ ] Tests pass locally
   - [ ] Added new tests
   - [ ] Updated documentation

   ## Related Issues
   Closes #XXX

   ## Additional Notes
   Any additional context
   ```

### Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged

### After Merge

- Delete your feature branch:
  ```bash
  git branch -d feature/your-feature-name
  git push origin --delete feature/your-feature-name
  ```

---

## 🐛 Issue Guidelines

### Creating an Issue

Use appropriate issue templates:

- **Bug Report**: For reporting bugs
- **Feature Request**: For suggesting new features
- **Documentation**: For documentation improvements
- **Question**: For asking questions

### Issue Labels

Issues will be labeled for easy tracking:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `priority: high/medium/low`: Issue priority

---

## 🏷️ Commit Messages

Follow these guidelines for commit messages:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(pruning): add support for group-wise pruning

Implement group-wise pruning pattern that groups weights
by channels and prunes entire groups together.

Closes #123
```

```bash
fix(mask): correct mask extraction for 2:4 pattern

The previous implementation didn't correctly enforce the
2:4 constraint for edge cases where tensor dimensions
weren't divisible by 4.

Fixes #456
```

---

## 📚 Additional Resources

- [PyTorch Contributing Guide](https://pytorch.org/docs/stable/community/contribution_guide.html)
- [Transformers Contributing Guide](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

## 🙏 Recognition

All contributors will be recognized in:
- The project README
- Release notes
- Contributors page

Thank you for contributing to UniPruning! 🎉

---

## 📞 Questions?

If you have questions about contributing:
- Open a [Discussion](https://github.com/your-repo/discussions)
- Join our community chat
- Email the maintainers

We're here to help! 😊
