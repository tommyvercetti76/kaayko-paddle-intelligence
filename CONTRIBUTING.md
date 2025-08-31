# Contributing to Kaayko Paddle Intelligence

🎉 Thank you for your interest in contributing to Kaayko! We welcome contributions from the community and are excited to see how you can help improve paddle safety predictions worldwide.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/tommyvercetti76/kaayko-paddle-intelligence.git
   cd kaayko-paddle-intelligence
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt  # Unified file includes all dev tools
   ```

## 🛠️ Development Workflow

### Setting up Development Environment

```bash
# Install pre-commit hooks
pre-commit install

# Run tests to ensure everything works
pytest tests/ -v

# Run code formatting
black kaayko/ tests/ examples/
isort kaayko/ tests/ examples/

# Run linting
flake8 kaayko/ tests/ examples/
mypy kaayko/
```

### Making Changes

1. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Run the full test suite**:
   ```bash
   pytest tests/ -v --cov=kaayko
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

## 🧪 Testing

We maintain high test coverage and require all contributions to include tests.

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=kaayko --cov-report=html

# Run specific test file
pytest tests/test_predictor.py -v
```

### Writing Tests
- Place test files in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Include both unit tests and integration tests
- Mock external dependencies (weather APIs, file I/O)

## 📋 Code Standards

### Python Style
We follow [PEP 8](https://pep8.org/) with some modifications:
- **Line length**: 88 characters (Black default)
- **Import sorting**: Use isort
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings

### Code Formatting
```bash
# Format code
black kaayko/ tests/ examples/

# Sort imports  
isort kaayko/ tests/ examples/

# Check formatting
black --check kaayko/ tests/ examples/
```

### Commit Messages
We use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` adding tests
- `refactor:` code refactoring
- `perf:` performance improvements

Example: `feat: add regional weather specialist models`

## 🎯 Types of Contributions

### 🐛 Bug Reports
Use GitHub Issues with the "bug" label:
- Describe the bug clearly
- Include steps to reproduce
- Provide error messages and logs
- Specify your environment (OS, Python version)

### ✨ Feature Requests  
Use GitHub Issues with the "enhancement" label:
- Describe the feature and its benefits
- Provide use cases and examples
- Consider implementation complexity

### 📝 Documentation
- Fix typos and improve clarity
- Add examples and tutorials
- Update API documentation
- Translate documentation

### 🧠 Model Improvements
- Add new weather features
- Improve prediction algorithms
- Add regional specialist models
- Optimize training pipeline

### 🌍 Data Contributions
- Add new lake locations
- Provide regional weather patterns
- Contribute climate zone data
- Improve data quality

## 📚 Documentation

### Building Documentation
```bash
# All docs dependencies included in main requirements
pip install -r requirements.txt

# Serve docs locally (if mkdocs is added to requirements)
# mkdocs serve

# Build documentation (if mkdocs is added to requirements)
# mkdocs build
```

### Documentation Guidelines
- Use clear, concise language
- Include code examples
- Add diagrams for complex concepts
- Keep API documentation up to date

## 🌟 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub** contributor graphs and stats

## 📞 Getting Help

- **GitHub Discussions**: Ask questions and get help
- **GitHub Issues**: Report bugs and request features  
- **Discord**: Join our community chat (link coming soon)

## 📄 License

By contributing to Kaayko, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for helping make water sports safer for everyone!** 🌊🚣‍♀️
