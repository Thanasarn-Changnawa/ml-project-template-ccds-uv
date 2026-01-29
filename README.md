# ML Project Template with UV and PyTorch

A comprehensive template for Machine Learning projects using UV for dependency management, PyTorch for deep learning, and Ruff for code quality.

## Features

- 🚀 **UV**: Fast Python package installer and resolver
- 🔥 **PyTorch**: Deep learning framework
- 🧹 **Ruff**: Fast Python linter and formatter
- 📓 **Jupyter Notebooks**: Interactive experimentation environment
- 💻 **VS Code**: Pre-configured workspace settings
- 🧪 **Pytest**: Testing framework
- 📊 **ML-Ready**: Pre-configured project structure for ML workflows

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw, immutable data
│   ├── processed/        # Cleaned, preprocessed data
│   └── external/         # External data sources
├── models/               # Trained model files
├── notebooks/            # Jupyter notebooks for experimentation
│   └── 01_experiment.ipynb
├── src/                  # Source code
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model definitions
│   ├── utils/           # Utility functions
│   └── train.py         # Training script
├── tests/               # Unit tests
├── .vscode/             # VS Code configuration
├── .gitignore
├── pyproject.toml       # Project dependencies and configuration
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [UV](https://github.com/astral-sh/uv) package manager

### Installation

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone this repository**:
   ```bash
   git clone <your-repo-url>
   cd ml-project-template-ccds-uv
   ```

3. **Create virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

4. **Install development dependencies**:
   ```bash
   uv pip install -e ".[dev]"
   ```

## Usage

### Running Experiments in Jupyter Notebooks

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open the example notebook**:
   - Navigate to `notebooks/01_experiment.ipynb`
   - The notebook includes examples for:
     - Data loading and preprocessing
     - Model building with PyTorch
     - Training loops
     - Visualization
     - Model evaluation

### Training Models

Run the training script:
```bash
python src/train.py
```

### Running Tests

```bash
pytest
```

With coverage:
```bash
pytest --cov=src --cov-report=html
```

### Code Quality

**Format and lint code with Ruff**:
```bash
ruff check .
ruff format .
```

**Fix issues automatically**:
```bash
ruff check --fix .
```

## VS Code Setup

This template includes pre-configured VS Code settings:

1. **Open the workspace**:
   - File → Open Workspace from File
   - Select `.vscode/ml-project.code-workspace`

2. **Install recommended extensions** when prompted:
   - Python
   - Pylance
   - Ruff
   - Jupyter

3. **The workspace is configured with**:
   - Auto-formatting on save with Ruff
   - Python interpreter pointing to `.venv`
   - Jupyter notebook support
   - Debug configurations

## Development Workflow

### 1. Experimentation Phase

Use Jupyter notebooks for rapid experimentation:
- Load and explore data
- Try different preprocessing techniques
- Build and test model architectures
- Visualize results

### 2. Code Organization

Move stable code from notebooks to source modules:
- Data utilities → `src/data/`
- Models → `src/models/`
- Helper functions → `src/utils/`

### 3. Training

Create training scripts in `src/` for reproducible training:
```bash
python src/train.py
```

### 4. Testing

Add tests for your code:
```bash
pytest tests/
```

## Customization

### Adding Dependencies

**For regular dependencies**:
```bash
uv pip install package-name
```

Then add to `pyproject.toml`:
```toml
dependencies = [
    "package-name>=version",
]
```

**For development dependencies**:
```bash
uv pip install --dev package-name
```

Update `pyproject.toml`:
```toml
[project.optional-dependencies]
dev = [
    "package-name>=version",
]
```

### PyTorch with CUDA

For GPU support, install PyTorch with CUDA:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Best Practices

1. **Keep notebooks clean**: Use notebooks for exploration, move production code to `.py` files
2. **Version control**: Commit `.ipynb` files but consider clearing outputs before committing
3. **Data management**: Never commit large data files; use `.gitignore`
4. **Model checkpoints**: Save models to `models/` directory
5. **Documentation**: Document your code and maintain this README
6. **Testing**: Write tests for critical functionality
7. **Code quality**: Run Ruff before committing

## Common Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install/update dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format .

# Lint code
ruff check .

# Start Jupyter
jupyter notebook

# Train model
python src/train.py
```

## Troubleshooting

### UV Installation Issues
- Ensure UV is in your PATH after installation
- Try restarting your terminal

### PyTorch Installation
- For GPU support, install CUDA-enabled PyTorch
- Check PyTorch website for specific installation commands

### VS Code Extensions
- Make sure to install recommended extensions
- Restart VS Code after installing extensions

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request
