# ML Project Template - Setup Guide

## Initial Setup Steps

### 1. Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Virtual Environment
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install base dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import jupyter; print('Jupyter: OK')"
ruff --version
```

## VS Code Setup

### 1. Open the project in VS Code
Open VS Code and then open this folder:
```bash
code .
```

### 2. Install Extensions
When prompted, install recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Ruff (charliermarsh.ruff)
- Jupyter (ms-toolsai.jupyter)

### 3. Select Python Interpreter
- Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
- Type "Python: Select Interpreter"
- Choose the interpreter from `.venv` folder

## First Steps

### 1. Explore the Example Notebook
```bash
jupyter notebook notebooks/01_experiment.ipynb
```

### 2. Run Tests
```bash
pytest
```

### 3. Try the Training Script
```bash
python src/train.py
```

### 4. Lint Your Code
```bash
ruff check .
ruff format .
```

## Next Steps

1. **Add your data** to `data/raw/`
2. **Modify the notebook** `notebooks/01_experiment.ipynb` for your use case
3. **Update models** in `src/models/model.py`
4. **Customize preprocessing** in `src/data/dataset.py`
5. **Update dependencies** in `pyproject.toml` as needed

## Troubleshooting

### Issue: UV not found
**Solution**: Add UV to your PATH or restart your terminal

### Issue: PyTorch not working with GPU
**Solution**: Install PyTorch with CUDA support:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Jupyter kernel not found
**Solution**: Install ipykernel in the virtual environment:
```bash
uv pip install ipykernel
python -m ipykernel install --user --name=ml-project
```

### Issue: VS Code not detecting Python
**Solution**: 
1. Restart VS Code
2. Manually select interpreter (Cmd+Shift+P → "Python: Select Interpreter")
3. Choose `.venv/bin/python`
