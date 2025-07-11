# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This is a deep learning educational repository using Python 3.10.6. Always work within the virtual environment:

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Development Commands

### Jupyter Notebook Development
```bash
# Start Jupyter Lab (primary development environment)
jupyter lab

# Start Jupyter Notebook (alternative)
jupyter notebook
```

### Python Execution
```bash
# Run individual Python scripts
python script_name.py

# Run with GPU support (if available)
python script_name.py --device cuda
```

## Repository Architecture

This repository follows a **progressive learning structure** for deep learning education:

### Core Learning Paths

1. **`/d2l-zh/`** - Complete "Dive into Deep Learning" Chinese textbook implementation
   - Organized by chapters covering fundamentals to advanced topics
   - Dual implementations: PyTorch and TensorFlow
   - Self-contained modules with theory and practice

2. **`/pytorch/`** - Structured PyTorch learning progression
   - `practise_1_getstarted.ipynb` - Introduction with Fashion-MNIST
   - `week3/` - Core implementations (linear regression, MLP, CNN, RNN, Transformer)
   - `week4/` - Advanced topics (text generation, diffusion models)
   - `data/` - Local datasets (FashionMNIST, MNIST)

3. **`/pytorch_2025/`** - Latest techniques and implementations
   - `month_7/practise_1_flashattention.ipynb` - Flash Attention optimization
   - Progressive monthly learning structure

4. **`/tensorflow/`** - TensorFlow learning path
   - `week1/` - Fundamentals (data, linear algebra, calculus, probability)
   - `week2/` - Linear regression implementations

### Key Technologies

- **PyTorch 2.1.0** - Primary deep learning framework
- **TensorFlow** - Secondary framework for comparison
- **D2L library (1.0.3)** - Educational utilities from "Dive into Deep Learning"
- **Jupyter** - Interactive development environment
- **NumPy, Pandas, Matplotlib** - Data manipulation and visualization

### Implementation Patterns

**Notebook Structure Convention:**
1. Import dependencies
2. Data loading and preprocessing
3. Model definition (often from scratch implementation first)
4. Training loop with visualization
5. Evaluation and analysis
6. Advanced techniques (optimization, regularization)

**Model Implementation Pattern:**
```python
class ModelName(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Layer definitions
    
    def forward(self, x):
        # Forward pass implementation
        return output

# Training pattern with device handling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModelName().to(device)
```

### Learning Philosophy

This codebase emphasizes:
- **Theory → From Scratch → Framework Implementation** progression
- **Comparative Learning** - PyTorch vs TensorFlow implementations
- **Interactive Experimentation** - Jupyter-first development
- **Practical Application** - Real-world examples and datasets

### Code Conventions

- **Chinese comments** for educational explanations
- **English** for variable names and function names
- **Modular design** with reusable components
- **Progressive complexity** from basic to advanced implementations
- **Comprehensive documentation** within notebooks

### Hardware Considerations

- **CPU training** supported for all implementations
- **GPU acceleration** available and recommended for larger models
- **Memory-efficient implementations** for resource-constrained environments
- **Flash Attention** for memory optimization in transformer models

### File Organization

- **Jupyter notebooks** (.ipynb) for interactive development and learning
- **Python scripts** (.py) for utility functions and reusable components
- **Data directories** for local datasets (excluded from git via .gitignore)
- **Model checkpoints** and weights excluded from version control

When working on this repository, prioritize educational clarity over production optimization, and maintain the progressive learning structure that allows students to build understanding step by step.