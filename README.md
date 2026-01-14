# Simity

Notebook Similarity Checker for Plagiarism Detection

## Features
- Compares Jupyter notebooks for similarity using sentence-transformer embeddings
- Supports comparison of only markdown, only code, or both
- Visualizes similarity matrix as a heatmap
- CLI interface for flexible usage

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd simity
   ```
2. Install dependencies (Python >=3.11 recommended) using [uv](https://github.com/astral-sh/uv):
   ```bash
   uv sync
   ```
   or use old, but still gold:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python main.py <NOTEBOOK_ROOT_PATH> [--mode all|markdown|code] [--window]
```

- `<NOTEBOOK_ROOT_PATH>`: Root directory to search for Jupyter notebooks (`.ipynb`)
- `--mode`: Comparison mode (default: `all`)
    - `all`: Compare both code and markdown
    - `markdown`: Compare only markdown cells
    - `code`: Compare only code cells
- `--window`: Show the heatmap in a dedicated window (blocks execution)

### Example
```bash
python main.py Exercise-1 --mode markdown --window
```

## Output
- Lists pairs of submissions with high similarity (>85%)
- Shows a heatmap of all similarities

## Requirements
- Python >=3.11
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- sentence-transformers
- uv (for installation)

## License
MIT
