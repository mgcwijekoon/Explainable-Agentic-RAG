# Explainable Agentic RAG System

Starter project structure for the DS205.3 Data Science with Python assignment.

## Requirements
- Python 3.11 (or a compatible Python 3 version required by your course)
- Visual Studio Code

## Windows setup
Open this folder in VS Code, then open **Terminal → New Terminal** and run:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` is not recognized, try `py` instead:

```powershell
py -m venv venv
```

## Project folders
- `data/raw/`: original PDF files
- `data/processed/`: extracted or processed data
- `src/`: Python source code
- `tests/`: project tests
- `chroma_db/`: local vector database data (generated while running the app)

Do not put passwords or API keys in Git. Store private values in `.env`; `.env` is ignored by Git.
