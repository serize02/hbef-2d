
# HBEF 2D

## Installation

### Install `uv`

#### macOS
```bash
brew install uv
```

#### Windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Sync Environment

After installing `uv`, sync the project environment:

```bash
uv sync
```

This command installs all dependencies and creates a virtual environment as specified in your project configuration.

## Usage

Once synced, activate the environment

```bash
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
````

Then download the model artifacts:

```bash
uv run download_artifacts.py
```

Place `.avi` echocardiogram video files in the `test` directory before running.


Then run the main scrip:

```bash
uv run main.py
```