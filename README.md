# Tina
# Tina: Autonomous Bioinformatics Agent

**Tina** is an AI-powered multi-agent system designed to autonomously analyze large-scale bioinformatics data (e.g., FASTQ, genome, metatranscriptomes) with minimal user input. It iteratively performs **reasoning → task execution → quality evaluation**, using multiple specialized AI models for robust and flexible analysis.

## 🧠 Architecture Overview

- `Executor Agent`: Generates bash/Python/R commands and executes them on local or HPC environments.
- `Supervisor Agent`: Evaluates results, checks errors, and determines if a retry or correction is needed.
- `Critic MoA`: Ensemble voting model for multi-agent evaluation and feedback loop.
- `Chat Agent`: Interface for dynamic human-AI interaction and override.

## 🗂️ Folder Structure

```
.
├── README.md           # documentation and usage guide
├── Tina_MoA_v22.py     # version 22 of the agent
├── Tina_MoA_v23.py     # version 23 of the agent
└── Tina_MoA_v24.py     # version 24 of the agent
```

Runtime logs are stored in the `tina_logs/` directory when the agent runs.

## 🚀 Usage

Install dependencies and set your API key before running any version:

```bash
pip install numpy rich together
export TOGETHER_API_KEY=<your_token>
```

Each script can then be executed with Python:

```bash
python Tina_MoA_v22.py  # run version 22
python Tina_MoA_v23.py  # run version 23
python Tina_MoA_v24.py  # run version 24
```

You may also use `TINA_API_KEY` instead of `TOGETHER_API_KEY` to provide the
required Together API access token.
