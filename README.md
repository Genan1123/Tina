# Tina
# Tina: Autonomous Bioinformatics Agent

**Tina** is an AI-powered multi-agent system designed to autonomously analyze large-scale bioinformatics data (e.g., FASTQ, genome, metatranscriptomes) with minimal user input. It iteratively performs **reasoning → task execution → quality evaluation**, using multiple specialized AI models for robust and flexible analysis.

## 🧠 Architecture Overview

- `Executor Agent`: Generates bash/Python/R commands and executes them on local or HPC environments.
- `Supervisor Agent`: Evaluates results, checks errors, and determines if a retry or correction is needed.
- `Critic MoA`: Ensemble voting model for multi-agent evaluation and feedback loop.
- `Chat Agent`: Interface for dynamic human-AI interaction and override.

## 🗂️ Folder Structure
