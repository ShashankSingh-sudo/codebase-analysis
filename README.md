# Code Analyzer

Clone any Git repository and analyze ALL code on 5 parameters: Functions & Methods, Endpoints/APIs, Dependencies, Optimizations, and Potential Issues. Powered by Hugging Face LLMs with parallel processing.

## Features

- **Clone any Git repo** – Paste a GitHub/GitLab/Bitbucket URL and clone in one click
- **Browse folder structure** – See all folders and select a specific directory to analyze
- **LLM-powered analysis** – Uses Hugging Face models to analyze code and provide:
  - Functions/methods with descriptions
  - REST endpoints and API definitions
  - Key variables and constants
  - Dependencies and imports
  - Optimization opportunities (performance, readability, security)
  - Potential issues and anti-patterns

## Setup

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # or: source venv/bin/activate   # macOS/Linux
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Hugging Face token** – Already configured in the app for inference.

## Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. Enter a Git repository URL (e.g. `https://github.com/username/repo.git`)
2. Click **Clone & Explore**
3. Select a folder from the dropdown to analyze
4. Click **Run AI Analysis**
5. View the structured report with functions, endpoints, variables, and optimization suggestions

## Supported File Types

Python, JavaScript, TypeScript, JSX, TSX, Java, Go, Rust, C/C++, Ruby, PHP, Swift, Kotlin, Scala, R, SQL

## Models

You can choose from several Hugging Face models in the sidebar. Some may require:
- A Hugging Face Pro subscription for larger models
- Model acceptance on the Hub (e.g. Llama, CodeLlama)

Free-tier friendly options: `microsoft/Phi-3-mini-4k-instruct`, `HuggingFaceH4/zephyr-7b-beta`

