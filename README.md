# chatgpt-mcp-data-quality-assistant

This project is a **local Model Context Protocol (MCP) server** that lets ChatGPT
run **data quality checks and exploratory data analysis (EDA)** on my local
datasets via natural language.

I expose the server over HTTP (using `ngrok`) and register it as a custom MCP
connection in ChatGPT. Once connected, I can ask ChatGPT things like:

- “Use the Local Data Quality & EDA connector and list the available datasets.”
- “Profile the `titanic.csv` dataset and summarise the main findings.”
- “Run data quality checks on `titanic` and give me the issues and a quality score.”
- “Open the profiling report for `titanic` and summarise it.”

ChatGPT then calls my local MCP tools, which are implemented in Python on top of
`pandas` and profiling libraries, and returns the results back into the chat.

---

## Features

- **MCP server** implemented in Python, compatible with ChatGPT’s MCP connections.
- **Dataset discovery**: lists available datasets from the local `data/` folder.
- **Automatic EDA**:
  - Generates a profiling report for `titanic.csv` with summary statistics,
    missing values, and correlations.
  - Saves the report into `reports/` as Markdown.
- **Data quality checks**:
  - Basic checks on missing values, numeric ranges, and simple consistency rules.
  - Returns a **data quality score** and a list of issues.
- **Local-first**: all data stays on my machine; ChatGPT only sees the results
  that the MCP server returns.

---

## Tech Stack

- **Python**
- **pandas** for data manipulation
- **Profiling / EDA** libraries for automated dataset reports
- **FastAPI / Starlette** + **Server-Sent Events (SSE)** to implement the MCP transport
- **ngrok** to expose the local MCP server to ChatGPT
- **Model Context Protocol (MCP)** for the tool definitions and schema

---

## How it works

1. **Core logic** (`core.py`)
   - Loads datasets from the `data/` directory.
   - For `titanic.csv`, it generates a profiling report with:
     - Row/column counts (e.g. 887 rows, 8 columns for this Titanic file).
     - Summary statistics for numeric columns (`Age`, `Fare`, etc.).
     - Correlations between variables such as `Survived`, `Pclass`, `Fare`, etc. :contentReference[oaicite:1]{index=1}
   - Implements functions like:
     - `list_datasets()`
     - `profile_dataset(name)`
     - `run_quality_checks(name)` (returns issues + a quality score)
     - `get_report(name)` (loads the Markdown report for ChatGPT to summarise)

2. **MCP server** (`server.py`)
   - Wraps the core functions as **MCP tools**.
   - Exposes them over HTTP using FastAPI + SSE (`text/event-stream`), so that
     ChatGPT can connect as an MCP client.
   - Uses JSON-RPC 2.0 under the hood to conform to MCP expectations.

3. **Exposing it to ChatGPT**
   - Start the MCP server locally:
     ```bash
     python server.py
     ```
   - Expose it via `ngrok`:
     ```bash
     ngrok http 8000
     ```
   - Use the `https://<your-subdomain>.ngrok.app/mcp` URL in the ChatGPT MCP
     configuration, with `Accept: text/event-stream`.
   - Create a new MCP connection in ChatGPT and point it to this URL.

4. **Using it from ChatGPT**
   - Example prompts once the connector is enabled:
     - “Use the Local Data Quality & EDA connector and list the available datasets.”
     - “Profile the titanic.csv dataset and summarise the main findings.”
     - “Run data quality checks on titanic and give me the issues and the quality score.”
     - “Open the profiling report for titanic and summarise it.”

---

## Example: Titanic dataset

As a demo, I use a **Titanic passenger dataset** stored at `data/titanic.csv`.

The profiling report includes:

- **887 rows** and **8 columns** with fields like `Survived`, `Pclass`, `Age`,
  `Fare`, and family relationships.
- Descriptive statistics (mean, std, min, max, quartiles).
- Correlation matrix to see how `Pclass`, `Age`, `Fare`, etc. relate to survival. :contentReference[oaicite:2]{index=2}

The generated Markdown report lives in `reports/titanic_profile.md` and can be
opened and summarised by ChatGPT via the MCP connector.

---

## Setup & Run

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # On macOS/Linux
   pip install -r requirements.txt
