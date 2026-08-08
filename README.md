# Counterfactual Causal Biomarker Discovery (CCBD) Engine

The Counterfactual Causal Biomarker Discovery (CCBD) Engine is an AI tool built with LangGraph, Google Gemini 3.5 Flash, and Streamlit. It automates the extraction of biological causal networks from medical literature, simulates Pearlian do-calculus interventions, and evaluates the biological plausibility of hypothetical biomarker targets.

By shifting from simple observational correlation to formal causal inference, CCBD helps distinguish true upstream drivers of disease from downstream symptoms.

---

## Overview

Medical literature often reports strong statistical correlations that do not imply direct therapeutic targets. Lowering a correlated protein level may only mask a symptom rather than treat the underlying disease. 

CCBD addresses this challenge by orchestrating a three-agent state machine:

1. Agent 1 (DAG Extractor): Parses raw biomedical text into a structured Directed Acyclic Graph (DAG) representing causal relationships between genes, proteins, and phenotypes.
2. Agent 2 (Counterfactual Mutator): Selects upstream nodes and simulates targeted suppression using Pearlian do-calculus to predict non-obvious downstream effects.
3. Agent 3 (Cohort Validator): Evaluates the hypothesis against biological conservation laws and assigns a quantitative plausibility score.

If validation fails, a conditional router directs the workflow to attempt alternative mutation paths until a viable hypothesis is identified or the safety iteration limit is reached.

---

## Key Features

- Multi-Agent Orchestration: Powered by LangGraph to maintain cycle execution, conditional branching, and state management.
- Structured Graph Output: Enforces strict JSON schemas using Pydantic models to construct reliable biological DAGs.
- Interactive Dashboard: Built with Streamlit for user-friendly configuration, key management, and visual step-by-step trace inspection.
- Secure API Integration: Allows users to supply their own Google Gemini API key dynamically via the UI without hardcoding secrets.
- Robust Text Parsing: Built-in sanitization routines to handle non-standard model responses and maintain clean visual rendering.

---

## Architecture Flow

~~~
+------------------------+
| Input Medical Context  |
+------------------------+
           |
           v
+------------------------+
|   Agent 1: Extract DAG |
+------------------------+
           |
           v
+------------------------+
| Agent 2: Mutate Graph  | <-------+
+------------------------+         |
           |                       |
           v                       | (Retry on low score)
+------------------------+         |
| Agent 3: Verify Cohort |         |
+------------------------+         |
           |                       |
           v                       |
+------------------------+         |
|   Conditional Router   | --------+
+------------------------+
           |
           | (Score > 0.75 OR Iterations >= 3)
           v
+------------------------+
|      Final Output      |
+------------------------+
~~~

---

## Prerequisites

- Python 3.10 or higher
- A Google Gemini API Key (Available via Google AI Studio)

---

## Installation and Local Setup

1. Clone the repository:
~~~bash
git clone https://github.com/your-username/ccbd-engine.git
cd ccbd-engine
~~~

2. Create and activate a virtual environment (optional but recommended):
~~~bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
~~~

3. Install required dependencies:
~~~bash
pip install -r requirements.txt
~~~

4. Launch the Streamlit application:
~~~bash
streamlit run app.py
~~~

5. Open the local URL provided by Streamlit (typically `http://localhost:8501`), enter your Gemini API key in the sidebar, and begin running analyses.

---

## Example Usage

### Input
- Target Disease: Severe Asthma
- Literature Context:
  "Studies show that environmental allergens trigger the release of IL-33 from airway cells. This IL-33 upregulates the activity of ILC2 cells. Consequently, activated ILC2 cells produce high levels of IL-5, which directly leads to eosinophilic inflammation in the lungs."

### Generated Outputs
- Causal DAG: Structured node/edge array linking Allergens -> IL-33 -> ILC2 -> IL-5 -> Eosinophilic Inflammation.
- Counterfactual Hypothesis: Evaluation of `do(IL-33 = Suppressed)` showing total block of downstream inflammatory cascades despite allergen presence.
- Verification Verdict: Quantitative endorsement score (e.g., 0.85) with step-by-step immunological rationale.

---

## License

Distributed under the MIT License. See LICENSE for more information.