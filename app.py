import os
import json
import streamlit as st
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 1. Page Configuration & UI Setup
# ==========================================
st.set_page_config(
    page_title="CCBD Engine | Causal Biomarker Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 CCBD: Counterfactual Causal Biomarker Discovery")
st.markdown("""
This application uses a multi-agent **LangGraph** architecture to extract causal biological pathways from literature, 
simulate 'What-If' interventions (Pearlian *do-calculus*), and verify plausible novel drug targets.
""")

# ==========================================
# 2. Sidebar: API Key Management
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    st.markdown("[Get a free Gemini API key here](https://aistudio.google.com/app/apikey)")
    
    if not api_key:
        st.warning("Please enter your API key to proceed.")
        st.stop()
    else:
        st.success("API Key provided.")
        os.environ["GOOGLE_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. **Extract:** Maps biological entities from text into a causal graph.")
    st.markdown("2. **Mutate:** Simulates suppressing a target node.")
    st.markdown("3. **Verify:** Scores the biological plausibility of the intervention.")

# ==========================================
# 3. LangGraph Architecture Definitions
# ==========================================

# Define Causal Graph Schemas
class CausalEdge(BaseModel):
    source: str = Field(description="The source biological entity (e.g., Gene, Protein)")
    target: str = Field(description="The target biological entity affected")
    effect_direction: str = Field(description="'upregulate' or 'downregulate'")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")

class CausalDAG(BaseModel):
    nodes: List[str] = Field(description="List of all unique biological entities mentioned")
    edges: List[CausalEdge] = Field(description="List of causal relationships between nodes")

# Define the State for the Graph
class CounterfactualState(TypedDict):
    literature_context: str
    target_disease: str
    causal_dag: Dict[str, Any]
    counterfactual_hypothesis: str
    intervention_result: str
    verification_score: float
    iterations: int # Safety counter to prevent infinite loops

# Initialize LLMs (Using current 3.5 flash for chat, and structured output for DAG)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    dag_llm = llm.with_structured_output(CausalDAG)
except Exception as e:
    st.error(f"Error initializing LLM: {e}. Please check your API key.")
    st.stop()

# ==========================================
# 4. Agent Nodes
# ==========================================

def dag_extractor_node(state: CounterfactualState):
    """Agent 1: Extracts the causal map from text."""
    prompt = f"""
    Extract a Directed Acyclic Graph (DAG) of causal biological pathways for {state['target_disease']}
    from the following context: {state['literature_context']}
    Ensure you capture specific genes, proteins, and phenotypes.
    """
    try:
        dag: CausalDAG = dag_llm.invoke(prompt)
        return {"causal_dag": dag.model_dump(), "iterations": state.get("iterations", 0) + 1}
    except Exception as e:
        return {"causal_dag": {"nodes": [], "edges": [], "error": str(e)}, "iterations": state.get("iterations", 0) + 1}

def counterfactual_mutation_node(state: CounterfactualState):
    """Agent 2: Proposes a 'What-If' intervention."""
    if "error" in state["causal_dag"]:
        return {"counterfactual_hypothesis": "Failed to extract DAG. Cannot mutate."}

    prompt = f"""
    Given this causal DAG: {json.dumps(state['causal_dag'])}
    Apply Pearlian do-calculus intervention: Select ONE upstream node (a cause, not a symptom) and simulate suppressing it.
    Identify counterfactual predictions that contradict a naive correlation. 
    Explain what happens to the downstream targets if this node is suppressed.
    """
    response = llm.invoke([
        SystemMessage(content="You are a Biomedical Causal Inference Agent. Be concise and analytical. CRITICAL RULE: ABSOLUTELY NO LaTeX, NO MathJax, NO math symbols (like $, -, or _). DO NOT format biological entities with any special characters. DO NOT use 'do()' notation in your output text. Write in plain, simple English sentences only."),
        HumanMessage(content=prompt)
    ])
    
    # Safely handle list responses
    content = response.content
    if isinstance(content, list):
        content = " ".join([str(c) for c in content if isinstance(c, str) or isinstance(c, dict) and c.get('text')])
    elif isinstance(content, dict):
        content = str(content.get('text', content))
        
    return {"counterfactual_hypothesis": str(content).replace('$', '').replace('\\text', '').replace('{', '').replace('}', '')}

def cohort_verification_node(state: CounterfactualState):
    """Agent 3: Verifies if the intervention makes biological sense."""
    if "error" in state["causal_dag"]:
        return {"intervention_result": "Failed.", "verification_score": 0.0, "iterations": state.get("iterations", 0) + 1}

    prompt = f"""
    Evaluate this counterfactual intervention hypothesis against biological conservation laws:
    HYPOTHESIS: {state['counterfactual_hypothesis']}
    DAG: {json.dumps(state['causal_dag'])}
    
    Assign a biological plausibility score between 0.0 and 1.0. 
    If the score is greater than 0.8, you MUST include the word "ENDORSE" in your response.
    Explain your reasoning.
    """
    response = llm.invoke([
        SystemMessage(content="You are a rigorous Biomedical Validator. Verify causal soundness. CRITICAL RULE: ABSOLUTELY NO LaTeX, NO MathJax, NO math symbols (like $, -, or _). DO NOT format biological entities with any special characters. Write in plain, simple English sentences only."), 
        HumanMessage(content=prompt)
    ])
    
    # Safely handle list responses
    content = response.content
    if isinstance(content, list):
        # Extract text from list blocks if necessary
        text_blocks = []
        for item in content:
            if isinstance(item, str):
                text_blocks.append(item)
            elif isinstance(item, dict) and 'text' in item:
                text_blocks.append(item['text'])
        content = " ".join(text_blocks)
    elif isinstance(content, dict):
        content = str(content.get('text', content))
    
    content = str(content).replace('$', '').replace('\\text', '').replace('{', '').replace('}', '')
    score = 0.85 if "ENDORSE" in content.upper() else 0.4
    
    # Increment iteration here to prevent infinite loops on retry
    return {"intervention_result": content, "verification_score": score, "iterations": state.get("iterations", 0) + 1}

def causal_router(state: CounterfactualState):
    """Decides whether to end the process or try mutating a different node."""
    if state.get("iterations", 0) >= 3: # Safety limit
        return "end"
    if state["verification_score"] > 0.75:
        return "end" 
    else:
        return "retry"

# ==========================================
# 5. Build the LangGraph
# ==========================================
workflow = StateGraph(CounterfactualState)
workflow.add_node("extract_dag", dag_extractor_node)
workflow.add_node("mutate_dag", counterfactual_mutation_node)
workflow.add_node("verify_cohort", cohort_verification_node)

workflow.set_entry_point("extract_dag")
workflow.add_edge("extract_dag", "mutate_dag")
workflow.add_edge("mutate_dag", "verify_cohort")
# If verification fails, it loops back to try a different mutation
workflow.add_conditional_edges("verify_cohort", causal_router, {"end": END, "retry": "mutate_dag"})

ccbd_app = workflow.compile()

# ==========================================
# 6. Streamlit User Interface
# ==========================================

st.subheader("Input Medical Literature")

col1, col2 = st.columns([1, 2])
with col1:
    target_disease = st.text_input("Target Disease", value="Severe Asthma")
with col2:
    default_text = "Studies show that environmental allergens trigger the release of IL-33 from airway cells. This IL-33 upregulates the activity of ILC2 cells. Consequently, activated ILC2 cells produce high levels of IL-5, which directly leads to eosinophilic inflammation in the lungs."
    literature = st.text_area("Literature Context", value=default_text, height=150)

if st.button("Run Causal Discovery Analysis", type="primary"):
    with st.spinner("Initializing Multi-Agent Workflow..."):
        
        # Prepare initial state
        initial_state = {
            "target_disease": target_disease,
            "literature_context": literature,
            "iterations": 0
        }
        
        # Execute Graph
        try:
            final_state = ccbd_app.invoke(initial_state)
            
            st.success("Analysis Complete!")
            
            # Display Results in Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Extracted Causal DAG", "🧪 Counterfactual Hypothesis", "✅ Verification Verdict"])
            
            with tab1:
                st.markdown("### Causal Map Extracted by Agent 1")
                if "error" in final_state.get("causal_dag", {}):
                    st.error(final_state["causal_dag"]["error"])
                else:
                    st.json(final_state["causal_dag"])
                    
            with tab2:
                st.markdown("### Intervention Proposed by Agent 2")
                st.info(final_state.get("counterfactual_hypothesis", "No hypothesis generated."))
                
            with tab3:
                st.markdown("### Plausibility Checked by Agent 3")
                score = final_state.get("verification_score", 0.0)
                
                if score > 0.75:
                    st.metric(label="Plausibility Score", value=f"{score:.2f}", delta="ENDORSED", delta_color="normal")
                else:
                    st.metric(label="Plausibility Score", value=f"{score:.2f}", delta="REJECTED", delta_color="inverse")
                    
                st.markdown("**Reasoning:**")
                st.write(final_state.get("intervention_result", "No result generated."))
                
                st.caption(f"Total Iterations (retries): {final_state.get('iterations', 0)}")

        except Exception as e:
            st.error(f"An error occurred during workflow execution: {e}")