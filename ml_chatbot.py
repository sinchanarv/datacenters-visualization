"""
ml_chatbot.py
─────────────
Feature 4: AI Chatbot powered by OpenAI GPT
- Reads your API key from .env file (OPENAI_API_KEY)
- Sends a summary of the filtered dataframe as context to GPT
- Answers any natural language question about the data
"""

import pandas as pd
import streamlit as st
import os
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _get_client():
    """Get OpenAI client using key from env or streamlit secrets."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # Also check st.secrets if deployed
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _build_context(df: pd.DataFrame) -> str:
    """Summarise the dataframe into a compact text block for GPT context."""
    total        = len(df)
    regions      = df['Region'].nunique()
    avg_stress   = df['Water_Stress_Score'].mean()
    total_mw     = df['Estimated_MW'].sum()
    total_water  = df['Daily_Water_Liters'].sum() / 1e9
    total_lives  = df['Equivalent_Human_Lives'].sum() / 1e6
    critical     = (df['Risk_Category'] == 'Critical').sum()
    high         = (df['Risk_Category'] == 'High').sum()

    # Top 10 most stressed hubs
    top_hubs = (
        df.nlargest(10, 'Water_Stress_Score')
          [['Hub_Name', 'Region', 'Water_Stress_Score', 'Estimated_MW', 'Risk_Category']]
          .to_string(index=False)
    )

    # Per-region summary
    region_agg = (
        df.groupby('Region')
          .agg(
              Hubs=('Hub_Name', 'count'),
              Avg_Stress=('Water_Stress_Score', 'mean'),
              Critical=('Risk_Category', lambda x: (x == 'Critical').sum()),
              Total_MW=('Estimated_MW', 'sum'),
          )
          .sort_values('Avg_Stress', ascending=False)
          .reset_index()
          .to_string(index=False)
    )

    return f"""
You are an expert water-risk analyst for AI infrastructure. 
The user is exploring the AI Hydro Risk Atlas — a dataset of data centres and their water stress levels.

=== DATASET SUMMARY (currently filtered view) ===
- Total data centres: {total:,}
- Regions: {regions}
- Average water stress score: {avg_stress:.3f} / 5  (WRI Aqueduct scale)
- Total AI capacity: {total_mw:,} MW
- Total daily water consumption: {total_water:.2f} billion litres
- Human water equivalent: {total_lives:.1f} million people/day
- Critical risk hubs (stress > 3.5): {critical}
- High risk hubs (stress > 2.5): {high}

=== TOP 10 MOST STRESSED HUBS ===
{top_hubs}

=== REGIONAL BREAKDOWN ===
{region_agg}

Answer the user's question using this data. Be concise, analytical, and specific.
Use numbers from the dataset where possible. If you don't know something, say so.
""".strip()


def _ask_gpt(question: str, df: pd.DataFrame, history: list) -> str:
    """Send question + context to GPT and return the answer."""
    client = _get_client()
    if client is None:
        return (
            "⚠️ **OpenAI API key not found.**\n\n"
            "Please create a `.env` file in your project folder with:\n"
            "```\nOPENAI_API_KEY=sk-...\n```\n"
            "Then restart the app with:\n"
            "```\npip install python-dotenv openai\n"
            "```\n"
            "And add this at the top of `app.py`:\n"
            "```python\nfrom dotenv import load_dotenv\nload_dotenv()\n```"
        )

    context = _build_context(df)

    messages = [{"role": "system", "content": context}]

    # Include last 6 messages of history for multi-turn memory
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",       # fast + cheap; change to gpt-4o for better answers
            messages=messages,
            max_tokens=600,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI API error: {str(e)}"


# ── Streamlit Chatbot UI ──────────────────────────────────────────────────────
def render_chatbot_section(df: pd.DataFrame):
    st.markdown("<div class='section-title'>▸ AI Analyst Chatbot — Powered by GPT</div>",
                unsafe_allow_html=True)

    # API key input (if not in env)
    api_key_env = os.getenv("OPENAI_API_KEY", "")
    if not api_key_env:
        st.markdown("""
        <div style='background:#0a1628;border:1px solid #ffaa00;border-radius:10px;
             padding:14px 20px;margin-bottom:16px;font-size:0.82rem;color:#ffaa00;'>
        ⚠️ No API key found in environment. Enter it below (only stored in session, never saved).
        </div>""", unsafe_allow_html=True)
        key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            key="openai_key_input",
        )
        if key_input:
            os.environ["OPENAI_API_KEY"] = key_input
            st.success("✅ Key set for this session!")
    else:
        st.markdown("""
        <div style='background:#0a1628;border:1px solid #00ff88;border-radius:10px;
             padding:10px 20px;margin-bottom:16px;font-size:0.82rem;color:#00ff88;'>
        ✅ OpenAI API key loaded · Model: gpt-4o-mini
        </div>""", unsafe_allow_html=True)

    # Suggested questions
    st.markdown("<p style='font-size:0.75rem;color:#4a6080;margin-bottom:8px;'>QUICK QUESTIONS</p>",
                unsafe_allow_html=True)

    quick_qs = [
        "Which region has the worst water stress?",
        "Summarise the dataset for me",
        "Which hubs are most at risk?",
        "How much water do these data centres use daily?",
        "Recommend the safest regions for a new data centre",
        "Compare water stress across regions",
    ]

    cols = st.columns(3)
    for i, q in enumerate(quick_qs):
        if cols[i % 3].button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state["chat_prefill"] = q

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Render conversation
    for msg in st.session_state["chat_history"]:
        role  = msg["role"]
        text  = msg["content"]
        align = "right" if role == "user" else "left"
        bg    = "#0f2040" if role == "user" else "#0a1628"
        border= "#00d4ff" if role == "user" else "#1a2d4a"
        label = "You" if role == "user" else "🤖 GPT Analyst"
        st.markdown(f"""
        <div style='text-align:{align};margin:6px 0;'>
          <div style='display:inline-block;max-width:88%;background:{bg};
               border:1px solid {border};border-radius:10px;padding:10px 16px;
               text-align:left;font-size:0.85rem;line-height:1.65;white-space:pre-wrap;'>
            <div style='font-size:0.65rem;color:#4a6080;margin-bottom:4px;
                 text-transform:uppercase;letter-spacing:1px;'>{label}</div>
            {text}
          </div>
        </div>""", unsafe_allow_html=True)

    # Input row
    prefill = st.session_state.pop("chat_prefill", "")
    user_input = st.text_input(
        "Ask anything about the data...",
        value=prefill,
        placeholder="e.g. Which country should we avoid for new data centres?",
        key="chat_input",
        label_visibility="collapsed",
    )

    col_send, col_clear = st.columns([1, 5])
    with col_send:
        send = st.button("Send ↗", use_container_width=True)
    with col_clear:
        if st.button("Clear chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    if send and user_input.strip():
        with st.spinner("GPT is thinking..."):
            response = _ask_gpt(
                user_input.strip(), df,
                st.session_state["chat_history"]
            )
        st.session_state["chat_history"].append({"role": "user",     "content": user_input.strip()})
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        st.rerun()