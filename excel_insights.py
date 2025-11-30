import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import google.generativeai as genai
import io
import os
import time
import random

# Page Config
st.set_page_config(page_title="- EDI -", layout="wide")

def get_ai_insights(df, api_key, insight_type="quality"):
    """
    Sends a data summary to Gemini for analysis.
    insight_type: 'quality' (technical) or 'business' (strategic)
    """
    if not api_key:
        return "⚠️ Please enter a Gemini API Key in the sidebar to unlock AI insights."
    
    try:
        genai.configure(api_key=api_key)
        
        # Create a token-efficient summary
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        desc = df.describe().to_markdown()
        head = df.head(5).to_markdown()
        
        base_prompt = f"""
        Dataset Summary:
        1. Columns & Types:
        {info_str}
        
        2. Statistical Summary:
        {desc}
        
        3. First 5 Rows:
        {head}
        """

        if insight_type == "quality":
            prompt = base_prompt + """
            Act as a Lead Data Engineer. Analyze this dataset summary for data quality and statistical health.
            Focus on:
            1. Completeness (missing data patterns and their impact).
            2. Consistency (potential outliers, anomalies, or weird values).
            3. Statistical distributions (skewness, variance issues).
            
            Provide 3-5 sharp, technical bullet points. Be direct and objective.
            Important: Be direct and professional. Do not use conversational filler (e.g.,"As a Lead Data Engineer", "Good morning", "Let's look at").
            Avoid jargon.
            """
        else:
            prompt = base_prompt + """
            Analyze this dataset summary for strategic business insights.
            Focus on:
            1. Key trends and patterns affecting business goals.
            2. Opportunities for forecasting, optimization, or growth.
            3. Actionable takeaways for decision-making.
            
            Provide 3-5 clear, objective business conclusions.
            Important: Be direct and professional. Do not use conversational filler (e.g., "Good morning", "Let's look at").
            Start directly with the insights. Avoid jargon.
            """
        
        # List of models to try in order of preference (Fastest -> Most Capable -> Legacy)
        models_to_try = [
            'gemini-2.0-flash-exp',       # Often has largest free context window in labs
            'gemini-2.5-flash',           # Newest Flash model
            'gemini-2.0-flash',           # Standard 2.0 Flash
            'gemini-exp-1206',            # High-reasoning experimental model
            'gemini-2.0-pro-exp',         # Pro experimental
            'gemini-1.5-flash',           # Reliable fallback
            'gemini-1.5-pro',             # Reliable fallback
        ]
        
        last_error = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_error = e
                continue
        
        # If all models fail, attempt to list what IS available to help debug
        try:
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            if available_models:
                models_str = "\n".join([f"- `{m}`" for m in available_models])
                return (f"⚠️ **Connection Error:** Could not connect to standard models.\n\n"
                        f"**However, the API successfully listed these available models:**\n{models_str}\n\n"
                        f"*Last error details: {str(last_error)}*")
            else:
                return f"⚠️ **Connection Error:** No models supporting generation were found. Last error: {str(last_error)}"
                
        except Exception as list_error:
            return f"⚠️ **Connection Error:** Could not connect to AI or list models. \n\nOriginal Error: {str(last_error)}\nList Error: {str(list_error)}"

    except Exception as e:
        return f"Error communicating with AI: {str(e)}"

def find_significant_relations(df, numeric_cols, categorical_cols):
    """
    Algorithmic check for interesting visualizations using statistical tests.
    Returns a list of dicts: {'type': 'boxplot'|'scatter', 'x': col, 'y': col, 'score': p-value/corr}
    """
    significant_plots = []
    
    # 1. ANOVA: Categorical vs Numeric (Find where categories actually matter)
    # We only check categories with a reasonable number of unique values (2-15) to avoid messy charts
    valid_cats = [c for c in categorical_cols if 2 <= df[c].nunique() <= 15]
    
    for cat in valid_cats:
        for num in numeric_cols:
            # Group samples by category
            groups = [group[num].dropna().values for name, group in df.groupby(cat)]
            if len(groups) > 1:
                # One-way ANOVA
                f_stat, p_val = stats.f_oneway(*groups)
                # If p < 0.05, the means are statistically different
                if p_val < 0.05:
                    significant_plots.append({
                        'type': 'boxplot',
                        'x': cat,
                        'y': num,
                        'score': p_val, # lower is better
                        'reason': f"Significant variation (p={p_val:.4f})"
                    })

    # 2. Pearson Correlation: Numeric vs Numeric
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        # Iterate through the upper triangle to avoid duplicates
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                r_val = corr_matrix.loc[col1, col2]
                
                # Threshold: Only show if correlation is strong (> 0.6)
                if r_val > 0.6:
                    significant_plots.append({
                        'type': 'scatter',
                        'x': col1,
                        'y': col2,
                        'score': 1 - r_val, # flip so we can sort by 'score' (lower is better/stronger)
                        'reason': f"Strong Correlation (r={r_val:.2f})"
                    })

    # Sort by score (lowest p-value or highest correlation) and take top 6
    significant_plots.sort(key=lambda x: x['score'])
    return significant_plots[:6]

def main():
    st.title("Excel Dynamic Insights")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        try:
            sys_api_key = st.secrets.get("GEMINI_API_KEY")
        except (FileNotFoundError, AttributeError):
            sys_api_key = os.environ.get("GEMINI_API_KEY")

        if sys_api_key:
            api_key = sys_api_key
            st.success("✅ Token Loaded")
        else:
            api_key = st.text_input("Gemini API Key (for AI Insights)", type="password")
            st.caption("Get a free key at aistudio.google.com")
            
        st.divider()
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'xls', 'csv'])

    if not uploaded_file:
        st.markdown(
            """
            <div style='text-align: center; padding: 50px; border: 2px dashed #cccccc; border-radius: 10px; opacity: 0.6;'>
                <h3>Waiting for Data</h3>
                <p>Upload a CSV or Excel file to begin automated analysis.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.stop()  # Stop execution here if no file is uploaded

    # --- File Processing & Animation ---
    if 'last_loaded_file' not in st.session_state or st.session_state.last_loaded_file != uploaded_file.name:
        progress_bar = st.progress(0, text="Initiating sequence...")
        
        all_funny_messages = [
                "Summoning the data spirits...",
                "Convincing Excel to cooperate...",
                "Crunching the numbers (and some cookies)...",
                "Aligning the cosmic data rays...",
                "Teaching pandas to do kung fu...",
                "Analyzing chaos...",
                "Searching StakeOverflow...",
                "Untangling the spaghetti code...",
                "Reading Reddit...",
                "Waking up the neural networks...",
                "Locating the missing semicolon...",
                "Brewing coffee for the algorithm...",
                "Translating binary to English...",
                "Reading the tea leaves...",
                "Polishing the pixels..."
                "wc lvl?"
                "Smoking Window..."
                "Rushing B..."
                "AFKing..."
                "Vibe Coding..."
        ]
        
        selected_messages = random.sample(all_funny_messages, 7)
        for i, msg in enumerate(selected_messages):
            time.sleep(0.4)
            progress = int(((i + 1) / len(selected_messages)) * 100)
            progress_bar.progress(progress, text=msg)
        
        time.sleep(0.2)
        progress_bar.empty()
        st.session_state.last_loaded_file = uploaded_file.name

    df = None
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Separate columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

    # --- Main Dashboard ---
    # Top Section: AI & Stats
    col_ai, col_stats = st.columns([1, 1])
    
    with col_stats:
        st.subheader("Data Health")
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{df.shape[0]:,}")
        m2.metric("Columns", df.shape[1])
        m3.metric("Missing Data", f"{(missing_cells/total_cells)*100:.1f}%")
        
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows.")
        
        with st.expander("View Data Preview"):
            st.dataframe(df.head(), width='stretch')

    with col_ai:
        st.subheader("✨ AI Intelligence")
        if api_key:
            tab_quality, tab_business = st.tabs(["🛠️ Data Quality", "💼 Business Value"])
            
            with tab_quality:
                st.caption("Audit data for outliers, missing values, and statistical anomalies.")
                if st.button("Run Quality Audit", key="btn_quality"):
                    result_container_q = st.empty()
                    with st.spinner("Auditing data quality..."):
                        insight_text = get_ai_insights(df, api_key, "quality")
                    result_container_q.markdown(insight_text)
            
            with tab_business:
                st.caption("Generate strategic insights and key takeaways for stakeholders.")
                if st.button("Generate Business Brief", key="btn_business"):
                    result_container_b = st.empty()
                    with st.spinner("Synthesizing business strategy..."):
                        insight_text = get_ai_insights(df, api_key, "business")
                    result_container_b.markdown(insight_text)
        else:
            st.info("Please configure an API key to unlock AI analysis.")
            st.markdown("*The AI will look for patterns, anomalies, and business context hidden in the numbers.*")

    st.divider()

    # Middle Section: Smart Visualizations
    st.subheader("🔍 Statistically Significant Findings")
    st.markdown("These charts were automatically selected because they show **strong correlations** or **significant differences**.")
    
    significant_finds = find_significant_relations(df, numeric_cols, categorical_cols)
    
    if significant_finds:
        cols = st.columns(2)
        for idx, plot in enumerate(significant_finds):
            with cols[idx % 2]:
                st.markdown(f"**{plot['x']} vs {plot['y']}**")
                st.caption(plot['reason'])
                
                fig, ax = plt.subplots(figsize=(8, 5))
                
                if plot['type'] == 'boxplot':
                    sns.boxplot(data=df, x=plot['x'], y=plot['y'], hue=plot['x'], legend=False, ax=ax, palette="viridis")
                    plt.xticks(rotation=45)
                elif plot['type'] == 'scatter':
                    sns.scatterplot(data=df, x=plot['x'], y=plot['y'], ax=ax, color='teal', alpha=0.6)
                    sns.regplot(data=df, x=plot['x'], y=plot['y'], scatter=False, ax=ax, color='red')
                
                st.pyplot(fig)
    else:
        st.info("No statistically significant relationships (strong correlations or group differences) were detected.")

    # Bottom Section: Manual Exploration
    st.divider()
    st.subheader("🎨 Manual Exploration")
    tab1, tab2 = st.tabs(["Distributions", "Correlations"])
    
    with tab1:
        if numeric_cols:
            selected_col = st.selectbox("Visualize Distribution:", numeric_cols)
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.histplot(df[selected_col], kde=True, ax=ax, color='#6C63FF')
            st.pyplot(fig)
    
    with tab2:
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()