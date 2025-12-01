# 📊 Excel Dynamic Insights (EDI)

**Work In Progress: This project is currently under active development.**

A Python-based (Streamlit) dashboard that automatically analyzes Excel and CSV datasets. It combines statistical analysis with Generative AI (Google Gemini) to provide both technical data quality audits and strategic business insights.

## 🚀 Features
#### Automated Data Health Check: 
Instantly calculates missing values, duplicates, and row/column counts.

#### AI-Powered Insights:
- **Quality Audit**: Behaves like a Data Engineer, identifying anomalies, outliers, and data consistency issues.
- **Business Brief**: Behaves like a Business Analyst, highlighting trends, drivers, and meaningful business takeaways.

#### Visualizations:
- **Smart Visualization**: Automatically detects and plots statistically significant relationships  
  - ANOVA for categorical variables  
  - Pearson correlation for numerical variables
- **Manual Exploration Mode**: Interactive tools including histograms and correlation heatmaps for deeper self-service analysis.


## 🛠️ Installation
- Clone the repository

- Install the required dependencies:

```pip install streamlit pandas numpy matplotlib seaborn scipy google-generativeai openpyxl```


## ⚙️ Configuration 
This app requires a **Google Gemini API key.**
- **Get a key:** Generate a free Gemini API key on [Google AI Studio](https://aistudio.google.com/)
  
<p align="center">
  ⚠️ <strong>Do <u>not</u> upload your API key to GitHub.</strong>
</p>

Secure Setup:
- Create a folder named `.streamlit` in the root directory.

- Create a file inside it named `secrets.toml`
- Add your key:
    - GEMINI_API_KEY = "your_actual_api_key_here"


**Note:** If no secrets file is found, the app will prompt you to enter the key manually in the sidebar.

## ▶️ Usage
Run the Streamlit app:

```streamlit run excel_insights.py```


Upload your .csv or .xlsx file to begin the analysis.
## 🗺️ Roadmap & Planned Additions
- Time-Series Analysis
- Export Reports
- More Chart Types
- Advanced Cleaning
