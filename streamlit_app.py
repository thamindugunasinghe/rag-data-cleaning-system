import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
import numpy as np
from src.data_profiler import DataProfiler
from src.cleaner import DataCleaner
from src.utils import calculate_data_quality_score, export_to_csv
import io
import json

# Page config
st.set_page_config(
    page_title="RAG Data Cleaning System",
    page_icon="🧹",
    layout="wide"
)

# Load configuration
@st.cache_data
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        # Default config if file not found
        return {
            'cleaning_strategies': {
                'quantitative': ["Mean Imputation", "Median Imputation", "Mode Imputation", "Forward Fill", "Backward Fill", "LLM Prediction"],
                'qualitative': ["Mode Imputation", "LLM Context Prediction", "Forward Fill", "Backward Fill", "Remove Rows"]
            }
        }

config = load_config()

# Initialize components
@st.cache_resource
def get_components():
    return DataProfiler(), DataCleaner()

profiler, cleaner = get_components()

# Main app
def main():
    st.title("🧹 RAG Data Cleaning System")
    st.markdown("**Intelligent Data Cleaning with AI-Powered Strategies**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Data Upload & Profiling", "Cleaning Configuration", "Results & Download"])
    
    if page == "Data Upload & Profiling":
        data_upload_page()
    elif page == "Cleaning Configuration":
        cleaning_config_page()
    elif page == "Results & Download":
        results_page()

def data_upload_page():
    st.header("📁 Data Upload & Profiling")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file for cleaning"
    )
    
    if uploaded_file:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state['original_df'] = df
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Display sample data
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Generate and display profile
            with st.spinner("🔍 Analyzing data quality..."):
                profile = profiler.profile_dataset(df)
                st.session_state['profile'] = profile
            
            display_data_profile(profile)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def display_data_profile(profile):
    st.subheader("📈 Data Quality Analysis")
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", profile['total_rows'])
    with col2:
        st.metric("Total Columns", profile['total_columns'])
    with col3:
        avg_quality = np.mean([col['data_quality'] for col in profile['column_profiles'].values()])
        st.metric("Avg Quality", f"{avg_quality:.1f}%")
    with col4:
        total_nulls = sum([col['null_count'] + col['error_count'] for col in profile['column_profiles'].values()])
        st.metric("Total Issues", total_nulls)
    
    # Column-wise analysis
    st.subheader("🔍 Column Analysis")
    
    column_data = []
    for col_name, col_profile in profile['column_profiles'].items():
        column_data.append({
            'Column': col_name,
            'Data Type': col_profile['data_type'],
            'Data Quality (%)': round(col_profile['data_quality'], 2),
            'Missing Values': col_profile['null_count'],
            'Error Values': col_profile['error_count'],
            'Unique Values': col_profile['unique_values'],
            'Recommended Strategy': col_profile['recommended_strategy']
        })
    
    column_df = pd.DataFrame(column_data)
    st.dataframe(column_df, use_container_width=True)
    
    # Quality visualization
    fig = px.bar(
        column_df, 
        x='Column', 
        y='Data Quality (%)',
        color='Data Type',
        title='Data Quality by Column'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def cleaning_config_page():
    st.header("⚙️ Cleaning Configuration")
    
    if 'original_df' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    df = st.session_state['original_df']
    profile = st.session_state.get('profile', {})
    
    st.subheader("🔧 Configure Cleaning Strategies")
    
    cleaning_strategies = {}
    
    # Create configuration for each column
    for column in df.columns:
        st.write(f"**{column}**")
        col_profile = profile.get('column_profiles', {}).get(column, {})
        data_type = col_profile.get('data_type', 'unknown')
        recommended = col_profile.get('recommended_strategy', 'Mode Imputation')
        
        # Get available strategies based on data type
        if data_type == 'quantitative':
            strategies = config['cleaning_strategies']['quantitative']
        else:
            strategies = config['cleaning_strategies']['qualitative']
        
        # Strategy selection
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            strategy = st.selectbox(
                f"Strategy for {column}",
                strategies,
                index=strategies.index(recommended) if recommended in strategies else 0,
                key=f"strategy_{column}"
            )
            cleaning_strategies[column] = strategy
        
        with col2:
            st.metric("Quality", f"{col_profile.get('data_quality', 0):.1f}%")
        
        with col3:
            issues = col_profile.get('null_count', 0) + col_profile.get('error_count', 0)
            st.metric("Issues", issues)
        
        st.divider()
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Start Cleaning", type="primary", use_container_width=True):
            with st.spinner("🔄 Cleaning data..."):
                try:
                    cleaned_df = cleaner.clean_dataset(df, cleaning_strategies)
                    st.session_state['cleaned_df'] = cleaned_df
                    st.session_state['cleaning_strategies'] = cleaning_strategies
                    
                    # Generate report
                    report = cleaner.generate_cleaning_report(df, cleaned_df)
                    st.session_state['cleaning_report'] = report
                    
                    st.success("✅ Data cleaning completed!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error during cleaning: {str(e)}")
    
    with col2:
        if st.button("🔄 Reset to Recommended", use_container_width=True):
            st.rerun()

def results_page():
    st.header("📊 Results & Download")
    
    if 'cleaned_df' not in st.session_state:
        st.warning("⚠️ Please clean the dataset first!")
        return
    
    original_df = st.session_state['original_df']
    cleaned_df = st.session_state['cleaned_df']
    report = st.session_state.get('cleaning_report', {})
    
    # Results comparison
    st.subheader("📈 Before vs After Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Dataset**")
        st.dataframe(original_df.head(), use_container_width=True)
        orig_quality = calculate_data_quality_score(original_df)
        st.metric("Overall Quality", f"{orig_quality:.1f}%")
        
    with col2:
        st.write("**Cleaned Dataset**")
        st.dataframe(cleaned_df.head(), use_container_width=True)
        clean_quality = calculate_data_quality_score(cleaned_df)
        st.metric("Overall Quality", f"{clean_quality:.1f}%")
    
    # Improvement metrics
    if report and 'improvements' in report:
        st.subheader("📊 Quality Improvements")
        
        improvements_data = []
        for column, improvement in report['improvements'].items():
            improvements_data.append({
                'Column': column,
                'Original Quality (%)': round(improvement['original_quality'], 2),
                'Final Quality (%)': round(improvement['final_quality'], 2),
                'Improvement (%)': round(improvement['quality_improvement'], 2)
            })
        
        improvements_df = pd.DataFrame(improvements_data)
        st.dataframe(improvements_df, use_container_width=True)
        
        # Visualization
        fig = go.Figure(data=[
            go.Bar(name='Original', x=improvements_df['Column'], y=improvements_df['Original Quality (%)']),
            go.Bar(name='Cleaned', x=improvements_df['Column'], y=improvements_df['Final Quality (%)'])
        ])
        fig.update_layout(barmode='group', title='Data Quality: Before vs After', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Download section
    st.subheader("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download cleaned dataset
        csv = export_to_csv(cleaned_df)
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download cleaning report
        if report:
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                label="📥 Download Cleaning Report (JSON)",
                data=report_json,
                file_name="cleaning_report.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col3:
        # Download comparison report
        comparison_data = {
            'original_shape': original_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'original_quality': orig_quality,
            'cleaned_quality': clean_quality,
            'improvement': clean_quality - orig_quality,
            'strategies_used': st.session_state.get('cleaning_strategies', {})
        }
        
        comparison_json = json.dumps(comparison_data, indent=2)
        st.download_button(
            label="📥 Download Summary Report (JSON)",
            data=comparison_json,
            file_name="summary_report.json",
            mime="application/json",
            use_container_width=True
        )

if __name__ == "__main__":
    main()