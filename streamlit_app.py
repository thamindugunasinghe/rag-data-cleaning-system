import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
import numpy as np
from src.data_profiler import DataProfiler
from src.cleaner import DataCleaner
from src.anomaly_detector import AnomalyDetector  # ← ADDED THIS
import io
import sys
from io import StringIO

# Page config
st.set_page_config(
    page_title="SOPAKA.AI",
    page_icon="🧹",
    layout="wide"
)

# Capture print statements
class StreamlitCapture:
    def __init__(self):
        self.logs = []
    
    def write(self, txt):
        self.logs.append(txt)
        if hasattr(st, 'session_state'):
            st.session_state['debug_logs'] = self.logs

    def flush(self):
        pass

# Load configuration
@st.cache_data
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'cleaning_strategies': {
                'quantitative': ['Mean Imputation', 'Median Imputation', 'Mode Imputation'],
                'qualitative': ['Mode Imputation', 'LLM Context Prediction']
            }
        }

config = load_config()

# Initialize components - UPDATED THIS
@st.cache_resource
def get_components():
    return DataProfiler(), DataCleaner(), AnomalyDetector()

profiler, cleaner, anomaly_detector = get_components()  # ← UPDATED THIS LINE

# Main app - UPDATED THIS
def main():
    st.title("🧹 SOPAKA.AI")
    st.markdown("**Intelligent Data Cleaning with AI-Powered Strategies**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Data Upload & Profiling", 
                                "Anomaly Detection",      # ← ADDED THIS
                                "Cleaning Configuration", 
                                "Results & Download"])
    
    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Show Debug Logs", value=False)
    
    if page == "Data Upload & Profiling":
        data_upload_page()
    elif page == "Anomaly Detection":           # ← ADDED THIS
        anomaly_detection_page()                # ← ADDED THIS
    elif page == "Cleaning Configuration":
        cleaning_config_page(show_debug)
    elif page == "Results & Download":
        results_page()

# ADDED THIS ENTIRE FUNCTION
def anomaly_detection_page():
    st.header("🔍 Anomaly Detection")
    
    if 'original_df' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    df = st.session_state['original_df']
    
    st.subheader("📊 Anomaly Detection Analysis")
    
    # Detection method selection
    col1, col2 = st.columns(2)
    
    with col1:
        detection_methods = st.multiselect(
            "Select Detection Methods",
            ["Z-Score", "IQR", "Isolation Forest", "Business Rules"],
            default=["Z-Score", "IQR"]
        )
    
    with col2:
        z_threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
    
    if st.button("🔍 Detect Anomalies", type="primary"):
        with st.spinner("🔍 Detecting anomalies..."):
            
            # Initialize anomaly detector if needed
            if 'anomaly_results' not in st.session_state:
                st.session_state['anomaly_results'] = {}
            
            # Statistical anomalies detection
            anomalies = {}
            
            for column in df.columns:
                # Only process numeric columns
                if df[column].dtype in ['int64', 'float64']:
                    column_anomalies = []
                    
                    # Clean the column first (remove non-numeric values)
                    numeric_data = pd.to_numeric(df[column], errors='coerce')
                    clean_data = numeric_data.dropna()
                    
                    if len(clean_data) == 0:
                        continue
                    
                    if "Z-Score" in detection_methods:
                        z_scores = np.abs((numeric_data - clean_data.mean()) / clean_data.std())
                        z_outliers = df[z_scores > z_threshold].index.tolist()
                        column_anomalies.extend(z_outliers)
                    
                    if "IQR" in detection_methods:
                        Q1 = clean_data.quantile(0.25)
                        Q3 = clean_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        iqr_outliers = df[(numeric_data < lower_bound) | (numeric_data > upper_bound)].index.tolist()
                        column_anomalies.extend(iqr_outliers)
                    
                    if "Isolation Forest" in detection_methods:
                        try:
                            from sklearn.ensemble import IsolationForest
                            iso_forest = IsolationForest(contamination=0.1, random_state=42)
                            outliers = iso_forest.fit_predict(clean_data.values.reshape(-1, 1))
                            iso_outliers = clean_data[outliers == -1].index.tolist()
                            column_anomalies.extend(iso_outliers)
                        except ImportError:
                            st.warning("⚠️ Scikit-learn not installed. Skipping Isolation Forest.")
                    
                    if column_anomalies:
                        anomalies[column] = list(set(column_anomalies))
            
            # Business Rules Validation
            if "Business Rules" in detection_methods:
                business_anomalies = []
                
                # Check if we have the required columns for business rule validation
                required_cols = ['Price Per Unit', 'Quantity', 'Total Spent']
                if all(col in df.columns for col in required_cols):
                    
                    for idx, row in df.iterrows():
                        try:
                            price = pd.to_numeric(row['Price Per Unit'], errors='coerce')
                            quantity = pd.to_numeric(row['Quantity'], errors='coerce')
                            total = pd.to_numeric(row['Total Spent'], errors='coerce')
                            
                            if pd.notna(price) and pd.notna(quantity) and pd.notna(total):
                                expected_total = price * quantity
                                if abs(total - expected_total) > 0.01:  # Allow small rounding differences
                                    business_anomalies.append(idx)
                        except:
                            continue
                    
                    if business_anomalies:
                        anomalies['Business Logic Violations'] = business_anomalies
            
            # Store results
            st.session_state['anomaly_results'] = anomalies
            
            # Display results
            st.subheader("📈 Anomaly Detection Results")
            
            if anomalies:
                total_anomalies = sum(len(indices) for indices in anomalies.values())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Anomalies", total_anomalies)
                with col2:
                    st.metric("Affected Columns", len(anomalies))
                with col3:
                    anomaly_percentage = (total_anomalies / len(df)) * 100
                    st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
                
                # Show anomalies by column
                for column, indices in anomalies.items():
                    with st.expander(f"📊 {column}: {len(indices)} anomalies", expanded=True):
                        
                        if len(indices) > 0:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Anomaly Statistics:**")
                                if column != 'Business Logic Violations':
                                    anomaly_values = df.loc[indices, column]
                                    stats_df = pd.DataFrame({
                                        'Statistic': ['Count', 'Mean', 'Std', 'Min', 'Max'],
                                        'Value': [
                                            len(anomaly_values),
                                            f"{pd.to_numeric(anomaly_values, errors='coerce').mean():.2f}" if pd.to_numeric(anomaly_values, errors='coerce').notna().any() else 'N/A',
                                            f"{pd.to_numeric(anomaly_values, errors='coerce').std():.2f}" if pd.to_numeric(anomaly_values, errors='coerce').notna().any() else 'N/A',
                                            f"{pd.to_numeric(anomaly_values, errors='coerce').min():.2f}" if pd.to_numeric(anomaly_values, errors='coerce').notna().any() else 'N/A',
                                            f"{pd.to_numeric(anomaly_values, errors='coerce').max():.2f}" if pd.to_numeric(anomaly_values, errors='coerce').notna().any() else 'N/A'
                                        ]
                                    })
                                    st.dataframe(stats_df, use_container_width=True)
                                else:
                                    st.write(f"Found {len(indices)} business rule violations")
                            
                            with col2:
                                # Create visualization for numeric columns
                                if column != 'Business Logic Violations' and df[column].dtype in ['int64', 'float64']:
                                    try:
                                        # Box plot showing outliers
                                        fig = px.box(df, y=column, title=f"Outliers in {column}")
                                        
                                        # Add anomaly points
                                        anomaly_data = df.loc[indices, column]
                                        fig.add_scatter(
                                            x=[0] * len(anomaly_data), 
                                            y=pd.to_numeric(anomaly_data, errors='coerce'),
                                            mode='markers',
                                            marker=dict(color='red', size=10, symbol='x'),
                                            name='Detected Anomalies',
                                            showlegend=True
                                        )
                                        
                                        fig.update_layout(height=300)
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.write(f"Could not create visualization: {str(e)}")
                                        
                                        # Fallback: show histogram
                                        try:
                                            fig = px.histogram(df, x=column, title=f"Distribution of {column}")
                                            st.plotly_chart(fig, use_container_width=True)
                                        except:
                                            st.write("Unable to create visualization for this column.")
                            
                            # Show sample anomalous rows
                            if st.button(f"Show Sample Anomalous Rows for {column}", key=f"show_{column}"):
                                sample_size = min(10, len(indices))
                                sample_indices = indices[:sample_size]
                                st.write(f"**Sample of {sample_size} anomalous rows:**")
                                st.dataframe(df.loc[sample_indices], use_container_width=True)
                
                # Anomaly handling options
                st.subheader("🔧 Anomaly Handling Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Export Anomaly Report", use_container_width=True):
                        # Create anomaly report
                        anomaly_report = {
                            'detection_summary': {
                                'total_anomalies': total_anomalies,
                                'affected_columns': len(anomalies),
                                'anomaly_rate': f"{anomaly_percentage:.2f}%",
                                'detection_methods': detection_methods
                            },
                            'anomalies_by_column': {}
                        }
                        
                        for column, indices in anomalies.items():
                            anomaly_report['anomalies_by_column'][column] = {
                                'count': len(indices),
                                'indices': indices,
                                'sample_values': df.loc[indices[:5], column].tolist() if column != 'Business Logic Violations' else []
                            }
                        
                        import json
                        report_json = json.dumps(anomaly_report, indent=2, default=str)
                        st.download_button(
                            label="📥 Download Anomaly Report (JSON)",
                            data=report_json,
                            file_name="anomaly_report.json",
                            mime="application/json"
                        )
                
                with col2:
                    if st.button("🗑️ Remove Anomalous Rows", use_container_width=True):
                        all_anomaly_indices = set()
                        for indices in anomalies.values():
                            all_anomaly_indices.update(indices)
                        
                        cleaned_df = df.drop(list(all_anomaly_indices))
                        st.session_state['anomaly_cleaned_df'] = cleaned_df
                        st.success(f"✅ Removed {len(all_anomaly_indices)} anomalous rows. Dataset size: {len(df)} → {len(cleaned_df)}")
                
                with col3:
                    if st.button("🏷️ Flag Anomalies", use_container_width=True):
                        flagged_df = df.copy()
                        flagged_df['anomaly_flag'] = False
                        
                        all_anomaly_indices = set()
                        for indices in anomalies.values():
                            all_anomaly_indices.update(indices)
                        
                        flagged_df.loc[list(all_anomaly_indices), 'anomaly_flag'] = True
                        st.session_state['flagged_df'] = flagged_df
                        st.success(f"✅ Flagged {len(all_anomaly_indices)} rows as anomalous")
                
            else:
                st.success("✅ No anomalies detected in the dataset!")
                st.info("This could mean your data is very clean, or you might want to try different detection methods or adjust thresholds.")

# Your existing functions remain the same...
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

def cleaning_config_page(show_debug=False):
    st.header("⚙️ Cleaning Configuration")
    
    if 'original_df' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    df = st.session_state['original_df']
    profile = st.session_state.get('profile', {})
    
    st.subheader("🔧 Configure Cleaning Strategies")
    
    # Warning for large datasets with LLM
    total_issues = sum([profile.get('column_profiles', {}).get(col, {}).get('null_count', 0) + 
                       profile.get('column_profiles', {}).get(col, {}).get('error_count', 0) 
                       for col in df.columns])
    
    if total_issues > 1000:
        st.warning(f"⚠️ Large dataset detected ({total_issues:,} issues). LLM predictions may take significant time and cost. Consider using statistical methods for large missing value counts.")
    
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
            
            # Show warning for LLM with many missing values
            issues = col_profile.get('null_count', 0) + col_profile.get('error_count', 0)
            if 'LLM' in strategy and issues > 50:
                st.caption(f"⚠️ {issues} missing values - will be limited to 50 LLM calls for performance")
        
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
            
            # Initialize session state for progress tracking
            st.session_state['cleaning_progress'] = 0.0
            st.session_state['llm_progress'] = 0.0
            st.session_state['debug_logs'] = []
            
            # Create progress placeholders
            progress_container = st.container()
            debug_container = st.container()
            
            with progress_container:
                st.write("🔄 **Cleaning Progress**")
                main_progress = st.progress(0)
                llm_progress = st.progress(0)
                status_text = st.empty()
            
            # Debug logs container
            if show_debug:
                with debug_container:
                    st.write("🐛 **Debug Logs**")
                    debug_log = st.empty()
            
            # Redirect stdout to capture prints
            old_stdout = sys.stdout
            sys.stdout = StreamlitCapture()
            
            try:
                with st.spinner("🔄 Cleaning data..."):
                    
                    # Update status
                    status_text.text("Initializing cleaning process...")
                    
                    # Start cleaning
                    cleaned_df = cleaner.clean_dataset(df, cleaning_strategies)
                    st.session_state['cleaned_df'] = cleaned_df
                    st.session_state['cleaning_strategies'] = cleaning_strategies
                    
                    # Update progress
                    main_progress.progress(0.8)
                    status_text.text("Generating cleaning report...")
                    
                    # Generate report
                    report = cleaner.generate_cleaning_report(df, cleaned_df)
                    st.session_state['cleaning_report'] = report
                    
                    # Complete
                    main_progress.progress(1.0)
                    llm_progress.progress(1.0)
                    status_text.text("✅ Cleaning completed successfully!")
                    
                    st.success("✅ Data cleaning completed!")
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Error during cleaning: {str(e)}")
                if show_debug:
                    st.exception(e)
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                
                # Show debug logs
                if show_debug and 'debug_logs' in st.session_state:
                    with debug_container:
                        debug_text = "\n".join(st.session_state['debug_logs'])
                        debug_log.text_area("Debug Output", debug_text, height=300)
    
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
        
    with col2:
        st.write("**Cleaned Dataset**")
        st.dataframe(cleaned_df.head(), use_container_width=True)
    
    # Improvement metrics
    if report:
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
        fig.update_layout(barmode='group', title='Data Quality: Before vs After')
        st.plotly_chart(fig, use_container_width=True)
    
    # Download section
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download cleaned dataset
        csv = cleaned_df.to_csv(index=False)
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
            import json
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                label="📥 Download Cleaning Report (JSON)",
                data=report_json,
                file_name="cleaning_report.json",
                mime="application/json",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
