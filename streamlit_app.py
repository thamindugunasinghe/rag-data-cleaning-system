import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
import numpy as np
from src.data_profiler import DataProfiler
from src.cleaner import DataCleaner
from src.anomaly_detector import AnomalyDetector
import io
import sys
from io import StringIO
from scipy import stats
import json

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

# Initialize components
@st.cache_resource
def get_components():
    return DataProfiler(), DataCleaner(), AnomalyDetector()

profiler, cleaner, anomaly_detector = get_components()

# Main app
def main():
    st.title("🧹 SOPAKA.AI")
    st.markdown("**Intelligent Data Cleaning with AI-Powered Strategies**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Data Upload & Profiling", 
                                "Anomaly Detection",
                                "Cleaning Configuration", 
                                "Results & Download"])
    
    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Show Debug Logs", value=False)
    
    if page == "Data Upload & Profiling":
        data_upload_page()
    elif page == "Anomaly Detection":
        anomaly_detection_page()
    elif page == "Cleaning Configuration":
        cleaning_config_page(show_debug)
    elif page == "Results & Download":
        results_page()

# COMPLETE ANOMALY DETECTION FUNCTION WITH WORKING BUTTONS
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
        z_threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
    
    if st.button("🔍 Detect Anomalies", type="primary", key="detect_anomalies_btn"):
        with st.spinner("🔍 Detecting anomalies..."):
            
            anomalies = {}
            error_markers = ["ERROR", "UNKNOWN", "N/A", "NULL", "", "error", "unknown"]
            
            # Process each column
            for column in df.columns:
                clean_data = df[column][~df[column].astype(str).isin(error_markers)].dropna()
                numeric_data = pd.to_numeric(clean_data, errors='coerce')
                valid_numeric = numeric_data.dropna()
                
                if len(valid_numeric) < 10:
                    continue
                
                column_anomalies = []
                
                if "Z-Score" in detection_methods:
                    try:
                        z_scores = np.abs(stats.zscore(valid_numeric))
                        z_outlier_mask = z_scores > z_threshold
                        z_outlier_indices = valid_numeric[z_outlier_mask].index.tolist()
                        column_anomalies.extend(z_outlier_indices)
                    except Exception as e:
                        st.warning(f"Z-Score analysis failed for {column}: {str(e)}")
                
                if "IQR" in detection_methods:
                    try:
                        Q1 = valid_numeric.quantile(0.25)
                        Q3 = valid_numeric.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        if IQR > 0:
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            iqr_outlier_mask = (valid_numeric < lower_bound) | (valid_numeric > upper_bound)
                            iqr_outlier_indices = valid_numeric[iqr_outlier_mask].index.tolist()
                            column_anomalies.extend(iqr_outlier_indices)
                    except Exception as e:
                        st.warning(f"IQR analysis failed for {column}: {str(e)}")
                
                if "Isolation Forest" in detection_methods:
                    try:
                        from sklearn.ensemble import IsolationForest
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outliers = iso_forest.fit_predict(valid_numeric.values.reshape(-1, 1))
                        iso_outlier_indices = valid_numeric[outliers == -1].index.tolist()
                        column_anomalies.extend(iso_outlier_indices)
                    except ImportError:
                        st.warning("⚠️ Scikit-learn not installed. Skipping Isolation Forest.")
                    except Exception as e:
                        st.warning(f"Isolation Forest failed for {column}: {str(e)}")
                
                if column_anomalies:
                    anomalies[column] = list(set(column_anomalies))
            
            # Business Rules Validation
            if "Business Rules" in detection_methods:
                business_violations = []
                required_cols = ['Price Per Unit', 'Quantity', 'Total Spent']
                
                if all(col in df.columns for col in required_cols):
                    for idx in range(len(df)):
                        try:
                            price_val = df.loc[idx, 'Price Per Unit']
                            qty_val = df.loc[idx, 'Quantity']
                            total_val = df.loc[idx, 'Total Spent']
                            
                            if any(str(val) in error_markers for val in [price_val, qty_val, total_val]):
                                continue
                                
                            price = pd.to_numeric(price_val, errors='coerce')
                            quantity = pd.to_numeric(qty_val, errors='coerce')
                            total = pd.to_numeric(total_val, errors='coerce')
                            
                            if pd.notna(price) and pd.notna(quantity) and pd.notna(total):
                                expected_total = price * quantity
                                if abs(total - expected_total) > 0.01:
                                    business_violations.append({
                                        'row': idx,
                                        'price': float(price),
                                        'quantity': float(quantity),
                                        'expected_total': float(expected_total),
                                        'actual_total': float(total),
                                        'difference': float(abs(total - expected_total))
                                    })
                        except Exception:
                            continue
                    
                    if business_violations:
                        anomalies['Business Logic Violations'] = business_violations
            
            # Store results
            st.session_state['anomaly_results'] = anomalies
            
            # Display results
            st.subheader("📈 Anomaly Detection Results")
            
            if anomalies:
                # Calculate total anomalies
                total_anomalies = 0
                for key, value in anomalies.items():
                    total_anomalies += len(value)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Anomalies", total_anomalies)
                with col2:
                    st.metric("Affected Columns", len(anomalies))
                with col3:
                    anomaly_percentage = (total_anomalies / len(df)) * 100
                    st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
                
                # Show anomalies by column
                for column, data in anomalies.items():
                    if column == 'Business Logic Violations':
                        with st.expander(f"💼 {column}: {len(data)} violations", expanded=True):
                            st.write("**Sample Business Rule Violations:**")
                            if data:
                                violation_df = pd.DataFrame(data[:10])
                                st.dataframe(violation_df, use_container_width=True)
                    else:
                        indices = data
                        with st.expander(f"📊 {column}: {len(indices)} anomalies", expanded=True):
                            
                            col_left, col_right = st.columns(2)
                            
                            with col_left:
                                st.write("**Anomaly Statistics:**")
                                anomaly_values = pd.to_numeric(df.loc[indices, column], errors='coerce').dropna()
                                
                                if len(anomaly_values) > 0:
                                    st.write(f"**Count:** {len(anomaly_values)}")
                                    st.write(f"**Mean:** {anomaly_values.mean():.2f}")
                                    st.write(f"**Std:** {anomaly_values.std():.2f}")
                                    st.write(f"**Min:** {anomaly_values.min():.2f}")
                                    st.write(f"**Max:** {anomaly_values.max():.2f}")
                                    
                                    unique_anomalies = sorted(anomaly_values.unique())
                                    st.write(f"**Unique Values:** {unique_anomalies[:10]}")
                            
                            with col_right:
                                try:
                                    all_numeric = pd.to_numeric(df[column], errors='coerce').dropna()
                                    
                                    if len(all_numeric) > 0 and len(anomaly_values) > 0:
                                        fig = go.Figure()
                                        
                                        fig.add_trace(go.Histogram(
                                            x=all_numeric,
                                            name='All Data',
                                            opacity=0.7,
                                            marker_color='lightblue',
                                            nbinsx=30
                                        ))
                                        
                                        fig.add_trace(go.Histogram(
                                            x=anomaly_values,
                                            name='Anomalies',
                                            marker_color='red',
                                            opacity=0.8,
                                            nbinsx=30
                                        ))
                                        
                                        fig.update_layout(
                                            title=f"{column} Distribution",
                                            xaxis_title=column,
                                            yaxis_title="Count",
                                            height=300,
                                            showlegend=True,
                                            barmode='overlay'
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                except Exception as e:
                                    st.write(f"Visualization error: {str(e)}")
                            
                            if st.button(f"Show Sample Rows", key=f"show_{column}_samples"):
                                sample_size = min(10, len(indices))
                                sample_indices = indices[:sample_size]
                                st.write("**Sample Anomalous Rows:**")
                                st.dataframe(df.loc[sample_indices], use_container_width=True)
                
                # FIXED HANDLING BUTTONS WITH PROPER CALLBACKS
                st.subheader("🔧 Anomaly Handling")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export Report Button with immediate callback
                    export_report = st.button("📊 Export Report", use_container_width=True, key="export_report_btn")
                    
                    if export_report:
                        try:
                            anomaly_report = {
                                'summary': {
                                    'total_anomalies': total_anomalies,
                                    'affected_columns': len(anomalies),
                                    'methods_used': detection_methods,
                                    'z_threshold': float(z_threshold),
                                    'anomaly_rate': f"{anomaly_percentage:.2f}%",
                                    'dataset_size': len(df)
                                },
                                'anomalies_by_column': {}
                            }
                            
                            for column, data in anomalies.items():
                                if column == 'Business Logic Violations':
                                    anomaly_report['anomalies_by_column'][column] = {
                                        'type': 'business_violations',
                                        'count': len(data),
                                        'sample_violations': data[:5]
                                    }
                                else:
                                    sample_values = []
                                    if len(data) > 0:
                                        try:
                                            sample_vals = df.loc[data[:5], column]
                                            sample_values = [float(x) for x in pd.to_numeric(sample_vals, errors='coerce').dropna()]
                                        except:
                                            sample_values = []
                                    
                                    anomaly_report['anomalies_by_column'][column] = {
                                        'type': 'statistical_outliers',
                                        'count': len(data),
                                        'sample_indices': data[:10],
                                        'sample_values': sample_values
                                    }
                            
                            report_json = json.dumps(anomaly_report, indent=2, default=str)
                            
                            st.download_button(
                                label="📥 Download Anomaly Report (JSON)",
                                data=report_json,
                                file_name="anomaly_report.json",
                                mime="application/json",
                                key="download_anomaly_report"
                            )
                            
                            st.success("✅ Anomaly report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
                
                with col2:
                    # Remove Anomalies Button with immediate callback
                    remove_anomalies = st.button("🗑️ Remove Anomalies", use_container_width=True, key="remove_anomalies_btn")
                    
                    if remove_anomalies:
                        try:
                            all_indices = set()
                            
                            for column, data in anomalies.items():
                                if column == 'Business Logic Violations':
                                    violation_indices = [v['row'] for v in data if isinstance(v, dict) and 'row' in v]
                                    all_indices.update(violation_indices)
                                else:
                                    all_indices.update(data)
                            
                            if all_indices:
                                # Remove anomalous rows
                                cleaned_df = df.drop(list(all_indices))
                                st.session_state['anomaly_cleaned_df'] = cleaned_df
                                
                                # Show success message with details
                                st.success(f"✅ Successfully removed {len(all_indices)} anomalous rows!")
                                
                                # Show impact metrics
                                removal_percentage = (len(all_indices) / len(df)) * 100
                                st.info(f"""
                                **Removal Impact:**
                                • Original size: {len(df):,} rows
                                • Cleaned size: {len(cleaned_df):,} rows  
                                • Removed: {len(all_indices):,} rows ({removal_percentage:.1f}%)
                                """)
                                
                                # Show sample of removed data
                                if st.checkbox("Preview removed data", key="preview_removed"):
                                    removed_sample = df.loc[list(all_indices)].head()
                                    st.write("**Sample of removed anomalous data:**")
                                    st.dataframe(removed_sample, use_container_width=True)
                                
                                # Offer to download cleaned data
                                cleaned_csv = cleaned_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Cleaned Dataset",
                                    data=cleaned_csv,
                                    file_name="cleaned_dataset_no_anomalies.csv",
                                    mime="text/csv",
                                    key="download_cleaned_no_anomalies"
                                )
                            else:
                                st.warning("No anomalous rows found to remove.")
                                
                        except Exception as e:
                            st.error(f"Error removing anomalies: {str(e)}")
                
                with col3:
                    # Flag Anomalies Button with immediate callback
                    flag_anomalies = st.button("🏷️ Flag Anomalies", use_container_width=True, key="flag_anomalies_btn")
                    
                    if flag_anomalies:
                        try:
                            flagged_df = df.copy()
                            flagged_df['is_anomaly'] = False
                            flagged_df['anomaly_type'] = ''
                            flagged_df['anomaly_column'] = ''
                            flagged_df['anomaly_method'] = ''
                            
                            total_flagged = 0
                            
                            for column, data in anomalies.items():
                                if column == 'Business Logic Violations':
                                    violation_indices = [v['row'] for v in data if isinstance(v, dict) and 'row' in v]
                                    if violation_indices:
                                        flagged_df.loc[violation_indices, 'is_anomaly'] = True
                                        flagged_df.loc[violation_indices, 'anomaly_type'] = 'Business Rule Violation'
                                        flagged_df.loc[violation_indices, 'anomaly_column'] = 'Multiple Columns'
                                        flagged_df.loc[violation_indices, 'anomaly_method'] = 'Business Rules'
                                        total_flagged += len(violation_indices)
                                else:
                                    if data:
                                        flagged_df.loc[data, 'is_anomaly'] = True
                                        flagged_df.loc[data, 'anomaly_type'] = 'Statistical Outlier'
                                        flagged_df.loc[data, 'anomaly_column'] = column
                                        
                                        # Determine which method detected it
                                        methods_used = []
                                        if "Z-Score" in detection_methods:
                                            methods_used.append("Z-Score")
                                        if "IQR" in detection_methods:
                                            methods_used.append("IQR")
                                        if "Isolation Forest" in detection_methods:
                                            methods_used.append("Isolation Forest")
                                        
                                        flagged_df.loc[data, 'anomaly_method'] = ", ".join(methods_used)
                                        total_flagged += len(data)
                            
                            st.session_state['flagged_df'] = flagged_df
                            
                            st.success(f"✅ Successfully flagged {total_flagged} rows as anomalous!")
                            
                            # Show flagging statistics
                            flag_percentage = (total_flagged / len(df)) * 100
                            st.info(f"""
                            **Flagging Results:**
                            • Total rows: {len(df):,}
                            • Flagged as anomalous: {total_flagged:,} ({flag_percentage:.1f}%)
                            • Normal rows: {len(df) - total_flagged:,} ({100 - flag_percentage:.1f}%)
                            """)
                            
                            # Show preview of flagged data
                            if st.checkbox("Preview flagged data", key="preview_flagged"):
                                flagged_preview = flagged_df[flagged_df['is_anomaly'] == True].head()
                                st.write("**Sample of flagged anomalous data:**")
                                display_columns = ['Transaction ID', 'Total Spent', 'is_anomaly', 'anomaly_type', 'anomaly_column', 'anomaly_method']
                                available_columns = [col for col in display_columns if col in flagged_preview.columns]
                                st.dataframe(flagged_preview[available_columns], use_container_width=True)
                            
                            # Offer to download flagged data
                            flagged_csv = flagged_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Flagged Dataset",
                                data=flagged_csv,
                                file_name="dataset_with_anomaly_flags.csv",
                                mime="text/csv",
                                key="download_flagged_dataset"
                            )
                            
                        except Exception as e:
                            st.error(f"Error flagging anomalies: {str(e)}")
            
            else:
                st.success("✅ No anomalies detected in the dataset!")
                st.info(f"""
                **Possible reasons:**
                • Your data is very clean and well-distributed
                • Z-Score threshold ({z_threshold}) might be too high
                • Try lowering the threshold to {z_threshold-0.5:.1f}
                • Consider using different detection methods
                """)
                
                # Show data distribution for context
                st.write("**Data Distribution Analysis:**")
                numeric_cols = []
                for col in df.columns:
                    if pd.to_numeric(df[col], errors='coerce').notna().sum() > 10:
                        numeric_cols.append(col)
                
                if numeric_cols:
                    selected_col = st.selectbox("Select column to analyze:", numeric_cols)
                    if selected_col:
                        clean_data = pd.to_numeric(df[selected_col], errors='coerce').dropna()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{selected_col} Statistics:**")
                            st.write(f"Mean: {clean_data.mean():.2f}")
                            st.write(f"Std: {clean_data.std():.2f}")
                            st.write(f"Min: {clean_data.min():.2f}")
                            st.write(f"Max: {clean_data.max():.2f}")
                        
                        with col2:
                            fig = px.histogram(x=clean_data, title=f"Distribution of {selected_col}", nbins=30)
                            st.plotly_chart(fig, use_container_width=True)

def data_upload_page():
    st.header("📁 Data Upload & Profiling")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file for cleaning"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state['original_df'] = df
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            with st.spinner("🔍 Analyzing data quality..."):
                profile = profiler.profile_dataset(df)
                st.session_state['profile'] = profile
            
            display_data_profile(profile)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def display_data_profile(profile):
    st.subheader("📈 Data Quality Analysis")
    
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
    
    total_issues = sum([profile.get('column_profiles', {}).get(col, {}).get('null_count', 0) + 
                       profile.get('column_profiles', {}).get(col, {}).get('error_count', 0) 
                       for col in df.columns])
    
    if total_issues > 1000:
        st.warning(f"⚠️ Large dataset detected ({total_issues:,} issues). LLM predictions may take significant time and cost.")
    
    cleaning_strategies = {}
    
    for column in df.columns:
        st.write(f"**{column}**")
        col_profile = profile.get('column_profiles', {}).get(column, {})
        data_type = col_profile.get('data_type', 'unknown')
        recommended = col_profile.get('recommended_strategy', 'Mode Imputation')
        
        if data_type == 'quantitative':
            strategies = config['cleaning_strategies']['quantitative']
        else:
            strategies = config['cleaning_strategies']['qualitative']
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            strategy = st.selectbox(
                f"Strategy for {column}",
                strategies,
                index=strategies.index(recommended) if recommended in strategies else 0,
                key=f"strategy_{column}"
            )
            cleaning_strategies[column] = strategy
            
            issues = col_profile.get('null_count', 0) + col_profile.get('error_count', 0)
            if 'LLM' in strategy and issues > 50:
                st.caption(f"⚠️ {issues} missing values - will be limited to 50 LLM calls")
        
        with col2:
            st.metric("Quality", f"{col_profile.get('data_quality', 0):.1f}%")
        
        with col3:
            issues = col_profile.get('null_count', 0) + col_profile.get('error_count', 0)
            st.metric("Issues", issues)
        
        st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Start Cleaning", type="primary", use_container_width=True):
            
            st.session_state['cleaning_progress'] = 0.0
            st.session_state['llm_progress'] = 0.0
            st.session_state['debug_logs'] = []
            
            progress_container = st.container()
            debug_container = st.container()
            
            with progress_container:
                st.write("🔄 **Cleaning Progress**")
                main_progress = st.progress(0)
                llm_progress = st.progress(0)
                status_text = st.empty()
            
            if show_debug:
                with debug_container:
                    st.write("🐛 **Debug Logs**")
                    debug_log = st.empty()
            
            old_stdout = sys.stdout
            sys.stdout = StreamlitCapture()
            
            try:
                with st.spinner("🔄 Cleaning data..."):
                    
                    status_text.text("Initializing cleaning process...")
                    
                    cleaned_df = cleaner.clean_dataset(df, cleaning_strategies)
                    st.session_state['cleaned_df'] = cleaned_df
                    st.session_state['cleaning_strategies'] = cleaning_strategies
                    
                    main_progress.progress(0.8)
                    status_text.text("Generating cleaning report...")
                    
                    report = cleaner.generate_cleaning_report(df, cleaned_df)
                    st.session_state['cleaning_report'] = report
                    
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
                sys.stdout = old_stdout
                
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
    
    st.subheader("📈 Before vs After Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Dataset**")
        st.dataframe(original_df.head(), use_container_width=True)
        
    with col2:
        st.write("**Cleaned Dataset**")
        st.dataframe(cleaned_df.head(), use_container_width=True)
    
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
        
        fig = go.Figure(data=[
            go.Bar(name='Original', x=improvements_df['Column'], y=improvements_df['Original Quality (%)']),
            go.Bar(name='Cleaned', x=improvements_df['Column'], y=improvements_df['Final Quality (%)'])
        ])
        fig.update_layout(barmode='group', title='Data Quality: Before vs After')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = cleaned_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if report:
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
