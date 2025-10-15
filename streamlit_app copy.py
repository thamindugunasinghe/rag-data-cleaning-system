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
    
    # Reset button in sidebar
    if st.sidebar.button("🔄 Reset All Data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    if page == "Data Upload & Profiling":
        data_upload_page()
    elif page == "Anomaly Detection":
        anomaly_detection_page()
    elif page == "Cleaning Configuration":
        cleaning_config_page(show_debug)
    elif page == "Results & Download":
        results_page()

# ENHANCED ANOMALY DETECTION FUNCTION WITH MULTIPLE METHODS
def anomaly_detection_page():
    st.header("🔍 Advanced Anomaly Detection")
    
    if 'original_df' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    df = st.session_state['original_df']
    
    st.subheader("📊 Anomaly Detection Configuration")
    
    # Enhanced detection method selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔍 Statistical Methods:**")
        statistical_methods = st.multiselect(
            "Statistical Outlier Detection",
            ["Z-Score", "Modified Z-Score", "IQR", "Percentile-based", "Standard Deviation"],
            default=["Z-Score", "IQR"],
            key="statistical_methods_selector"
        )
        
        st.write("**🤖 Machine Learning Methods:**")
        ml_methods = st.multiselect(
            "ML-based Anomaly Detection", 
            ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "DBSCAN Clustering"],
            default=["Isolation Forest"],
            key="ml_methods_selector"
        )
        
        st.write("**💼 Business Logic Methods:**")
        business_methods = st.multiselect(
            "Domain-specific Validation",
            ["Business Rules", "Data Type Validation", "Range Validation", "Pattern Matching"],
            default=["Business Rules"],
            key="business_methods_selector"
        )
    
    with col2:
        st.write("**⚙️ Detection Parameters:**")
        
        # Statistical parameters
        z_threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1, key="z_threshold_slider")
        iqr_multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1, key="iqr_multiplier_slider")
        percentile_lower = st.slider("Lower Percentile", 0.1, 10.0, 1.0, 0.1, key="percentile_lower_slider")
        percentile_upper = st.slider("Upper Percentile", 90.0, 99.9, 99.0, 0.1, key="percentile_upper_slider")
        
        # ML parameters
        contamination = st.slider("Contamination Rate", 0.01, 0.5, 0.1, 0.01, key="contamination_slider")
        
        # Advanced options
        with st.expander("🔬 Advanced Options"):
            multivariate_detection = st.checkbox("Multivariate Detection", value=False, key="multivariate_check")
            time_series_detection = st.checkbox("Time Series Anomalies", value=False, key="timeseries_check")
            categorical_detection = st.checkbox("Categorical Anomalies", value=True, key="categorical_check")
    
    # Combine all selected methods
    detection_methods = statistical_methods + ml_methods + business_methods
    
    # Reset anomaly detection button
    if st.button("🔄 Reset Detection", key="reset_detection_btn"):
        for key in ['anomaly_results', 'detection_methods_used', 'detection_params', 'show_export', 'show_removal', 'show_flagging']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # ENHANCED DETECT ANOMALIES BUTTON
    if st.button("🔍 Detect Anomalies", type="primary", key="detect_anomalies_btn"):
        if not detection_methods:
            st.warning("⚠️ Please select at least one detection method!")
            return
            
        with st.spinner("🔍 Running comprehensive anomaly detection..."):
            
            anomalies = {}
            error_markers = ["ERROR", "UNKNOWN", "N/A", "NULL", "", "error", "unknown"]
            detection_summary = {}
            
            # Process each column with enhanced methods
            for column in df.columns:
                clean_data = df[column][~df[column].astype(str).isin(error_markers)].dropna()
                numeric_data = pd.to_numeric(clean_data, errors='coerce')
                valid_numeric = numeric_data.dropna()
                
                if len(valid_numeric) < 10:
                    continue
                
                column_anomalies = []
                detection_summary[column] = {}
                
                # STATISTICAL METHODS
                if "Z-Score" in detection_methods:
                    try:
                        z_scores = np.abs(stats.zscore(valid_numeric))
                        z_outlier_mask = z_scores > z_threshold
                        z_outlier_indices = valid_numeric[z_outlier_mask].index.tolist()
                        column_anomalies.extend(z_outlier_indices)
                        detection_summary[column]['Z-Score'] = len(z_outlier_indices)
                    except Exception as e:
                        st.warning(f"Z-Score analysis failed for {column}: {str(e)}")
                
                if "Modified Z-Score" in detection_methods:
                    try:
                        median = valid_numeric.median()
                        mad = np.median(np.abs(valid_numeric - median))
                        modified_z_scores = 0.6745 * (valid_numeric - median) / mad
                        mod_z_outlier_mask = np.abs(modified_z_scores) > z_threshold
                        mod_z_outlier_indices = valid_numeric[mod_z_outlier_mask].index.tolist()
                        column_anomalies.extend(mod_z_outlier_indices)
                        detection_summary[column]['Modified Z-Score'] = len(mod_z_outlier_indices)
                    except Exception as e:
                        st.warning(f"Modified Z-Score analysis failed for {column}: {str(e)}")
                
                if "IQR" in detection_methods:
                    try:
                        Q1 = valid_numeric.quantile(0.25)
                        Q3 = valid_numeric.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        if IQR > 0:
                            lower_bound = Q1 - iqr_multiplier * IQR
                            upper_bound = Q3 + iqr_multiplier * IQR
                            iqr_outlier_mask = (valid_numeric < lower_bound) | (valid_numeric > upper_bound)
                            iqr_outlier_indices = valid_numeric[iqr_outlier_mask].index.tolist()
                            column_anomalies.extend(iqr_outlier_indices)
                            detection_summary[column]['IQR'] = len(iqr_outlier_indices)
                    except Exception as e:
                        st.warning(f"IQR analysis failed for {column}: {str(e)}")
                
                if "Percentile-based" in detection_methods:
                    try:
                        lower_percentile = np.percentile(valid_numeric, percentile_lower)
                        upper_percentile = np.percentile(valid_numeric, percentile_upper)
                        percentile_mask = (valid_numeric < lower_percentile) | (valid_numeric > upper_percentile)
                        percentile_indices = valid_numeric[percentile_mask].index.tolist()
                        column_anomalies.extend(percentile_indices)
                        detection_summary[column]['Percentile'] = len(percentile_indices)
                    except Exception as e:
                        st.warning(f"Percentile analysis failed for {column}: {str(e)}")
                
                if "Standard Deviation" in detection_methods:
                    try:
                        mean = valid_numeric.mean()
                        std = valid_numeric.std()
                        std_outlier_mask = np.abs(valid_numeric - mean) > (z_threshold * std)
                        std_outlier_indices = valid_numeric[std_outlier_mask].index.tolist()
                        column_anomalies.extend(std_outlier_indices)
                        detection_summary[column]['Std Dev'] = len(std_outlier_indices)
                    except Exception as e:
                        st.warning(f"Standard Deviation analysis failed for {column}: {str(e)}")
                
                # MACHINE LEARNING METHODS
                if "Isolation Forest" in detection_methods:
                    try:
                        from sklearn.ensemble import IsolationForest
                        iso_forest = IsolationForest(contamination=contamination, random_state=42)
                        outliers = iso_forest.fit_predict(valid_numeric.values.reshape(-1, 1))
                        iso_outlier_indices = valid_numeric[outliers == -1].index.tolist()
                        column_anomalies.extend(iso_outlier_indices)
                        detection_summary[column]['Isolation Forest'] = len(iso_outlier_indices)
                    except ImportError:
                        st.warning("⚠️ Scikit-learn not installed. Skipping Isolation Forest.")
                    except Exception as e:
                        st.warning(f"Isolation Forest failed for {column}: {str(e)}")
                
                if "Local Outlier Factor" in detection_methods:
                    try:
                        from sklearn.neighbors import LocalOutlierFactor
                        lof = LocalOutlierFactor(contamination=contamination)
                        outliers = lof.fit_predict(valid_numeric.values.reshape(-1, 1))
                        lof_outlier_indices = valid_numeric[outliers == -1].index.tolist()
                        column_anomalies.extend(lof_outlier_indices)
                        detection_summary[column]['LOF'] = len(lof_outlier_indices)
                    except ImportError:
                        st.warning("⚠️ Scikit-learn not installed. Skipping LOF.")
                    except Exception as e:
                        st.warning(f"LOF analysis failed for {column}: {str(e)}")
                
                if "One-Class SVM" in detection_methods:
                    try:
                        from sklearn.svm import OneClassSVM
                        svm = OneClassSVM(gamma='scale', nu=contamination)
                        outliers = svm.fit_predict(valid_numeric.values.reshape(-1, 1))
                        svm_outlier_indices = valid_numeric[outliers == -1].index.tolist()
                        column_anomalies.extend(svm_outlier_indices)
                        detection_summary[column]['One-Class SVM'] = len(svm_outlier_indices)
                    except ImportError:
                        st.warning("⚠️ Scikit-learn not installed. Skipping One-Class SVM.")
                    except Exception as e:
                        st.warning(f"One-Class SVM analysis failed for {column}: {str(e)}")
                
                if "DBSCAN Clustering" in detection_methods:
                    try:
                        from sklearn.cluster import DBSCAN
                        dbscan = DBSCAN(eps=0.5, min_samples=5)
                        clusters = dbscan.fit_predict(valid_numeric.values.reshape(-1, 1))
                        dbscan_outlier_indices = valid_numeric[clusters == -1].index.tolist()
                        column_anomalies.extend(dbscan_outlier_indices)
                        detection_summary[column]['DBSCAN'] = len(dbscan_outlier_indices)
                    except ImportError:
                        st.warning("⚠️ Scikit-learn not installed. Skipping DBSCAN.")
                    except Exception as e:
                        st.warning(f"DBSCAN analysis failed for {column}: {str(e)}")
                
                # Remove duplicates and store
                if column_anomalies:
                    anomalies[column] = list(set(column_anomalies))
            
            # CATEGORICAL ANOMALIES
            if categorical_detection:
                categorical_anomalies = {}
                for column in df.columns:
                    if df[column].dtype == 'object':
                        value_counts = df[column].value_counts()
                        # Find rare categories (appearing less than 1% of the time)
                        rare_threshold = len(df) * 0.01
                        rare_categories = value_counts[value_counts < rare_threshold].index
                        
                        if len(rare_categories) > 0:
                            rare_indices = df[df[column].isin(rare_categories)].index.tolist()
                            if rare_indices:
                                categorical_anomalies[f"{column}_rare_categories"] = rare_indices
                                detection_summary[f"{column}_categorical"] = {'Rare Categories': len(rare_indices)}
                
                anomalies.update(categorical_anomalies)
            
            # BUSINESS RULES VALIDATION
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
                        detection_summary['Business Rules'] = {'Violations': len(business_violations)}
            
            # DATA TYPE VALIDATION
            if "Data Type Validation" in detection_methods:
                dtype_violations = []
                for column in df.columns:
                    expected_dtype = df[column].dtype
                    
                    for idx, value in df[column].items():
                        try:
                            if expected_dtype == 'int64' and not str(value).replace('-', '').isdigit():
                                if str(value) not in error_markers:
                                    dtype_violations.append({'row': idx, 'column': column, 'value': value, 'expected_type': 'integer'})
                            elif expected_dtype == 'float64' and pd.isna(pd.to_numeric(value, errors='coerce')):
                                if str(value) not in error_markers:
                                    dtype_violations.append({'row': idx, 'column': column, 'value': value, 'expected_type': 'float'})
                        except:
                            continue
                
                if dtype_violations:
                    anomalies['Data Type Violations'] = dtype_violations
                    detection_summary['Data Type'] = {'Violations': len(dtype_violations)}
            
            # RANGE VALIDATION
            if "Range Validation" in detection_methods:
                range_violations = []
                
                # Define reasonable ranges for common business data
                range_rules = {
                    'Quantity': (0, 1000),  # Reasonable quantity range
                    'Price Per Unit': (0, 10000),  # Reasonable price range
                    'Total Spent': (0, 100000)  # Reasonable total range
                }
                
                for column, (min_val, max_val) in range_rules.items():
                    if column in df.columns:
                        numeric_col = pd.to_numeric(df[column], errors='coerce')
                        range_mask = (numeric_col < min_val) | (numeric_col > max_val)
                        range_indices = df[range_mask & numeric_col.notna()].index.tolist()
                        
                        if range_indices:
                            range_violations.extend([{'row': idx, 'column': column, 'value': df.loc[idx, column], 
                                                    'range': f'{min_val}-{max_val}'} for idx in range_indices])
                
                if range_violations:
                    anomalies['Range Violations'] = range_violations
                    detection_summary['Range Validation'] = {'Violations': len(range_violations)}
            
            # Store results in session state
            detection_params = {
                'z_threshold': z_threshold,
                'iqr_multiplier': iqr_multiplier,
                'percentile_lower': percentile_lower,
                'percentile_upper': percentile_upper,
                'contamination': contamination,
                'multivariate_detection': multivariate_detection,
                'time_series_detection': time_series_detection,
                'categorical_detection': categorical_detection
            }
            
            st.session_state['anomaly_results'] = anomalies
            st.session_state['detection_methods_used'] = detection_methods
            st.session_state['detection_params'] = detection_params
            st.session_state['detection_summary'] = detection_summary
            
            # Clear previous button states
            for key in ['show_export', 'show_removal', 'show_flagging']:
                if key in st.session_state:
                    del st.session_state[key]
    
    # DISPLAY RESULTS (Persistent - will show even after button clicks)
    if 'anomaly_results' in st.session_state:
        anomalies = st.session_state['anomaly_results']
        detection_methods_used = st.session_state.get('detection_methods_used', [])
        detection_params = st.session_state.get('detection_params', {})
        detection_summary = st.session_state.get('detection_summary', {})
        
        st.subheader("📈 Comprehensive Anomaly Detection Results")
        
        # Show detection summary
        st.info(f"""
        🔍 **Detection Configuration:** {len(detection_methods_used)} methods used  
        📊 **Parameters:** Z-Score: {detection_params.get('z_threshold', 2.5)}, IQR: {detection_params.get('iqr_multiplier', 1.5)}, Contamination: {detection_params.get('contamination', 0.1)}
        """)
        
        # Show method-wise detection summary
        with st.expander("📋 Detection Summary by Method", expanded=False):
            summary_data = []
            for column, methods in detection_summary.items():
                for method, count in methods.items():
                    summary_data.append({
                        'Column': column,
                        'Method': method,
                        'Anomalies Found': count
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        if anomalies:
            # Calculate total anomalies
            total_anomalies = 0
            for key, value in anomalies.items():
                total_anomalies += len(value)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Anomalies", total_anomalies)
            with col2:
                st.metric("Affected Features", len(anomalies))
            with col3:
                anomaly_percentage = (total_anomalies / len(df)) * 100
                st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
            with col4:
                st.metric("Methods Used", len(detection_methods_used))
            
            # Show anomalies by column/type
            for anomaly_type, data in anomalies.items():
                if 'Business Logic Violations' in anomaly_type:
                    with st.expander(f"💼 {anomaly_type}: {len(data)} violations", expanded=True):
                        st.write("**Business Rule Violations:**")
                        if data:
                            violation_df = pd.DataFrame(data[:10])
                            st.dataframe(violation_df, use_container_width=True)
                
                elif 'Data Type Violations' in anomaly_type:
                    with st.expander(f"🔧 {anomaly_type}: {len(data)} violations", expanded=True):
                        st.write("**Data Type Violations:**")
                        if data:
                            violation_df = pd.DataFrame(data[:10])
                            st.dataframe(violation_df, use_container_width=True)
                
                elif 'Range Violations' in anomaly_type:
                    with st.expander(f"📏 {anomaly_type}: {len(data)} violations", expanded=True):
                        st.write("**Range Violations:**")
                        if data:
                            violation_df = pd.DataFrame(data[:10])
                            st.dataframe(violation_df, use_container_width=True)
                
                elif 'rare_categories' in anomaly_type:
                    indices = data
                    column = anomaly_type.replace('_rare_categories', '')
                    with st.expander(f"🏷️ {column}: {len(indices)} rare category anomalies", expanded=True):
                        
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            st.write("**Rare Category Statistics:**")
                            rare_values = df.loc[indices, column].value_counts()
                            st.dataframe(rare_values.head(10), use_container_width=True)
                        
                        with col_right:
                            # Visualization
                            try:
                                fig = px.bar(x=rare_values.index[:10], y=rare_values.values[:10], 
                                           title=f"Rare Categories in {column}")
                                fig.update_layout(height=300, xaxis_tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.write(f"Visualization error: {str(e)}")
                
                else:
                    # Regular statistical anomalies
                    indices = data
                    with st.expander(f"📊 {anomaly_type}: {len(indices)} anomalies", expanded=True):
                        
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            st.write("**Anomaly Statistics:**")
                            anomaly_values = pd.to_numeric(df.loc[indices, anomaly_type], errors='coerce').dropna()
                            
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
                                if anomaly_type in df.columns:
                                    all_numeric = pd.to_numeric(df[anomaly_type], errors='coerce').dropna()
                                    
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
                                            title=f"{anomaly_type} Distribution",
                                            xaxis_title=anomaly_type,
                                            yaxis_title="Count",
                                            height=300,
                                            showlegend=True,
                                            barmode='overlay'
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            except Exception as e:
                                st.write(f"Visualization error: {str(e)}")
                        
                        # Show sample rows button
                        sample_key = f"show_sample_{anomaly_type}_{hash(str(indices))}"
                        if st.button(f"Show Sample Rows", key=sample_key):
                            sample_size = min(10, len(indices))
                            sample_indices = indices[:sample_size]
                            st.write("**Sample Anomalous Rows:**")
                            st.dataframe(df.loc[sample_indices], use_container_width=True)
            
            # ENHANCED HANDLING BUTTONS SECTION
            st.subheader("🔧 Advanced Anomaly Handling")
            st.write("Choose how you want to handle the detected anomalies:")
            
            # Create four columns for buttons
            col1, col2, col3, col4 = st.columns(4)
            
            # EXPORT REPORT BUTTON
            with col1:
                export_clicked = st.button("📊 Export Report", use_container_width=True, key="export_report_btn")
                
                if export_clicked:
                    st.session_state['show_export'] = True
                
                # Show export functionality
                if st.session_state.get('show_export', False):
                    try:
                        with st.container():
                            st.write("**📊 Export Report Results:**")
                            
                            # Generate comprehensive report
                            anomaly_report = {
                                'detection_metadata': {
                                    'timestamp': pd.Timestamp.now().isoformat(),
                                    'dataset_size': len(df),
                                    'total_anomalies': total_anomalies,
                                    'anomaly_rate': f"{anomaly_percentage:.2f}%",
                                    'methods_used': detection_methods_used,
                                    'parameters': detection_params
                                },
                                'detection_summary': detection_summary,
                                'anomalies_by_type': {}
                            }
                            
                            # Process each type of anomaly
                            for anomaly_type, data in anomalies.items():
                                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                                    # Complex anomalies (business rules, data type, etc.)
                                    anomaly_report['anomalies_by_type'][anomaly_type] = {
                                        'count': len(data),
                                        'type': 'complex_violation',
                                        'sample_violations': data[:5]
                                    }
                                else:
                                    # Simple anomalies (statistical outliers)
                                    sample_values = []
                                    if len(data) > 0:
                                        try:
                                            if anomaly_type in df.columns:
                                                sample_vals = df.loc[data[:5], anomaly_type]
                                                sample_values = [float(x) for x in pd.to_numeric(sample_vals, errors='coerce').dropna()]
                                        except:
                                            sample_values = []
                                    
                                    anomaly_report['anomalies_by_type'][anomaly_type] = {
                                        'count': len(data),
                                        'type': 'statistical_outlier',
                                        'sample_indices': data[:10],
                                        'sample_values': sample_values
                                    }
                            
                            # Create JSON report
                            report_json = json.dumps(anomaly_report, indent=2, default=str)
                            
                            # Display summary
                            st.success("✅ Comprehensive anomaly report generated!")
                            st.info(f"""
                            **Report Summary:**
                            • Total anomalies: {total_anomalies:,}
                            • Anomaly types: {len(anomalies)}
                            • Detection methods: {len(detection_methods_used)}
                            • File size: {len(report_json):,} characters
                            """)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Comprehensive Report (JSON)",
                                data=report_json,
                                file_name=f"comprehensive_anomaly_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                key="download_anomaly_report",
                                help="Download detailed anomaly analysis report with all detection methods"
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
            
            # REMOVE ANOMALIES BUTTON  
            with col2:
                remove_clicked = st.button("🗑️ Remove Anomalies", use_container_width=True, key="remove_anomalies_btn")
                
                if remove_clicked:
                    st.session_state['show_removal'] = True
                
                # Show removal functionality
                if st.session_state.get('show_removal', False):
                    try:
                        with st.container():
                            st.write("**🗑️ Remove Anomalies Results:**")
                            
                            # Collect all anomalous indices
                            all_indices = set()
                            
                            for anomaly_type, data in anomalies.items():
                                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                                    # Complex anomalies with row indices
                                    if 'row' in data[0]:
                                        violation_indices = [v['row'] for v in data]
                                        all_indices.update(violation_indices)
                                else:
                                    # Simple anomalies (list of indices)
                                    all_indices.update(data)
                            
                            if all_indices:
                                # Create cleaned dataset
                                cleaned_df = df.drop(list(all_indices))
                                st.session_state['anomaly_cleaned_df'] = cleaned_df
                                
                                # Show removal statistics
                                removal_percentage = (len(all_indices) / len(df)) * 100
                                
                                st.success(f"✅ Successfully removed {len(all_indices):,} anomalous rows!")
                                
                                # Display impact metrics
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Original Size", f"{len(df):,}")
                                with col_b:
                                    st.metric("Cleaned Size", f"{len(cleaned_df):,}")
                                with col_c:
                                    st.metric("Removed", f"{len(all_indices):,}")
                                
                                st.info(f"""
                                **Removal Impact:**
                                • Removed {removal_percentage:.1f}% of the dataset
                                • Data integrity: {100 - removal_percentage:.1f}% retained
                                • New dataset size: {len(cleaned_df):,} rows × {len(cleaned_df.columns)} columns
                                """)
                                
                                # Show preview of cleaned data
                                st.write("**Preview of cleaned dataset:**")
                                st.dataframe(cleaned_df.head(), use_container_width=True)
                                
                                # Create CSV for download
                                cleaned_csv = cleaned_df.to_csv(index=False)
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Cleaned Dataset (CSV)",
                                    data=cleaned_csv,
                                    file_name=f"cleaned_dataset_no_anomalies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key="download_cleaned_no_anomalies",
                                    help="Download dataset with anomalous rows removed"
                                )
                                
                                # NEW: Send to Cleaning Configuration Button
                                st.write("---")
                                if st.button("🔄 Send to Cleaning Configuration", 
                                           type="secondary", 
                                           use_container_width=True, 
                                           key="send_cleaned_to_config",
                                           help="Replace original data with cleaned data for further processing"):
                                    
                                    # Replace the original dataset with cleaned dataset
                                    st.session_state['original_df'] = cleaned_df.copy()
                                    
                                    # Clear previous profiles to trigger re-analysis
                                    if 'profile' in st.session_state:
                                        del st.session_state['profile']
                                    
                                    # Clear previous cleaning results
                                    for key in ['cleaned_df', 'cleaning_report', 'cleaning_strategies']:
                                        if key in st.session_state:
                                            del st.session_state[key]
                                    
                                    # Show success and redirect info
                                    st.success("✅ Dataset updated! Cleaned data is now your working dataset.")
                                    st.info("""
                                    **Next Steps:**
                                    1. Go to 'Cleaning Configuration' page
                                    2. Your anomaly-cleaned data will be analyzed
                                    3. Configure additional cleaning strategies if needed
                                    """)
                                    
                                    # Auto-redirect option
                                    if st.button("🎯 Go to Cleaning Configuration Now", 
                                               type="primary",
                                               key="auto_redirect_cleaning"):
                                        st.rerun()
                                
                            else:
                                st.warning("⚠️ No anomalous rows found to remove.")
                                
                    except Exception as e:
                        st.error(f"❌ Error removing anomalies: {str(e)}")
            
            # FLAG ANOMALIES BUTTON
            with col3:
                flag_clicked = st.button("🏷️ Flag Anomalies", use_container_width=True, key="flag_anomalies_btn")
                
                if flag_clicked:
                    st.session_state['show_flagging'] = True
                
                # Show flagging functionality
                if st.session_state.get('show_flagging', False):
                    try:
                        with st.container():
                            st.write("**🏷️ Flag Anomalies Results:**")
                            
                            # Create flagged dataset
                            flagged_df = df.copy()
                            flagged_df['is_anomaly'] = False
                            flagged_df['anomaly_type'] = ''
                            flagged_df['anomaly_category'] = ''
                            flagged_df['anomaly_method'] = ''
                            flagged_df['anomaly_severity'] = ''
                            flagged_df['anomaly_reason'] = ''
                            
                            total_flagged = 0
                            
                            # Flag each type of anomaly
                            for anomaly_type, data in anomalies.items():
                                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                                    # Complex anomalies
                                    if 'row' in data[0]:
                                        violation_indices = [v['row'] for v in data]
                                        if violation_indices:
                                            flagged_df.loc[violation_indices, 'is_anomaly'] = True
                                            flagged_df.loc[violation_indices, 'anomaly_category'] = 'Validation'
                                            flagged_df.loc[violation_indices, 'anomaly_type'] = anomaly_type
                                            flagged_df.loc[violation_indices, 'anomaly_method'] = 'Rule-based Validation'
                                            flagged_df.loc[violation_indices, 'anomaly_severity'] = 'High'
                                            flagged_df.loc[violation_indices, 'anomaly_reason'] = f'{anomaly_type} detected'
                                            total_flagged += len(violation_indices)
                                else:
                                    # Simple statistical anomalies
                                    if data:
                                        flagged_df.loc[data, 'is_anomaly'] = True
                                        flagged_df.loc[data, 'anomaly_category'] = 'Statistical'
                                        flagged_df.loc[data, 'anomaly_type'] = anomaly_type
                                        
                                        # Determine methods used
                                        methods_used = []
                                        for method in detection_methods_used:
                                            if method in ['Z-Score', 'Modified Z-Score', 'IQR', 'Percentile-based', 'Standard Deviation']:
                                                methods_used.append(method)
                                            elif method in ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM', 'DBSCAN Clustering']:
                                                methods_used.append(method)
                                        
                                        flagged_df.loc[data, 'anomaly_method'] = ", ".join(methods_used) if methods_used else 'Multiple'
                                        
                                        # Set severity based on anomaly type
                                        if 'rare_categories' in anomaly_type:
                                            flagged_df.loc[data, 'anomaly_severity'] = 'Low'
                                            flagged_df.loc[data, 'anomaly_reason'] = 'Rare categorical value'
                                        else:
                                            flagged_df.loc[data, 'anomaly_severity'] = 'Medium'
                                            flagged_df.loc[data, 'anomaly_reason'] = 'Statistical outlier'
                                        
                                        total_flagged += len(data)
                            
                            # Store flagged dataset
                            st.session_state['flagged_df'] = flagged_df
                            
                            # Show flagging statistics
                            flag_percentage = (total_flagged / len(df)) * 100
                            
                            st.success(f"✅ Successfully flagged {total_flagged:,} rows as anomalous!")
                            
                            # Display flagging metrics
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Total Rows", f"{len(df):,}")
                            with col_b:
                                st.metric("Flagged Anomalies", f"{total_flagged:,}")
                            with col_c:
                                st.metric("Normal Rows", f"{len(df) - total_flagged:,}")
                            
                            st.info(f"""
                            **Flagging Results:**
                            • Anomaly rate: {flag_percentage:.1f}%
                            • Normal data rate: {100 - flag_percentage:.1f}%
                            • New columns added: 6 (comprehensive anomaly metadata)
                            """)
                            
                            # Show preview of flagged data
                            st.write("**Preview of flagged dataset:**")
                            flagged_preview = flagged_df[flagged_df['is_anomaly'] == True].head()
                            
                            if not flagged_preview.empty:
                                display_columns = ['Transaction ID', 'Total Spent', 'is_anomaly', 'anomaly_type', 'anomaly_category', 'anomaly_method', 'anomaly_severity']
                                available_columns = [col for col in display_columns if col in flagged_preview.columns]
                                st.dataframe(flagged_preview[available_columns], use_container_width=True)
                            
                            # Create CSV for download
                            flagged_csv = flagged_df.to_csv(index=False)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Flagged Dataset (CSV)",
                                data=flagged_csv,
                                file_name=f"dataset_with_comprehensive_anomaly_flags_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_flagged_dataset",
                                help="Download dataset with comprehensive anomaly flags added as new columns"
                            )
                            
                            # NEW: Send to Cleaning Configuration Button
                            st.write("---")
                            if st.button("🔄 Send to Cleaning Configuration", 
                                       type="secondary", 
                                       use_container_width=True, 
                                       key="send_flagged_to_config",
                                       help="Replace original data with flagged data for further processing"):
                                
                                # Replace the original dataset with flagged dataset
                                st.session_state['original_df'] = flagged_df.copy()
                                
                                # Clear previous profiles to trigger re-analysis
                                if 'profile' in st.session_state:
                                    del st.session_state['profile']
                                
                                # Clear previous cleaning results
                                for key in ['cleaned_df', 'cleaning_report', 'cleaning_strategies']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                
                                # Show success and redirect info
                                st.success("✅ Dataset updated! Flagged data is now your working dataset.")
                                st.info("""
                                **Next Steps:**
                                1. Go to 'Cleaning Configuration' page
                                2. Your anomaly-flagged data will be analyzed
                                3. Configure cleaning strategies (anomaly flags will be preserved)
                                """)
                                
                                # Auto-redirect option
                                if st.button("🎯 Go to Cleaning Configuration Now", 
                                           type="primary",
                                           key="auto_redirect_flagged"):
                                    st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Error flagging anomalies: {str(e)}")
            
            # NEW: BOTH DATASETS DOWNLOAD BUTTON
            with col4:
                if (st.session_state.get('show_removal', False) and st.session_state.get('anomaly_cleaned_df') is not None) or \
                   (st.session_state.get('show_flagging', False) and st.session_state.get('flagged_df') is not None):
                    
                    download_both_clicked = st.button("📦 Download Both", use_container_width=True, key="download_both_btn")
                    
                    if download_both_clicked:
                        st.session_state['show_both_downloads'] = True
                    
                    if st.session_state.get('show_both_downloads', False):
                        with st.container():
                            st.write("**📦 Download Both Datasets:**")
                            
                            # Create download buttons for both datasets
                            if st.session_state.get('anomaly_cleaned_df') is not None:
                                cleaned_csv = st.session_state['anomaly_cleaned_df'].to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Cleaned Dataset",
                                    data=cleaned_csv,
                                    file_name=f"cleaned_no_anomalies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key="download_both_cleaned",
                                    use_container_width=True
                                )
                            
                            if st.session_state.get('flagged_df') is not None:
                                flagged_csv = st.session_state['flagged_df'].to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Flagged Dataset", 
                                    data=flagged_csv,
                                    file_name=f"flagged_comprehensive_anomalies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key="download_both_flagged",
                                    use_container_width=True
                                )
                            
                            st.success("✅ Both datasets are ready for download!")
        
        else:
            st.success("✅ No anomalies detected in the dataset!")
            st.info(f"""
            **Possible reasons:**
            • Your data is very clean and well-distributed
            • Detection thresholds might be too strict
            • Try adjusting parameters or using different methods
            • Consider enabling categorical anomaly detection
            """)
            
            # Show data distribution for context
            st.write("**📊 Data Distribution Analysis:**")
            numeric_cols = []
            for col in df.columns:
                if pd.to_numeric(df[col], errors='coerce').notna().sum() > 10:
                    numeric_cols.append(col)
            
            if numeric_cols:
                selected_col = st.selectbox("Select column to analyze:", numeric_cols, key="no_anomaly_analysis")
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
                        fig.update_layout(height=300)
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
            
            # NEW: Show dataset source info
            if 'is_anomaly' in df.columns:
                anomaly_count = df['is_anomaly'].sum()
                st.info(f"📊 **Anomaly-flagged dataset detected:** {anomaly_count:,} anomalies flagged with comprehensive metadata")
            elif len(df) < 10000:  # Assuming original had more rows
                st.info(f"🗑️ **Anomaly-cleaned dataset detected:** Anomalous rows have been removed")
            else:
                st.info(f"📁 **Original dataset:** Ready for anomaly detection and processing")
            
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
    
    # Show dataset type info
    if 'is_anomaly' in df.columns:
        anomaly_count = df['is_anomaly'].sum()
        st.info(f"📊 **Working with anomaly-flagged dataset:** {anomaly_count:,} rows flagged as anomalous")
    elif 'anomaly_cleaned_df' in st.session_state:
        st.info(f"🗑️ **Working with anomaly-cleaned dataset:** Anomalous rows have been removed")
    
    st.subheader("🔧 Configure Cleaning Strategies")
    
    total_issues = sum([profile.get('column_profiles', {}).get(col, {}).get('null_count', 0) + 
                       profile.get('column_profiles', {}).get(col, {}).get('error_count', 0) 
                       for col in df.columns])
    
    if total_issues > 1000:
        st.warning(f"⚠️ Large dataset detected ({total_issues:,} issues). LLM predictions may take significant time and cost.")
    
    cleaning_strategies = {}
    
    for column in df.columns:
        # Skip anomaly flag columns
        if column.startswith('anomaly_') or column == 'is_anomaly':
            continue
            
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
            file_name="final_cleaned_dataset.csv",
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
