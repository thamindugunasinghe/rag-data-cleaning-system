import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st
import time
from .llm_handler import LLMHandler
from .data_profiler import DataProfiler

class DataCleaner:
    def __init__(self):
        self.llm_handler = LLMHandler()
        self.profiler = DataProfiler()
        self.error_markers = ["ERROR", "UNKNOWN", "N/A", "NULL", "", "error", "unknown"]
    
    def clean_dataset(self, df: pd.DataFrame, cleaning_strategies: Dict[str, str]) -> pd.DataFrame:
        """Clean entire dataset based on specified strategies"""
        
        print(f"🚀 Starting data cleaning process...")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🔧 Cleaning strategies: {cleaning_strategies}")
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Get data profile
        print("📈 Generating data profile...")
        profile = self.profiler.profile_dataset(df)
        
        # Clean each column based on strategy
        total_columns = len(cleaning_strategies)
        
        for idx, (column, strategy) in enumerate(cleaning_strategies.items(), 1):
            if column in cleaned_df.columns:
                print(f"\n🔄 Processing column {idx}/{total_columns}: '{column}'")
                print(f"📋 Strategy: {strategy}")
                
                column_profile = profile['column_profiles'][column]
                issues_count = column_profile['null_count'] + column_profile['error_count']
                print(f"⚠️  Issues to fix: {issues_count}")
                
                # Update progress in Streamlit
                if hasattr(st, 'session_state'):
                    progress = idx / total_columns
                    st.session_state['cleaning_progress'] = progress
                
                start_time = time.time()
                
                cleaned_df[column] = self._clean_column(
                    cleaned_df[column], 
                    strategy, 
                    column_profile,
                    cleaned_df,
                    column_name=column
                )
                
                end_time = time.time()
                print(f"✅ Column '{column}' cleaned in {end_time - start_time:.2f} seconds")
        
        print(f"\n🎉 Data cleaning completed successfully!")
        return cleaned_df
    
    def _clean_column(self, series: pd.Series, strategy: str, profile: Dict, 
                     full_df: pd.DataFrame, column_name: str) -> pd.Series:
        """Clean individual column based on strategy"""
        
        print(f"  🔍 Analyzing column '{column_name}'...")
        
        # First, mark error values as NaN
        cleaned_series = series.copy()
        error_mask = cleaned_series.astype(str).isin(self.error_markers)
        error_count = error_mask.sum()
        
        if error_count > 0:
            print(f"  🚫 Found {error_count} error markers, converting to NaN")
            cleaned_series[error_mask] = np.nan
        
        # Count missing values
        missing_count = cleaned_series.isnull().sum()
        print(f"  📊 Missing values to handle: {missing_count}")
        
        if missing_count == 0:
            print(f"  ✅ No missing values in '{column_name}', skipping cleaning")
            return cleaned_series
        
        # Apply cleaning strategy
        print(f"  🛠️  Applying strategy: {strategy}")
        
        if strategy == "Mean Imputation" and profile['data_type'] == 'quantitative':
            return self._mean_imputation(cleaned_series, column_name)
        
        elif strategy == "Median Imputation" and profile['data_type'] == 'quantitative':
            return self._median_imputation(cleaned_series, column_name)
        
        elif strategy == "Mode Imputation":
            return self._mode_imputation(cleaned_series, column_name)
        
        elif strategy == "Forward Fill":
            print(f"  ➡️  Applying forward fill...")
            return cleaned_series.fillna(method='ffill')
        
        elif strategy == "Backward Fill":
            print(f"  ⬅️  Applying backward fill...")
            return cleaned_series.fillna(method='bfill')
        
        elif strategy == "LLM Prediction" or strategy == "LLM Context Prediction":
            return self._llm_imputation(cleaned_series, column_name, profile, full_df)
        
        elif strategy == "Remove Rows":
            print(f"  🗑️  Rows with missing values will be removed at dataset level")
            return cleaned_series
        
        else:
            print(f"  ⚠️  Unknown strategy '{strategy}', defaulting to mode imputation")
            return self._mode_imputation(cleaned_series, column_name)
    
    def _mean_imputation(self, series: pd.Series, column_name: str) -> pd.Series:
        """Fill missing values with mean"""
        print(f"  📊 Calculating mean for '{column_name}'...")
        numeric_series = pd.to_numeric(series, errors='coerce')
        mean_value = numeric_series.mean()
        
        if pd.isna(mean_value):
            print(f"  ⚠️  Cannot calculate mean (no valid numeric values), using 0")
            mean_value = 0
        else:
            print(f"  ✅ Mean value: {mean_value:.4f}")
        
        filled_series = numeric_series.fillna(mean_value)
        filled_count = (numeric_series.isna() & filled_series.notna()).sum()
        print(f"  🔧 Filled {filled_count} missing values with mean")
        
        return filled_series
    
    def _median_imputation(self, series: pd.Series, column_name: str) -> pd.Series:
        """Fill missing values with median"""
        print(f"  📊 Calculating median for '{column_name}'...")
        numeric_series = pd.to_numeric(series, errors='coerce')
        median_value = numeric_series.median()
        
        if pd.isna(median_value):
            print(f"  ⚠️  Cannot calculate median (no valid numeric values), using 0")
            median_value = 0
        else:
            print(f"  ✅ Median value: {median_value:.4f}")
        
        filled_series = numeric_series.fillna(median_value)
        filled_count = (numeric_series.isna() & filled_series.notna()).sum()
        print(f"  🔧 Filled {filled_count} missing values with median")
        
        return filled_series
    
    def _mode_imputation(self, series: pd.Series, column_name: str) -> pd.Series:
        """Fill missing values with mode"""
        print(f"  📊 Calculating mode for '{column_name}'...")
        mode_values = series.mode()
        
        if len(mode_values) == 0:
            print(f"  ⚠️  Cannot calculate mode (no valid values), keeping NaN")
            return series
        
        mode_value = mode_values.iloc[0]
        print(f"  ✅ Mode value: '{mode_value}'")
        
        filled_series = series.fillna(mode_value)
        filled_count = (series.isna() & filled_series.notna()).sum()
        print(f"  🔧 Filled {filled_count} missing values with mode")
        
        return filled_series
    
    def _llm_imputation(self, series: pd.Series, column_name: str, profile: Dict, 
                       full_df: pd.DataFrame) -> pd.Series:
        """Fill missing values using LLM predictions with progress tracking"""
        
        filled_series = series.copy()
        missing_indices = series[series.isnull()].index.tolist()
        total_missing = len(missing_indices)
        
        print(f"  🤖 Starting LLM imputation for {total_missing} missing values...")
        
        if total_missing == 0:
            return filled_series
        
        # Limit LLM calls for performance (max 50 predictions)
        if total_missing > 50:
            print(f"  ⚠️  Large number of missing values ({total_missing}). Limiting to first 50 for performance.")
            missing_indices = missing_indices[:50]
            # Fill remaining with mode/median
            remaining_indices = series[series.isnull()].index.tolist()[50:]
            if profile['data_type'] == 'quantitative':
                fallback_value = pd.to_numeric(series, errors='coerce').median()
                if pd.isna(fallback_value):
                    fallback_value = 0
            else:
                fallback_value = series.mode().iloc[0] if len(series.mode()) > 0 else "Unknown"
            
            print(f"  📊 Filling remaining {len(remaining_indices)} values with fallback: {fallback_value}")
            for idx in remaining_indices:
                filled_series.loc[idx] = fallback_value
        
        # Process LLM predictions with progress tracking
        successful_predictions = 0
        failed_predictions = 0
        
        for i, idx in enumerate(missing_indices, 1):
            print(f"  🔄 LLM Prediction {i}/{len(missing_indices)} (Row {idx})")
            
            # Update Streamlit progress if available
            if hasattr(st, 'session_state'):
                sub_progress = i / len(missing_indices)
                st.session_state['llm_progress'] = sub_progress
            
            try:
                # Get context from other columns
                context = {}
                for col in full_df.columns:
                    if col != column_name and pd.notna(full_df.loc[idx, col]):
                        context[col] = full_df.loc[idx, col]
                
                print(f"    📋 Context: {dict(list(context.items())[:3])}...")  # Show first 3 items
                
                # Get LLM prediction
                predicted_value = self.llm_handler.predict_missing_value(
                    context, column_name, profile['data_type']
                )
                
                if predicted_value:
                    # Convert to appropriate type
                    if profile['data_type'] == 'quantitative':
                        try:
                            predicted_value = float(predicted_value)
                            print(f"    ✅ Predicted: {predicted_value}")
                        except ValueError:
                            print(f"    ⚠️  Invalid numeric prediction: '{predicted_value}', using median")
                            predicted_value = profile.get('median', 0)
                    else:
                        print(f"    ✅ Predicted: '{predicted_value}'")
                    
                    filled_series.loc[idx] = predicted_value
                    successful_predictions += 1
                else:
                    print(f"    ❌ LLM returned empty prediction")
                    failed_predictions += 1
                    
            except Exception as e:
                print(f"    ❌ LLM prediction failed: {str(e)}")
                failed_predictions += 1
                
                # Use fallback value
                if profile['data_type'] == 'quantitative':
                    fallback = profile.get('median', 0)
                else:
                    fallback = profile.get('mode', 'Unknown')
                filled_series.loc[idx] = fallback
                print(f"    🔄 Using fallback value: {fallback}")
        
        print(f"  📊 LLM Imputation completed:")
        print(f"    ✅ Successful predictions: {successful_predictions}")
        print(f"    ❌ Failed predictions: {failed_predictions}")
        
        return filled_series
    
    def generate_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
        """Generate comprehensive cleaning report"""
        
        print(f"📋 Generating cleaning report...")
        
        original_profile = self.profiler.profile_dataset(original_df)
        cleaned_profile = self.profiler.profile_dataset(cleaned_df)
        
        report = {
            'original_stats': original_profile,
            'cleaned_stats': cleaned_profile,
            'improvements': {}
        }
        
        # Calculate improvements
        for column in original_df.columns:
            if column in cleaned_df.columns:
                orig_quality = original_profile['column_profiles'][column]['data_quality']
                clean_quality = cleaned_profile['column_profiles'][column]['data_quality']
                improvement = clean_quality - orig_quality
                
                report['improvements'][column] = {
                    'quality_improvement': improvement,
                    'original_quality': orig_quality,
                    'final_quality': clean_quality
                }
                
                print(f"  📊 {column}: {orig_quality:.1f}% → {clean_quality:.1f}% (+{improvement:.1f}%)")
        
        print(f"✅ Cleaning report generated successfully!")
        return report
