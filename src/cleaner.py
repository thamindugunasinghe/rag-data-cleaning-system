import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .llm_handler import LLMHandler
from .data_profiler import DataProfiler

class DataCleaner:
    def __init__(self):
        self.llm_handler = LLMHandler()
        self.profiler = DataProfiler()
        self.error_markers = ["ERROR", "UNKNOWN", "N/A", "NULL", "", "error", "unknown"]
    
    def clean_dataset(self, df: pd.DataFrame, cleaning_strategies: Dict[str, str]) -> pd.DataFrame:
        """Clean entire dataset based on specified strategies"""
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Get data profile
        profile = self.profiler.profile_dataset(df)
        
        # Clean each column based on strategy
        for column, strategy in cleaning_strategies.items():
            if column in cleaned_df.columns:
                column_profile = profile['column_profiles'][column]
                cleaned_df[column] = self._clean_column(
                    cleaned_df[column], 
                    strategy, 
                    column_profile,
                    cleaned_df
                )
        
        return cleaned_df
    
    def _clean_column(self, series: pd.Series, strategy: str, profile: Dict, full_df: pd.DataFrame) -> pd.Series:
        """Clean individual column based on strategy"""
        
        # First, mark error values as NaN
        cleaned_series = series.copy()
        error_mask = cleaned_series.astype(str).isin(self.error_markers)
        cleaned_series[error_mask] = np.nan
        
        # Apply cleaning strategy
        if strategy == "Mean Imputation" and profile['data_type'] == 'quantitative':
            return self._mean_imputation(cleaned_series)
        
        elif strategy == "Median Imputation" and profile['data_type'] == 'quantitative':
            return self._median_imputation(cleaned_series)
        
        elif strategy == "Mode Imputation":
            return self._mode_imputation(cleaned_series)
        
        elif strategy == "Forward Fill":
            return cleaned_series.fillna(method='ffill')
        
        elif strategy == "Backward Fill":
            return cleaned_series.fillna(method='bfill')
        
        elif strategy == "LLM Prediction" or strategy == "LLM Context Prediction":
            return self._llm_imputation(cleaned_series, series.name, profile, full_df)
        
        elif strategy == "Remove Rows":
            # This will be handled at dataset level
            return cleaned_series
        
        else:
            # Default to mode imputation
            return self._mode_imputation(cleaned_series)
    
    def _mean_imputation(self, series: pd.Series) -> pd.Series:
        """Fill missing values with mean"""
        numeric_series = pd.to_numeric(series, errors='coerce')
        mean_value = numeric_series.mean()
        return numeric_series.fillna(mean_value)
    
    def _median_imputation(self, series: pd.Series) -> pd.Series:
        """Fill missing values with median"""
        numeric_series = pd.to_numeric(series, errors='coerce')
        median_value = numeric_series.median()
        return numeric_series.fillna(median_value)
    
    def _mode_imputation(self, series: pd.Series) -> pd.Series:
        """Fill missing values with mode"""
        mode_value = series.mode()
        if len(mode_value) > 0:
            return series.fillna(mode_value.iloc[0])
        return series
    
    def _llm_imputation(self, series: pd.Series, column_name: str, profile: Dict, full_df: pd.DataFrame) -> pd.Series:
        """Fill missing values using LLM predictions"""
        
        filled_series = series.copy()
        missing_indices = series.isnull()
        
        for idx in series[missing_indices].index:
            # Get context from other columns
            context = {}
            for col in full_df.columns:
                if col != column_name and pd.notna(full_df.loc[idx, col]):
                    context[col] = full_df.loc[idx, col]
            
            # Get LLM prediction
            predicted_value = self.llm_handler.predict_missing_value(
                context, column_name, profile['data_type']
            )
            
            if predicted_value:
                # Convert to appropriate type
                if profile['data_type'] == 'quantitative':
                    try:
                        predicted_value = float(predicted_value)
                    except:
                        predicted_value = profile.get('median', 0)
                
                filled_series.loc[idx] = predicted_value
        
        return filled_series
    
    def generate_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
        """Generate comprehensive cleaning report"""
        
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
                report['improvements'][column] = {
                    'quality_improvement': clean_quality - orig_quality,
                    'original_quality': orig_quality,
                    'final_quality': clean_quality
                }
        
        return report