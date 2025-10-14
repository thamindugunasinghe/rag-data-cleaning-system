import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DataProfiler:
    def __init__(self):
        self.error_markers = ["ERROR", "UNKNOWN", "N/A", "NULL", "", "error", "unknown"]
    
    def profile_dataset(self, df: pd.DataFrame) -> Dict:
        """Comprehensive data profiling"""
        profile = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_profiles': {}
        }
        
        for column in df.columns:
            profile['column_profiles'][column] = self._profile_column(df[column])
        
        return profile
    
    def _profile_column(self, series: pd.Series) -> Dict:
        """Profile individual column"""
        total_count = len(series)
        
        # Count missing values and errors
        null_count = series.isnull().sum()
        error_count = series.astype(str).isin(self.error_markers).sum()
        valid_count = total_count - null_count - error_count
        
        # Determine data type
        data_type = self._determine_data_type(series)
        
        profile = {
            'data_type': data_type,
            'total_count': total_count,
            'valid_count': valid_count,
            'null_count': null_count,
            'error_count': error_count,
            'data_quality': (valid_count / total_count) * 100,
            'unique_values': series.nunique(),
            'recommended_strategy': self._recommend_strategy(series, data_type)
        }
        
        if data_type == 'quantitative':
            valid_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(valid_series) > 0:
                profile.update({
                    'mean': valid_series.mean(),
                    'median': valid_series.median(),
                    'mode': valid_series.mode().iloc[0] if len(valid_series.mode()) > 0 else None,
                    'std': valid_series.std()
                })
        else:
            valid_series = series[~series.astype(str).isin(self.error_markers + [str(np.nan)])].dropna()
            if len(valid_series) > 0:
                profile['mode'] = valid_series.mode().iloc[0] if len(valid_series.mode()) > 0 else None
                profile['top_values'] = valid_series.value_counts().head(5).to_dict()
        
        return profile
    
    def _determine_data_type(self, series: pd.Series) -> str:
        """Determine if column is quantitative or qualitative"""
        # Clean series for analysis
        clean_series = series[~series.astype(str).isin(self.error_markers)].dropna()
        
        if len(clean_series) == 0:
            return 'unknown'
        
        # Try converting to numeric
        numeric_series = pd.to_numeric(clean_series, errors='coerce')
        numeric_count = numeric_series.notna().sum()
        
        # If >80% can be converted to numeric, consider quantitative
        if numeric_count / len(clean_series) > 0.8:
            return 'quantitative'
        else:
            return 'qualitative'
    
    def _recommend_strategy(self, series: pd.Series, data_type: str) -> str:
        """Recommend cleaning strategy based on data analysis"""
        if data_type == 'quantitative':
            valid_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(valid_series) > 0:
                # Check for skewness to recommend mean vs median
                skewness = valid_series.skew()
                if abs(skewness) < 0.5:
                    return "Mean Imputation"
                else:
                    return "Median Imputation"
        else:
            # For qualitative data, check uniqueness
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:  # Low uniqueness
                return "Mode Imputation"
            else:
                return "LLM Context Prediction"
        
        return "Mode Imputation"