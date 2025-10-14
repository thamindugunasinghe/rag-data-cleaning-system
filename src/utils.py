import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import io

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate if dataframe is suitable for processing"""
    if df is None or df.empty:
        return False
    
    if len(df.columns) == 0:
        return False
    
    return True

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect column types automatically"""
    column_types = {}
    
    for column in df.columns:
        # Try to infer if numeric
        numeric_count = pd.to_numeric(df[column], errors='coerce').notna().sum()
        total_non_null = df[column].notna().sum()
        
        if total_non_null == 0:
            column_types[column] = 'unknown'
        elif numeric_count / total_non_null > 0.8:
            column_types[column] = 'quantitative'
        else:
            column_types[column] = 'qualitative'
    
    return column_types

def calculate_data_quality_score(df: pd.DataFrame) -> float:
    """Calculate overall data quality score (0-100)"""
    if df.empty:
        return 0.0
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    
    # Count error markers
    error_markers = ["ERROR", "UNKNOWN", "N/A", "NULL", "", "error", "unknown"]
    error_cells = 0
    
    for column in df.columns:
        error_cells += df[column].astype(str).isin(error_markers).sum()
    
    quality_score = ((total_cells - missing_cells - error_cells) / total_cells) * 100
    return max(0.0, quality_score)

def export_to_csv(df: pd.DataFrame) -> str:
    """Export dataframe to CSV string"""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def export_to_excel(df: pd.DataFrame) -> bytes:
    """Export dataframe to Excel bytes"""
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
    return excel_buffer.getvalue()

def generate_column_summary(series: pd.Series) -> Dict[str, Any]:
    """Generate summary statistics for a column"""
    summary = {
        'name': series.name,
        'count': len(series),
        'non_null_count': series.notna().sum(),
        'null_count': series.isnull().sum(),
        'unique_count': series.nunique(),
        'dtype': str(series.dtype)
    }
    
    # Add type-specific statistics
    if pd.api.types.is_numeric_dtype(series):
        summary.update({
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max()
        })
    else:
        summary.update({
            'top_value': series.mode().iloc[0] if len(series.mode()) > 0 else None,
            'top_frequency': series.value_counts().iloc[0] if len(series.value_counts()) > 0 else 0
        })
    
    return summary

def detect_outliers(series: pd.Series) -> List[int]:
    """Detect outliers using IQR method"""
    if not pd.api.types.is_numeric_dtype(series):
        return []
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
    return outliers

def suggest_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Suggest appropriate data types for columns"""
    suggestions = {}
    
    for column in df.columns:
        series = df[column].dropna()
        
        if len(series) == 0:
            suggestions[column] = 'object'
            continue
        
        # Check if all values can be converted to datetime
        try:
            pd.to_datetime(series, errors='raise')
            suggestions[column] = 'datetime64[ns]'
            continue
        except:
            pass
        
        # Check if all values can be converted to numeric
        try:
            pd.to_numeric(series, errors='raise')
            # Check if integers
            if all(float(x).is_integer() for x in series if pd.notna(x)):
                suggestions[column] = 'int64'
            else:
                suggestions[column] = 'float64'
            continue
        except:
            pass
        
        # Check if boolean
        if set(series.unique()).issubset({True, False, 'True', 'False', '1', '0', 1, 0}):
            suggestions[column] = 'bool'
            continue
        
        # Default to object/string
        suggestions[column] = 'object'
    
    return suggestions

def create_sample_dataset(rows: int = 100, columns: int = 5, missing_rate: float = 0.1) -> pd.DataFrame:
    """Create a sample dataset for testing"""
    np.random.seed(42)
    
    data = {}
    
    # Create different types of columns
    for i in range(columns):
        if i % 3 == 0:  # Numeric columns
            col_data = np.random.randn(rows) * 100
            col_name = f'numeric_col_{i}'
        elif i % 3 == 1:  # Categorical columns
            categories = ['A', 'B', 'C', 'D']
            col_data = np.random.choice(categories, size=rows)
            col_name = f'category_col_{i}'
        else:  # Mixed columns
            col_data = [f'item_{j}' for j in range(rows)]
            col_name = f'text_col_{i}'
        
        # Introduce missing values
        missing_indices = np.random.choice(rows, size=int(rows * missing_rate), replace=False)
        col_data = pd.Series(col_data)
        col_data.iloc[missing_indices] = np.nan
        
        data[col_name] = col_data
    
    return pd.DataFrame(data)