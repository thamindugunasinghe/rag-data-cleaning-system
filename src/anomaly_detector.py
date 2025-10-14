import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self):
        self.error_markers = ["ERROR", "UNKNOWN", "N/A", "NULL", "", "error", "unknown"]
        self.anomalies_found = {}
        
    def detect_all_anomalies(self, df: pd.DataFrame) -> Dict:
        """Comprehensive anomaly detection across all data types"""
        
        print(f"🔍 Starting comprehensive anomaly detection...")
        
        anomalies = {
            'statistical_outliers': self._detect_statistical_outliers(df),
            'categorical_anomalies': self._detect_categorical_anomalies(df),
            'business_logic_violations': self._detect_business_violations(df),
            'data_quality_issues': self._detect_data_quality_issues(df),
            'temporal_anomalies': self._detect_temporal_anomalies(df),
            'correlation_anomalies': self._detect_correlation_anomalies(df)
        }
        
        # Generate summary
        total_anomalies = sum(len(v) if isinstance(v, list) else 1 if v else 0 for v in anomalies.values())
        
        anomalies['summary'] = {
            'total_anomalies': total_anomalies,
            'dataset_health_score': max(0, 100 - (total_anomalies / len(df)) * 10),
            'recommendations': self._generate_recommendations(anomalies)
        }
        
        print(f"✅ Anomaly detection completed. Found {total_anomalies} total anomaly types.")
        return anomalies
    
    def _detect_statistical_outliers(self, df: pd.DataFrame) -> List[Dict]:
        """Detect statistical outliers in numeric columns"""
        
        print(f"  📊 Detecting statistical outliers...")
        outliers = []
        
        for column in df.columns:
            clean_col = self._clean_numeric_column(df[column])
            
            if len(clean_col) < 10:  # Need sufficient data
                continue
            
            # IQR Method
            Q1 = clean_col.quantile(0.25)
            Q3 = clean_col.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # Handle case where all values are the same
                continue
                
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = clean_col[(clean_col < lower_bound) | (clean_col > upper_bound)]
            
            # Z-Score Method
            try:
                z_scores = np.abs(stats.zscore(clean_col))
                z_outliers = clean_col[z_scores > 3]
            except:
                z_outliers = pd.Series([], dtype=float)
            
            # Business Range Method
            business_outliers = self._detect_business_range_outliers(column, clean_col)
            
            # Machine Learning Methods
            ml_outliers = self._detect_ml_outliers(clean_col)
            
            if len(iqr_outliers) > 0 or len(z_outliers) > 0 or len(business_outliers) > 0 or len(ml_outliers) > 0:
                severity = 'high' if len(iqr_outliers) > len(clean_col) * 0.1 else 'medium'
                
                outliers.append({
                    'column': column,
                    'type': 'statistical_outlier',
                    'total_values': len(clean_col),
                    'iqr_outliers': len(iqr_outliers),
                    'iqr_values': sorted(iqr_outliers.unique().tolist()[:10]),
                    'z_outliers': len(z_outliers),
                    'business_outliers': len(business_outliers),
                    'ml_outliers': len(ml_outliers),
                    'outlier_percentage': round((len(iqr_outliers) / len(clean_col)) * 100, 2),
                    'bounds': {'lower': round(lower_bound, 2), 'upper': round(upper_bound, 2)},
                    'severity': severity,
                    'recommendation': f"Review {len(iqr_outliers)} outlier values in {column}"
                })
        
        print(f"    Found {len(outliers)} columns with statistical outliers")
        return outliers
    
    def _detect_categorical_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in categorical columns"""
        
        print(f"  📝 Detecting categorical anomalies...")
        anomalies = []
        
        for column in df.columns:
            clean_col = self._clean_categorical_column(df[column])
            
            if len(clean_col) < 10 or clean_col.nunique() < 2:
                continue
            
            value_counts = clean_col.value_counts()
            total_valid = len(clean_col)
            
            # Detect rare categories (< 5% of data)
            rare_threshold = max(1, total_valid * 0.05)
            rare_categories = value_counts[value_counts < rare_threshold]
            
            # Check for unexpected categories
            expected_categories = self._get_expected_categories(column)
            unexpected = []
            if expected_categories:
                unexpected = [cat for cat in value_counts.index if cat not in expected_categories]
            
            # Check for suspicious patterns
            suspicious_patterns = self._detect_suspicious_patterns(clean_col)
            
            if len(rare_categories) > 0 or len(unexpected) > 0 or len(suspicious_patterns) > 0:
                anomalies.append({
                    'column': column,
                    'type': 'categorical_anomaly',
                    'total_categories': len(value_counts),
                    'rare_categories': len(rare_categories),
                    'rare_details': dict(rare_categories.head(5)),
                    'unexpected_categories': unexpected[:5],
                    'suspicious_patterns': suspicious_patterns,
                    'distribution': dict(value_counts.head(5)),
                    'severity': 'high' if len(unexpected) > 0 else 'medium',
                    'recommendation': f"Review {len(rare_categories)} rare categories and {len(unexpected)} unexpected values in {column}"
                })
        
        print(f"    Found {len(anomalies)} columns with categorical anomalies")
        return anomalies
    
    def _detect_business_violations(self, df: pd.DataFrame) -> List[Dict]:
        """Detect business logic violations"""
        
        print(f"  💼 Detecting business logic violations...")
        violations = []
        
        # Check Price × Quantity = Total calculation
        if all(col in df.columns for col in ['Quantity', 'Price Per Unit', 'Total Spent']):
            calculation_errors = self._check_calculation_errors(df)
            if calculation_errors:
                violations.append(calculation_errors)
        
        # Check for impossible combinations
        impossible_combos = self._detect_impossible_combinations(df)
        violations.extend(impossible_combos)
        
        # Check for negative values where they shouldn't exist
        negative_violations = self._check_negative_values(df)
        violations.extend(negative_violations)
        
        print(f"    Found {len(violations)} business logic violations")
        return violations
    
    def _detect_data_quality_issues(self, df: pd.DataFrame) -> List[Dict]:
        """Detect data quality issues"""
        
        print(f"  🗃️ Detecting data quality issues...")
        issues = []
        
        for column in df.columns:
            # Count different types of issues
            null_count = df[column].isnull().sum()
            error_count = df[column].astype(str).isin(self.error_markers).sum()
            
            # Check for formatting issues
            formatting_issues = self._detect_formatting_issues(df[column])
            
            # Check for duplicates in ID columns
            duplicates = 0
            if any(id_term in column.lower() for id_term in ['id', 'transaction', 'order']):
                duplicates = df[column].duplicated().sum()
            
            total_issues = null_count + error_count + len(formatting_issues) + duplicates
            
            if total_issues > 0:
                quality_score = max(0, 100 - (total_issues / len(df)) * 100)
                
                issues.append({
                    'column': column,
                    'type': 'data_quality_issue',
                    'null_count': int(null_count),
                    'error_markers': int(error_count),
                    'formatting_issues': len(formatting_issues),
                    'duplicates': int(duplicates),
                    'total_issues': int(total_issues),
                    'quality_score': round(quality_score, 1),
                    'severity': 'high' if quality_score < 70 else 'medium' if quality_score < 90 else 'low',
                    'recommendation': f"Clean {total_issues} data quality issues in {column}"
                })
        
        print(f"    Found {len(issues)} columns with data quality issues")
        return issues
    
    def _detect_temporal_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect temporal anomalies in date columns"""
        
        print(f"  📅 Detecting temporal anomalies...")
        anomalies = []
        
        # Identify potential date columns
        date_columns = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'created', 'updated'])]
        
        for column in date_columns:
            clean_dates = self._clean_date_column(df[column])
            
            if len(clean_dates) < 10:
                continue
            
            try:
                date_series = pd.to_datetime(clean_dates, errors='coerce').dropna()
                
                if len(date_series) == 0:
                    continue
                
                temporal_issues = []
                
                # Check for future dates
                future_dates = date_series[date_series > pd.Timestamp.now()]
                if len(future_dates) > 0:
                    temporal_issues.append(f"{len(future_dates)} future dates")
                
                # Check for very old dates (before 2000)
                old_dates = date_series[date_series < pd.Timestamp('2000-01-01')]
                if len(old_dates) > 0:
                    temporal_issues.append(f"{len(old_dates)} dates before 2000")
                
                # Check for weekend patterns (might be unusual for some businesses)
                weekends = date_series[date_series.dt.dayofweek >= 5]
                weekend_percentage = (len(weekends) / len(date_series)) * 100
                
                if temporal_issues or weekend_percentage > 60:  # If > 60% weekend activity
                    if weekend_percentage > 60:
                        temporal_issues.append(f"High weekend activity ({weekend_percentage:.1f}%)")
                    
                    anomalies.append({
                        'column': column,
                        'type': 'temporal_anomaly',
                        'issues': temporal_issues,
                        'date_range': f"{date_series.min().strftime('%Y-%m-%d')} to {date_series.max().strftime('%Y-%m-%d')}",
                        'future_dates': len(future_dates),
                        'old_dates': len(old_dates),
                        'weekend_percentage': round(weekend_percentage, 1),
                        'severity': 'high' if len(future_dates) > 0 else 'medium',
                        'recommendation': f"Review temporal patterns in {column}"
                    })
                    
            except Exception as e:
                print(f"    Error analyzing dates in {column}: {e}")
        
        print(f"    Found {len(anomalies)} columns with temporal anomalies")
        return anomalies
    
    def _detect_correlation_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect correlation anomalies between related columns"""
        
        print(f"  🔗 Detecting correlation anomalies...")
        anomalies = []
        
        # Get numeric columns for correlation analysis
        numeric_data = {}
        for col in df.columns:
            clean_col = self._clean_numeric_column(df[col])
            if len(clean_col) > 10:
                numeric_data[col] = clean_col
        
        if len(numeric_data) < 2:
            return anomalies
        
        try:
            # Align all series to common index
            common_indices = None
            for col, series in numeric_data.items():
                if common_indices is None:
                    common_indices = set(series.index)
                else:
                    common_indices = common_indices.intersection(set(series.index))
            
            if len(common_indices) < 10:
                return anomalies
            
            # Create aligned dataframe
            aligned_df = pd.DataFrame()
            for col, series in numeric_data.items():
                aligned_df[col] = series.loc[list(common_indices)]
            
            corr_matrix = aligned_df.corr()
            
            # Check for expected strong correlations that are weak
            expected_correlations = [
                ('Quantity', 'Total Spent', 0.3),
                ('Price Per Unit', 'Total Spent', 0.3),
            ]
            
            for col1, col2, expected_min in expected_correlations:
                if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                    actual_corr = corr_matrix.loc[col1, col2]
                    
                    if abs(actual_corr) < expected_min:
                        anomalies.append({
                            'type': 'weak_expected_correlation',
                            'columns': [col1, col2],
                            'expected_correlation': f'>{expected_min}',
                            'actual_correlation': round(float(actual_corr), 3),
                            'severity': 'medium',
                            'recommendation': f"Investigate weak correlation between {col1} and {col2}"
                        })
            
            # Check for unexpected strong correlations
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates
                        corr_value = corr_matrix.loc[col1, col2]
                        
                        # Flag unexpected strong correlations (>0.95)
                        if abs(corr_value) > 0.95 and col1 != col2:
                            anomalies.append({
                                'type': 'unexpected_strong_correlation',
                                'columns': [col1, col2],
                                'correlation': round(float(corr_value), 3),
                                'severity': 'medium',
                                'recommendation': f"Investigate strong correlation between {col1} and {col2}"
                            })
                            
        except Exception as e:
            print(f"    Error in correlation analysis: {e}")
        
        print(f"    Found {len(anomalies)} correlation anomalies")
        return anomalies
    
    # Helper methods
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean and convert column to numeric"""
        clean_series = series[~series.astype(str).isin(self.error_markers)].dropna()
        return pd.to_numeric(clean_series, errors='coerce').dropna()
    
    def _clean_categorical_column(self, series: pd.Series) -> pd.Series:
        """Clean categorical column"""
        return series[~series.astype(str).isin(self.error_markers)].dropna()
    
    def _clean_date_column(self, series: pd.Series) -> pd.Series:
        """Clean date column"""
        return series[~series.astype(str).isin(self.error_markers)].dropna()
    
    def _detect_business_range_outliers(self, column: str, series: pd.Series) -> pd.Series:
        """Detect business logic range violations"""
        column_lower = column.lower()
        
        if 'quantity' in column_lower:
            return series[(series <= 0) | (series > 50)]  # Reasonable quantity range
        elif 'price' in column_lower:
            return series[(series <= 0) | (series > 200)]  # Reasonable price range
        elif any(term in column_lower for term in ['total', 'spent', 'amount']):
            return series[(series <= 0) | (series > 1000)]  # Reasonable total range
        else:
            return pd.Series([], dtype=float)
    
    def _detect_ml_outliers(self, series: pd.Series) -> pd.Series:
        """Use ML methods to detect outliers"""
        if len(series) < 10:
            return pd.Series([], dtype=float)
        
        try:
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            # Return outliers (labeled as -1)
            outlier_indices = series.index[outlier_labels == -1]
            return series.loc[outlier_indices]
            
        except Exception as e:
            return pd.Series([], dtype=float)
    
    def _detect_suspicious_patterns(self, series: pd.Series) -> List[str]:
        """Detect suspicious patterns in categorical data"""
        patterns = []
        
        # Check for potential injection attacks
        malicious_patterns = ['select', 'insert', 'update', 'delete', 'drop', 'union', '<script>', 'javascript:', 'eval(']
        for pattern in malicious_patterns:
            if series.astype(str).str.contains(pattern, case=False, na=False).any():
                patterns.append(f"Potential injection pattern: '{pattern}'")
        
        # Check for excessive length
        max_length = series.astype(str).str.len().max()
        if max_length > 100:
            patterns.append(f"Unusually long text values (max: {max_length} chars)")
        
        return patterns
    
    def _get_expected_categories(self, column: str) -> Optional[List[str]]:
        """Get expected categories for known column types"""
        expected_categories = {
            'payment': ['Cash', 'Credit Card', 'Digital Wallet', 'Debit Card', 'Check'],
            'location': ['In-store', 'Takeaway', 'Delivery', 'Online', 'Drive-through'],
            'item': ['Coffee', 'Tea', 'Juice', 'Cake', 'Cookie', 'Sandwich', 'Salad', 'Smoothie', 'Muffin', 'Bagel'],
            'status': ['Active', 'Inactive', 'Pending', 'Completed', 'Cancelled']
        }
        
        column_lower = column.lower().replace(' ', '').replace('_', '')
        for key, categories in expected_categories.items():
            if key in column_lower:
                return categories
        
        return None
    
    def _check_calculation_errors(self, df: pd.DataFrame) -> Optional[Dict]:
        """Check for calculation errors in business logic"""
        qty_clean = self._clean_numeric_column(df['Quantity'])
        price_clean = self._clean_numeric_column(df['Price Per Unit'])
        total_clean = self._clean_numeric_column(df['Total Spent'])
        
        # Find common valid indices
        common_indices = set(qty_clean.index) & set(price_clean.index) & set(total_clean.index)
        
        if len(common_indices) < 10:
            return None
        
        calculation_errors = []
        sample_size = min(1000, len(common_indices))
        sample_indices = list(common_indices)[:sample_size]
        
        for idx in sample_indices:
            qty = qty_clean.loc[idx]
            price = price_clean.loc[idx]
            total = total_clean.loc[idx]
            
            expected = qty * price
            difference = abs(total - expected)
            
            if difference > 0.01:  # Allow small rounding errors
                calculation_errors.append({
                    'row': int(idx),
                    'quantity': float(qty),
                    'price': float(price),
                    'expected_total': round(float(expected), 2),
                    'actual_total': float(total),
                    'difference': round(float(difference), 2)
                })
        
        if calculation_errors:
            return {
                'type': 'calculation_mismatch',
                'description': 'Price × Quantity ≠ Total Spent',
                'violations_found': len(calculation_errors),
                'sample_violations': calculation_errors[:5],
                'severity': 'high',
                'recommendation': 'Recalculate Total Spent or verify Price/Quantity data'
            }
        
        return None
    
    def _detect_impossible_combinations(self, df: pd.DataFrame) -> List[Dict]:
        """Detect impossible business combinations"""
        violations = []
        
        # Check for zero quantity with non-zero total
        if all(col in df.columns for col in ['Quantity', 'Total Spent']):
            qty_clean = self._clean_numeric_column(df['Quantity'])
            total_clean = self._clean_numeric_column(df['Total Spent'])
            
            zero_qty_violations = []
            for idx in df.index:
                if idx in qty_clean.index and idx in total_clean.index:
                    if qty_clean.loc[idx] == 0 and total_clean.loc[idx] > 0:
                        zero_qty_violations.append(int(idx))
            
            if zero_qty_violations:
                violations.append({
                    'type': 'impossible_combination',
                    'description': 'Zero quantity with non-zero total',
                    'violations_found': len(zero_qty_violations),
                    'sample_rows': zero_qty_violations[:5],
                    'severity': 'high',
                    'recommendation': 'Review transactions with zero quantity but positive total'
                })
        
        return violations
    
    def _check_negative_values(self, df: pd.DataFrame) -> List[Dict]:
        """Check for negative values where they shouldn't exist"""
        violations = []
        
        # Columns that shouldn't have negative values
        positive_only_columns = ['Quantity', 'Price Per Unit', 'Total Spent']
        
        for column in positive_only_columns:
            if column in df.columns:
                clean_col = self._clean_numeric_column(df[column])
                negative_values = clean_col[clean_col < 0]
                
                if len(negative_values) > 0:
                    violations.append({
                        'type': 'negative_values',
                        'description': f'Negative values in {column}',
                        'column': column,
                        'violations_found': len(negative_values),
                        'negative_values': negative_values.tolist()[:10],
                        'severity': 'high',
                        'recommendation': f'Review negative values in {column} - should be positive'
                    })
        
        return violations
    
    def _detect_formatting_issues(self, series: pd.Series) -> List[str]:
        """Detect formatting inconsistencies"""
        issues = []
        
        if series.dtype == 'object':
            # Check for mixed case in categorical data
            unique_vals = series.dropna().unique()
            if len(unique_vals) > 1:
                lower_vals = [str(v).lower() for v in unique_vals]
                if len(set(lower_vals)) < len(unique_vals):
                    issues.append("Mixed case formatting")
            
            # Check for leading/trailing whitespace
            if series.astype(str).str.match(r'^\s|\s$').any():
                issues.append("Leading/trailing whitespace")
        
        return issues
    
    def _generate_recommendations(self, anomalies: Dict) -> List[str]:
        """Generate actionable recommendations based on detected anomalies"""
        recommendations = []
        
        if anomalies['statistical_outliers']:
            recommendations.append("🔍 Review statistical outliers - consider capping, transformation, or removal")
        
        if anomalies['categorical_anomalies']:
            recommendations.append("📝 Standardize categorical values and investigate rare categories")
        
        if anomalies['business_logic_violations']:
            recommendations.append("💼 Fix business logic violations - recalculate derived fields")
        
        if anomalies['data_quality_issues']:
            recommendations.append("🧹 Address data quality issues - clean error markers and missing values")
        
        if anomalies['temporal_anomalies']:
            recommendations.append("📅 Validate date ranges and fix temporal inconsistencies")
        
        if anomalies['correlation_anomalies']:
            recommendations.append("🔗 Investigate unexpected correlations for potential data issues")
        
        if not any(anomalies[key] for key in anomalies.keys() if key != 'summary'):
            recommendations.append("✅ No significant anomalies detected - data quality looks good!")
        
        return recommendations