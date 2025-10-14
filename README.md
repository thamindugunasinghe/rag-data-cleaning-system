# 🧹 RAG Data Cleaning System

An intelligent data cleaning application powered by Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques.

## Features

- **Smart Data Profiling**: Automatic detection of data quality issues
- **Multiple Cleaning Strategies**: Traditional statistical methods + AI-powered predictions
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Generalized Solution**: Works with any CSV/Excel dataset
- **Comprehensive Reporting**: Before/after analysis with quality metrics
- **LLM Integration**: Context-aware missing value prediction using OpenAI GPT

## Screenshots

### Data Upload & Profiling
Upload your dataset and get instant quality analysis with recommendations.

### Cleaning Configuration
Choose from multiple strategies for each column with intelligent recommendations.

### Results & Download
View improvements and download cleaned datasets with detailed reports.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/thamindugunasinghe/rag-data-cleaning-system.git
cd rag-data-cleaning-system
```

2. **Create virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_actual_openai_api_key" > .env
```

5. **Run the application:**
```bash
streamlit run streamlit_app.py
```

## Usage

1. **Upload Dataset**: Upload CSV or Excel file through the web interface
2. **Review Profile**: Analyze data quality issues and get automatic recommendations
3. **Configure Cleaning**: Choose strategies for each column from dropdown menus
4. **Clean Data**: Execute cleaning process with real-time progress
5. **Download Results**: Get cleaned dataset and comprehensive reports

## Supported Cleaning Strategies

### Quantitative Data
- **Mean Imputation**: Replace missing values with column mean
- **Median Imputation**: Replace with median (robust to outliers)
- **Mode Imputation**: Replace with most frequent value
- **Forward Fill**: Use previous valid value
- **Backward Fill**: Use next valid value
- **LLM Prediction**: AI-powered context-aware prediction

### Qualitative Data
- **Mode Imputation**: Replace with most frequent category
- **LLM Context Prediction**: AI prediction based on other column values
- **Forward/Backward Fill**: Sequential filling
- **Remove Rows**: Delete rows with missing values

## Project Structure

```
rag-data-cleaning-system/
├── src/
│   ├── __init__.py
│   ├── data_profiler.py      # Data quality analysis
│   ├── cleaner.py            # Cleaning strategies implementation
│   ├── llm_handler.py        # OpenAI LLM integration
│   └── utils.py              # Helper functions
├── streamlit_app.py          # Main web application
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration settings
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## Configuration

Edit `config.yaml` to customize:

```yaml
llm:
  model: "gpt-3.5-turbo"     # OpenAI model to use
  temperature: 0.1            # LLM creativity (0-1)
  max_tokens: 500            # Response length limit

cleaning_strategies:
  quantitative: [...]
  qualitative: [...]

error_markers:
  - "ERROR"
  - "UNKNOWN"
  - "N/A"
```

## API Integration

The system uses OpenAI's GPT models for intelligent data imputation:

- **Context-aware predictions**: Uses surrounding column data to predict missing values
- **Business rule validation**: Ensures predictions make logical sense
- **Confidence scoring**: Provides uncertainty estimates for predictions

## Data Quality Metrics

- **Overall Quality Score**: Percentage of clean data points
- **Column-wise Analysis**: Individual column quality assessment
- **Error Detection**: Automatic identification of error markers
- **Improvement Tracking**: Before/after comparison metrics

## Requirements

- Python 3.8+
- OpenAI API key
- 2GB+ RAM (for larger datasets)
- Internet connection (for LLM API calls)

See `requirements.txt` for full dependency list.

## Examples

### Basic Usage
```python
from src.data_profiler import DataProfiler
from src.cleaner import DataCleaner

# Initialize components
profiler = DataProfiler()
cleaner = DataCleaner()

# Profile your data
profile = profiler.profile_dataset(df)

# Configure cleaning strategies
strategies = {
    'numeric_column': 'Median Imputation',
    'category_column': 'Mode Imputation',
    'text_column': 'LLM Context Prediction'
}

# Clean the data
cleaned_df = cleaner.clean_dataset(df, strategies)
```

### Custom Error Markers
```python
# Add custom error markers
profiler.error_markers.extend(['MISSING', 'INVALID', 'TBD'])
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please:

1. Check the [Issues](https://github.com/thamindugunasinghe/rag-data-cleaning-system/issues) page
2. Create a new issue with detailed description
3. Contact: thamindu.gunasinghe@gmail.com

## Roadmap

- [ ] Support for additional file formats (JSON, Parquet)
- [ ] Advanced anomaly detection
- [ ] Custom cleaning rule definition
- [ ] Batch processing for large datasets
- [ ] Database integration
- [ ] REST API endpoints
- [ ] Docker containerization

## Acknowledgments

- OpenAI for GPT API
- Streamlit for the web framework
- Pandas/NumPy for data processing
- Plotly for visualizations

---

**Built with ❤️ by [Thamindu Gunasinghe](https://github.com/thamindugunasinghe)**