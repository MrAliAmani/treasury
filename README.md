# Treasury Futures Analysis Dashboard

## Project Description

A sophisticated data visualization and analysis tool designed to explore the relationship between economic indicators and Treasury yield curves. Built with Streamlit and Plotly, this dashboard provides interactive visualizations and real-time analysis capabilities.

### Key Features

- **Interactive Dashboard**: Real-time visualization of economic data with customizable filters
- **Data Analysis**: 
  - Economic indicator impact analysis
  - Yield curve visualization
  - Surprise calculation and tracking
  - Statistical correlations and metrics
- **Export Capabilities**: 
  - Excel export for raw data
  - PDF reports with visualizations
- **Error Handling**: Robust error management and user feedback
- **Responsive Design**: Adapts to different screen sizes and devices

## Installation

### Prerequisites

- Python 3.13.1 or higher
- Git (for cloning the repository)
- Virtual environment management tool (venv)

### Setup Instructions

1. Clone the repository:
```sh
git clone [repository-url]
cd treasury_futures
```

2. Create and activate virtual environment:
```sh
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```sh
pip install -r requirements.txt
```

4. Set up environment variables:
```sh
# Copy template file
cp .env.template .env

# Edit .env file and add your FRED API key
FRED_API_KEY=your_api_key_here
```

## Usage

### Running the Application

Start the Streamlit app:
```sh
streamlit run frontend/app.py
```

The dashboard will be available at `http://localhost:8501`

### Features Guide

1. **Data Visualization**:
   - View economic indicator trends
   - Analyze yield curve changes
   - Explore surprise impacts

2. **Data Export**:
   - Export data to Excel for further analysis
   - Generate PDF reports with visualizations

3. **Data Filtering**:
   - Select date ranges
   - Choose specific indicators
   - Filter by impact levels

## Development

### Project Structure
```
treasury_futures/
├── backend/
│   ├── analysis.py
│   ├── data_loader.py
│   ├── data_processor.py
│   └── visualizations.py
├── frontend/
│   ├── app.py
│   └── __init__.py
└── tests/
    ├── integration/
    │   └── test_app_flow.py
    └── unit/
        └── test_data_processing.py
```

### Running Tests

1. Unit Tests:
```sh
pytest tests/unit/test_data_processing.py
```

2. Integration Tests:
```sh
pytest tests/integration/test_app_flow.py
```

3. Run with Coverage:
```sh
pytest --cov=your_package tests/
coverage html  # Generate HTML report
```

### Test Categories

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test component interactions and data flow
- **End-to-End Tests**: Test complete user workflows

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data visualization powered by [Plotly](https://plotly.com/)
- Economic data provided by [FRED API](https://fred.stlouisfed.org/docs/api/fred/)

To run the dockerized application:

Build and start the containers:
docker-compose up --build

Stop the containers:
docker-compose down
