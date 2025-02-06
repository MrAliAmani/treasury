# Treasury Futures Analysis Dashboard

## Overview
A data visualization and analysis platform that explores relationships between economic indicators and Treasury yield curves. Built with Streamlit and Plotly, this dashboard offers interactive visualizations and real-time analysis tools.

## Features
- Interactive visualizations of economic data and yield curves
- Real-time analysis of economic indicator impacts
- Statistical correlation tracking and metrics
- Data export to Excel and PDF formats
- Responsive design for all devices

## Prerequisites
- Python 3.9 or higher
- Git
- Docker (optional)

## Quick Start

### Local Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd treasury_futures
```

2. Set up Python environment:
```bash
# Create environment
conda create -n treasury python=3.9
conda activate treasury

# Install dependencies
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.template .env
# Edit .env and add your FRED API key
```

4. Run the application:
```bash
streamlit run frontend/app.py
```

### Docker Installation

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down
```

## Project Structure
```
treasury_futures/
├── backend/           # Data processing and analysis
├── frontend/         # Streamlit interface
└── tests/           # Test suites
```

## Testing

Run specific test categories:
```bash
# Unit tests
pytest -m "unit"

# Integration tests
pytest -m "integration"

# All tests with coverage
pytest --cov=backend --cov-report=term-missing
```

## Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Include docstrings for all functions
- Add tests for new features

## API Documentation

### FRED API
- Requires API key from [FRED API](https://fred.stlouisfed.org/docs/api/fred/)
- Set key in `.env` file

## License
MIT License - See LICENSE file for details

## Tech Stack
- [Streamlit](https://streamlit.io/) - Frontend framework
- [Plotly](https://plotly.com/) - Data visualization
- [FRED API](https://fred.stlouisfed.org/docs/api/fred/) - Economic data source

## Support
For issues and feature requests, please use the GitHub issue tracker.
