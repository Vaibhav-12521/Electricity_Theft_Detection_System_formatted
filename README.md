# Electricity Theft Detection System

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Generate sample data:
```bash
python data_generation.py
```
3. Train the model:
```bash
python src/model_training.py
```
4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

- `data/` - generated sample datasets
- `models/` - serialized trained model
- `src/` - preprocessing, model training, and detection logic
- `app.py` - Streamlit dashboard
- `requirements.txt` - Python dependencies
