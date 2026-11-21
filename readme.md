# 🧠 Personality Predictor

A Streamlit web app that predicts whether a profile looks **Introvert**, **Ambivert** or **Extrovert**, using a Logistic Regression model trained in `logistic_solutions.ipynb`.

## Features

- **Predict:** adjust trait sliders and get a prediction with a confidence score and a probability chart.
- **Model Insights:** coefficient heatmap and overall feature-importance chart.
- **Explore Data (optional):** upload the dataset CSV to see summary stats, class distribution, histograms and a correlation heatmap.

## Project structure

```
.
├── app.py                    # Streamlit app
├── requirements.txt          # Python dependencies
├── logistic_solutions.ipynb  # Notebook used to train the model
├── personality_model.pkl     # Trained model (created by the notebook)
└── scaler.pkl                # Fitted StandardScaler (created by the notebook)
```

## Setup

### 1. Create the model files

Run `logistic_solutions.ipynb` from top to bottom. The last cells save:

- `personality_model.pkl`
- `scaler.pkl`

Put both files in the same folder as `app.py`.

> If they are missing, the app still starts and lets you upload them from the sidebar.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Optional: use a virtual environment first.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` if the browser doesn't open automatically.

## How it works

1. The sliders are built from the feature names and averages stored in `scaler.pkl`.
2. Your inputs are scaled with the same `StandardScaler` used in training.
3. The model returns a probability for each class, and the highest one is shown as the result.

### Class labels

The notebook encodes labels with `LabelEncoder`, which sorts them alphabetically:

| Code | Personality |
|------|-------------|
| 0 | Ambivert |
| 1 | Extrovert |
| 2 | Introvert |

If your dataset uses different labels, edit `CLASS_NAMES` at the top of `app.py`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit is not recognized` | Use `python -m streamlit run app.py` |
| `No module named streamlit` | Run `python -m pip install -r requirements.txt` with the same Python you use to run the app |
| App asks for model files | Place `personality_model.pkl` and `scaler.pkl` next to `app.py`, or upload them in the sidebar |
| Pickle / version warning or error | Install the same scikit-learn version you used in the notebook |
| Slider ranges look wrong | Adjust the `slider_range()` function in `app.py` |

## Tech stack

Python · Streamlit · scikit-learn · pandas · NumPy · Altair

## Note

The dataset is synthetic, and the predictions are for learning and demonstration only, not a psychological assessment.
