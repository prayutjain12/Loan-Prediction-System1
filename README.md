# FinAI - Advanced Loan Prediction & Financial Analysis System

A comprehensive financial technology platform for loan eligibility prediction, affordability analysis, and investment modeling.

## 🚀 Features
- **Credit ML Radar**: Simulated neural assessment of credit scores with granular recovery roadmaps.
- **Affordability Lab**: Institutional-grade repayment capacity analysis (FOIR modeling).
- **Loan Quantitative Lab**: High-precision amortization modeling with EIR analysis.
- **FD/Investment Compounding**: Multi-stage growth modeling for fixed income.
- **AI Financial Assistant**: Neural chatbot trained on Indian banking and financial regulations.

## 🧠 Machine Learning Component
The core ML logic is located in the `/ml` directory:
- **`generate_dataset.py`**: Synthetic data generator for training.
- **`train_model.py`**: Random Forest training script.
- **`loan_prediction_model.pkl`**: The actual trained model file for loan eligibility.

### Running the ML Training
1. `pip install -r ml/requirements.txt`
2. `python ml/train_model.py`

## 🛠️ Tech Stack
- **Frontend**: React 18, Vite, Tailwind CSS, Framer Motion, Lucide Icons.
- **Backend/ML**: Python 3.x, Scikit-Learn, Pandas, NumPy.

## 📈 Integration
The frontend uses logic derived from the trained Random Forest model. For production, the `.pkl` model can be deployed as an API endpoint.
