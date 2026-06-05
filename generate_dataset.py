import pandas as pd
import numpy as np

def generate_loan_data(n=1000):
    np.random.seed(42)
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Married': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n),
        'Self_Employed': np.random.choice(['Yes', 'No'], n),
        'ApplicantIncome': np.random.randint(2500, 80000, n),
        'CoapplicantIncome': np.random.randint(0, 40000, n),
        'LoanAmount': np.random.randint(50, 600, n),
        'Loan_Amount_Term': np.random.choice([120, 180, 240, 300, 360], n),
        'Credit_History': np.random.choice([0, 1], n, p=[0.2, 0.8]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n)
    }
    
    df = pd.DataFrame(data)
    
    # Logic for Loan_Status (to make it learnable)
    # Simple rule: Approved if Credit_History is 1 and Income is high enough for LoanAmount
    def determine_status(row):
        score = 0
        if row['Credit_History'] == 1: score += 5
        if row['ApplicantIncome'] + row['CoapplicantIncome'] > row['LoanAmount'] * 100: score += 3
        if row['Education'] == 'Graduate': score += 1
        
        return 'Y' if score >= 6 else 'N'

    df['Loan_Status'] = df.apply(determine_status, axis=1)
    
    df.to_csv('ml/loan_data.csv', index=False)
    print("Synthetic dataset created at ml/loan_data.csv")

if __name__ == "__main__":
    generate_loan_data()
