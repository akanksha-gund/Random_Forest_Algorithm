import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'random.pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Advanced HTML + CSS Layout with Background Image, Vibrant Colors & Glassmorphism
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced AI Impact & Layoff Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #8b5cf6;
            --primary-hover: #7c3aed;
            --accent: #ec4899;
            --card-bg: rgba(15, 23, 42, 0.75);
            --border-color: rgba(255, 255, 255, 0.15);
            --text-main: #f8fafc;
            --text-muted: #cbd5e1;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            /* Advanced Professional Tech Background Image with Dark Gradient Overlay */
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 27, 75, 0.85)), 
                        url('https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=2000&q=80') no-repeat center center fixed;
            background-size: cover;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 50px 20px;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 50px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 40px rgba(139, 92, 246, 0.2);
        }

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to right, #c084fc, #f43f5e, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
            letter-spacing: -0.5px;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.05rem;
            font-weight: 400;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 24px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        input, select {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px 18px;
            color: var(--text-main);
            font-size: 1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        input:hover {
            border-color: rgba(139, 92, 246, 0.5);
        }

        input:focus {
            outline: none;
            border-color: var(--primary);
            background: rgba(15, 23, 42, 0.9);
            box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.25);
        }

        .btn-submit {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 16px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 15px;
            box-shadow: 0 10px 20px rgba(139, 92, 246, 0.3);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px rgba(236, 72, 153, 0.4);
            filter: brightness(1.1);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-card {
            margin-top: 35px;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(236, 72, 153, 0.15));
            border: 1px solid rgba(139, 92, 246, 0.4);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            animation: fadeIn 0.5s ease-in-out;
        }

        .result-card h3 {
            font-size: 1.2rem;
            color: #c084fc;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .result-card p {
            font-size: 1.6rem;
            font-weight: 800;
            color: #ffffff;
        }

        .error-card {
            margin-top: 25px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #fca5a5;
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            font-weight: 500;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI Impact & Layoff Predictor</h1>
            <p>Advanced Machine Learning Analytics & Organizational Risk Dashboard</p>
        </header>

        <form method="POST">
            <div class="form-grid">
                <div class="input-group">
                    <label>Company Size</label>
                    <input type="number" name="company_size" required placeholder="e.g. 5000">
                </div>
                <div class="input-group">
                    <label>Month</label>
                    <input type="number" name="month" min="1" max="12" required placeholder="e.g. 10">
                </div>
                <div class="input-group">
                    <label>Year</label>
                    <input type="number" name="year" required placeholder="e.g. 2026">
                </div>
                <div class="input-group">
                    <label>Layoffs Count</label>
                    <input type="number" name="layoffs_count" required placeholder="e.g. 150">
                </div>
                <div class="input-group">
                    <label>Layoff Percentage (%)</label>
                    <input type="number" step="any" name="layoff_percentage" required placeholder="e.g. 5.5">
                </div>
                <div class="input-group">
                    <label>Open Roles</label>
                    <input type="number" name="open_roles" required placeholder="e.g. 25">
                </div>
                <div class="input-group">
                    <label>Remote Jobs Percentage (%)</label>
                    <input type="number" step="any" name="remote_jobs_percentage" required placeholder="e.g. 40">
                </div>
                <div class="input-group">
                    <label>Stock Growth Percent (%)</label>
                    <input type="number" step="any" name="stock_growth_percent" required placeholder="e.g. 12.4">
                </div>
                <div class="input-group">
                    <label>Revenue Growth Percent (%)</label>
                    <input type="number" step="any" name="revenue_growth_percent" required placeholder="e.g. 8.1">
                </div>
                <div class="input-group">
                    <label>Salary Budget Change (%)</label>
                    <input type="number" step="any" name="salary_budget_change" required placeholder="e.g. -2.0">
                </div>
                <div class="input-group">
                    <label>AI Replacement Risk Score (0-1)</label>
                    <input type="number" step="any" name="ai_replacement_risk" required placeholder="e.g. 0.45">
                </div>
                <div class="input-group">
                    <label>Employee Sentiment Score (0-1)</label>
                    <input type="number" step="any" name="employee_sentiment" required placeholder="e.g. 0.65">
                </div>
                <div class="input-group">
                    <label>Job Security Score (0-1)</label>
                    <input type="number" step="any" name="job_security_score" required placeholder="e.g. 0.70">
                </div>
                <div class="input-group">
                    <label>AI Adoption Level (Encoded/Numeric)</label>
                    <input type="number" step="any" name="ai_adoption_level" required placeholder="e.g. 2">
                </div>
                <div class="input-group">
                    <label>AI Automation Impact (Encoded/Numeric)</label>
                    <input type="number" step="any" name="ai_automation_impact" required placeholder="e.g. 1">
                </div>
            </div>

            <button type="submit" class="btn-submit">Run Advanced Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Prediction Result</h3>
            <p>{{ prediction }}</p>
        </div>
        {% endif %}

        {% if error %}
        <div class="error-card">
            <p>{{ error }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None

    if request.method == 'POST':
        if model is None:
            error = "Model file (random.pkl) could not be loaded properly."
        else:
            try:
                form_data = {
                    'company_size': float(request.form.get('company_size', 0)),
                    'month': float(request.form.get('month', 0)),
                    'year': float(request.form.get('year', 0)),
                    'layoffs_count': float(request.form.get('layoffs_count', 0)),
                    'layoff_percentage': float(request.form.get('layoff_percentage', 0)),
                    'open_roles': float(request.form.get('open_roles', 0)),
                    'remote_jobs_percentage': float(request.form.get('remote_jobs_percentage', 0)),
                    'stock_growth_percent': float(request.form.get('stock_growth_percent', 0)),
                    'revenue_growth_percent': float(request.form.get('revenue_growth_percent', 0)),
                    'salary_budget_change': float(request.form.get('salary_budget_change', 0)),
                    'ai_replacement_risk': float(request.form.get('ai_replacement_risk', 0)),
                    'employee_sentiment': float(request.form.get('employee_sentiment', 0)),
                    'job_security_score': float(request.form.get('job_security_score', 0)),
                    'ai_adoption_level': float(request.form.get('ai_adoption_level', 0)),
                    'ai_automation_impact': float(request.form.get('ai_automation_impact', 0))
                }

                feature_names = model.feature_names_in_
                input_df = pd.DataFrame(columns=feature_names)
                input_df.loc[0] = 0

                for key, val in form_data.items():
                    if key in input_df.columns:
                        input_df[key] = val

                pred_val = model.predict(input_df)[0]
                prediction = f"Predicted Outcome Category: {pred_val}"

            except Exception as e:
                error = f"An error occurred during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)