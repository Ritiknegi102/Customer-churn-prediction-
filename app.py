
from flask import Flask, request, render_template
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask("__name__")
from flask import Flask, request, render_template
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask("name")
# Load the initial dataset and model
df_1 = pd.read_csv("first_telc.csv")
model = pickle.load(open("model.sav", "rb"))

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Extract input queries from the form
    input_queries = {f'query{i}': request.form[f'query{i}'] for i in range(1, 20)}
    
    data = [[input_queries[f'query{i}'] for i in range(1, 20)]]
    # Create a DataFrame for the new input data
    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])
    
    

    # Clean and convert the tenure column
    new_df['tenure'] = new_df['tenure'].apply(lambda x: int(x) if x.isdigit() else 0)
    # Combine with the original dataset to ensure consistency in dummy variable creation
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2['tenure'], range(1, 80, 12), right=False, labels=labels)
    
    # Drop the 'tenure' column
    df_2.drop(columns=['tenure'], axis=1, inplace=True)
    # One-hot encode categorical variables
    df_2 = pd.get_dummies(df_2, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                         'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'])
    # Ensure the new input data has all the columns that were used during training
    missing_cols = set(model.feature_names_in_) - set(df_2.columns)
    for c in missing_cols:
        df_2[c] = 0
    df_2 = df_2[model.feature_names_in_]
    # Predict and calculate the probability
    single = model.predict(df_2.tail(1))
    probability = model.predict_proba(df_2.tail(1))[:, 1]
    
    # calculating tenure in years
    tenure_years = int(new_df['tenure'].values[0]) // 12

    
    # Generate a plot for the customer's churn risk over time
    plt.figure(figsize=(6, 4))
    x = range(tenure_years + 1)
    y = [0.5] * (tenure_years + 1)
    y[-1] = probability[0]
    plt.plot(x, y)
    plt.title("Churn Risk Over Time")
    plt.xlabel("Years")
    plt.ylabel("Churn Risk")
    plt.ylim(0, 1)
    plot_data = io.BytesIO()
    plt.savefig(plot_data, format='png')
    plot_data.seek(0)
    plot_url = base64.b64encode(plot_data.getvalue()).decode('utf-8')
    
    # Generate a bar chart for service usage
    service_usage = {
        'PhoneService': new_df['PhoneService'].values[0],
        'InternetService': new_df['InternetService'].values[0],
        'OnlineSecurity': new_df['OnlineSecurity'].values[0],
        'OnlineBackup': new_df['OnlineBackup'].values[0],
        'DeviceProtection': new_df['DeviceProtection'].values[0],
        'TechSupport': new_df['TechSupport'].values[0],
        'StreamingTV': new_df['StreamingTV'].values[0],
        'StreamingMovies': new_df['StreamingMovies'].values[0]
    }

    services = list(service_usage.keys())
    usage = list(service_usage.values())

    plt.figure(figsize=(8, 6))
    plt.bar(services, usage)
    plt.xlabel('Services')
    plt.ylabel('Usage')
    plt.title('Customer Service Usage')
    plt.xticks(rotation=45)

    service_usage_plot = io.BytesIO()
    plt.savefig(service_usage_plot, format='png', bbox_inches='tight')
    service_usage_plot.seek(0)
    service_usage_plot_url = base64.b64encode(service_usage_plot.getvalue()).decode('utf-8')

    revenue_contribution = {
        'PhoneService': 20,
        'InternetService': 30,
        'OnlineSecurity': 10,
        'OnlineBackup': 15,
        'DeviceProtection': 5,
        'TechSupport': 10,
        'StreamingTV': 5,
        'StreamingMovies': 5
    }

    labels = list(revenue_contribution.keys())
    values = list(revenue_contribution.values())

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Revenue Contribution by Service')

    revenue_contribution_plot = io.BytesIO()
    plt.savefig(revenue_contribution_plot, format='png', bbox_inches='tight')
    revenue_contribution_plot.seek(0)
    revenue_contribution_plot_url = base64.b64encode(revenue_contribution_plot.getvalue()).decode('utf-8')
    
    if single == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    return render_template('home.html', output1=o1, output2=o2, plot_url=plot_url, service_usage_plot_url=service_usage_plot_url,
                            revenue_contribution_plot_url=revenue_contribution_plot_url, **input_queries)



if __name__ == '__main__':
    app.run(debug=True)
