import pandas as pd  
import numpy as np  
from tabulate import tabulate  
from sklearn.preprocessing import StandardScaler  
from sklearn.pipeline import make_pipeline  
from sklearn.linear_model import LogisticRegression  

# Define stable variables  
data_file = 'model_data.csv'  # Data file name  
specified_dates = [  
    '2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08',  
    '2021-01-11', '2021-01-12', '2021-01-13', '2021-01-14', '2021-01-15',  
    '2021-01-19', '2021-01-20', '2021-01-21', '2021-01-22', '2021-01-25',  
    '2021-01-26', '2021-01-28', '2021-01-29', '2021-02-01', '2021-02-02',  
    '2021-02-03', '2021-02-04', '2021-02-05', '2021-02-08', '2021-02-09',  
    '2021-02-10', '2021-02-11', '2021-02-12', '2021-02-16', '2021-02-17',  
    '2021-02-18', '2021-02-19', '2021-02-22', '2021-02-23', '2021-02-24',  
    '2021-02-25', '2021-02-26', '2021-03-01', '2021-03-02', '2021-03-03',  
    '2021-03-04', '2021-03-05', '2021-03-08', '2021-03-09', '2021-03-10',  
    '2021-03-11', '2021-03-12', '2021-03-15', '2021-03-16', '2021-03-17',  
    '2021-03-18', '2021-03-19', '2021-03-22', '2021-03-23', '2021-03-24',  
    '2021-03-25', '2021-03-26', '2021-03-29', '2021-03-30', '2021-03-31'  
]  

# Define variables  
feature_columns = [  
    'relevant_positive_pct',  
    'relevant_negative_pct',  
    'relevant_positive_comments_pct',  
    'relevant_negative_comments_pct',  
    'relevant_positive_score_pct',  
    'relevant_negative_score_pct',  
]  

# Define start and end dates for predictions  
start_date = '2021-01-04'  
end_date = '2021-02-17'  

def calculate_metrics(actual, predicted):  
    actual = np.array(actual)  
    predicted = np.array(predicted)  
      
    # Accuracy  
    accuracy = np.mean(actual == predicted)  
      
    # Confusion Matrix  
    classes = np.unique(np.concatenate((actual, predicted)))  
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)  
    for a, p in zip(actual, predicted):  
        confusion_matrix[a][p] += 1  
      
    # Precision, Recall, F1-score per class  
    precision = np.zeros(len(classes))  
    recall = np.zeros(len(classes))  
    f1_score = np.zeros(len(classes))  
      
    for i in range(len(classes)):  
        true_positive = confusion_matrix[i][i]  
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive  
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive  
          
        precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0  
        recall[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0  
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0  
      
    return {  
        'accuracy': accuracy,  
        'precision': precision,  
        'recall': recall,  
        'f1_score': f1_score,  
        'classes': classes  
    }  

def run_rolling_predictions(data_file, feature_columns, specified_dates, start_date, end_date):  
    # Load and prepare the data  
    df = pd.read_csv(data_file)  
    df['date'] = pd.to_datetime(df['date'])  
    df = df[df['date'].isin(pd.to_datetime(specified_dates))].sort_values('date')  

    # Create model pipelines  
    models = {  
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))  
    }  

    print("\nFeature Columns:", ", ".join(feature_columns))  

    results = {model_name: [] for model_name in models.keys()}  

    # Get the index range for predictions  
    start_idx = specified_dates.index(start_date)  
    end_idx = specified_dates.index(end_date)  
    window_size = end_idx - start_idx  
    last_possible_start = len(specified_dates) - window_size - 2  # Ensure we can always have a next date  

    # Perform rolling predictions  
    for i in range(start_idx, last_possible_start + 1):  
        current_start_date = specified_dates[i]  
        current_end_date = specified_dates[i + window_size]  
        next_date = specified_dates[i + window_size + 1]  

        train = df[(df['date'] >= pd.to_datetime(current_start_date)) &    
                   (df['date'] <= pd.to_datetime(current_end_date))]  
        test = df[df['date'] == pd.to_datetime(next_date)]  

        if test.empty:  
            break  

        if train['movement'].nunique() < 2:  
            continue  

        for model_name, model in models.items():  
            model.fit(train[feature_columns], train['movement'])  
              
            X_test, y_test = test[feature_columns], test['movement']  
            predictions = model.predict(X_test)  
              
            results[model_name].append({  
                'date': next_date,  
                'actual': y_test.iloc[0],  
                'predicted': predictions[0]  
            })  

    # Print results and calculate overall metrics  
    for model_name, model_results in results.items():  
        print(f"\nResults for {model_name}:")  
          
        # Prepare data for the table  
        table_data = [[result['date'], result['actual'], result['predicted']] for result in model_results]  
        headers = ["Date", "Actual", "Predicted"]  
        print(tabulate(table_data, headers=headers, tablefmt="grid"))  

        # Calculate overall metrics  
        all_actual = [result['actual'] for result in model_results]  
        all_predicted = [result['predicted'] for result in model_results]  
          
        metrics = calculate_metrics(all_actual, all_predicted)  

        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")  

# Call the function to run rolling predictions  
run_rolling_predictions(data_file, feature_columns, specified_dates, start_date, end_date)
