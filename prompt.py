import boto3
import pandas as pd
import time
import sys
import threading
import json
from botocore.exceptions import ClientError
from limiter import Limiter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define Variables
# Input
file_path = "[YOUR_FILE_PATH_HERE]"

# Model related
# Running on AWS SageMaker – no API key required  
# Modify based on your environment and model choice  
model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
temperature = 0
max_tokens = 128

# System prompt definition
system_prompt = "[YOUR_PROMPT_HERE]"

# Multi-thread and quota related
num_workers = 10  # Number of concurrent workers
limiter = Limiter(rate=16, capacity=1000, consume=1)  # Initialize token bucket

@limiter  # Apply the decorator to automatically handle rate limiting
def evaluate_content(row, idx):
    # Initialize the Bedrock API client
    bedrock = boto3.client(service_name='bedrock-runtime')

    # Prepare the request body with specific instructions and data
    messages = [{"role": "user", "content": row['content']}]
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": messages
    })

    # Make the API call and handle potential errors
    try:
        response = bedrock.invoke_model(body=body, modelId=model_id)
        response_body = json.loads(response.get('body').read())
        return idx, response_body['content'][0]['text']
    except ClientError:
        return idx, "none, none"  # Return a default value in case of an error

 # Load CSV data
data = pd.read_csv(file_path)
total_rows = len(data)
processed_count = 0

# Function to process a single row
def process_row(row, idx):
    global processed_count
    result = evaluate_content(row, idx)
    with thread_lock:
        processed_count += 1
        # Update progress in the console
        sys.stdout.write(f"\rProcessed: {processed_count}/{total_rows}")
        sys.stdout.flush()
    return result

# Set up a thread lock for safely updating the processed count and use ThreadPoolExecutor to process rows in parallel
thread_lock = threading.Lock()
start_time = time.time()  # Start timing
results = []
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    # Create future tasks and store them with their indices
    futures = {executor.submit(process_row, row, idx): idx for idx, row in data.iterrows()}
    for future in as_completed(futures):
        idx, res = future.result()
        results.append((idx, res))

end_time = time.time()  # End timing

# Sort results and transform into DataFrame to maintain order
results.sort()  # Sort by index
data['results'] = [res for idx, res in results]

# Use try-except structure to handle potential errors in data processing
for index, result in enumerate(data['results']):
    try:
        relevance, sentiment = result.split(', ', 1)
        data.at[index, 'relevance'] = relevance
        data.at[index, 'sentiment'] = sentiment
    except ValueError:
        # If splitting fails (e.g., no ", " found), set default values
        data.at[index, 'relevance'] = "none"
        data.at[index, 'sentiment'] = "none"

# Drop 'results' column as it has been split into other columns
data.drop(columns=['results'], inplace=True)

# Save the updated data back to the same CSV file
data.to_csv(file_path, index=False)

# Calculate and print the total processing time
total_time_taken = end_time - start_time  # Calculate total processing time
print(f"Processing complete. The original file has been updated. Total time taken: {total_time_taken:.2f} seconds.")