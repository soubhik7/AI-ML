import torch
from transformers import pipeline

# Check if a GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load a pipeline for natural language inference using the RoBERTa model
nli_model = pipeline("text-classification", model="roberta-large-mnli", device=device)
#nli_model2 = pipeline("test-classification", model="ro")

def detect_truthfulness(statements):
    true_false_results = []

    for statement in statements:
        # Apply the NLI model and get the detailed output
        results = nli_model(statement)

        # Print detailed model output for debugging
        print(f"Model output for '{statement}': {results}")

        # Interpret the model's output
        result = results[0]
        if result['label'] == 'ENTAILMENT':
            true_false_results.append((statement, "True"))
        elif result['label'] == 'CONTRADICTION':
            true_false_results.append((statement, "False"))
        else:
            true_false_results.append((statement, "Uncertain"))

    return true_false_results

# Example usage
statements = [
    "The Eiffel Tower is in Paris.",
    "The sun orbits the Earth.",
    "Water boils at 100 degrees Celsius at sea level.",
    "My name is soubhik"
]

results = detect_truthfulness(statements)

for statement, truth_value in results:
    print(f"Statement: {statement} | Truthfulness: {truth_value}")