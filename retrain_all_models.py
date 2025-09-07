# retrain_all_models.py


import subprocess
import os

# Define model training scripts relative to project root
model_scripts = [
    # "models/logreg_model.py",
    # "models/roberta_model.py",
    # "models/deberta_model.py",
    "models/rnn_model.py"
]

for script in model_scripts:
    print(f"\nRunning: {script}")
    
    # Ensure the path exists
    if not os.path.isfile(script):
        print(f"Script not found: {script}")
        continue
    
    # Run the script and capture outputs
    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


#Then run from terminal:
#python retrain_all_models.py