import subprocess

# List of Python scripts to run
scripts = [
    'decision_tree_regression.py',
    'multiple_linear_regression.py',
    'polynomial_regression.py',
    'random_forest_regression.py',
    'support_vector_regression.py'
]

# Run each script simultaneously
processes = []
for script in scripts:
    process = subprocess.Popen(['python', script])
    processes.append(process)

# Wait for all scripts to finish
for process in processes:
    process.communicate()

print("All models have been run successfully.")
