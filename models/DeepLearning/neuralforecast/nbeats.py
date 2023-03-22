

import importlib.util
import os

# Get the relative path to the 'data_imputation.py' file
module_path = os.path.join(os.path.dirname(
    __file__), '..', '..', 'sample', 'data_imputation.py')

# Load the module using the relative path
spec = importlib.util.spec_from_file_location("data_imputation", module_path)
data_imputation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_imputation)


# Use the imported functions
data = data_imputation.impute_missing_values_mean(data)
