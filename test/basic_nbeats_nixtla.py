
# %%

## Going to use the NeuralForecast package from Nixtla. Focus on the NBEATS model with covariates (exogenous variables)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS



