
# %%
## Import the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import NBEATSModel
from darts import TimeSeries
from darts.metrics import rmse
from darts.dataprocessing.transformers import Scaler
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("darts.models.forecasting.pl_forecasting_module").setLevel(logging.CRITICAL)
logging.getLogger("darts.models.forecasting.nbeats").setLevel(logging.CRITICAL)
logging.getLogger("darts.models.forecasting.torch_forecasting_model").setLevel(logging.CRITICAL)
logging.getLogger("darts.models.forecasting.torch_forecasting_model").setLevel(logging.CRITICAL)
# %%
