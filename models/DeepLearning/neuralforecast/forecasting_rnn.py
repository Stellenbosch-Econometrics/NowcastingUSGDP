
### Load packages ###

import pickle


from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoRNN
from neuralforecast.losses.pytorch import MAE


### Import data and best config ###

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


df = load_object('pickle_files/rnn_df.pickle')
futr_df = load_object('pickle_files/rnn_futr_df.pickle')
pcc_list = load_object('pickle_files/rnn_pcc_list.pickle')
fcc_list = load_object('pickle_files/rnn_fcc_list.pickle')
best_config = load_object('pickle_files/rnn_best_config.pickle')


### RNN model forecast ###

horizon = 4

model = AutoRNN(h=horizon, loss=MAE(), config=best_config, num_samples=30)
nf = NeuralForecast(models=[model], freq='Q')
nf.fit(df=df)

# fcst_model = nf.predict(futr_df=futr_df)

Y_hat_df = nf.predict(futr_df=futr_df).reset_index()
Y_hat_df.head()


### Plot the results ###

# # Concatenate the train and forecast dataframes
# plot_df = pd.concat([df, Y_hat_df]).set_index('ds')

# plt.figure(figsize=(12, 3))
# plot_df[['y', 'AutoRNN']].plot(linewidth=2)

# plt.title('AirPassengers Forecast', fontsize=10)
# plt.ylabel('Monthly Passengers', fontsize=10)
# plt.xlabel('Timestamp [t]', fontsize=10)
# plt.axvline(x=plot_df.index[-horizon], color='k', linestyle='--', linewidth=2)
# plt.legend(prop={'size': 10})
# plt.show()


### RNN model across all vintages ###


### Cross validation ###


# 10495415
