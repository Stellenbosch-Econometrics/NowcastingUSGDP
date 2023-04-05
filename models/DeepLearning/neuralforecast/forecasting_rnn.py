
### Import data and best config ###



### RNN model forecast ###

# model = AutoRNN(h=horizon, loss=MAE(), config=best_config, num_samples=30)

# nf = NeuralForecast(models=[model], freq='Q')
# nf.fit(df=df)


fcst_model = nf.predict(futr_df=futr_df)

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
