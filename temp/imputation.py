

def impute_missing_values_mean(data):
    imputed_data = data.copy()
    imputed_data.apply(lambda x: x.fillna(x.mean()), axis=0)
    return imputed_data


def impute_missing_values_median(data):
    imputed_data = data.copy()
    imputed_data.apply(lambda x: x.fillna(x.median()), axis=0)
    return imputed_data


def impute_missing_values_rolling_mean(data, window_size):
    imputed_data = data.copy()
    imputed_data.fillna(data.rolling(window=window_size,
                        min_periods=1, center=True).mean(), inplace=True)
    return imputed_data


def impute_missing_values_rolling_median(data, window_size):
    imputed_data = data.copy()
    imputed_data.fillna(data.rolling(window=window_size,
                        min_periods=1, center=True).median(), inplace=True)
    return imputed_data


def impute_missing_values_bfill_ffill(data):
    imputed_data = data.copy()
    imputed_data.fillna(method='bfill', inplace=True)
    imputed_data.fillna(method='ffill', inplace=True)
    return imputed_data


def impute_missing_values_interpolate(data, method='linear'):
    imputed_data = data.copy()
    imputed_data.interpolate(method=method, inplace=True)
    return imputed_data

def impute_missing_values_kalman_smoother(data, order=1):
    imputed_data = data.copy()

    for col in data.columns:
        y = imputed_data[col].values
        n = len(y)
        mask = np.isnan(y)
        ksm = KalmanSmoother(n, k_endog=1, k_states=order,
                             initialization='approximate_diffuse')
        ksm['design'] = np.eye(order)
        ksm['obs_cov'] = np.eye(1)
        ksm['transition'] = np.eye(order)
        ksm['state_cov'] = np.eye(order)

        y_diff = diff(y, order, axis=0)
        y_diff = np.concatenate((np.zeros(order), y_diff))

        y[mask] = 0
        ksm.bind(y[:, None])
        ksm.initialize_known(np.zeros(order), np.eye(order))

        smoothed_state = ksm.smooth()

        y_imputed = np.cumsum(y_diff * ~mask) + y[mask]
        imputed_data[col] = y_imputed

    return imputed_data