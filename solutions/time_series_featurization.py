def prepare_data_for_supervised_learning(data, prediction_horizon=1, dropna=True):
    """
    
    """
    ## Input data
    # lags & diffs
    lag_data = []
    diff_data = []
    for lag in range(MAX_LAG_FEATURES, 0, -1):
        x = data.shift(lag)
        diff = x - data.shift(lag+1)
        x.columns = [f"{col}-lag-{lag}" for col in x.columns]
        diff.columns = [f"{col}-diff-{lag}" for col in train_data.columns]
        lag_data.append(x)
        diff_data.append(diff)
    
    
    lags = pd.concat(lag_data, axis=1)
    diffs = pd.concat(diff_data, axis=1)

    # categorical features
    quarters = pd.DataFrame(data.index.quarter.values, index=data.index, columns=['quarter'])
    months = pd.DataFrame(data.index.month.values, index=data.index, columns=['month']) 
    weeks = pd.DataFrame(data.index.isocalendar().week.values, index=data.index, columns=['week'])
    week_of_months = data.index.to_series().apply(lambda x: (x.day - 1) // 7 + 1).to_frame()
    week_of_months.columns = ['week_of_month']

    # rolling statistics over 4 weeks
    rolling_mean = data.rolling(window=WINDOW_FOR_ROLLING_STATS).mean()
    rolling_mean.columns = [f"{col}-rolling_mean" for col in rolling_mean.columns]
    
    rolling_min = data.rolling(window=WINDOW_FOR_ROLLING_STATS).min()
    rolling_min.columns = [f"{col}-rolling_min" for col in rolling_min.columns]
    
    rolling_max = data.rolling(window=WINDOW_FOR_ROLLING_STATS).max()
    rolling_max.columns = [f"{col}-rolling_max" for col in rolling_max.columns]

    X = pd.concat([lags, diffs, quarters, months, weeks, week_of_months, rolling_mean, rolling_min, rolling_max], axis=1)

    ## Target data
    target_data = []
    for fwd in range(1, prediction_horizon+1):
        x = data.shift(-fwd)
        x.columns = [f"{col}-target-{fwd}" for col in x.columns]
        target_data.append(x)
    Y = pd.concat(target_data, axis=1)
    
    data = pd.concat([X, Y], axis=1)
    if dropna:
        data = data.dropna()
    return data