import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import utils_tfb

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

plt.style.use('tableau-colorblind10')
PLOTTING_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

CURRENT_DIR = pathlib.Path(__file__).resolve().parent

def load_tutotrial_data(dataset, log_transform=False):
    """Loads dataset for tutorial."""
    TS_DATA_FOLDER = pathlib.Path(CURRENT_DIR / "forecasting").resolve()
    if dataset == 'exchange_rate':
        dataset = TS_DATA_FOLDER / "Exchange.csv"
        data = utils_tfb.read_data(str(dataset))
        data.index.freq = 'D'  # since we know that the frequency is daily
        data = data.resample("W").mean()
    else:
        raise ValueError(f"Unrecognized dataset: {dataset}")


    if log_transform:
        transformed_data = np.log(data / data.shift(1)).dropna()
        return data, transformed_data
    
    return data


def get_cols(time_series_data, cols):
    if cols is None:
        cols = time_series_data.columns

    if len(cols) > 10:
        return random.sample(cols, 10)
    
    return cols


def plot_raw_data(time_series_data, cols=None, figsize=(15, 10), ax=None):
    """Plots the time series in data. If there are more than 10, it samples `n` of them.
    
    Args:
        time_series_data: pandas dataframe containing time series in each column and index as a PeriodIndex
        cols: list of columns names in time_series_data to plot
    """
    cols = get_cols(time_series_data, cols)
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=100)

    for idx, col in enumerate(cols):
        ax.plot(time_series_data[col], label=f'{col}')
    
    ax.grid()
    ax.legend()

    if fig is not None:
        fig.suptitle("Raw data")
    return fig, ax


def plot_seasonality_decompose(time_series_data, cols=None, figsize=(15, 15), axs=None):
    """Seasonality decomposition for the time series. 
    Args:
        time_series_data: pandas dataframe containing time series in each column and index as a PeriodIndex
        cols: list of columns names in time_series_data to plot
    """
    fig = None
    if axs is None:
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=figsize, dpi=100)
    cols = get_cols(time_series_data, cols)
    
    for idx, col in enumerate(cols):
        ts = time_series_data[col]
        color = PLOTTING_COLORS[idx % len(PLOTTING_COLORS)]
        
        # original series
        ax = axs[0]
        ax.plot(ts.values, color=color, label=f'{col}')
        ax.set_title('Original')
        ax.set_xticks([])
        
        # seasonal decomposition
        result = seasonal_decompose(ts, model="additive")
        
        # trend
        ax = axs[1]
        ax.plot(result.trend, color=color, label=f'{col}')
        ax.set_title('Trend')
        ax.set_xticks([])
        ax.legend()
        
        # seasonality
        ax = axs[2]
        ax.plot(result.seasonal, color=color, label=f'{col}', alpha=1/(idx+1))
        ax.set_title('Seasonality')
        ax.set_xticks([])
        # ax.legend()
        
        # residual
        ax = axs[3]
        ax.plot(result.resid, color=color, label=f'{col}', alpha=1/(idx+1))
        ax.set_title('Residuals')
        # ax.legend()
        
        for ax in axs:
            ax.grid()

    if fig is not None:
        fig.suptitle("Seasonality decomposition")
    return fig, axs


def plot_acf_pacf(time_series_data, cols=None, figsize=(10, 10), axs=None, n_lags=150):
    """Plots autocorrelation and partial autocorrelations."""
    # more the number of lags, more time it will take
    if n_lags > 50:
        lags = np.arange(0, n_lags, 2)
    else:
        lags = np.arange(0, n_lags)

    fig = None
    if axs is None:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=100)

    cols = get_cols(time_series_data, cols)

    ax = axs[0]
    for idx, col in enumerate(cols):
        color = PLOTTING_COLORS[idx % len(PLOTTING_COLORS)]
        ts = time_series_data[col]
        autocorrs = [ts.autocorr(lag=lag) for lag in lags]
        ax.plot(lags, autocorrs, label=f'{col}', color=color)

    ax.set_xlim(0, max(lags))
    ax.hlines(0, 0, max(lags), color='red', linewidth=3, linestyle='--')
    ax.legend()
    ax.grid()
    ax.set_title("Autocorrelations")

    ax = axs[1]
    for idx, col in enumerate(cols):
        color = PLOTTING_COLORS[idx % len(PLOTTING_COLORS)]
        ts = time_series_data[col]
        _ = plot_pacf(ts, method='ywm', lags=lags, ax=ax, title=None, label=f'{col}')

    ax.set_title("Partial Autocorrelations")
    return fig, axs


def check_stationarity(time_series_data, cols=None):
    """Checks whether the time series is stationary using ADF test."""
    cols = get_cols(time_series_data, cols)
    alpha = 0.05

    for col in cols:
        ts = time_series_data[col]
        results = adfuller(ts)
        pvalue = results[1]
        print(f"Col: {col}\tP-value: {pvalue:0.4f}\tStationary: {pvalue < alpha}")


def mean_absolute_error(forecast, target):
    """Computes Mean Absolute Error."""
    return np.mean(np.abs(forecast - target))


def mean_squared_error(forecast, target):
    """Computes the mean squared error."""
    return np.mean(np.power(forecast - target, 2))


def root_relative_squared_error(forecast, target, train):
    """Returns the scaled MSE, where the scale is mean on the training data as forecast"""
    mse = mean_squared_error(forecast, target)
    mean = np.mean(train)
    naive_mse = mean_squared_error(target, mean)
    return mse / naive_mse


def mean_absolute_scaled_error(forecast, target, insample_data, seasonality=1):
    """
    Computes Mean Absolute Scaled Error. 
    
    Args:
        target (list, np.array, pd.Series): ground truth time series
        forecast (list, np.array, pd.Series): forecasted values for target
        insample_data (list, np.array, pd.Series): training data for forecast
    """
    insample_naive_forecast = insample_data[:-seasonality] # our naive predictions
    insample_targets = insample_data[seasonality:] # our targets are the values after the shift by seasonality
    #
    mae = mean_absolute_error(forecast, target)
    mae_naive = mean_absolute_error(insample_naive_forecast, insample_targets)
    #
    eps = np.finfo(np.float64).eps
    #
    mase = mae / np.maximum(mae_naive, eps)
    return mase


def get_all_metrics(forecast, target, insample_data, seasonality=1):
    mse = mean_squared_error(forecast, target)
    rse = root_relative_squared_error(forecast, target, insample_data)
    mae = mean_absolute_error(forecast, target)
    mase = mean_absolute_scaled_error(forecast, target, insample_data, seasonality)
    return {'mae':mae, 'mase':mase, 'rse': rse, 'mse':mse}


def get_mase(forecasted_df, target_df, train_df, horizon=-1):
    """Multivariate Time Series: Computes mase for each column.

    Args:
        forecasted_df (pd.DataFrame): forecasts of the same shape as target_df
        target_df (pd.DataFrame): target for forecasts
        train_df (pd.DataFrame): historical data for training
        horizon (int): horizon over which to compute MASE
    """
    if horizon < 0:
        horizon = forecasted_df.shape[0]

    all_mase = {}
    for col in train_df.columns:
        mase = mean_absolute_scaled_error(
            forecasted_df[col].values[:horizon], 
            target_df[col].values[:horizon], 
            train_df[col].values
        )
        all_mase[col] = mase

    return all_mase

def highlight_min(data, color='white', font_weight='bold'):
    """Returns the style for each cell."""
    attr = f'background-color: {color}; font-weight: {font_weight}'
    if data.ndim == 1:  # Apply column-wise
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # Apply element-wise if needed
        is_min = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''), index=data.index, columns=data.columns)


def plot_forecasted_series(model_results, train_data, test_data, n_history_to_plot=100, forecasting_horizon=100, col=None):
    """Plots forecasted series in continuation with the historical data.
    Args:
        model_results (statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper): trained model results.
        train_data (pd.DataFrame): pandas dataframe containing time series in each column and index as a PeriodIndex
        test_data (pd.DataFrame): pandas dataframe containing time series for targets in each column and index as a PeriodIndex
        n_hisrtory_to_plot (int): number of points from train_data to plot
        forecasting_horizon (int): number of points to project forward in time
        col (str): a valid column in train_data
    
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    color = PLOTTING_COLORS[0]
    
    train_data[col][-n_history_to_plot:].plot(ax=ax, color=color, label='train series')
    
    # forecasts
    predictions = model_results.get_forecast(steps=forecasting_horizon)
    mean = predictions.predicted_mean
    se = predictions.se_mean
    
    #
    mean.plot(ax=ax, color=color, linestyle=':', linewidth=2, label='estimated mean')
    ax.fill_between(se.index, mean.values - se.values, se.values + mean.values, color=color, alpha=0.4)
    test_data[col][:forecasting_horizon].plot(ax=ax, color=color, linestyle='--', label='target')
    (mean-se).plot(ax=ax, color='red', linestyle='--')
    (mean+se).plot(ax=ax, color='red', linestyle='--')
    
    ax.legend()

    fig.suptitle(f"Column: {col}")
    return fig, ax


def display_results(path="", models=[], metric='mase'):
    """Displays results so that the minimum is highlighted across different horizons."""
    assert path != "", "Directory path, where results are saved, is required"
    path = pathlib.Path(path).resolve()

    all_results = []
    for results_folder in path.iterdir():
        if (results_folder / "metrics.csv").exists():
            results = pd.read_csv(str(results_folder / "metrics.csv"))
            all_results.append(results)

    results = pd.concat(all_results)

    if models:
        results = results[results['model'].isin(models)]

    mean_df = results.groupby(['horizon', 'model']).mean(['mase', 'mae', 'rse', 'mse']).reset_index()
    mean_df['col'] = 'overall'

    results = pd.concat([results, mean_df])

    results_df = results.pivot_table(index=['model'], columns=['horizon', 'col'], values=metric)
    for h in results.horizon.unique():
        print("#"*50, f"Horizon: {h}", "#"*50)
        df = results_df[h]
        styled_df = df.style.apply(highlight_min, axis=0)
        display(styled_df)
        print("\n")


def get_mase_metrics(historical_data, test_predictions, target_data, columns, forecasting_horizons, model_name, return_records=True):
    """
    Returns a dictionary containing metrics for the test_predictions.
    
    Args:
        historical_data (pd.DataFrame): Time series used for training the model.
        test_predictions (pd.DataFrame): Forecasts for the rest of the time series.
        target_data (pd.DataFrame): Actual values for the rest of the time series.
        columns (list): List of column identifiers for the time series.
        forecasting_horizons (list): Horizon to be used to compute MASE.
        return_records (bool): If a flattened list is to be returned. 
    """
    # compute metrics
    model_metrics = {}
    for col in columns:
        model_metrics[col] = {}
        
        ts = historical_data[col]
        predictions = test_predictions[col]
        for horizon in forecasting_horizons:
            forecast = predictions[:horizon]
            actual = target_data[col][:horizon].values
            model_metrics[col][horizon] = get_all_metrics(forecast, actual, insample_data=ts.values)
    
    records = []
    if return_records:
        records = []
        for col in columns:
            for horizon, metrics in model_metrics[col].items():
                records.append({
                    'col': col,
                    'horizon': horizon,
                    'model': model_name,
                    **metrics
                })

    return model_metrics, records


def plot_forecasts(historical_data, 
                   forecast_directory_path,
                   target_data, 
                   columns, 
                   n_history_to_plot=100, 
                   forecasting_horizon=12, 
                   dpi=200, 
                   models=[], 
                   exclude_models=[],
                   plot_se=True):
    """
    Plots the time series and its forecast for the tutorial on exchange rate predictions.

    Args:
        historical_data (pd.DataFrame, pd.Series): Past data on which models are trained
        forecasts (pd.DataFrame, pd.Series): Forecasts by the model
        target_data (pd.DataFrame): Target data for the time series
        results: output of `update_results`
        columns (list): A list of columns that are to be plotted.
        n_history_to_plot (int): number of last time steps from the historical data to plot
        forecasting_horizon (int): number of forecasted time steps to plot from forecasts
        dpi (int): matplotlib argument for display clarity
        models (list): models to plot
        exclude_models (list): don't plot these models
        plot_se (bool): if True and present, plots 1 standard error.

    """
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), dpi=dpi)

    path = pathlib.Path(forecast_directory_path).resolve()

    # load metrics and predictions
    all_results, all_predictions = [], []
    for results_folder in path.iterdir():
        if not results_folder.is_dir():
            continue

        if (results_folder / "metrics.csv").exists():
            results = pd.read_csv(str(results_folder / "metrics.csv"))
            all_results.append(results)
        
        if (results_folder / "predictions.csv").exists():
            filename = str(results_folder / "predictions.csv")
            all_predictions.append(pd.read_csv(filename, index_col='date', parse_dates=['date']))

    results = pd.concat(all_results)
    forecasts = pd.concat(all_predictions, axis=1)

    # plot
    max_horizon = forecasting_horizon
    models = models if models else results['model'].unique()
    models = [x for x in models if x not in exclude_models]
    for idx, col_name in enumerate(columns):
        ax = axs[idx % 2, idx // 2]

        # training data
        historical_data[col_name][-n_history_to_plot:].plot(ax=ax, color='blue', label=f'history')

        # a dotted line for the transition between train and forecast
        pd.concat([historical_data[col_name][-1:], 
                   target_data[col_name][:1]]).plot(ax=ax, color='blue', linestyle=':', label='')

        for j_idx, model in enumerate(models):
            color = PLOTTING_COLORS[j_idx % len(PLOTTING_COLORS)]

            # metric for display
            mase = results[(results['col'] == col_name) & (results['model'] == model) & (results['horizon'] == max_horizon)]['mase'].item()

            # forecasting mean
            mean = forecasts[f"{model}_{col_name}_mean"]
            mean.plot(ax=ax, color=color, linestyle='--', linewidth=2, label=f'{model}. MASE (T={max_horizon}): {mase: 0.2f}')

            # forecasting se
            if plot_se and f"{model}_{col_name}_se" in forecasts.columns:
                se = forecasts[f"{model}_{col_name}_se"]
                ax.fill_between(se.index, mean.values - se.values, se.values + mean.values, color=color, alpha=0.4)

        # target data
        target_data[col_name].plot(ax=ax, color='red', linestyle='-', label='target', alpha=0.7)

        # legends and misc.
        ax.legend(prop={'size':6})
        ax.tick_params(axis='x', labelsize=8)
        if j_idx // 2 != 4:
            ax.set_xlabel('')

        ax.set_title(f'Forecasts for {col_name}')
    
    return fig, axs



#### LEGACY FUNCTIONS

def update_test_predictions(test_predictions, save=False, model_name=None, prediction_type=''):
    """
    Updates test predictions in the CSV. 
    Returns the hisotorical test predictions if the column in `test_predictions` already exist.

    Args:
        test_predictions (pd.DataFrame, dict(str, pd.DataFrame), dict(str, np.ndarray)): 
            Contains (mean or se) predictions for each time index. A model is either specified in its columns or as a key of dictionary. 
        save (bool): whether to save to the file that contains other predictions or not. 
    """

    prev_test_predictions = pd.read_csv("test_predictions_tutorial_exchange_rate.csv", index_col='date', parse_dates=['date'])

    if isinstance(test_predictions, dict):
        more_test_predictions = pd.DataFrame(test_predictions)
    elif isinstance(test_predictions, pd.DataFrame):
        assert test_predictions.shape[0] == prev_test_predictions.shape[0], f"Number of rows in test_predictions:{test_predictions.shape[0]}\t Number of rows expected:{prev_test_predictions.shape[0]}"
        test_predictions = test_predictions.copy()
        test_predictions.columns = [f"{model_name}_{col}_{prediction_type}" for col in test_predictions.columns]
        more_test_predictions = test_predictions
    else:
        raise ValueError(f"Unrecognized type of test_predictions: {type(test_predictions)}")
    
    more_test_predictions.index = prev_test_predictions.index

    if sum(x in prev_test_predictions.columns for x in more_test_predictions.columns) > 0:
        print("Columns already exists. Returning the existing test predictions.")
        return prev_test_predictions

    test_predictions = pd.concat([prev_test_predictions, more_test_predictions], axis=1)
    print("New columns:", test_predictions.columns)

    if save:
        test_predictions.to_csv("test_predictions_tutorial_exchange_rate.csv", index=True)
    return test_predictions




def update_results(records, save=False, rewrite=False):
    """Updates results with records. Ensures that the models are not repeated."""
    results = pd.read_csv('metrics_tutorial_exchange_rate.csv')
    print(f"Existing size of the results: {results.shape}")
    print(f"Existing methods in results: {results.model.unique()}")

    new_methods = set([x['model'] for x in records])

    if sum(x in results.model.unique() for x in new_methods) > 0:
        if not rewrite:
            print( "The model already exist in the results. Returning original.")
            return results
        else:
            for x in new_methods:
                if x in results['model'].unique():
                    print(f"dropping {x}")
                    results = results[results['model'] != x]

    # Store these metrics to be able to compare them against other methods later on
    more_results = pd.DataFrame(records)
    results = pd.concat([results, more_results])
    
    if save:
        results.to_csv('metrics_tutorial_exchange_rate.csv', index=False)

    print(f"New size of the results: {results.shape}")
    print(f"New methods in results: {results.model.unique()}")

    return results



## Dataloader 
class SlidingFixedWindow(torch.utils.data.Dataset):
    """
    Returns a time series data iterator with sliding window of fixed size.

    Args:
        data: Pandas dataframe 
        seq_length (int): number of past time steps to include in the training sequence.
    """
    def __init__(self, data, seq_length=100):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index:index+self.seq_length].values, dtype=torch.float),
            torch.tensor(self.data.iloc[index+self.seq_length].values, dtype=torch.float)
        )