"""This module contains functions that calculate metrics, plot results, and save progress"""
import pathlib
import numpy as np
import plotly.subplots as sb
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
import pandas as pd
from calibrate import residuals


def plot_data(train: pd.DataFrame, test: pd.DataFrame, save_name: str = 'original_data') -> None:
    """PLot temperature and heating data and save to html file in Results folder.

    Args:
        train (pd.DataFrame): The training dataset.
        test (pd.DataFrame): The test dataset.
        save_name (str, optional): The name of the file that will be saved in Results folder.
            Defaults to 'original_data'.
    """

    n_zones = (len(train.columns) - 2) // 4
    # Create the plotly figure
    fig = sb.make_subplots(rows=n_zones+1,
                           cols=2,
                           shared_xaxes=True, vertical_spacing=0.02,
                           horizontal_spacing=0.05)
    name = 'T<sub>ext</sub> (<sup>o</sup>C)'
    # Plot Text for the training and test set
    fig.add_trace(go.Scatter(
        x=train.index, y=train.loc[:, 'Text'],
        line=dict(color=px.colors.qualitative.T10[0]),
        name='Train', legendgroup='Train',
        showlegend=False),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=test.index, y=test.loc[:, 'Text'],
        line=dict(color=px.colors.qualitative.T10[1]),
        name='Test', legendgroup='Test',
        showlegend=False),
        row=1, col=1)
    fig.update_yaxes(title_text=name, row=1, col=1)
    # Plot GHI for the training and test set
    name = 'GHI (kW/m<sup>2</sup>)'
    fig.add_trace(go.Scatter(
        x=train.index, y=train.loc[:, 'GHI'],
        line=dict(color=px.colors.qualitative.T10[0]),
        name='Train', legendgroup='Train',
        showlegend=False),
        row=1, col=2)
    fig.add_trace(go.Scatter(
        x=test.index, y=test.loc[:, 'GHI'],
        line=dict(color=px.colors.qualitative.T10[1]),
        name='Test', legendgroup='Test',
        showlegend=False),
        row=1, col=2)
    fig.update_yaxes(title_text=name, row=1, col=2)
    # For each zone
    for idx in range(n_zones):
        # Plot zone temperature
        name = f'T<sub>{idx+1:02d}</sub> (<sup>o</sup>C)'
        fig.add_trace(go.Scatter(
            x=train.index, y=train.loc[:, f'T{idx+1:02d}_TEMP'],
            line=dict(color=px.colors.qualitative.T10[0]),
            name='Train', legendgroup='Train',
            showlegend=False),
            row=idx+2, col=1)
        fig.add_trace(go.Scatter(
            x=test.index, y=test.loc[:, f'T{idx+1:02d}_TEMP'],
            line=dict(color=px.colors.qualitative.T10[1]),
            name='Test', legendgroup='Test',
            showlegend=False),
            row=idx+2, col=1)
        fig.update_yaxes(title_text=name, row=idx+2, col=1)
        # Plot zone heating
        name = f'Q<sub>{idx+1:02d}</sub> (kWh)'
        fig.add_trace(go.Scatter(
            x=train.index, y=train.loc[:, f'T{idx+1:02d}_Wh'],
            line=dict(color=px.colors.qualitative.T10[0]),
            name='Train', legendgroup='Train',
            showlegend=False),
            row=idx+2, col=2)
        fig.add_trace(go.Scatter(
            x=test.index, y=test.loc[:, f'T{idx+1:02d}_Wh'],
            line=dict(color=px.colors.qualitative.T10[1]),
            name='Test', legendgroup='Test',
            showlegend=False),
            row=idx+2, col=2)
        fig.update_yaxes(title_text=name, row=idx+2, col=2)
    fig['data'][0]['showlegend'] = True
    fig['data'][1]['showlegend'] = True
    fig.update_layout(legend=dict(
        yanchor="top", y=1.05, xanchor="center", x=0.5, orientation='h'))
    fig.update_xaxes(matches='x')
    fig.write_html(pathlib.Path(f'Results/{save_name}.html'))
    return


def get_predictions(parameters: pd.DataFrame, data_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get temperature predictions

    Args:
        parameters (pd.DataFrame): The model parameters
        data_np (np.ndarray): The data in a numpy format

    Returns:
        tuple[np.ndarray, np.ndarray]: measured, predicted temperatures
    """

    # Get arrays of values, variables, and indices of the parameters
    param_values = parameters.loc[parameters.vary, 'value'].values
    param_variables = parameters.loc[parameters.vary, 'variable'].values
    param_index1 = parameters.loc[parameters.vary, 'index1'].values
    param_index2 = parameters.loc[parameters.vary, 'index2'].values
    n_zones = (data_np.shape[2]-2) // 2
    # Calculate the residuals
    errors, _ = residuals(
        param_values, param_variables, param_index1, param_index2, data_np, flattened=False)
    # Predicted temperatures are measured plus errors
    predicted = errors + data_np[:, 1:, :n_zones//2]
    return data_np[:, 1:, :n_zones//2], predicted


def get_metrics(parameters: pd.DataFrame, measured: np.ndarray, predicted: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Calculate the BIC, RMSE, variance, r2 and FIT.

    Args:
        parameters (pd.DataFrame): The model parameters
        measured (np.ndarray): The measured temperatures
        predicted (np.ndarray): The predicted temperatures

    Returns:
        tuple[np.ndarray, np.ndarray, pd.DataFrame]: The multi-step rmse and fit until 24
            hours and the rest of the metrics in a dataframe
    """

    # Calculate the multi-step RMSE of each zone
    rmses = np.zeros(measured.shape[1:])
    for i in range(rmses.shape[1]):
        rmses[:, i] = mean_squared_error(
            measured[:, :, i], predicted[:, :, i], squared=False, multioutput='raw_values')
    # Calculate the multi-step FIT of each zone
    num = np.sqrt(np.sum((measured - predicted) ** 2, axis=0))
    den = np.sqrt(np.sum((measured - np.mean(measured, axis=0)) ** 2, axis=0))
    fits = (1 - num/den) * 100
    # Reshape the data so that it only has 2 dimensions, one being the zone
    measured = measured.reshape(-1, measured.shape[-1])
    predicted = predicted.reshape(-1, predicted.shape[-1])
    # Calculate the total RMSE of all zones (assume the same weights)
    rmse = mean_squared_error(measured, predicted, squared=False)
    # Calculate the variance and weigh each zone based on the variance
    variance = explained_variance_score(
        measured, predicted, multioutput='variance_weighted')
    # Calculate the bic
    bic = measured.size * np.log(mean_squared_error(measured, predicted)) \
        + np.log(measured.size) * parameters.vary.sum()
    # Calculate the r2 and weigh each zone based on the variance
    r2 = r2_score(measured, predicted, multioutput='variance_weighted')
    # Calculate the FIT
    num = np.sqrt(np.sum((measured - predicted) ** 2, axis=0))
    den = np.sqrt(np.sum((measured - np.mean(measured, axis=0)) ** 2, axis=0))
    weights = np.mean((measured - np.mean(measured, axis=0)) ** 2, axis=0)
    fit = np.average((1 - num/den) * 100, weights=weights)
    metrics = pd.Series(index=['BIC', 'RMSE', 'variance', 'r2', 'FIT'], data=[
                        bic, rmse, variance, r2, fit])
    return rmses, fits, metrics


def get_metrics_and_plots(parameters: pd.DataFrame, data_np: np.ndarray, data: pd.DataFrame,
                          save_name: str = 'example', save_metrics: bool = True,
                          plot_metrics: bool = False) -> pd.DataFrame:
    """Get metrics and (optionally) save and plot them

    Args:
        parameters (pd.DataFrame): The model parameters
        data_np (np.ndarray): The data in numpy format
        data (pd.DataFrane): The data in pandas format
        save_name (str, optional): Name of the saved files. Defaults to 'example'.
        save_metrics (bool, optional): Whether you want to save the metrics in folders
            Parameters and Metrics_csvs. Defaults to True.
        plot_metrics (bool, optional): Whether you want to save the plots in folders 
            Metrics_plots and Temepratures. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """

    # Get predictions
    measured, predicted = get_predictions(parameters, data_np)
    # Get metrics
    rmses, fits, metrics = get_metrics(parameters, measured, predicted)
    metrics.name = save_name

    if plot_metrics:
        plot_temperatures(predicted, data, save_name)
        plot_rmse_fit(rmses, fits, save_name)
    if save_metrics:
        parameters.to_csv(pathlib.Path(f'Results/Parameters/{save_name}.csv'))
        metrics.to_csv(pathlib.Path(f'Results/Metrics_csvs/{save_name}.csv'))
    return metrics


def plot_temperatures(predicted: np.ndarray, data: pd.DataFrame, save_name: str) -> None:
    """Plot measured and predicted temperatures and save them as an html file

    Args:
        predicted (np.ndarray): The predicted temperatures in numpy format
        data (pd.DataFrame): The measured temperatures in pandas format
        save_name (str): The name of the file to save the plot
    """

    fig = sb.make_subplots(rows=predicted.shape[2], cols=1,
                           shared_xaxes=True, shared_yaxes='columns')
    for idx in range(predicted.shape[2]):
        name = f'T<sub>{idx+1}</sub> (<sup>o</sup>C)'
        fig.add_trace(go.Scatter(
            x=data.index, y=data.loc[:, f'T{idx+1:02d}_TEMP'],
            line=dict(color=px.colors.qualitative.T10[0]),
            name='Measured', legendgroup='Measured',
            showlegend=False),
            row=idx+1, col=1)
        for j in range(predicted.shape[0]):
            fig.add_trace(go.Scatter(
                x=data.index[j+1:j+1+predicted.shape[1]
                             ], y=predicted[j, :, idx],
                line=dict(color=px.colors.qualitative.T10[1]),
                opacity=0.3,
                name='Predicted', legendgroup='Predicted',
                showlegend=False),
                row=idx+1, col=1)
        fig.update_yaxes(title_text=name, row=idx+1,
                         col=1)
    fig['data'][0]['showlegend'] = True
    fig['data'][-1]['showlegend'] = True
    fig.update_layout(legend=dict(
        yanchor="top", y=1.05, xanchor="center", x=0.5, orientation='h'))
    fig.update_xaxes(matches='x')
    fig.write_html(pathlib.Path(f'Results/Temperatures/{save_name}.html'))
    return


def plot_rmse_fit(rmses: np.ndarray, fits: np.ndarray, save_name: str) -> None:
    """Plot the multi-step RMSE and FIT up to 24 hours 

    Args:
        rmses (np.ndarray): The array of rmses
        fits (np.ndarray): The array of fits
        save_name (str): The name of the file to save the plot
    """

    fig = sb.make_subplots(rows=2, cols=1,
                           vertical_spacing=0.05, horizontal_spacing=0.05)
    for i in range(rmses.shape[1]):
        name = f'T<sub>{i+1:02d}</sub>'
        fig.add_trace(go.Scatter(
            x=np.arange(1, rmses.shape[0]+1), y=rmses[:, i],
            name=name, legendgroup=name,
            showlegend=False),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=np.arange(1, fits.shape[0]+1), y=fits[:, i],
            name=name, legendgroup=name,
            showlegend=False),
            row=2, col=1)
    fig.update_layout(legend=dict(
        yanchor="top", y=1.05, xanchor="center", x=0.5, orientation='h'))
    fig.update_xaxes(range=[1, 24], tick0=0, dtick=4, row=1, col=1)
    fig.update_xaxes(range=[1, 24], tick0=0, dtick=4, row=2, col=1)
    fig.update_yaxes(title_text="RMSE (<sup>o</sup>C)",
                     title_standoff=20, row=1, col=1)
    fig.update_yaxes(title_text="FIT (%)", title_standoff=0, row=2, col=1)
    fig.update_layout(hovermode="x unified")
    fig.write_html(pathlib.Path(f'Results/Metrics_plots/{save_name}.html'))
    return
