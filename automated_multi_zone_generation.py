"""This module uses an automated forward and backward selection procedure to 
find the structure of thermal networks"""
import pathlib
import shutil
import multiprocessing as mp
import pandas as pd
import numpy as np
from plots_and_metrics import plot_data, get_metrics_and_plots
from calibrate import calibrate, create_matrices


def load_data(data_path: pathlib.Path) -> pd.DataFrame:
    """Load temperatures, powers and weather data from csv file.

    Args:
        data_path (pathlib.Path): he path of the data folder.

    Returns:
        pd.DataFrame: The resulting dataset. It containes first the temperatures of the air and 
        envelope zones, then the power inputs of the air and envelope zones and finally
        the exterior temperature and solar radiation.
    """
    # Open file and read data from csv
    with data_path.open(mode='rb') as file:
        data = pd.read_csv(file, index_col=0, parse_dates=True, dtype='float')
    data.index.freq = pd.infer_freq(data.index)
    TEMP = np.sort([column for column in data.columns if 'TEMP' in column])
    PWR = np.sort([column for column in data.columns if 'Wh' in column])
    # Initialize envelope node temperature and heating
    temperatures = data.loc[:, TEMP].copy()
    temperatures.loc[:, [
        f'T{i+len(TEMP)+1:02d}_TEMP' for i in range(len(TEMP))]] = 0
    powers = data.loc[:, PWR].copy() / 1000
    powers.loc[:, [f'T{i+len(TEMP)+1:02d}_Wh' for i in range(len(PWR))]] = 0
    weather = data.loc[:, ['Text', 'GHI']]
    weather.loc[:, 'GHI'] /= 1000
    return pd.concat((temperatures, powers, weather), axis=1)


def split_dataset(dataset: pd.DataFrame, test_percentage: float = 0.2) \
        -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split the dataset into training and test set (in days).

    Args:
        dataset (pd.DataFrame): The dataset to split.
        test_percentage (float, optional): The percentage of the test set. Defaults to 0.2.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: The training set 
            and the test set in dataframe format and 
        numpy array for the MRI method.
    """
    # Split dataset into training and test set.
    day = int(86400 / dataset.index.freq.delta.total_seconds())
    split = max((dataset.shape[0] * test_percentage) // day, 1)
    train = dataset.iloc[:-int(split*day), :]
    test = dataset.iloc[-int(split*day)-1:, :]
    # Create 3d numpy array with a rolling window of 1 day
    groups = []
    for i in range(train.shape[0]-day):
        groups.append(train.iloc[i:i+day+1, :].values)
    train_day = np.array(groups)
    groups = []
    for i in range(dataset.shape[0]-day):
        groups.append(dataset.iloc[i:i+day+1, :].values)
    test_day = np.array(groups)
    return train, test, train_day, test_day


def create_parameters_and_metrics(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the parameter and metric dataframes.

    If the procedure stopped half way then recover the latest iteration and start from there.

    Args:
        dataset (pd.DataFrame): The timeseries dataset. Used to infer the number of zones 
            and the timestep.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The parameters and metrics dataframes.
    """
    # If the path exists then the process finished half way. Read the saved files and continue
    if pathlib.Path('Results/Parameters/0.csv').exists():
        iteration = sorted([int(file.stem) for file in pathlib.Path('Results/Parameters').iterdir()
                            if file.is_file()])[-1]
        metrics = pd.concat([pd.read_csv(file, index_col=0)
                             for file in pathlib.Path('Results/Metrics_csvs').iterdir()
                             if file.is_file()], axis=1, sort=True).T
        params = pd.read_csv(pathlib.Path(
            f'Results/Parameters/{iteration}.csv'), index_col=0, header=0)
        iteration += 1
    # Otherwise create the parameters
    else:
        params = pd.DataFrame(
            columns=['value', 'vary', 'min', 'max', 'variable', 'index1', 'index2'])
        # variable: 0 for capacitance, 1 for conductances, 2 for solar, 3 for initial temp
        n_zones = (len(dataset.columns) - 2) // 2
        timestep = dataset.index.freq.delta.total_seconds()
        for i in range(1, n_zones // 2 + 1):
            for j in range(i, n_zones // 2 + 1):
                params.loc[f'U{i:d}{j:d}'] = [1e-2, False, 1e-3, 1e4, 1, i, j]
                params.loc[f'U{i:d}{i+n_zones//2:d}'] = [1,
                                                         False, 1e-1, 1e4, 1, i, i+n_zones//2]
            params.loc[f'U{i+n_zones//2:d}{i+n_zones//2:d}'] = \
                [1, False, 1e-3, 1e2, 1, i+n_zones//2, i+n_zones//2]
            params.loc[f'Cdt{i:d}'] = [1e3/timestep, True,
                                       1e-2/timestep, 1e6/timestep, 0, i, i]
            params.loc[f'Cdt{i+n_zones//2:d}'] = \
                [1e5/timestep, False, 1/timestep, 1e6 /
                    timestep, 0, i+n_zones//2, i+n_zones//2]
            params.loc[f'T{i+n_zones//2:d}_init'] = \
                [dataset.loc[dataset.index[0], f'T{i:02d}_TEMP'] - 3, False,
                 dataset.loc[dataset.index[0], 'Text'],
                 dataset.loc[dataset.index[0], f'T{i:02d}_TEMP'], 3, i+n_zones//2, i+n_zones//2]
            params.loc[f'alpha{i+n_zones//2:d}'] = [1e-1,
                                                    False, 1e-5, 1e1, 2, i+n_zones//2, i+n_zones//2]
            params.loc[f'alpha{i:d}'] = [1e-1, False, 1e-5, 1e2, 2, i, i]
        iteration = 0
        metrics = pd.DataFrame(
            columns=['BIC', 'RMSE', 'variance', 'r2', 'FIT'])
    params = params.astype({'value': 'float64', 'vary': bool,
                           'variable': 'int64', 'index1': 'int64', 'index2': 'int64'})
    return params, metrics, iteration


def print_update(iteration: int, parameter_name: str, metrics: pd.DataFrame) -> None:
    """Print an update when the iteration finishes

    Args:
        iteration (int): Iteration number
        parameter_name (str): Name of parameter that was added or removed
        metrics (pd.DataFrame): Dataframe of metrics
    """
    print(f'{str(iteration):^6s}|{parameter_name:^9s}|{metrics.BIC.iloc[-1]:^11.0f}' +
          f'|{metrics.RMSE.iloc[-1]:^11.5f}|{metrics.r2.iloc[-1]:^11.5f}|' +
          f'{metrics.variance.iloc[-1]:^11.5f}|{metrics.FIT.iloc[-1]:^11.3f}|')


def add_or_remove(params: pd.DataFrame, new_parameter: str, train_np: np.ndarray,
                  train: pd.DataFrame, number_of_samples: int, add: bool = True) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add or remove the parameter, calibrate teh model and compute its metrics

    Args:
        params (pd.DataFrame): The model parameters
        new_parameter (str): The parameter to add or remove
        train_np (np.ndarray): The training set in numpy format
        train (pd.DataFrame): The training set in pandas format
        number_of_samples (int): The number of initial guesses to sample from the latin hypercube.
            The optimization runs as many times as the samples of the initial guesses
        add (bool, optional): Whether to add the paramteter or to remove it. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the new parameter dataframe and the corresponding metrics
    """
    params_temp = params.copy()
    n_zones = (train_np.shape[2]-2) // 2
    # If the node is a thermal capacitance then add or remove all the varibales associated with it
    if new_parameter[:3] == 'Cdt':
        i = int(new_parameter[3:])
        if i > n_zones//2:
            new_parameter = [new_parameter, f'U{i}{i}',
                             f'U{i-n_zones//2}{i}',
                             f'alpha{i}', f'T{i:d}_init']
    params_temp.loc[new_parameter, 'vary'] = add
    # Calibrate the model and calculate the new parameters
    params_temp.loc[params_temp.vary, 'value'] = calibrate(
        params_temp, train_np, number_of_samples)
    # Calculate the metrics
    new_metrics = get_metrics_and_plots(params_temp, train_np, train, None, False, False)
    return params_temp, new_metrics


def forward_loop(params: pd.DataFrame, metrics: pd.DataFrame, iteration: int, train_np: np.ndarray,
                 train: pd.DataFrame, number_of_samples: int, save_metrics: bool = True,
                 plot_metrics: bool = False, parallel: bool = False) \
        -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Perform a forward selection iteration

    Args:
        params (pd.DataFrame): The model parameters
        metrics (pd.DataFrame): The metrics database
        iteration (int): The iteration number
        train_np (np.ndarray): The training set in numpy format
        train (pd.DataFrame): The training set in pandas format
        number_of_samples (int): The number of initial guesses to sample from the latin hypercube.
            The optimization runs as many times as the samples of the initial guesses
        save_metrics (bool, optional): Whether you want to save the metrics in folders
            Parameters and Metrics_csvs. Defaults to True.
        plot_metrics (bool, optional): Whether you want to save the plots in folders 
            Metrics_plots and Temepratures. Defaults to False.
        parallel (bool, optional): Whether to run in parallel using all but one cores of CPU.
            Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, int]: the new parameter dataframe, the corresponding 
            metrics and the iteration number
    """
    # Generate the parameter list
    param_list = params[~params.vary].index.to_list()
    n_zones = (train_np.shape[2]-2) // 2
    for i in range(n_zones//2+1, n_zones+1):
        if f'Cdt{i}' in param_list:
            param_list.remove(f'U{i}{i}')
            param_list.remove(f'U{i-n_zones//2}{i}')
            param_list.remove(f'alpha{i}')
            param_list.remove(f'T{i:d}_init')
    # If there are no more parameters to add exit
    if not param_list:
        return params, metrics, iteration
    new_parameters = []
    new_metrics = []
    # If in parallel run a pool with multiprocessing
    if parallel:
        with mp.Pool(mp.cpu_count() - 1) as pool:
            for result in pool.starmap(add_or_remove, [
                    [params, param, train_np, train, number_of_samples] for param in param_list]):
                new_parameters.append(result[0])
                new_metrics.append(result[1])
    # Otherwise manually iterate over all parameters to add and calibrate the model
    # and get its metrics
    else:
        for param in param_list:
            new_param, metric = add_or_remove(
                params, param, train_np, train, number_of_samples)
            new_parameters.append(new_param)
            new_metrics.append(metric)
    new_metrics = np.array(new_metrics)
    # Compare the lowest BIC to the base BIC. If it is lower
    if new_metrics[:, 0].min() < metrics.loc[metrics.index[-1], 'BIC']:
        # Choose the best model
        chosen = new_metrics[:, 0].argmin()
        # calculate again its metrics so that you can save them and plot them if needed
        metrics.loc[f'{iteration}', :] = get_metrics_and_plots(
            new_parameters[chosen], train_np, train, iteration, save_metrics, plot_metrics)
        # Print a message about the result
        print_update(iteration, f'+{param_list[chosen]}', metrics)
        iteration = iteration + 1
        return new_parameters[chosen], metrics, iteration
    # Otherwise exit
    return params, metrics, iteration


def backward_loop(params: pd.DataFrame, metrics: pd.DataFrame, iteration: int, train_np: np.ndarray,
                  train: pd.DataFrame, number_of_samples: int, save_metrics: bool = True,
                  plot_metrics: bool = False, parallel: bool = False) \
        -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Perform a backward selection iteration

    Args:
        params (pd.DataFrame): The model parameters
        metrics (pd.DataFrame): The metrics database
        iteration (int): The iteration number
        train_np (np.ndarray): The training set in numpy format
        train (pd.DataFrame): The training set in pandas format
        number_of_samples (int): The number of initial guesses to sample from the latin hypercube.
            The optimization runs as many times as the samples of the initial guesses
        save_metrics (bool, optional): Whether you want to save the metrics in folders
            Parameters and Metrics_csvs. Defaults to True.
        plot_metrics (bool, optional): Whether you want to save the plots in folders 
            Metrics_plots and Temepratures. Defaults to False.
        parallel (bool, optional): Whether to run in parallel using all but one cores of CPU.
            Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, int]: the new parameter dataframe, the corresponding 
            metrics and the iteration number
    """
    # Generate the parameter list
    param_list = params[params.vary].index.to_list()
    n_zones = (train_np.shape[2]-2) // 2
    for i in range(n_zones//2+1, n_zones+1):
        # If you have an envelope node then delete either the alpha or the Unne
        # or the node itself separately
        if f'Cdt{i}' in param_list:
            param_list.remove(f'U{i-n_zones//2}{i}')
            param_list.remove(f'T{i:d}_init')
            if f'alpha{i}' in param_list and f'U{i}{i}' not in param_list:
                param_list.remove(f'alpha{i}')
            elif f'alpha{i}' not in param_list and f'U{i}{i}' in param_list:
                param_list.remove(f'U{i}{i}')
    if not param_list:
        return params, metrics, iteration
    new_parameters = []
    new_metrics = []
    # If in parallel run a pool with multiprocessing
    if parallel:
        with mp.Pool(mp.cpu_count() - 1) as pool:
            for result in pool.starmap(add_or_remove, [
                [params, param, train_np, train, number_of_samples, False]
                    for param in param_list]):
                new_parameters.append(result[0])
                new_metrics.append(result[1])
    # Otherwise manually iterate over all parameters to add and calibrate the model
    # and get its metrics
    else:
        for param in param_list:
            new_param, metric = add_or_remove(
                params, param, train_np, train, number_of_samples, add=False)
            new_parameters.append(new_param)
            new_metrics.append(metric)
    new_metrics = np.array(new_metrics)
    # Compare the lowest BIC to the base BIC. If it is lower
    if new_metrics[:, 0].min() < metrics.loc[metrics.index[-1], 'BIC']:
        # Choose the best model
        chosen = new_metrics[:, 0].argmin()
        # calculate again its metrics so that you can save them and plot them if needed
        metrics.loc[f'{iteration}', :] = get_metrics_and_plots(
            new_parameters[chosen], train_np, train, iteration, save_metrics, plot_metrics)
        # Print a message about the result
        print_update(iteration, f'-{param_list[chosen]}', metrics)
        iteration = iteration + 1
        return new_parameters[chosen], metrics, iteration
    # Otherwise exit
    else:
        return params, metrics, iteration


def finalize(params: pd.DataFrame, metrics: pd.DataFrame, test_np: np.ndarray,
             test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finalize the model.

    Get its final metrics, plot the temperatures and save the A, B, and C matrices.

    Args:
        params (pd.DataFrame): The model parameters
        metrics (pd.DataFrame): The metrics database
        test_np (np.ndarray): The test set in numpy format
        test (pd.DataFrame): The test set in pandas format

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The A, B, and C matrices
    """
    # Create folder to save final metrics
    pathlib.Path('Results/final').mkdir(parents=True, exist_ok=True)
    # Get metrics and plots
    metrics.loc['final', :] = get_metrics_and_plots(
        params, test_np, test, 'final', True, True)
    # Copy them from the respective folders to the final one
    shutil.copy2(pathlib.Path('Results/Metrics_plots/final.html'),
                 pathlib.Path('Results/final/metrics.html'))
    shutil.copy2(pathlib.Path('Results/Metrics_csvs/final.csv'),
                 pathlib.Path('Results/final/metrics.csv'))
    shutil.copy2(pathlib.Path('Results/Parameters/final.csv'),
                 pathlib.Path('Results/final/parameters.csv'))
    shutil.copy2(pathlib.Path('Results/Temperatures/final.html'),
                 pathlib.Path('Results/final/temperatures.html'))
    pathlib.Path('Results/Metrics_plots/final.html').unlink()
    pathlib.Path('Results/Metrics_csvs/final.csv').unlink()
    pathlib.Path('Results/Parameters/final.csv').unlink()
    pathlib.Path('Results/Temperatures/final.html').unlink()
    # Print an update message
    print_update('final', '---', metrics)
    param_values = params.loc[params.vary, 'value'].values
    param_variables = params.loc[params.vary, 'variable'].values
    param_index1 = params.loc[params.vary, 'index1'].values
    param_index2 = params.loc[params.vary, 'index2'].values
    n_zones = (test_np.shape[2] - 2) // 2
    # Calculate A, B, and C matrices
    matrix_A, matrix_B, matrix_C, _ = create_matrices(
        param_values, param_variables, param_index1, param_index2, n_zones)
    # Save them in the final folder
    np.savez(pathlib.Path('Results/final/matrices.npz'),
             A=matrix_A, B=matrix_B, C=matrix_C)
    return matrix_A, matrix_B, matrix_C


if __name__ == '__main__':
    # Define constants for the run
    HOUSE_PATH = pathlib.Path('data_in/house_data.csv')
    NUMBER_OF_SAMPLES = 1
    SAVE_METRICS = True
    PLOT_METRICS = True
    PARALLEL = True

    # Load house data
    house_data = load_data(HOUSE_PATH)
    # Split it into training set and test set in pandas and numpy format
    train_set, test_set, train_set_np, test_set_np = split_dataset(house_data)
    # Create folders to save progress
    for folder in ['Results/Metrics_plots', 'Results/Parameters', 'Results/Metrics_csvs',
                   'Results/Temperatures']:
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    # Create the parameters, metrics and iteration either from scratch or
    # read them from saved progress
    parameters, house_metrics, house_iteration = create_parameters_and_metrics(
        house_data)
    # Plot an interactive file of the available data
    if not pathlib.Path('Results/original_data.html').exists():
        plot_data(train_set, test_set, 'original_data')
    if not pathlib.Path('Results/final/matrices.npz').exists():
        print(' ite  |  param  |    BIC    |   RMSE    |    r2     | variance  |    FIT    |')
        # If it's the first iteration then calibrate the simplest model possible
        # and calculate its metrics
        if house_iteration == 0:
            parameters.loc[parameters.vary, 'value'] = calibrate(
                parameters, train_set_np, NUMBER_OF_SAMPLES)
            house_metrics.loc[f'{house_iteration}', :] = get_metrics_and_plots(
                parameters, train_set_np, train_set, house_iteration, SAVE_METRICS, PLOT_METRICS)
            print_update(house_iteration, '---', house_metrics)
            house_iteration += 1
        # Start outer loop
        iteration_outer_loop = house_iteration - 1
        while iteration_outer_loop != house_iteration:
            iteration_outer_loop = house_iteration
            # Move forward once
            parameters, house_metrics, house_iteration = forward_loop(
                parameters, house_metrics, house_iteration, train_set_np, train_set,
                NUMBER_OF_SAMPLES, SAVE_METRICS, PLOT_METRICS, PARALLEL)
            # Start inner loop. For every forward step we check for redundant parameters
            iteration_inner_loop = house_iteration - 1
            while iteration_inner_loop != house_iteration:
                iteration_inner_loop = house_iteration
                parameters, house_metrics, house_iteration = backward_loop(
                    parameters, house_metrics, house_iteration, train_set_np, train_set,
                    NUMBER_OF_SAMPLES, SAVE_METRICS, PLOT_METRICS, PARALLEL)
        A, B, C = finalize(parameters, house_metrics, test_set_np, test_set)
    else:
        A, B, C = np.load(pathlib.Path('Results/final/matrices.npz'))
