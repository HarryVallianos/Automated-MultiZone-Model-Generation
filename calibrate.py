"""This module contains functions to calibrate thermal models using the implicit method 
    and a custom jacobian."""
import pandas as pd
import scipy.stats as st
import scipy.optimize as op
import numpy as np
import numpy.linalg as nlg
from numba import njit


def calibrate(parameters: pd.DataFrame, train_np: np.ndarray, n_samples: int = 5) -> np.ndarray:
    """Calibrate the model using the varying parameters and the training dataset

    Args:
        parameters (pd.DataFrame): The model parameters
        train (np.ndarray): The training dataset in numpy format
        n_samples (int, optional): How many time to run the optimization with different
            initial conditions form a latin hypercube. Defaults to 5.

    Returns:
        np.ndarray: The calibrated model parameters
    """
    # Get bounds, values, variables and indices for parameters
    bounds = parameters.loc[parameters.vary, ['min', 'max']].values
    x0 = parameters.loc[parameters.vary, 'value'].values
    param_variables = parameters.loc[parameters.vary, 'variable'].values
    param_index1 = parameters.loc[parameters.vary, 'index1'].values
    param_index2 = parameters.loc[parameters.vary, 'index2'].values
    # If we want more than one sample then get it from a latin hypercube
    if n_samples > 1:
        n_samples_total = n_samples - 1
        sampler = st.qmc.LatinHypercube(
            d=len(x0), optimization='random-cd', seed=0)
        lower = np.clip(x0 * 10 ** (-3 * np.sign(x0)),
                        bounds[:, 0]+1e-10, bounds[:, 1]-1e-10)
        upper = np.clip(x0 * 10 ** (3 * np.sign(x0)),
                        bounds[:, 0]+1e-10, bounds[:, 1]-1e-10)
        samples = st.qmc.scale(sampler.random(n_samples_total), lower, upper)
        samples = np.vstack((x0, samples))
    else:
        samples = [x0]
    results = np.ones(n_samples) * 1e12
    result_parameters = np.zeros((n_samples, len(x0)))
    for i, sample in enumerate(samples):
        try:
            # Calibrate the model using least_squares and the custom jacobian
            res = op.least_squares(residuals, sample,
                                   args=(param_variables, param_index1,
                                         param_index2, train_np),
                                   bounds=(bounds[:, 0], bounds[:, 1]),
                                   jac=residuals_jac,  # verbose=2,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=2000)
            # Calibrate the model using least_squares without a custom jacobian and using as initial
            # condition the result of the previous calibration. Just in case the custom jacobian is
            # a bit off
            res = op.least_squares(residuals, res.x,
                                   args=(param_variables, param_index1,
                                         param_index2, train_np),
                                   bounds=(bounds[:, 0], bounds[:, 1]),
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=2000)
            results[i] = res.cost
            result_parameters[i, :] = res.x
        except:
            pass
    return result_parameters[results.argmin(), :]

# Custom numba function for np.ix_ found from https://github.com/numba/numba/issues/5894


@njit('float64[:,::1](float64[:,::1], int64[::1], int64[::1])', cache=True)
def numba_ix(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.

    Args:
        arr (np.ndarray): 2D array to be indexed
        rows (np.ndarray): Row indices
        cols (np.ndarray): Column indices

    Returns:
        np.ndarray: array with the given rows and columns of the input array
    """

    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start: start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))

# Use numba to jit the function and make it much faster also cache to make it even faster.
# Comment cache out if you want to debug.


@njit('Tuple((float64[:,::1], float64[:,::1], float64[:,::1], int64[::1]))(float64[::1], \
      int64[::1], int64[::1], int64[::1], int64)', cache=True)
def create_matrices(param_values: np.ndarray, param_variables: np.ndarray,
                    param_index1: np.ndarray, param_index2: np.ndarray, n_zones: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create matrices A, B, and C of state space T[t+1] = AT[t] + BQ_h + C[T_ext,Q_sol].
        The discretization is done with the implicit method.

    Args:
        param_values (np.ndarray): Array with the value of each parameter
        param_variables (np.ndarray): Array with an integer indicating the kind 
            of variable each parameter is
        param_index1 (np.ndarray): Array with an integer indicating the first index
            associated with the variable
        param_index2 (np.ndarray): Array with an integer indicating the second index
            associated with the variable
        n_zones (int64): Number of zones

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Matrices A, B, and C
            and an array indicating which indices to keep
            (omitting envelope nodes that don't exist)
    """
    # Initialize the matrices of AT[t+1] = B_tempT[t] + B_extT_ext + B_solQ_sol+Q_h

    A = np.zeros((n_zones, n_zones), dtype=np.float64)
    B_ext = np.zeros(n_zones, dtype=np.float64)
    B_sol = np.zeros(n_zones, dtype=np.float64)
    B_temp = np.zeros((n_zones, n_zones), dtype=np.float64)
    # Add values from capacitances
    index = param_index1[param_variables == 0] - 1
    for ind, value in zip(index, param_values[param_variables == 0]):
        B_temp[ind, ind] += value
        A[ind, ind] += value
    # Add values from conductances with the same indices, e.g., U11
    index = param_index1[np.logical_and(
        param_variables == 1, param_index1 == param_index2)] - 1
    B_ext[index] += param_values[np.logical_and(
        param_variables == 1, param_index1 == param_index2)]
    for ind, value in zip(index, param_values[np.logical_and(
            param_variables == 1, param_index1 == param_index2)]):
        B_temp[ind, ind] -= value
    # Add values from conductances with different indices, e.g., U12
    index1 = param_index1[np.logical_and(
        param_variables == 1, param_index1 != param_index2)] - 1
    index2 = param_index2[np.logical_and(
        param_variables == 1, param_index1 != param_index2)] - 1
    for in1, in2, value in zip(index1, index2, param_values[np.logical_and(
            param_variables == 1, param_index1 != param_index2)]):
        A[in1, in2] -= value
        A[in2, in1] -= value
        A[in1, in1] += value
        A[in2, in2] += value
    # Add values from solar factors
    index = param_index1[param_variables == 2] - 1
    B_sol[index] += param_values[param_variables == 2]
    index_to_keep = np.union1d(np.where(A.sum(axis=0) != 0)[
                               0], np.arange(n_zones//2))
    # Invert matrix A
    try:
        A_inv = nlg.inv(numba_ix(A, index_to_keep, index_to_keep))
    except:
        A_inv = nlg.pinv(numba_ix(A, index_to_keep, index_to_keep))
    A = A_inv @ numba_ix(B_temp, index_to_keep, index_to_keep)
    B = A_inv
    C = np.column_stack(
        (A_inv @ B_ext[index_to_keep], A_inv @ B_sol[index_to_keep]))
    return A, np.ascontiguousarray(B), C, index_to_keep


@njit('Tuple((float64[:,:,::1], float64[:,:,::1], float64[:,:], float64[:,:], \
    float64[:,:,::1]))(float64[:,::1], float64[:,::1], float64[:,::1], float64[:,:,:],\
        int64)', cache=True)
def get_results(A: np.ndarray, B: np.ndarray, C: np.ndarray, data: np.ndarray,
                n_zones: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return errors, temperatures, exterior temperature, solar radiation and sources.

    The errors are used for the optimization and the rest are used to calculate the jacobian.

    Args:
        A (np.ndarray): A matrix of T[t+1] = AT[t] + BQ_h + C[T_ext,Q_sol]
        B (np.ndarray): B matrix of T[t+1] = AT[t] + BQ_h + C[T_ext,Q_sol]
        C (np.ndarray): C matrix of T[t+1] = AT[t] + BQ_h + C[T_ext,Q_sol]
        data (np.ndarray): The data in numpy format
        n_zones (int): The number of zones

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Errors, 
        temperatures, exterior temperature, solar radiation and sources.
    """
    # Initialize arrays
    n_zones_air = n_zones // 2
    n_zones_tot = (data.shape[2]-2) // 2
    temperatures = np.zeros((*data.shape[:2], n_zones_tot))
    Text = np.zeros((data.shape[0], data.shape[1]-1))
    Qsol = np.zeros((data.shape[0], data.shape[1]-1))
    sources = np.zeros((data.shape[0], data.shape[1]-1, n_zones_tot))
    # Iterate over the timesteps
    for idx in range(data.shape[0]):
        # Calculate sources over 24 hours
        source = np.ascontiguousarray(
            data[idx, :-1, n_zones_tot:]) @ np.vstack((B, C.T))
        sources[idx, :, :] = source
        temperature = np.zeros((data.shape[1], n_zones_tot))
        temperature[0, :] = np.ascontiguousarray(data[idx, 0, :n_zones_tot])
        # Calculate temperature over 24 hours
        for t in range(data.shape[1]-1):
            temperature[t+1, :] = A @ temperature[t, :] + source[t, :]
        # Initial temperature of envelope nodes at the next timestep is equal to what was
        # calculated at the previous timestep
        data[idx, :, n_zones_air:n_zones_tot] = temperature[:,
                                                            n_zones_air:n_zones_tot]
        if idx < data.shape[0] - 1:
            data[idx+1, 0, n_zones_air:n_zones_tot] = temperature[1,
                                                                  n_zones_air:n_zones_tot]
        temperatures[idx, :, :] = temperature
    errors = temperatures[:, 1:, :n_zones_air].copy()
    errors -= data[:, 1:, :n_zones_air]
    Text = data[:, :-1, -2]
    Qsol = data[:, :-1, -1]
    return errors, temperatures, Text, Qsol, sources


def residuals(param_values: np.ndarray, param_variables: np.ndarray, param_index1: np.ndarray,
              param_index2: np.ndarray, data_np: np.ndarray, flattened: bool = True) \
        -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Calculate the residuals

    Args:
        param_values (np.ndarray): Array with the value of each parameter
        param_variables (np.ndarray): Array with an integer indicating the kind of variable
            each parameter is
        param_index1 (np.ndarray): Array with an integer indicating the first index associated
            with the variable
        param_index2 (np.ndarray): Array with an integer indicating the second index associated
            with the variable
        data_np (np.ndarray): The data in numpy format
        flattened (bool, optional): Whether the errors should be flattened to be used in the
            least squares. Defaults to True.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: Either the flattened residuals to be used
            in the least squares or a tuple of the unflattened residuals and the predicted
            temperatures
    """
    data = data_np.copy()
    # Initialize the envelope nodes
    data[0, 0, param_index1[param_variables == 3] -
         1] = param_values[param_variables == 3]
    n_zones = (data.shape[2] - 2) // 2
    # Calculate state-space matrices
    A, B, C, index_to_keep = create_matrices(
        param_values, param_variables, param_index1, param_index2, n_zones)
    # Reduce the dataset to only used envelope nodes
    index_to_keep = np.concatenate(
        (index_to_keep, index_to_keep+n_zones, n_zones*2+np.array([0, 1])))
    data = data[:, :, index_to_keep]
    # Get the residuals
    errors, temperatures, _, _, _ = get_results(A, B, C, data, n_zones)
    if flattened:
        return errors.flatten()
    return errors, temperatures


def residuals_jac(param_values: np.ndarray, param_variables: np.ndarray,
                  param_index1: np.ndarray, param_index2: np.ndarray, data_np: np.ndarray) \
        -> np.ndarray:
    """Calculate the jacobian of the residuals for the parameter values

    Args:
        param_values (np.ndarray): Array with the value of each parameter
        param_variables (np.ndarray): Array with an integer indicating the kind of variable
            each parameter is
        param_index1 (np.ndarray): Array with an integer indicating the first index associated 
            with the variable
        param_index2 (np.ndarray): Array with an integer indicating the second index associated 
            with the variable
        data_np (np.ndarray): The data in numpy format

    Returns:
        np.ndarray: The jacobian of the residuals
    """
    data = data_np.copy()
    # Initialize the envelope nodes
    data[0, 0, param_index1[param_variables == 3] -
         1] = param_values[param_variables == 3]
    n_zones = (data.shape[2] - 2) // 2
    # Calculate state-space matrices
    A, B, C, index_to_keep = create_matrices(
        param_values, param_variables, param_index1, param_index2, n_zones)
    # Reduce the dataset to only used envelope nodes
    index1 = np.concatenate(
        [np.where(par_index1 == index_to_keep+1)[0] for par_index1 in param_index1]) + 1
    index2 = np.concatenate(
        [np.where(par_index2 == index_to_keep+1)[0] for par_index2 in param_index2]) + 1
    index_to_keep = np.concatenate(
        (index_to_keep, index_to_keep + n_zones, n_zones*2+np.array([0, 1])))
    data = data[:, :, index_to_keep]
    # Get the residuals
    _, temperatures, Text, Qsol, sources = get_results(A, B, C, data, n_zones)
    # Get the jacobian
    jac = get_jac_plain(A, B, data, param_variables, index1,
                        index2, temperatures, Text, Qsol, sources)
    # Return only the air nodes and flatten them
    return jac[:, :n_zones//2, :].reshape(-1, jac.shape[-1])


@njit('float64[:,:,::1](float64[:,::1], float64[:,::1], float64[:,:,:], int64[::1], \
      int64[::1], int64[::1], float64[:,:,::1], float64[:,:], float64[:,:], float64[:,:,::1])',
      cache=True)
def get_jac_plain(A: np.ndarray, B: np.ndarray, data_np: np.ndarray, param_variables: np.ndarray,
                  param_index1: np.ndarray, param_index2: np.ndarray, temperatures: np.ndarray,
                  Text: np.ndarray, Qsol: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Calculate the jacobian matrix having the temperature results

    Args:
        A (np.ndarray): A matrix of T[t+1] = AT[t] + BQ_h + C[T_ext,Q_sol]
        B (np.ndarray): B matrix of T[t+1] = AT[t] + BQ_h + C[T_ext,Q_sol]
        data_np (np.ndarray): The data in numpy format
        param_variables (np.ndarray): Array with an integer indicating the kind of variable each 
            parameter is
        param_index1 (np.ndarray): Array with an integer indicating the first index associated
            with the variable
        param_index2 (np.ndarray): Array with an integer indicating the second index associated 
            with the variable
        temperatures (np.ndarray): The predicted zone temperatures
        Text (np.ndarray): The exterior temperature
        Qsol (np.ndarray): The solar radiation
        sources (np.ndarray): The heating sources

    Returns:
        np.ndarray: jacobian matrix
    """
    # Initialize matrices
    horizon = data_np.shape[1] - 1
    n_zones_tot = (data_np.shape[2]-2) // 2
    temperatures = temperatures.reshape(-1, n_zones_tot)
    Text = Text.flatten()
    Qsol = Qsol.flatten()
    sources = sources.reshape(-1, n_zones_tot)
    jac = np.zeros(
        (sources.shape[0], sources.shape[1], param_variables.shape[0]))
    # Iterate over the 24 hours
    for t in range(horizon):
        # Get the indexes that correspond to the horizon (e.g., all instances of second-step
        # prediction) to use advanced indexing and speed thigns up
        indexer = np.arange(t, Text.shape[0], horizon)
        indexer_total = np.arange(t, temperatures.shape[0], horizon+1)

        # Calculate jacobian from thermal capacitance
        par = param_variables == 0
        index = param_index1[par] - 1
        for i, ind in zip(np.where(par)[0], index):
            for j in range(jac.shape[1]):
                jac[indexer, j, i] += (temperatures[indexer_total, ind] -
                                       temperatures[indexer_total+1, ind]) * B[ind, j]

        # Calculate jacobian from thermal conductance when the two indeces are the same, e.g., U11
        par = np.logical_and(param_variables == 1,
                             param_index1 == param_index2)
        index = param_index1[par] - 1
        for i, ind in zip(np.where(par)[0], index):
            for j in range(jac.shape[1]):
                jac[indexer, j, i] += (-temperatures[indexer_total,
                                       ind] + Text[indexer]) * B[ind, j]

        # Calculate jacobian from thermal conductance when the indeces are not the same, e.g., U12
        par = np.logical_and(param_variables == 1,
                             param_index1 != param_index2)
        index1 = param_index1[par] - 1
        index2 = param_index2[par] - 1
        for i, ind1, ind2 in zip(np.where(par)[0], index1, index2):
            for j in range(jac.shape[1]):
                jac[indexer, j, i] += (
                    temperatures[indexer_total+1, ind2] - temperatures[indexer_total+1, ind1]) \
                    * (B[ind1, j] - B[ind2, j])

        # Calculate jacobian from solar factors
        par = param_variables == 2
        index = param_index1[par] - 1
        for i, ind in zip(np.where(par)[0], index):
            for j in range(jac.shape[1]):
                jac[indexer, j, i] += Qsol[indexer] * B[ind, j]

    for t in range(data_np.shape[0]):
        for j in range(horizon):
            # Calculate jacobian from initial guess of envelope nodes
            if t == 0 and j == 0:
                par = param_variables == 3
                index = param_index1[par] - 1
                for i, ind in zip(np.where(par)[0], index):
                    jac[0, :, i] += A[:, ind]
            # Calculate jacobian from initial guess of envelope nodes and recurrent terms
            elif t > 0 and j == 0:
                par = param_variables == 3
                index = param_index1[par] - 1
                for i, ind in zip(np.where(par)[0], index):
                    jac[horizon*t, ...] += (
                        np.ascontiguousarray(np.expand_dims(A[:, ind], 1)) @
                        np.ascontiguousarray(np.expand_dims(jac[horizon*(t-1), ind, :], 0)))
            # Calculate jacobian from recurrent terms
            else:
                jac[horizon*t+j, ...] += A @ jac[horizon*t+j-1, ...]
    return jac
