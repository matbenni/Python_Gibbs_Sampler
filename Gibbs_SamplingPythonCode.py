import numpy as np
import pandas as pd


def gibbs_horseshoe(y, X, N_sims):

    Xt = np.transpose(X)
    XtX = np.dot(Xt, X)
    Xty = np.dot(Xt, y)
    p = np.shape(X)[1]  # Number of columns
    n = np.shape(X)[0]  # Number of rows

    tau_out = beta_out = np.zeros((p, N_sims + 1))
    sigma_sq_out = np.zeros((N_sims + 1))

    tau_loc = lambda_loc = np.ones(p)
    phi_loc = eta_loc = 1

    diag_p = np.zeros((p, p))
    diag_tau = np.zeros((p, p))

    diag_p[np.diag_indices_from(diag_p)] = p

    beta_loc = np.dot(np.linalg.inv(XtX + (diag_p) / n), Xty)
    sigma_sq_loc = (1 / n) * (np.dot(y, y) - np.dot(Xty, beta_loc))  # [1,1]

    tau_out[:, 1] = tau_loc
    beta_out[:, 1] = beta_loc
    sigma_sq_out[1] = sigma_sq_loc

    # begin for loop
    for i in range(0, N_sims):
        diag_tau[np.diag_indices_from(diag_tau)] = tau_loc
        # tau_loc is the diagonal of p x p matrix
        SIGMA = np.linalg.inv(XtX + diag_tau)
        beta_loc = beta_out[:, i] = np.array(
            [np.random.multivariate_normal(np.dot(SIGMA, Xty), sigma_sq_loc * SIGMA, 1)])
        beta_loc = beta_loc.flatten()
        sigma_sq_loc = sigma_sq_out[i] = 1 / np.random.gamma((n + p + 1) / 2, (1 / 2) * np.dot(y - np.dot(X, beta_loc),
                                                                                               y - np.dot(X,
                                                                                                          beta_loc)) + np.dot(
            tau_loc * beta_loc, beta_loc) + 1, 1)
        tau_loc = tau_out[:, i] = np.random.exponential(1, p) / (beta_loc ** 2 / (2 * sigma_sq_loc) + lambda_loc / 2)
        lambda_loc = np.random.exponential(1, p) / (tau_loc / 2 + phi_loc / 2)
        phi_loc = np.random.gamma((p + 1) / 2, sum(lambda_loc) / 2 + eta_loc / 2, 1)
        eta_loc = np.random.exponential(phi_loc / 2 + 1 / 2, 1)

    return (beta_out, sigma_sq_out, tau_out)
    # end for loop

filename = 'diabetes.csv'
data = pd.read_csv(filename)
y_loc = data['y']
x2_loc = data.drop(columns=['y', 'Unnamed: 0'])

out = gibbs_horseshoe(y_loc, x2_loc, 100)
out
