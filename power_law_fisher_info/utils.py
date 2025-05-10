import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import gamma, zeta


def finite_zeta(alpha: NDArray[np.float64], N: int, sgm0: float) -> NDArray[np.float64]:
    n = np.arange(1, N + 1).reshape(-1, 1)
    return np.sum((n ** (-alpha)) * np.exp(-(sgm0**2) * (n**2)), axis=0)


def calc_I_ii(
    alpha: NDArray[np.float64], d: int, sgm0: float, sgmi: float
) -> NDArray[np.float64]:
    Vd = np.pi ** (d / 2.0) / gamma(d / 2.0 + 1.0)
    k = 2.0 / d
    I = np.where(
        alpha > 1 + k,
        zeta(alpha - k)
        / ((sgm0**2 * d * Vd ** (2.0 / d) / 4) + (sgmi**2) * zeta(alpha - k)),
        1.0 / (sgmi**2),
    )
    return I


def calc_Vd(d: int):
    return np.pi ** (d / 2.0) / gamma(d / 2.0 + 1.0)


def dzeta_da(alpha: float, eps: float = 1e-6):
    # Numerically calculate the derivative of the zeta function
    return np.diff(zeta([alpha, alpha + eps])) / eps


def calc_theoretical_gamma(d: int, sgm0: float, sgmi: float):
    alpha_c = 1 + 2.0 / d
    Vd = calc_Vd(d)
    return -(sgm0**2) * d * Vd ** (2.0 / d) / (4 * sgmi**4 * dzeta_da(alpha_c))


def generate_xi(K: int, N: int) -> NDArray[np.float64]:
    # neuronal noise
    return np.random.randn(K, N)


def generate_eta(K: int) -> NDArray[np.float64]:
    # stimulus noise
    return np.random.randn(K, 1)


def calc_r(
    alpha: float,
    sgm0: float,
    sgm1: float,
    theta: float,
    eta: NDArray[np.float64],
    xi: NDArray[np.float64],
    n: NDArray[np.float64],
    N: int = 100,
    K: int = 10000,
) -> NDArray[np.float64]:
    xn = n ** (-alpha / 2.0) * np.cos(n * (theta + sgm1 * eta))
    yn = n ** (-alpha / 2.0) * np.sin(n * (theta + sgm1 * eta))
    rn = np.zeros((K, N))
    rn[:, 0::2] = xn
    rn[:, 1::2] = yn
    rn = rn + sgm0 * xi
    return rn


def get_fdt_df(
    alpha_list: NDArray[np.float64],
    theta_list: NDArray[np.float64],
    sgm0: float,
    sgm1: float,
    eps: float,
    K: int,
    N: int,
    M: int,
) -> pd.DataFrame:
    df = pd.DataFrame()
    for i, alpha in enumerate(alpha_list):
        xs1 = []
        ys1 = []
        xs2 = []
        ys2 = []
        for theta in theta_list:
            eta = np.random.randn(K, 1)
            xi = np.random.randn(K, N)
            n = np.arange(1, M + 1).reshape(1, M)

            r0n = calc_r(alpha, sgm0, sgm1, theta, eta, xi, n)
            r1n = calc_r(alpha, sgm0, sgm1, theta + eps, eta, xi, n)

            m0n = r0n.mean(axis=0)
            m1n = r1n.mean(axis=0)

            mun = (m1n - m0n) / eps

            cov = np.cov(r0n.T)

            mm = np.outer(mun, mun)
            # xs = sgm0**2 * mm.reshape(-1)
            length = mm.shape[0]
            non_diag_cov = cov[np.eye(length) == 0].reshape(-1)
            diag_cov = np.diag(cov).reshape(-1)
            non_diag_mm = mm[np.eye(length) == 0].reshape(-1)
            diag_mm = np.diag(mm).reshape(-1)
            # non_diag_cov から length 個分一様サンプル
            idx = np.random.choice(non_diag_cov.shape[0], length)
            idx = np.random.choice(non_diag_cov.shape[0], length)

            xs1 = np.concatenate([xs1, diag_mm])
            xs2 = np.concatenate([xs2, non_diag_mm[idx]])

            ys1 = np.concatenate([ys1, diag_cov])
            ys2 = np.concatenate([ys2, non_diag_cov[idx]])

        for i in range(N):
            _df = pd.DataFrame(
                {
                    "N": N,
                    "theta": theta,
                    "alpha": alpha,
                    "sgm0": sgm0,
                    "sgm1": sgm1,
                    "xs1": xs1[i],
                    "ys1": ys1[i],
                    "xs2": xs2[i],
                    "ys2": ys2[i],
                },
                index=[0],
            )
            df = pd.concat([df, _df], axis=0, ignore_index=True)

    return df
