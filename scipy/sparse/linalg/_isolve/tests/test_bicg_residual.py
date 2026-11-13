"""Regression tests for BiCG residual replacement (gh-26504)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scipy.linalg import norm
from scipy.sparse import csr_array, diags_array
from scipy.sparse.linalg import LinearOperator, bicg


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64,
                                   np.complex128])
@pytest.mark.parametrize("preconditioned", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_bicg_large_initial_guess(dtype, preconditioned, sparse):
    # Recursively updated residuals lose the low-order digits of the solution
    # when the initial guess is large. A restart can recover the requested
    # accuracy, since this system itself is well conditioned.
    low_precision = np.empty((), dtype=dtype).real.dtype == np.dtype(np.float32)
    magnitude = 1e5 if low_precision else 1e12
    rtol = 1e-5 if low_precision else 1e-10
    A = np.array([[4, 1], [2, 3]], dtype=dtype)
    b = np.array([1, 2], dtype=dtype)
    x0 = np.array([magnitude, -magnitude], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        A *= 1 + 0.25j
        b *= 1 - 0.5j
        x0 *= 1 + 0.75j
    M = np.diag(1 / A.diagonal()) if preconditioned else None
    if sparse:
        A = csr_array(A)
        M = csr_array(M) if M is not None else None
    initial = x0.copy()
    x, info = bicg(A, b, x0=x0, M=M, rtol=rtol, maxiter=50)
    assert info == 0
    assert np.linalg.norm(b - A @ x) <= rtol * np.linalg.norm(b)
    assert_allclose(x0, initial, rtol=0, atol=0)
    assert x.dtype == dtype


def test_bicg_unattainable_tolerance():
    # Original input from gh-26504. The true residual stagnates well above the
    # recurrence's residual. It must not be reported as successful convergence.
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((50, 50)))
    A = Q @ np.diag(np.r_[np.ones(25), np.full(25, 1e10)]) @ Q.T
    A = (A + A.T) / 2
    b = A @ Q[:, 0]
    iterations = []
    x, info = bicg(A, b, rtol=1e-8, maxiter=100,
                       callback=lambda x: iterations.append(x.copy()))
    assert info != 0 or np.linalg.norm(b - A @ x) <= 1e-8 * np.linalg.norm(b)
    assert len(iterations) <= 100
    assert np.isfinite(x).all()


@pytest.mark.parametrize("maxiter", [1, 2, 3])
def test_bicg_restart_iteration_budget(maxiter):
    A = np.array([[4., 1.], [2., 3.]])
    b = np.array([1., 2.])
    iterations = []
    x, info = bicg(A, b, x0=[1e12, -1e12], rtol=1e-10,
                       maxiter=maxiter,
                       callback=lambda x: iterations.append(x.copy()))
    assert len(iterations) <= maxiter
    assert info != 0 or np.linalg.norm(b - A @ x) <= 1e-10 * np.linalg.norm(b)
    if info > 0:
        assert info == maxiter


def test_bicg_last_iteration_convergence():
    # This reaches tolerance on the last allowed iteration.
    A = np.diag([1., 2.])
    b = np.array([1., 1.])
    x, info = bicg(A, b, rtol=0.4, maxiter=1)
    assert info == 0
    assert np.linalg.norm(b - A @ x) <= 0.4 * np.linalg.norm(b)


def test_bicg_verification_matvec_cost():
    # Verifying success needs only one additional operator application on a
    # normally converging system; it must not add one on every iteration.
    n = 100
    matrix = diags_array([-np.ones(n-1), 4*np.ones(n), -np.ones(n-1)],
                         offsets=[-1, 0, 1], format="csr")
    calls = []
    iterations = []

    def matvec(x):
        calls.append(None)
        return matrix @ x

    A = LinearOperator(matrix.shape, matvec=matvec, rmatvec=matvec, dtype=matrix.dtype)
    b = np.ones(n)
    x, info = bicg(A, b, rtol=1e-10,
                       callback=lambda x: iterations.append(None))
    assert info == 0
    assert np.linalg.norm(b - matrix @ x) <= 1e-10 * np.linalg.norm(b)
    # Every step applies A and its adjoint once; verification applies A once.
    assert len(calls) <= 2 * len(iterations) + 1


@pytest.mark.parametrize("maxiter", [0, -1])
def test_bicg_nonpositive_iteration_budget(maxiter):
    with pytest.raises(ValueError, match="maxiter must be greater than 0"):
        bicg(np.eye(2), [1., 2.], maxiter=maxiter)


def test_bicg_overflowed_recurrence_is_not_success():
    # The inputs are finite but dot products in the recurrence can overflow.
    # In particular, inf <= inf must not accept the zero initial guess.
    A = np.diag([2., 3.])
    b = np.array([1e200, 2e200])
    with np.errstate(over="ignore", invalid="ignore"):
        x, info = bicg(A, b, rtol=1e-8, maxiter=5)
    assert info != 0 or norm(b - A @ x) <= 1e-8 * norm(b)


def test_bicg_large_finite_exact_initial_guess():
    # Stable residual norms must still accept an exact, finite large solution.
    A = np.diag([2., 3.])
    x0 = np.array([1e200, 2e200])
    b = A @ x0
    x, info = bicg(A, b, x0=x0, rtol=1e-8)
    assert info == 0
    assert_allclose(x, x0, rtol=0, atol=0)


@pytest.mark.parametrize("dtype, magnitude", [(np.float32, 3e38),
                                             (np.float64, 1.3e308)])
@pytest.mark.parametrize("exact_guess", [False, True])
def test_bicg_overflowed_rhs_norm(dtype, magnitude, exact_guess):
    # All entries are finite, but their norm is unrepresentable. Only a
    # verified zero residual may establish convergence in this case.
    A = np.eye(2, dtype=dtype)
    b = np.full(2, magnitude, dtype=dtype)
    x0 = b if exact_guess else None
    with np.errstate(over="ignore", invalid="ignore"):
        x, info = bicg(A, b, x0=x0, rtol=1e-5, maxiter=1)
    if exact_guess:
        assert info == 0
        assert_allclose(x, b, rtol=0, atol=0)
    else:
        assert info != 0
