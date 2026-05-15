from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .exceptions import InvalidConfigurationError, InvalidCycleError
from .validation import validate_series


# Periodogram and frequency-domain helpers.


def compute_document_periodogram(x: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the periodogram with the normalisation used in the notes.

    I(λ_j) = |FFT(x)_j|² / (2πT),  λ_j = 2πj/T,  j = 0, ..., T-1.

    This matches the sin/cos sum formula; |·|² removes the
    phase shift from 0-based vs 1-based indexing.
    """
    arr = validate_series(x, min_length=2)
    T = len(arr)
    fft_vals = np.fft.fft(arr)
    periodogram = np.abs(fft_vals) ** 2 / (2.0 * np.pi * T)
    lambdas = 2.0 * np.pi * np.arange(T, dtype=float) / T
    return lambdas, periodogram


def find_periodogram_peak(
    periodogram: np.ndarray,
    exclude_zero: bool = True,
) -> int:
    """Return R* = argmax I(λ_j).

    With exclude_zero=True, frequency 0 is skipped so the mean does not dominate.
    """
    _validate_periodogram(periodogram, min_length=2 if exclude_zero else 1)
    if exclude_zero:
        return int(np.argmax(periodogram[1:])) + 1
    return int(np.argmax(periodogram))


def find_top_periodogram_peaks(
    periodogram: np.ndarray,
    n_peaks: int,
    exclude_zero: bool = True,
) -> np.ndarray:
    """Return the strongest periodogram peaks, largest first."""
    _validate_find_top_peaks(periodogram, n_peaks, exclude_zero)
    candidates = periodogram[1:] if exclude_zero else periodogram
    offset = 1 if exclude_zero else 0
    top_local = np.argsort(candidates)[-n_peaks:][::-1]
    return top_local + offset


def compute_psi_single_cycle(
    T: int,
    R: int,
    drop_singular_frequency: bool = True,
) -> np.ndarray:
    """Compute ψ(λ_j, R) = log(|2(cos(λ_j) - cos(λ_R))|) for j = 0, ..., T-1.

    The expression is singular at j = R and at its mirrored frequency T-R.
    When drop_singular_frequency=True, both positions are set to 0.0.
    """
    _validate_psi_single_cycle(T, R, drop_singular_frequency)
    j = np.arange(T, dtype=float)
    lambda_j = 2.0 * np.pi * j / T
    lambda_R = 2.0 * np.pi * R / T
    with np.errstate(divide="ignore"):
        psi = np.log(np.abs(2.0 * (np.cos(lambda_j) - np.cos(lambda_R))))
    if drop_singular_frequency:
        psi[R] = 0.0
        mirror = T - R
        if mirror != R:
            psi[mirror] = 0.0
    return psi


def compute_psi_multi_cycle(
    T: int,
    cycles: object,
    drop_singular_frequency: bool = True,
) -> np.ndarray:
    """Reserved for the multi-cycle ψ case."""
    raise NotImplementedError(
        "Multi-cycle psi computation is not implemented yet. "
        "Use stochastic_cycle_mode='single' for the current release."
    )


def compute_xaa_single_cycle(psi: np.ndarray) -> float:
    """Compute XAA(R) = (2/T) * Σ_{j=0}^{T-1} ψ(λ_j, R)².  T = len(psi)."""
    _validate_xaa_single_cycle(psi)
    T = len(psi)
    return float((2.0 / T) * np.sum(psi ** 2))


def compute_xaa_multi_cycle(psi_multi: np.ndarray) -> float:
    """Reserved for the multi-cycle XAA case."""
    raise NotImplementedError(
        "Multi-cycle XAA computation is not implemented yet. "
        "Use stochastic_cycle_mode='single' for the current release."
    )


def compute_residual_periodogram(residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the periodogram of the regression residuals.

    Delegates to compute_document_periodogram; same normalisation I(λ_j) = |FFT|²/(2πT).
    """
    _validate_compute_residual_periodogram(residuals)
    return compute_document_periodogram(residuals)


def compute_frequency_variance_single_cycle(
    I_residuals: np.ndarray,
    R: int,
    drop_frequency: bool = True,
) -> float:
    """Compute VAR*(R,D) = (2π/T) Σ_j I_residuals[j], optionally excluding j = R."""
    _validate_compute_frequency_variance_single_cycle(I_residuals, R)
    T = len(I_residuals)
    mask = np.ones(T, dtype=bool)
    if drop_frequency:
        mask[R] = False
    return float((2.0 * np.pi / T) * np.sum(I_residuals[mask]))


def compute_frequency_variance_multi_cycle(
    I_residuals: np.ndarray,
    cycles: object,
    drop_frequency: bool = True,
) -> float:
    """Compute VAR*(R,D) = (2π/T) Σ_j I_residuals[j], excluding j = R_q for all cycles."""
    _validate_compute_frequency_variance_multi_cycle(I_residuals, cycles)
    T = len(I_residuals)
    mask = np.ones(T, dtype=bool)
    if drop_frequency:
        for cycle in cycles:
            if 0 <= cycle.R < T:
                mask[cycle.R] = False
    return float((2.0 * np.pi / T) * np.sum(I_residuals[mask]))


def compute_frequency_variance_dynamic(
    I_residuals: np.ndarray,
    cycles: tuple,
    mode: str = "single",
    drop_frequency: bool = True,
) -> float:
    """Dispatch to the single- or multi-cycle frequency variance based on mode.

    "multi_peak_single_cycle" uses the single-cycle path; peak selection already happened upstream.
    """
    _validate_compute_frequency_variance_dynamic(cycles, mode)
    if mode in ("single", "multi_peak_single_cycle"):
        return compute_frequency_variance_single_cycle(
            I_residuals, cycles[0].R, drop_frequency
        )
    return compute_frequency_variance_multi_cycle(I_residuals, cycles, drop_frequency)


def compute_psi_dynamic(
    T: int,
    cycles: tuple,
    stochastic_cycle_mode: str,
    drop_singular_frequency: bool = True,
) -> np.ndarray:
    """Choose the ψ calculation for the selected cycle mode.

    "multi_peak_single_cycle" shares the single-cycle path here because the
    peak selection has already happened upstream.
    """
    _validate_psi_dynamic(cycles, stochastic_cycle_mode)
    if stochastic_cycle_mode in ("single", "multi_peak_single_cycle"):
        return compute_psi_single_cycle(T, cycles[0].R, drop_singular_frequency)
    return compute_psi_multi_cycle(T, cycles, drop_singular_frequency)


def compute_xaa_dynamic(
    psi: np.ndarray,
    stochastic_cycle_mode: str,
) -> float:
    """Choose the XAA calculation for the selected cycle mode.

    "multi_peak_single_cycle" uses the same single-cycle path as compute_psi_dynamic.
    """
    if stochastic_cycle_mode in ("single", "multi_peak_single_cycle"):
        return compute_xaa_single_cycle(psi)
    if stochastic_cycle_mode == "multi_cycle":
        return compute_xaa_multi_cycle(psi)
    raise InvalidConfigurationError(
        f"Unknown stochastic_cycle_mode: {stochastic_cycle_mode!r}."
    )

# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know  the exact 
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_periodogram(periodogram: Any, min_length: int = 1) -> None:
    if not isinstance(periodogram, np.ndarray):
        try:
            periodogram = np.asarray(periodogram, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"periodogram must be a numeric array: {exc}"
            ) from exc
    if periodogram.ndim != 1:
        raise InvalidConfigurationError(
            f"periodogram must be 1-dimensional, got shape {periodogram.shape}."
        )
    if periodogram.size == 0:
        raise InvalidConfigurationError("periodogram must not be empty.")
    if periodogram.size < min_length:
        raise InvalidConfigurationError(
            f"periodogram has {periodogram.size} elements; at least {min_length} required."
        )
    if not np.all(np.isfinite(periodogram)):
        raise InvalidConfigurationError(
            "periodogram contains non-finite values (NaN or inf)."
        )


def _validate_find_top_peaks(
    periodogram: Any, n_peaks: int, exclude_zero: bool
) -> None:
    if isinstance(n_peaks, bool) or not isinstance(n_peaks, int):
        raise InvalidConfigurationError(
            f"n_peaks must be an int, got {type(n_peaks).__name__}."
        )
    if n_peaks < 1:
        raise InvalidConfigurationError(f"n_peaks must be >= 1, got {n_peaks}.")
    _validate_periodogram(periodogram, min_length=1)
    n_available = len(periodogram) - (1 if exclude_zero else 0)
    if n_peaks > n_available:
        raise InvalidConfigurationError(
            f"n_peaks={n_peaks} exceeds the number of available frequencies "
            f"({n_available})."
        )


def _validate_psi_single_cycle(T: int, R: int, drop_singular_frequency: bool) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T < 2:
        raise InvalidConfigurationError(f"T must be >= 2, got {T}.")
    if isinstance(R, bool) or not isinstance(R, int):
        raise InvalidConfigurationError(f"R must be an int, got {type(R).__name__}.")
    if R < 1 or R > T - 1:
        raise InvalidConfigurationError(
            f"R must satisfy 1 <= R <= T-1={T - 1}, got R={R}."
        )
    if not isinstance(drop_singular_frequency, bool):
        raise InvalidConfigurationError(
            f"drop_singular_frequency must be a bool, "
            f"got {type(drop_singular_frequency).__name__}."
        )


def _validate_xaa_single_cycle(psi: Any) -> None:
    if not isinstance(psi, np.ndarray):
        try:
            psi = np.asarray(psi, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"psi must be a numeric array: {exc}"
            ) from exc
    if psi.ndim != 1:
        raise InvalidConfigurationError(
            f"psi must be 1-dimensional, got shape {psi.shape}."
        )
    if psi.size == 0:
        raise InvalidConfigurationError("psi must not be empty.")
    if not np.all(np.isfinite(psi)):
        raise InvalidConfigurationError(
            "psi contains non-finite values. "
            "Ensure compute_psi_single_cycle was called with "
            "drop_singular_frequency=True."
        )


def _validate_compute_residual_periodogram(residuals: Any) -> None:
    if not isinstance(residuals, np.ndarray):
        try:
            residuals = np.asarray(residuals, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"residuals must be a numeric array: {exc}"
            ) from exc
    if residuals.ndim != 1 or residuals.size == 0:
        raise InvalidConfigurationError("residuals must be a non-empty 1-D array.")
    if not np.all(np.isfinite(residuals)):
        raise InvalidConfigurationError("residuals contains non-finite values.")


def _validate_compute_frequency_variance_single_cycle(
    I_residuals: Any, R: int
) -> None:
    if not isinstance(I_residuals, np.ndarray):
        try:
            I_residuals = np.asarray(I_residuals, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"I_residuals must be a numeric array: {exc}"
            ) from exc
    if I_residuals.ndim != 1 or I_residuals.size == 0:
        raise InvalidConfigurationError("I_residuals must be a non-empty 1-D array.")
    T = len(I_residuals)
    if isinstance(R, bool) or not isinstance(R, int):
        raise InvalidConfigurationError(f"R must be an int, got {type(R).__name__}.")
    if R < 0 or R >= T:
        raise InvalidConfigurationError(f"R must satisfy 0 <= R < T={T}, got R={R}.")


def _validate_compute_frequency_variance_multi_cycle(
    I_residuals: Any, cycles: object
) -> None:
    if not isinstance(I_residuals, np.ndarray):
        try:
            I_residuals = np.asarray(I_residuals, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"I_residuals must be a numeric array: {exc}"
            ) from exc
    if I_residuals.ndim != 1 or I_residuals.size == 0:
        raise InvalidConfigurationError("I_residuals must be a non-empty 1-D array.")
    try:
        cycle_list = list(cycles)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"cycles must be iterable, got {type(cycles).__name__}."
        ) from exc
    if len(cycle_list) == 0:
        raise InvalidConfigurationError("cycles must not be empty.")


def _validate_compute_frequency_variance_dynamic(
    cycles: tuple, mode: str
) -> None:
    _VALID_MODES = {"single", "multi_peak_single_cycle", "multi_cycle"}
    if mode not in _VALID_MODES:
        raise InvalidConfigurationError(
            f"Unknown mode: {mode!r}. Expected one of {sorted(_VALID_MODES)}."
        )
    if mode in ("single", "multi_peak_single_cycle") and len(cycles) != 1:
        raise InvalidCycleError(
            f"mode={mode!r} requires exactly 1 cycle, got {len(cycles)}."
        )


def _validate_psi_dynamic(cycles: tuple, stochastic_cycle_mode: str) -> None:
    _VALID_MODES = {"single", "multi_peak_single_cycle", "multi_cycle"}
    if stochastic_cycle_mode not in _VALID_MODES:
        raise InvalidConfigurationError(
            f"Unknown stochastic_cycle_mode: {stochastic_cycle_mode!r}."
        )
    if stochastic_cycle_mode in ("single", "multi_peak_single_cycle") and len(cycles) != 1:
        raise InvalidCycleError(
            f"stochastic_cycle_mode={stochastic_cycle_mode!r} requires exactly "
            f"1 cycle, got {len(cycles)}."
        )
