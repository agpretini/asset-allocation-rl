"""Regularized multivariate Student-t distribution."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MultivariateStudentT:
    """Multivariate Student-t location, scale, and degrees of freedom."""

    mean: np.ndarray
    scale_matrix: np.ndarray
    degrees_freedom: float

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        degrees_freedom: float = 8.0,
        regularization: float = 1e-6,
        fallback_covariance: np.ndarray | None = None,
    ) -> "MultivariateStudentT":
        """Estimate location and a regularized scale matrix.

        The sample covariance is converted to the Student-t scale convention,
        whose covariance is ``scale * nu / (nu - 2)``.
        """
        data = np.asarray(observations, dtype=float)
        if data.ndim != 2 or data.shape[0] < 1:
            raise ValueError("observations must be a non-empty two-dimensional array")
        if not np.isfinite(data).all():
            raise ValueError("observations must be finite")
        if degrees_freedom <= 2.0:
            raise ValueError("degrees_freedom must be greater than two")
        if regularization <= 0.0:
            raise ValueError("regularization must be positive")

        dimension = data.shape[1]
        if data.shape[0] >= 2:
            covariance = np.atleast_2d(np.cov(data, rowvar=False, ddof=1))
        elif fallback_covariance is not None:
            covariance = np.asarray(fallback_covariance, dtype=float)
        else:
            covariance = np.eye(dimension, dtype=float) * regularization
        if covariance.shape != (dimension, dimension):
            raise ValueError("fallback covariance has incompatible shape")

        covariance = _regularize_symmetric_matrix(covariance, regularization)
        scale_matrix = covariance * ((degrees_freedom - 2.0) / degrees_freedom)
        scale_matrix = _regularize_symmetric_matrix(scale_matrix, regularization)
        return cls(
            mean=data.mean(axis=0),
            scale_matrix=scale_matrix,
            degrees_freedom=degrees_freedom,
        )

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one vector from the fitted multivariate Student-t."""
        gaussian = rng.multivariate_normal(
            mean=np.zeros(len(self.mean)),
            cov=self.scale_matrix,
            check_valid="raise",
        )
        chi_square = rng.chisquare(self.degrees_freedom)
        return self.mean + gaussian / np.sqrt(chi_square / self.degrees_freedom)


def _regularize_symmetric_matrix(
    matrix: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Return a finite symmetric positive-definite matrix."""
    symmetric = (
        np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T
    ) / 2.0
    if not np.isfinite(symmetric).all():
        raise ValueError("covariance matrix must be finite")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, regularization)
    regularized = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    return (regularized + regularized.T) / 2.0
