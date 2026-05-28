"""
BAYESIAN RISK MODEL
===================

Model risiko Bayesian menggunakan Beta-Binomial conjugacy untuk probabilitas
kejadian dan Naive Bayes untuk integrasi multi-source evidence.

Sitasi:
    Bayes (1763). An Essay towards solving a Problem in the Doctrine of Chances.
        Philosophical Transactions.
    Gelman dkk (2013). Bayesian Data Analysis (3rd ed). CRC Press.
    Berger (1985). Statistical Decision Theory and Bayesian Analysis. Springer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import math


@dataclass
class BayesianBeliefState:
    posterior_mean: float
    posterior_variance: float
    credible_interval_95: tuple
    n_observations: int
    evidence_log: List[Dict] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict:
        return {
            "posterior_mean": round(self.posterior_mean, 4),
            "posterior_variance": round(self.posterior_variance, 6),
            "credible_interval_95": [
                round(self.credible_interval_95[0], 4),
                round(self.credible_interval_95[1], 4),
            ],
            "n_observations": self.n_observations,
            "evidence_log": self.evidence_log,
            "explanation": self.explanation,
        }


class BayesianRiskModel:
    """
    Beta-Binomial Bayesian model untuk probabilitas event bencana.

    Args:
        alpha_prior: hyperparameter Beta(alpha, beta) yang menyatakan
            jumlah pseudo-observation event (semakin tinggi semakin
            informatif terhadap prior tinggi).
        beta_prior: jumlah pseudo-observation non-event.
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
    ) -> None:
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("alpha_prior dan beta_prior harus > 0")
        self.alpha = float(alpha_prior)
        self.beta = float(beta_prior)
        self._n_obs = 0
        self._log: List[Dict] = []

    def update(self, n_events: int, n_observations: int, source: str = "data") -> None:
        if n_events < 0 or n_observations < n_events:
            raise ValueError("n_events <= n_observations dan keduanya >= 0")
        self.alpha += n_events
        self.beta += (n_observations - n_events)
        self._n_obs += n_observations
        self._log.append({
            "source": source,
            "n_events": n_events,
            "n_observations": n_observations,
            "alpha_after": self.alpha,
            "beta_after": self.beta,
        })

    def update_with_likelihood_ratio(
        self,
        likelihood_ratio: float,
        source: str = "expert",
    ) -> None:
        """
        Bayes update sederhana terhadap mean: pseudo-count berdasarkan LR.
        """
        lr = max(1e-6, float(likelihood_ratio))
        # convert LR ke pseudo-event vs pseudo-non-event
        pseudo_pos = math.log(lr) if lr > 1 else 0.0
        pseudo_neg = -math.log(lr) if lr < 1 else 0.0
        self.alpha += pseudo_pos
        self.beta += pseudo_neg
        self._log.append({
            "source": source,
            "likelihood_ratio": lr,
            "alpha_after": self.alpha,
            "beta_after": self.beta,
        })

    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def posterior_variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def credible_interval(self, level: float = 0.95) -> tuple:
        """
        Approximate Beta credible interval menggunakan normal approximation
        (cukup akurat saat alpha+beta besar). Untuk produksi pakai SciPy.
        """
        mean = self.posterior_mean()
        var = self.posterior_variance()
        sd = math.sqrt(var)
        z = 1.959963984540054  # 95%
        if level == 0.90:
            z = 1.6448536269514722
        elif level == 0.99:
            z = 2.5758293035489004
        lo = max(0.0, mean - z * sd)
        hi = min(1.0, mean + z * sd)
        return (lo, hi)

    def get_belief(self) -> BayesianBeliefState:
        mean = self.posterior_mean()
        var = self.posterior_variance()
        ci = self.credible_interval()
        explanation = (
            f"Posterior Beta(alpha={self.alpha:.2f}, beta={self.beta:.2f}). "
            f"Posterior mean = {mean:.3f}, sd = {math.sqrt(var):.3f}. "
            f"95% credible interval = [{ci[0]:.3f}, {ci[1]:.3f}]. "
            f"Konjugat Beta-Binomial mengikuti Gelman dkk (2013)."
        )
        return BayesianBeliefState(
            posterior_mean=mean,
            posterior_variance=var,
            credible_interval_95=ci,
            n_observations=self._n_obs,
            evidence_log=list(self._log),
            explanation=explanation,
        )
