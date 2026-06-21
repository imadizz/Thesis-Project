"""
kalman_filter.py
Adaptive scalar Kalman filter for fusing PCU camera score with GPS features.

The measurement noise R is dynamically adjusted based on weather and time of day,
compensating for reduced camera reliability under adverse conditions.

R values:
  clear daytime   -> 0.05  (trust camera, low noise)
  overcast/rain   -> 0.15  (moderate degradation)
  fog/night       -> 0.20  (severe degradation, trust GPS more)
"""


class AdaptiveKalman:
    """
    Scalar Kalman filter with weather-adaptive measurement noise.

    State:  x  = estimated normalised congestion score (0-1)
    Input:  z  = PCU-weighted camera congestion score (noisy measurement)
    """

    R_TABLE = {
        ('clear',    'daytime'):  0.05,
        ('overcast', 'daytime'):  0.15,
        ('rainy',    'daytime'):  0.15,
        ('foggy',    'daytime'):  0.20,
        ('snowy',    'daytime'):  0.15,
        ('clear',    'night'):    0.20,
        ('overcast', 'night'):    0.20,
        ('rainy',    'night'):    0.20,
        ('foggy',    'night'):    0.20,
    }
    R_DEFAULT = 0.10
    Q_DEFAULT = 0.01   # process noise: how fast congestion changes between frames

    def __init__(self, Q: float = None):
        self.x = 0.0                         # state estimate
        self.P = 1.0                         # state covariance (uncertainty)
        self.Q = Q if Q is not None else self.Q_DEFAULT

    def _get_R(self, weather: str, timeofday: str) -> float:
        return self.R_TABLE.get((weather, timeofday), self.R_DEFAULT)

    def update(self, z: float, weather: str = 'clear', timeofday: str = 'daytime') -> dict:
        """
        Perform one Kalman predict+update step.

        Args:
            z:          Camera-derived normalised congestion score [0, 1]
            weather:    BDD100K weather label (clear/overcast/rainy/foggy/snowy)
            timeofday:  BDD100K time label (daytime/night/dawn/dusk)

        Returns:
            dict with keys: state, gain, innovation, P_prior
        """
        R = self._get_R(weather, timeofday)

        # Predict step
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update step
        K      = P_pred / (P_pred + R)         # Kalman gain
        innov  = z - x_pred                    # innovation (residual)
        self.x = x_pred + K * innov
        self.P = (1.0 - K) * P_pred

        return {
            'state':      self.x,
            'gain':       K,
            'innovation': innov,
            'P_prior':    P_pred,
            'R_used':     R,
        }

    def reset(self):
        """Reset filter state (call between independent video sequences)."""
        self.x = 0.0
        self.P = 1.0
