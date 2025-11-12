from typing import List


class MockTankModel:
    """
    Simulates the future ML model
    """

    def __init__(self, learned_daily_loss_factor=0.005):
        # estimated factor
        self.loss_factor = learned_daily_loss_factor

    def predict(self, features_list_2d: List[List[float]]) -> List[float]:
        """
        Expects features in following order:
        [W_prev, Qin_l, Qout_l_final, Tmax_c]
        """
        results = []
        for features in features_list_2d:
            W_prev = features[0]
            Qin_l = features[1]
            Qout_l = features[2]
            # Tmax_c = features[3]

            # learned loss
            natural_loss = W_prev * self.loss_factor

            W_t_ml = W_prev + Qin_l - Qout_l - natural_loss
            results.append(W_t_ml)

        return results
