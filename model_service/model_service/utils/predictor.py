from typing import List


class MockTankModel:
    """
    Simulates the future ML model
    """

    def __init__(self, A_soil_m2=10.0, water_in_tank_loss_factor=0.0007, loss_through_et_factor=0.05):
        # estimated factors
        self.A_soil_m2 = A_soil_m2
        self.water_in_tank_loss_factor = water_in_tank_loss_factor  # natural water loss
        self.loss_through_et_factor = loss_through_et_factor  # loss through ET

    def predict(self, features_list: List[List[float]]) -> List[float]:
        """
        Expects features in following order:
        [W_prev, M_prev, Qin_l, Qout_l_final, Tmax_c]
        """
        results = []
        for features in features_list:
            W_prev = features[0]
            M_prev = features[1]
            Qin_l = features[2]
            Qout_l = features[3]
            Tmax_c = features[4]

            et_potential_mm = max(0.0, Tmax_c * self.loss_through_et_factor)

            et_actual_mm = min(et_potential_mm, M_prev)

            # loss in water tank
            tank_loss_l = max(0.0, W_prev * Tmax_c * self.water_in_tank_loss_factor)

            Qout_mm = Qout_l / self.A_soil_m2

            # new soil moisture
            M_t_ml = M_prev + Qout_mm - et_actual_mm
            M_t_ml = max(0.0, M_t_ml)

            # new tank water level
            W_t_ml = W_prev + Qin_l - Qout_l - tank_loss_l

            results.append([W_t_ml, M_t_ml])

        return results
