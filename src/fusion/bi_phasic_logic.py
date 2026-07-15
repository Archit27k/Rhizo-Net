class BiPhasicLogic:

    def diagnose(
        self,
        spectral_label,
        confidence,
        edge_density
    ):

        if edge_density > 6:

            if spectral_label == "Healthy":

                return {
                    "diagnosis": "Green Wilt / Drought Stress",
                    "action": "Check irrigation immediately."
                }

            else:

                return {
                    "diagnosis": "Advanced Pythium Soft Rot",
                    "action": "Quarantine infected crop."
                }

        else:

            if spectral_label == "Healthy":

                return {
                    "diagnosis": "Healthy Crop",
                    "action": "No intervention required."
                }

            else:

                return {
                    "diagnosis": "Early Foliar Infection",
                    "action": "Apply preventive fungicide."
                }