# utils/irrigation_xai.py
#
# Explainable AI for Irrigation Recommendation using LIME.

import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer


def _build_sentence(top_feature, crop, water_retention, drainage, irrigation_type):
    wr  = water_retention
    dr  = drainage
    irr = irrigation_type

    if top_feature == "Crop_Enc":
        return (
            f"The crop type ({crop}) is the primary factor recommending "
            f"{irr}; it is agronomically best suited for this irrigation "
            f"method given {wr.lower()} water retention and {dr.lower()} drainage."
        )
    elif top_feature == "WR_Enc":
        return (
            f"Water retention ({wr}) is the primary factor driving the "
            f"recommendation of {irr} for {crop}; "
            f"the soil's water-holding capacity makes this the most "
            f"suitable irrigation method under {dr.lower()} drainage."
        )
    elif top_feature == "Drainage_Enc":
        return (
            f"Drainage ({dr}) is the primary factor driving the "
            f"recommendation of {irr} for {crop}; "
            f"the soil's drainage rate makes this the most efficient "
            f"irrigation method given {wr.lower()} water retention."
        )
    return (
        f"{irr} is recommended for {crop} based on "
        f"{wr.lower()} water retention and {dr.lower()} drainage."
    )


def explain_irrigation(crop, water_retention, drainage, irrigation_type,
                        crop_enc, wr_enc, drainage_enc,
                        model, training_data, label_encoder):
    feature_names = ["Crop_Enc", "WR_Enc", "Drainage_Enc"]

    explainer = LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=label_encoder.classes_,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    instance       = np.array([crop_enc, wr_enc, drainage_enc], dtype=float)
    pred_class_idx = int(model.predict(instance.reshape(1, -1))[0])

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=3,
        num_samples=500,
        labels=(pred_class_idx,)
    )

    lime_weights        = explanation.as_list(label=pred_class_idx)
    feature_importances = {}
    for condition_str, weight in lime_weights:
        for fname in feature_names:
            if fname in condition_str:
                feature_importances[fname] = abs(weight)
                break

    top_feature = max(feature_importances, key=feature_importances.get) if feature_importances else "Crop_Enc"

    return _build_sentence(top_feature, crop, water_retention, drainage, irrigation_type)
