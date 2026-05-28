"""
Encoder step builder for categorical variables.
"""
from __future__ import annotations
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_encoder_step(strategy: str = "onehot", **kwargs):
    """
    Strategies:
    - 'onehot'         : standard one-hot encoding. Default. Increases dimensionality.
    - 'ordinal'        : integer encoding. Use only when categories have natural order.
    - 'target_encoding': target-mean encoding. RISK of leakage if not nested in CV.

    For target encoding, we recommend the category_encoders package or
    using it inside a custom CV-aware transformer. Default to onehot for safety.
    """
    if strategy == "onehot":
        # handle_unknown='ignore' is critical for preventing test-time errors
        # when a category appears that wasn't in training
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            **kwargs,
        )

    if strategy == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            **kwargs,
        )

    if strategy == "target_encoding":
        # Lazy fallback: target encoding requires careful CV handling.
        # For MVP we return one-hot which is leak-safe by default.
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
