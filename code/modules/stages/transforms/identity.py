"""Identity — No-op transform (placeholder for DPP)."""


class Identity:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, metadata=None, **kwargs):
        return X, y, metadata or {}

    def transform(self, X, y=None, metadata=None, **kwargs):
        return X, y, metadata or {}

