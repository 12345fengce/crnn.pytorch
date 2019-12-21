from .model import Model


def get_model(in_channels, num_class, config):
    return Model(in_channels, num_class, config)
