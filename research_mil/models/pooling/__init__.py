
from .methods import  bagAverage

poolings = {
    'bag_average' : bagAverage,
}


def load_pooling(pooling, in_channels, num_classes, embed):
    pooling_names = list(poolings.keys())
    if pooling not in pooling_names:
        raise ValueError('Invalid choice for pooling method - choices: {}'.format(' | '.join(pooling_names)))
    return poolings[pooling](in_channels, num_classes, embed)

