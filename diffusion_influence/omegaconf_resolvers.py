import math

import omegaconf


def register_custom_resolvers():
    """
    Register some custom resolvers for OmegaConf that allow for neat syntax in the config files, e.g.:

    ```yaml
    training:
        num_channels: 3
        extra_channels: 2
    model:
        input_channels: ${add:${dataset.num_channels}, ${dataset.extra_channels}}
        # input_channels will be 5
    ```

    """
    omegaconf.OmegaConf.register_new_resolver("add", lambda a, b: a + b)
    omegaconf.OmegaConf.register_new_resolver("subtract", lambda a, b: a - b)
    omegaconf.OmegaConf.register_new_resolver("multiply", lambda a, b: a * b)
    omegaconf.OmegaConf.register_new_resolver("divide", lambda a, b: a / b)
    omegaconf.OmegaConf.register_new_resolver("floor", lambda x: math.floor(x))
    omegaconf.OmegaConf.register_new_resolver("power", lambda a, b: a**b)
    omegaconf.OmegaConf.register_new_resolver(
        "conditional", lambda cond, a, b: a if cond else b
    )
