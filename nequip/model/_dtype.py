from nequip.nn import GraphModuleMixin
from nequip.nn import ComputeDtypeWrapper as ComputeDtypeWrapperModule


def ComputeDtypeWrapper(
    model: GraphModuleMixin,
    config,
) -> GraphModuleMixin:
    """Wrap the model up to this point in a module that converts its inputs into the right dtype and converts outputs back later."""
    # Wrap the model up to this point in something that
    return ComputeDtypeWrapperModule(
        func=model,
        compute_dtype=config["compute_dtype"],
        output_dtype=config["default_dtype"],
    )
