def int_to_float(n):
    """
    Casts an int into a float, safely.

    Args:
        n: int, required.  Input number as int.

    Returns:
        float, or n as-is if not an int
    """

    if type(n) is int:
        return float(n)
    else:
        return n


def validate_type(param, types=type(None)):
    """
    Validates the input param against a type or types. Note that the method
    type checks using the `type` method, which is more restrictive than the
    `isinstance` method that returns true for sub-classes of the type being
    checked.

    Args:
        param: Any, required.  Input parameter to be type-checked.
        types: type, type[], optional.  Type or types to check param against.

    Raises:
        TypeError
    """

    if type(types) is list:
        checks = [t for t in types if type(param) is not t]
        if len(checks) == len(types):
            raise TypeError(f"{param} must be of types {types}")
    else:
        t = types  # it's a single type at this point
        if type(param) is not t:
            raise TypeError(f"{param} must be of type {t}")


def validate_instance(param, instances=type(None)):
    """
    Validates the input param via the `isinstance` method, which is less
    restrictive than `type`. `isintance` returns true for sub-classes of the
    type being checked.

    Args:
        param: Any, required.  Input parameter to be type-checked.
        types: type, type[], optional.  Type or types to check param against.

    Raises:
        TypeError
    """
    if type(instances) is list:
        checks = [i for i in instances if not isinstance(param, i)]
        if len(checks) == len(instances):
            raise TypeError(f"{param} must be of instances {instances}")
    else:
        instance = instances  # it's a single type at this point
        if not isinstance(param, instance):
            raise TypeError(f"{param} must be of instance {instance}")
