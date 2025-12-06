import json
import click


def parse_kwargs(kwargs_tuple):
    """Parse key=value tuples into a dict with automatic type inference."""
    result = {}
    for item in kwargs_tuple:
        if '=' not in item:
            raise click.BadParameter(f"Expected key=value format, got: {item}")
        key, value = item.split('=', 1)
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result

