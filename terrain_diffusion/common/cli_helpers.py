import json
import re
import click


def parse_cache_size(value: str | None) -> int | None:
    """Parse human-readable size (e.g., '100M', '1G', '500K') to bytes."""
    if value is None:
        return None
    value = value.strip().upper()
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*([KMGT]?B?)', value)
    if not match:
        raise click.BadParameter(f"Invalid size format: {value}. Use e.g. 100M, 1G, 500K")
    num, suffix = float(match.group(1)), match.group(2).rstrip('B')
    multipliers = {'': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
    return int(num * multipliers.get(suffix, 1))


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

