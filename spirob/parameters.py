"""Shared JSON validation for the CLI and browser builder.

The public schema is also the website's field/help catalogue. This small
validator implements the subset used by that schema, without a runtime server
or an extra validation dependency.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import warnings

SCHEMA_PATH = Path(__file__).resolve().parents[1] / 'site/parameters.schema.json'


def _validate(value, schema, path, errors):
    kinds = schema.get('type', [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    kind = ('null' if value is None else 'boolean' if isinstance(value, bool)
            else 'integer' if isinstance(value, int) else 'number' if isinstance(value, float)
            else 'object' if isinstance(value, dict) else 'array' if isinstance(value, list)
            else 'string' if isinstance(value, str) else 'invalid')
    if kind not in kinds and not (kind == 'integer' and 'number' in kinds):
        errors.append(f'{path}: expected {" or ".join(kinds)}')
        return
    if kind in ('number', 'integer'):
        if not math.isfinite(value):
            errors.append(f'{path}: must be finite')
        for key, passes, symbol in [('minimum', lambda a,b:a>=b, '>='),
                                    ('maximum', lambda a,b:a<=b, '<='),
                                    ('exclusiveMinimum', lambda a,b:a>b, '>'),
                                    ('exclusiveMaximum', lambda a,b:a<b, '<')]:
            if key in schema and not passes(value, schema[key]):
                errors.append(f'{path}: must be {symbol} {schema[key]}')
    if 'enum' in schema and value not in schema['enum']:
        errors.append(f'{path}: choose {", ".join(map(str, schema["enum"]))}')
    if kind == 'object':
        for key in schema.get('required', []):
            if key not in value:
                errors.append(f'{path}.{key}: missing required field')
        for key, item in value.items():
            if key not in schema.get('properties', {}):
                errors.append(f'{path}.{key}: unknown parameter (check spelling)')
            else:
                _validate(item, schema['properties'][key], f'{path}.{key}', errors)
    if kind == 'array':
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', math.inf):
            errors.append(f'{path}: expected {schema.get("minItems")} values')
        for i, item in enumerate(value):
            _validate(item, schema['items'], f'{path}[{i}]', errors)


def validate_schema(params):
    errors = []
    _validate(params, json.loads(SCHEMA_PATH.read_text()), 'params', errors)
    if errors:
        raise ValueError('Invalid parameters:\n' + '\n'.join(errors))


def normalize_params(params):
    """Remove the retired task marker; retain every other explicit setting."""
    result = deepcopy(params)
    if isinstance(result, dict) and isinstance(result.get('post_gen'), dict):
        if 'target_site_pos' in result['post_gen']:
            del result['post_gen']['target_site_pos']
            warnings.warn('post_gen.target_site_pos is retired and has been removed.', UserWarning)
    validate_schema(result)
    quat = result.get('post_gen', {}).get('robot_quat')
    if quat is not None and not any(quat):
        raise ValueError('post_gen.robot_quat must be nonzero')
    return result
