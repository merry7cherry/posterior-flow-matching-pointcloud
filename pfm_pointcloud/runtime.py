"""Configuration, checkpoint and provenance helpers."""
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as handle:
        config = yaml.safe_load(handle)
    expected = {'seed', 'category', 'num_points', 'scale_mode', 'batch_size',
                'max_iters', 'lr', 'end_lr', 'sched_start', 'sched_end', 'weight_decay',
                'max_grad_norm', 'logvar_min', 'logvar_max',
                'eps_t', 'precision', 'allow_tf32', 'log_every',
                'checkpoint_every', 'validate_every', 'evaluation'}
    expected.add('kl_weight')
    if set(config) != expected:
        raise ValueError(f'configuration fields differ: {set(config) ^ expected}')
    if config['category'] not in ('chair', 'airplane') or config['scale_mode'] != 'shape_unit':
        raise ValueError('use chair/airplane with shape_unit training normalization')
    if not math.isfinite(config['kl_weight']) or config['kl_weight'] <= 0:
        raise ValueError('joint training requires positive KL weight')
    if not (math.isfinite(config['logvar_min']) and math.isfinite(config['logvar_max'])
            and config['logvar_min'] < config['logvar_max']):
        raise ValueError('invalid log-variance bounds')
    if type(config['num_points']) is not int or config['num_points'] < 2:
        raise ValueError('num_points must be an integer >= 2')
    if not math.isfinite(config['weight_decay']) or config['weight_decay'] < 0:
        raise ValueError('invalid weight_decay')
    for key in ('batch_size', 'max_iters', 'log_every', 'checkpoint_every', 'validate_every'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 < config['eps_t'] < 0.5 or config['precision'] not in ('fp32', 'bf16'):
        raise ValueError('invalid precision/time cutoff')
    if not 0 <= config['sched_start'] < config['sched_end'] <= config['max_iters']:
        raise ValueError('invalid scheduler range')
    for key in ('lr','end_lr','max_grad_norm'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('invalid seed')
    if type(config['allow_tf32']) is not bool:
        raise ValueError('execution flags must be boolean')
    ev = config['evaluation']
    fields = {'seed','batch_size','metric_batch_size','steps','terminal_time','diffusion_scale',
              'ode_precision','sde_precision','normalize','diagnostic_times'}
    if set(ev) != fields or ev['normalize'] != 'shape_bbox':
        raise ValueError('invalid evaluation fields/normalization')
    for key in ('batch_size','metric_batch_size','steps'):
        if type(ev[key]) is not int or ev[key] <= 0:
            raise ValueError(f'evaluation {key} must be positive integer')
    if not 0 < ev['terminal_time'] < 1 or not math.isfinite(ev['diffusion_scale']) or ev['diffusion_scale'] < 0:
        raise ValueError('invalid sampler cutoff/strength')
    if ev['ode_precision'] != ev['sde_precision'] or ev['ode_precision'] not in ('fp32','bf16'):
        raise ValueError('ODE/SDE must use matched supported precision')
    times = ev['diagnostic_times']
    rounded = torch.tensor(times, dtype=torch.float32)
    if len(times) == 0 or len(set(rounded.tolist())) != len(times) or not ((rounded > 0) & (rounded < 1)).all():
        raise ValueError('diagnostic times must be distinct representable interior times')
    return config


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def revision():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for file in sorted(root.glob('*.py')):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    # Content identity works for editable installs, wheels, and ZIP snapshots.
    return {'source_sha256': digest.hexdigest()}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def validate_condition_category(config, native_args):
    """A category uses its own official encoder/prior and native normalization."""
    if native_args.get('categories') != [config['category']]:
        raise ValueError('condition checkpoint category differs from configuration')
    if native_args.get('scale_mode') != config['scale_mode']:
        raise ValueError('condition checkpoint training normalization differs')
