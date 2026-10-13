"""Joint PFM training with complete checkpoint/resume state."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import torch

from .common import get_linear_scheduler
from .generator import PFMGenerator, sha256_file
from .pfm import delta_loss, network_context
from .dataset import ShapeNetCore
from .runtime import (load_config, config_digest, revision, seed_all, rng_state,
                           restore_rng, atomic_save, write_json, validate_condition_category)


BASE_STATS = ('loss', 'mse', 'variance', 'grad_norm', 'pfm_loss', 'latent_kl')

def train_step(model, x1, optimizer, scheduler, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    z, kl = model.encode_with_kl(x1)
    x0 = torch.randn_like(x1)
    eps = config['eps_t']
    t = eps + (1 - 2 * eps) * torch.rand(len(x1), device=x1.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    with network_context(x1.device, config['precision']):
        velocity, ell = model.net(xt, t, z)
    pfm_loss = delta_loss(velocity, ell, x0, x1)
    loss = pfm_loss + config.get('kl_weight', 0.0) * kl
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.training_parameters(), config['max_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    values = (loss.detach(), (x1-x0-velocity).square().mean().detach(),
              ell.exp().mean().detach(), grad.detach(), pfm_loss.detach(), kl.detach())
    return torch.stack(values)


@torch.no_grad()
def validate(model, points, config, device):
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(12345)
        model.eval()
        x1 = points[:32].to(device)
        z = model.encode(x1)
        x0 = torch.randn_like(x1)
        t = 0.001 + 0.998 * torch.rand(len(x1), device=device)
        xt = (1-t[:, None, None])*x0 + t[:, None, None]*x1
        with network_context(device, config['precision']):
            velocity, ell = model.net(xt, t, z)
        return {'loss': delta_loss(velocity, ell, x0, x1).item(),
                'variance': ell.exp().mean().item()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Stop at this iteration, keeping the configured schedule for resume')
    parser.add_argument('--limit-train', type=int, help='Use a fixed subset for CPU smoke runs; recorded in checkpoint identity')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error('threads and stop-after must be positive')
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = config['allow_tf32']
        torch.backends.cudnn.allow_tf32 = config['allow_tf32']
        torch.cuda.reset_peak_memory_stats()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = open(output / '.train.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    provenance = revision()
    identity = {'config_sha256': config_digest(config),
                'dataset_sha256': sha256_file(args.dataset),
                'condition_sha256': sha256_file(args.condition), 'limit_train': args.limit_train}
    seed_all(config['seed'])
    model = PFMGenerator.from_official(args.condition, config['logvar_min'], config['logvar_max']).to(device)
    validate_condition_category(config, model.native_args)
    optimizer = torch.optim.Adam(model.training_parameters(), lr=config['lr'],
                                 weight_decay=config['weight_decay'])
    scheduler = get_linear_scheduler(optimizer, config['sched_start'], config['sched_end'],
                                     config['lr'], config['end_lr'])
    dset = ShapeNetCore(args.dataset, [config['category']], 'train', config['scale_mode'], limit=args.limit_train)
    # Native deterministic dataset ordering, no augmentation, no per-epoch
    # shuffle, and the final partial batch are preserved. A tensor slice makes
    # the exact next batch recoverable from iteration alone.
    points = torch.stack([d['pointcloud'] for d in dset])
    if tuple(points.shape[1:]) != (config['num_points'], 3) or not torch.isfinite(points).all():
        raise ValueError('invalid category point clouds')
    if device.type == 'cuda':
        points = points.pin_memory()
    if len(points) < 2 or config['batch_size'] < 2 or len(points) % config['batch_size'] == 1:
        raise ValueError('PointNet BatchNorm requires at least two shapes per training batch; adjust batch_size or subset')
    batches = math.ceil(len(points) / config['batch_size'])
    start = 0
    if args.resume:
        checkpoint = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        if checkpoint['identity'] != identity or checkpoint['producer'] != provenance:
            raise ValueError('resume identity/revision mismatch')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start = checkpoint['iteration']
        restore_rng(checkpoint['rng'])
    elif (output / 'latest.pt').exists() or (output / 'run.json').exists():
        raise FileExistsError('existing run requires explicit --resume')
    write_json(output / 'run.json', {'config': config, 'identity': identity, 'producer': provenance,
               'native_args': model.native_args, 'train_size': len(points),
               'validation': 'disabled by matched protocol; fixed training probe only',
               'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
               'training_mode': 'joint',
               'initialization': 'category encoder/prior; seed-initialized PFM; no optimizer import',
               'condition_origin': model.condition_origin,
               'resume_iteration': start})
    end = min(config['max_iters'], args.stop_after or config['max_iters'])
    if end <= start:
        raise ValueError('requested endpoint is not after resume iteration')
    started = window_start = time.perf_counter()
    stats = BASE_STATS
    accumulator = torch.zeros(len(stats), device=device)
    window_count = 0
    with open(output / 'train.jsonl', 'a', buffering=1) as log:
        for iteration in range(start + 1, end + 1):
            offset = ((iteration - 1) % batches) * config['batch_size']
            x1 = points[offset:offset+config['batch_size']].to(device, non_blocking=True)
            accumulator += train_step(model, x1, optimizer, scheduler, config)
            window_count += 1
            if iteration % config['log_every'] == 0 or iteration == end:
                values = (accumulator / window_count).cpu().tolist()
                now = time.perf_counter()
                record = dict(zip(stats, values))
                record.update(iteration=iteration, seconds_per_step=(now-window_start)/window_count,
                              elapsed_seconds=now-started, lr=optimizer.param_groups[0]['lr'],
                              peak_vram_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
                line = json.dumps(record, allow_nan=False)
                print(line, flush=True)
                log.write(line + '\n')
                accumulator.zero_()
                window_count = 0
                window_start = time.perf_counter()
            if iteration % config['validate_every'] == 0 or iteration == end:
                metrics = validate(model, points, config, device)
                print(json.dumps({'training_probe': metrics, 'iteration': iteration}), flush=True)
            if iteration % config['checkpoint_every'] == 0 or iteration == end:
                checkpoint = {'format': 'pointcloud-pfm-v1', 'iteration': iteration,
                              'config': config, 'identity': identity, 'producer': provenance,
                              'native_args': model.native_args, 'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                              'rng': rng_state()}
                atomic_save(checkpoint, output / 'latest.pt')
                if iteration == end:
                    atomic_save(checkpoint, output / f'checkpoint_{iteration:06d}.pt')
    write_json(output / 'completion.json', {'iteration': end, 'formal_complete': end == config['max_iters'],
               'elapsed_seconds': time.perf_counter()-started, 'producer': provenance, 'identity': identity,
               'start_iteration': start, 'training_mode': 'joint'})


if __name__ == '__main__':
    main()
