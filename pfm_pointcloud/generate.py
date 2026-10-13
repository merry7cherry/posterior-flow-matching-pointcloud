"""Generate shapes from a self-contained PFM checkpoint without training assets."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .generator import PFMGenerator, sha256_file
from .pfm import sample
from .evaluate import save_plot
from .runtime import write_json, revision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--solver', choices=['ode', 'sde'], default='ode')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--diffusion-scale', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=9988)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.count < 1 or args.threads < 1:
        parser.error('count and threads must be positive')
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = torch.device(args.device)
    model = PFMGenerator.from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    config = checkpoint['config']
    rng = torch.Generator(device=device).manual_seed(args.seed)
    brownian = torch.Generator(device=device).manual_seed(args.seed+100)
    clouds, snapshots = [], []
    batch_size = config['evaluation']['batch_size']
    for start in range(0, args.count, batch_size):
        size = min(batch_size, args.count-start)
        z = model.prior_sample(torch.randn((size, model.native_args['latent_dim']), device=device, generator=rng))
        noise = torch.randn((size, config['num_points'], 3), device=device, generator=rng)
        points, stats = sample(model.net, z, noise, solver=args.solver, steps=args.steps,
                               terminal_time=config['evaluation']['terminal_time'],
                               diffusion_scale=args.diffusion_scale, brownian=brownian, capture=start == 0)
        clouds.append(points.cpu())
        if start == 0:
            snapshots = stats['snapshots']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    points = torch.cat(clouds)
    np.save(output/'points.npy', points.numpy())
    torch.save(snapshots, output/'trajectory.pt')
    save_plot(points, output/'samples.png')
    write_json(output/'sampling.json', {'checkpoint_sha256': sha256_file(args.checkpoint),
               'producer': revision(), 'solver': args.solver, 'steps': args.steps,
               'terminal_time': config['evaluation']['terminal_time'],
               'diffusion_scale': args.diffusion_scale, 'seed': args.seed, 'count': args.count,
               'forward_calls_per_batch': args.steps+1, 'device': str(device)})


if __name__ == '__main__':
    main()
