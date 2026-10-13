"""One explicit PFM checkpoint: generation quality and paired diagnostics."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .generator import PFMGenerator, sha256_file
from .pfm import sample
from .dataset import ShapeNetCore
from .runtime import revision, write_json, validate_condition_category
from .metrics import distChamfer, lgan_mmd_cov, knn


def bbox_normalize(x):
    low, high = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
    scale = (high-low).amax(-1, keepdim=True)/2
    if not torch.isfinite(x).all() or (scale <= 0).any():
        raise ValueError('nonfinite or degenerate point cloud')
    return (x-(low+high)/2)/scale


@torch.no_grad()
def pairwise_cd(a, b, batch_size):
    rows = []
    for index in range(len(a)):
        row = []
        for start in range(0, len(b), batch_size):
            refs = b[start:start+batch_size]
            left, right = distChamfer(a[index:index+1].expand(len(refs), -1, -1).contiguous(), refs)
            # Squared distances have small roundoff negatives at coincident points.
            row.append(left.clamp_min(0).mean(1) + right.clamp_min(0).mean(1))
        rows.append(torch.cat(row))
    return torch.stack(rows)


@torch.no_grad()
def cd_metrics(generated, reference, batch_size, reference_distances=None):
    cross = pairwise_cd(generated, reference, batch_size)
    gg = pairwise_cd(generated, generated, batch_size)
    rr = reference_distances if reference_distances is not None else pairwise_cd(reference, reference, batch_size)
    metrics = lgan_mmd_cov(cross)
    nearest = knn(gg, cross, rr, 1)
    result = {key + '-CD': value.item() for key, value in metrics.items()}
    result.update({'1-NNA-CD': nearest['acc'].item(),
                   '1-NNA-generated-CD': nearest['acc_t'].item(),
                   '1-NNA-reference-CD': nearest['acc_f'].item()})
    return result


@torch.no_grad()
def posterior_diagnostics(model, points, times, batch_size, seed, device):
    results = {}
    low, high = model.net.logvar_min, model.net.logvar_max
    for ti, time_value in enumerate([*times, None]):
        # One independent, reproducible stream per diagnostic time and suite.
        rng = torch.Generator(device=device).manual_seed(seed + 1000 + ti)
        totals = torch.zeros(8, dtype=torch.float64, device=device)
        bins = torch.zeros(10, 3, dtype=torch.float64, device=device)
        for start in range(0, len(points), batch_size):
            x1 = points[start:start+batch_size].to(device)
            z = model.encode(x1, rng)
            x0 = torch.randn(x1.shape, device=device, generator=rng)
            t = (0.001+0.998*torch.rand(len(x1), device=device, generator=rng)
                 if time_value is None else x1.new_full((len(x1),), time_value))
            xt = (1-t[:, None, None])*x0 + t[:, None, None]*x1
            velocity, ell = model.net(xt, t, z)
            gamma = ell.exp()
            residual = x1-x0-velocity
            error = residual.square()
            nll = 0.5 * (math.log(2*math.pi)+ell+error/gamma)
            endpoint_scale = (1-t[:, None, None]).square()
            entries = [error.sum(), gamma.sum(), nll.sum(), (endpoint_scale*error).sum(),
                       (endpoint_scale*gamma).sum(), (nll+torch.log1p(-t[:, None, None])).sum(),
                       (ell <= low+.01).sum(), (ell >= high-.01).sum()]
            totals += torch.stack(entries).double()
            indexes = ((ell-low)*10/(high-low)).long().clamp(0,9).flatten()
            bins[:,0].scatter_add_(0,indexes,torch.ones_like(error.flatten(),dtype=torch.float64))
            bins[:,1].scatter_add_(0,indexes,gamma.flatten().double())
            bins[:,2].scatter_add_(0,indexes,error.flatten().double())
        count = points.numel()
        values = (totals/count).cpu().tolist()
        keys = ['velocity_mse','velocity_variance','velocity_nll','endpoint_mse',
                'endpoint_variance','endpoint_nll','near_lower_bound_fraction','near_upper_bound_fraction']
        record = dict(zip(keys,values))
        record['logvar_bounds'] = [low, high]
        record['error_variance_ratio'] = values[0]/values[1]
        record['calibration_bins_count_variance_sum_error_sum'] = bins.cpu().tolist()
        record['coordinate_count'] = count
        results['random_time' if time_value is None else str(time_value)] = record
    return results


def save_plot(points, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count = min(8, len(points))
    columns = min(4, count)
    rows = math.ceil(count / columns)
    fig = plt.figure(figsize=(3 * columns, 3 * rows))
    for index, cloud in enumerate(points[:count].cpu().numpy()):
        ax = fig.add_subplot(rows, columns, index+1, projection='3d')
        ax.scatter(cloud[:,0], cloud[:,2], cloud[:,1], s=0.3)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=20, azim=35)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, help='Limit held-out test shapes for a bounded evaluation')
    parser.add_argument('--steps', type=int, help='Override integration steps')
    parser.add_argument('--diffusion-scale', type=float, help='Override the SDE scale coefficient')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.limit is not None and args.limit < 2):
        parser.error('threads must be positive; evaluation requires at least two shapes')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    protocol = config['evaluation'].copy()
    if args.steps is not None:
        protocol['steps'] = args.steps
    if args.diffusion_scale is not None:
        protocol['diffusion_scale'] = args.diffusion_scale
    if not args.limit and checkpoint['iteration'] != config['max_iters']:
        raise ValueError('full evaluation requires the final checkpoint; use --limit for a bounded run')
    if sha256_file(args.dataset) != checkpoint['identity']['dataset_sha256']:
        raise ValueError('dataset identity mismatch')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = PFMGenerator.from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    validate_condition_category(config, model.native_args)
    dset = ShapeNetCore(args.dataset, [config['category']], 'test', config['scale_mode'], limit=args.limit)
    points = torch.stack([d['pointcloud'] for d in dset])
    ids = [d['id'] for d in dset]
    reference = bbox_normalize(points).to(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    identity = {'checkpoint_sha256':sha256_file(args.checkpoint), 'iteration':checkpoint['iteration'],
                'category':config['category'],
                'training_mode':'joint',
                'logvar_bounds':[config['logvar_min'], config['logvar_max']],
                'training_identity':checkpoint['identity'], 'producer':revision(), 'protocol':protocol,
                'test_ids':ids, 'count':len(points), 'bounded':args.limit is not None,
                'emd':'unavailable', 'jsd':'excluded: upstream grid/normalization mismatch',
                'checkpoint_producer': checkpoint['producer']}
    write_json(output/'evaluation.json', identity)
    results = {}
    rr = pairwise_cd(reference, reference, protocol['metric_batch_size'])
    for solver in ('ode', 'sde'):
        start_time = time.perf_counter()
        # Matching initial draws across ODE/SDE, separate Brownian stream.
        rng = torch.Generator(device=device).manual_seed(protocol['seed'])
        brownian = torch.Generator(device=device).manual_seed(protocol['seed']+100)
        generated, nfe, vjp = [], 0, 0
        for start in range(0,len(points),protocol['batch_size']):
            batch = min(protocol['batch_size'],len(points)-start)
            w = torch.randn((batch,model.native_args['latent_dim']),device=device,generator=rng)
            z = model.prior_sample(w)
            x0 = torch.randn((batch,config['num_points'],3),device=device,generator=rng)
            precision = protocol[solver+'_precision'] if device.type == 'cuda' else 'fp32'
            x, stats = sample(model.net,z,x0,solver=solver,steps=protocol['steps'],
                              terminal_time=protocol['terminal_time'],diffusion_scale=protocol['diffusion_scale'],
                              precision=precision,brownian=brownian,capture=start == 0)
            if start == 0:
                torch.save(stats['snapshots'],output/(solver+'_trajectory.pt'))
            generated.append(x.cpu())
            nfe += stats['nfe']; vjp += stats['vjp']
        raw = torch.cat(generated)
        normalized = bbox_normalize(raw)
        generation_seconds = time.perf_counter()-start_time
        np.save(output/(solver+'_raw.npy'),raw.numpy())
        np.save(output/(solver+'_bbox.npy'),normalized.numpy())
        save_plot(normalized,output/(solver+'.png'))
        metrics = cd_metrics(normalized.to(device),reference,protocol['metric_batch_size'],rr)
        metrics.update(generation_seconds=generation_seconds,total_seconds=time.perf_counter()-start_time,
                       network_forward_calls=nfe,input_vjp_calls=vjp,
                       raw_abs_max=raw.abs().max().item(),
                       raw_center_norm=raw.mean(1).norm(dim=1).mean().item(),
                       raw_flat_std=raw.flatten(1).std(1).mean().item())
        results[solver] = metrics
        write_json(output/(solver+'_metrics.json'),metrics)
        print(json.dumps({solver:metrics}),flush=True)
    results['posterior'] = posterior_diagnostics(model,points,protocol['diagnostic_times'],
                                                  protocol['batch_size'],protocol['seed'],device)
    write_json(output/'results.json', {'identity':identity, 'results':results})


if __name__ == '__main__':
    main()
