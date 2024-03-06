import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def visualize_samples(model, num_samples=200, grid_size=(10, 20), save_path=None):
    """
    Generate and visualize samples from the VAE.

    Args:
        model: VAE: Trained VAE model
        num_samples: int: Number of samples to generate
        grid_size: tuple: Grid size for visualization (rows, columns)
        save_path: str or None: Path to save the visualization image (if None, show the plot)
    """
    model.eval()
    with torch.no_grad():
        z_samples = torch.randn(num_samples, model.z_dim).to(device)
        generated_samples = model.sample_x_given(z_samples).cpu().view(-1, 28, 28)

    fig, axes = plt.subplots(*grid_size, figsize=(15, 7.5))
    for i in range(num_samples):
        row, col = divmod(i, grid_size[1])
        axes[row, col].imshow(generated_samples[i], cmap='gray')
        axes[row, col].axis('off')

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved at {save_path}")
        plt.show()
    else:
        plt.show()



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,     help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
args = parser.parse_args()
layout = [
    ('model={:s}',  'vae'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
vae = VAE(z_dim=args.z, name=model_name).to(device)

if args.train:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train(model=vae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)
    ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=args.train == 2)

else:
    ut.load_model_by_name(vae, global_step=args.iter_max)
    ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=True)


# 使用已训练的模型进行可视化
visualize_samples(vae, num_samples=200, grid_size=(10, 20), save_path='./visualizing_result.png')


