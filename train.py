import torch
from dataloader import get_mnist_data
from learning import Training
from utils import get_args
from models import VAE

import warnings

warnings.filterwarnings("ignore")


def _main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_data(device, args.batch_size)

    if args.model_name == "vae":
        assert args.beta == 1, "beta should be one in VAE"
        assert args.conditional == False, "Conditional flag should be False in VAE"

    if args.model_name == "bvae":
        assert args.conditional == False, "Conditional flag should be False in beta-VAE"

    model = VAE(
        z_dim=args.latent_dimension,
        beta=args.beta,
        conditional=args.conditional,
        device=device,
    ).to(device)

    model, optimizer, report = Training(
        model=model,
        model_name=args.model_name,
        batch_size=args.batch_size,
        train_loader=train_loader,
        val_loader=test_loader,
        beta=args.beta,
        latent_dimension=args.latent_dimension,
        epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        load_saved_model=args.load_saved_model,
        ckpt_save_freq=args.ckpt_save_freq,
        ckpt_save_path=args.ckpt_save_path,
        ckpt_path=args.ckpt_path,
        report_root=args.report_root,
    )
    return model, optimizer, report


if __name__ == "__main__":
    args = get_args()
    _main(args)
