from utils import load_model, save_model
import torch
from torch import optim
import pandas as pd
from utils import AverageMeter
from tqdm import tqdm


def Training(
    model,
    model_name,
    batch_size,
    train_loader,
    val_loader,
    beta,
    latent_dimension,
    epochs,
    learning_rate,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_root,
):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if load_saved_model:
        model, optimizer = load_model(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer
        )

    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "beta",
            "latent_dimension",
            "epoch",
            "batch_size",
            "batch_index",
            "nelbo-batch",
            "avg_nelbo_till_current_batch",
            "kl-batch",
            "avg_kl_till_current_batch",
            "rec-batch",
            "avg_rec_till_current_batch",
        ]
    )

    for epoch in tqdm(range(1, epochs + 1)):
        avg_train_nelbo = AverageMeter()
        avg_train_kl = AverageMeter()
        avg_train_rec = AverageMeter()
        avg_val_nelbo = AverageMeter()
        avg_val_kl = AverageMeter()
        avg_val_rec = AverageMeter()
        model.train()
        mode = "train"
        loop_train = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc="train",
            position=0,
            leave=True,
        )
        for batch_idx, (xu, yu) in loop_train:
            optimizer.zero_grad()
            xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
            loss, summaries = model.loss(xu, yu)
            loss.backward()
            optimizer.step()
            avg_train_nelbo.update(round(summaries["train/loss"].item(), 3), xu.size(0))
            avg_train_kl.update(round(summaries["gen/kl_z"].item(), 3), xu.size(0))
            avg_train_rec.update(round(summaries["gen/rec"].item(), 3), xu.size(0))

            new_row = pd.DataFrame(
                {
                    "model_name": model_name,
                    "mode": mode,
                    "beta": beta,
                    "latent_dimension": latent_dimension,
                    "epoch": epoch,
                    "batch_size": batch_size,
                    "batch_index": batch_idx,
                    "nelbo-batch": avg_train_nelbo.val,
                    "avg_nelbo_till_current_batch": avg_train_nelbo.avg,
                    "kl-batch": avg_train_kl.val,
                    "avg_kl_till_current_batch": avg_train_kl.avg,
                    "rec-batch": avg_train_rec.val,
                    "avg_rec_till_current_batch": avg_train_rec.avg,
                },
                index=[0],
            )

            report.loc[len(report)] = new_row.values[0]

            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                nelbo_train="{:.4f}".format(avg_train_nelbo.avg),
                kl_train="{:.4f}".format(avg_train_kl.avg),
                rec_train="{:.4f}".format(avg_train_rec.avg),
                refresh=True,
            )
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_{beta}_epoch{epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )

        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            for batch_idx, (xu, yu) in loop_val:
                optimizer.zero_grad()
                xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                loss, summaries = model.loss(xu, yu)
                avg_val_nelbo.update(
                    round(summaries["train/loss"].item(), 3), xu.size(0)
                )
                avg_val_kl.update(round(summaries["gen/kl_z"].item(), 3), xu.size(0))
                avg_val_rec.update(round(summaries["gen/rec"].item(), 3), xu.size(0))

                new_row = pd.DataFrame(
                    {
                        "model_name": model_name,
                        "mode": mode,
                        "epoch": epoch,
                        "beta": beta,
                        "latent_dimension": latent_dimension,
                        "batch_size": batch_size,
                        "batch_index": batch_idx,
                        "nelbo-batch": avg_val_nelbo.val,
                        "avg_nelbo_till_current_batch": avg_val_nelbo.avg,
                        "kl-batch": avg_val_kl.val,
                        "avg_kl_till_current_batch": avg_val_kl.avg,
                        "rec-batch": avg_val_rec.val,
                        "avg_rec_till_current_batch": avg_val_rec.avg,
                    },
                    index=[0],
                )
                report.loc[len(report)] = new_row.values[0]

                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    nelbo_val="{:.4f}".format(avg_val_nelbo.avg),
                    kl_val="{:.4f}".format(avg_val_kl.avg),
                    rec_val="{:.4f}".format(avg_val_rec.avg),
                    refresh=True,
                )
    report.to_csv(f"{report_root}/{model_name}_{beta}_report.csv")
    return model, optimizer, report
