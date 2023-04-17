import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import get_mnist_data
from utils import sample_gaussian

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cpred(model, model_name, class_number, figsize, title_padding, title_fontsize):
    label_tensor = torch.ones(1, dtype=torch.int64) * class_number
    z_sample = model.sample_z(batch=1)
    x_sample = (
        model.sample_x_given(z_sample, y=label_tensor)
        .view(28, 28)
        .cpu()
        .detach()
        .numpy()
    )
    fig, ax = plt.subplots(1, figsize=figsize)
    fig.suptitle(f"Model: {model_name}", y=title_padding, fontsize=title_fontsize)
    ax.imshow(x_sample, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_samples(
    model,
    model_name,
    conditional,
    classes_for_plot,
    figsize,
    title_padding,
    title_fontsize,
    walk_over_dim="none",
    variable_dim=0,
    generate=True,
):

    if conditional:
        number_of_classes = len(classes_for_plot)
        for class_id, _class in enumerate(classes_for_plot):
            label_tensor = torch.ones(3, dtype=torch.int64) * _class
            z_sample = model.sample_z(batch=3)
            x_sample = model.sample_x_given(z_sample, y=label_tensor)
            if class_id == 0:
                all_class_samples = x_sample
            else:
                all_class_samples = torch.cat((all_class_samples, x_sample), dim=0)

        tensor_for_plot = all_class_samples.view(number_of_classes * 3, 1, 28, 28)
        plot_samples_given_tensor_samples(
            tensor_for_plot=tensor_for_plot,
            model_name=model_name,
            conditional=conditional,
            figsize=figsize,
            title_padding=title_padding,
            title_fontsize=title_fontsize,
        )
    else:
        if walk_over_dim.lower() == "one":

            z_sample = model.sample_z(batch=1)
            assert (
                variable_dim < z_sample.shape[1]
            ), f"Maximum variable dimension could be {z_sample.shape[1]}"
            value_range = (z_sample.max() - z_sample.min()).item()
            z_samples = z_sample.numpy()
            value_to_add = np.linspace(0, value_range, 100)[1:]
            for index in range(99):
                value_vector_to_add = [
                    value_to_add[index] if _dim == variable_dim else 0
                    for _dim in range(z_sample.shape[1])
                ]
                temp_samp = (z_samples[0] + value_vector_to_add).reshape(1, -1)
                z_samples = np.append(z_samples, temp_samp, axis=0)
            z_samples = torch.tensor(z_samples, dtype=torch.float32)
            x_sample = model.sample_x_given(z_samples, y=None)
            tensor_for_plot = x_sample.view(100, 1, 28, 28)
            plot_samples_given_tensor_samples(
                tensor_for_plot=tensor_for_plot,
                model_name=model_name,
                conditional=conditional,
                figsize=figsize,
                title_padding=title_padding,
                title_fontsize=title_fontsize,
            )
        elif walk_over_dim.lower() == "all":
            z_sample = model.sample_z(batch=1)
            value_range = (z_sample.max() - z_sample.min()).item()
            z_samples = z_sample.numpy()
            value_to_add = np.linspace(0, value_range, 100)[1:]
            for dim in range(10):
                z_samples = z_sample.numpy()
                for index in range(99):
                    value_vector_to_add = [
                        value_to_add[index] if _dim == dim else 0
                        for _dim in range(z_sample.shape[1])
                    ]
                    temp_samp = (z_samples[0] + value_vector_to_add).reshape(1, -1)
                    z_samples = np.append(z_samples, temp_samp, axis=0)
                z_samples = torch.tensor(z_samples, dtype=torch.float32)
                x_sample = model.sample_x_given(z_samples, y=None)
                tensor_for_plot = x_sample.view(100, 1, 28, 28)
                plot_samples_given_tensor_samples(
                    tensor_for_plot=tensor_for_plot,
                    model_name=model_name + f" - Variable dimension= {dim}",
                    conditional=conditional,
                    figsize=figsize,
                    title_padding=title_padding,
                    title_fontsize=title_fontsize,
                )
        elif walk_over_dim.lower() == "none":

            if generate:
                z_samples = model.sample_z(batch=100)
            else:
                _, test_loader = get_mnist_data(
                    device=device, batch_size=100, shuffle=False
                )
                test_batch = next(iter(test_loader))[0]
                test_batch_for_pass_net = test_batch.to(device).reshape(
                    test_batch.size(0), -1
                )
                q_mean, q_var = model.enc.encode(x=test_batch_for_pass_net, y=None)
                z_samples = sample_gaussian(
                    q_mean.expand(100, 10), q_var.expand(100, 10)
                )

            x_samples = model.sample_x_given(z_samples, y=None)
            tensor_for_plot = x_samples.view(100, 1, 28, 28)
            plot_samples_given_tensor_samples(
                tensor_for_plot=tensor_for_plot,
                model_name=model_name,
                conditional=conditional,
                figsize=figsize,
                title_padding=title_padding,
                title_fontsize=title_fontsize,
            )
        else:
            raise ValueError("walk_over_dim argument should be one,all or none")


def plot_samples_given_tensor_samples(
    tensor_for_plot, model_name, conditional, figsize, title_padding, title_fontsize=18
):
    samples_array = tensor_for_plot.cpu().detach().numpy()
    if conditional:
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=figsize)
        fig.suptitle(f"Model: {model_name}", y=title_padding, fontsize=title_fontsize)
        i = 0
        for row in range(3):
            for col in range(3):
                ax[row, col].imshow(samples_array[i][0], cmap="gray")
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])
                i += 1

    else:
        fig, ax = plt.subplots(nrows=10, ncols=10, figsize=figsize)
        fig.suptitle(f"Model: {model_name}", y=title_padding, fontsize=title_fontsize)
        i = 0
        for row in range(10):
            for col in range(10):
                ax[row, col].imshow(samples_array[i][0], cmap="gray")
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])
                i += 1


def plot_metric(
    model_name,
    dataframe,
    metric_name,
    figsize=(20, 10),
    xylabel_fontsize=24,
    xy_ticks_fontsize=16,
    legend_fontsize=18,
    title_fontsize=30,
    linewidth_plot=4,
):

    metric_names = {
        "rec": "Reconstruction-loss",
        "kl": "KL-divergence",
        "nelbo": "Negetive-ELBO",
    }
    ##########    Prepare metric data    ##############
    train_report = dataframe[dataframe["mode"] == "train"]
    test_report = dataframe[dataframe["mode"] == "test"]

    last_train_batch_id = train_report["batch_index"].max()
    last_test_batch_id = test_report["batch_index"].max()

    report_train_last_index = train_report[
        train_report.batch_index == last_train_batch_id
    ]
    report_test_last_index = test_report[test_report.batch_index == last_test_batch_id]
    current_metric_train_name = f"avg_{metric_name}_till_current_batch"
    current_metric_test_name = f"avg_{metric_name}_till_current_batch"
    current_metric_train = report_train_last_index[current_metric_train_name].values
    current_metric_test = report_test_last_index[current_metric_test_name].values

    ##############    Plot   ################
    fig, ax = plt.subplots(1, figsize=figsize)
    fig.suptitle(f"{model_name} {metric_names[metric_name]}", fontsize=title_fontsize)
    ax.plot(
        current_metric_train,
        color="b",
        linewidth=linewidth_plot,
        label=f"Train - {metric_names[metric_name]}",
    )
    ax.plot(
        current_metric_test,
        color="r",
        linewidth=linewidth_plot,
        label=f"Test - {metric_names[metric_name]}",
    )

    ##########    Change Properties    ##############
    ax.set_xlabel("Epoch", fontsize=xylabel_fontsize)
    ax.set_ylabel(metric_names[metric_name], fontsize=xylabel_fontsize)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(loc=0, prop={"size": legend_fontsize})
    ax.tick_params(axis="x", labelsize=xy_ticks_fontsize)
    ax.tick_params(axis="y", labelsize=xy_ticks_fontsize)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def compare_metrics(
    reports,
    models_name,
    metric_name,
    colors,
    figsize,
    xylabel_fontsize,
    xy_ticks_fontsize,
    legend_fontsize,
    title_fontsize,
    linewidth_plot,
):

    metric_names = {
        "rec": "Reconstruction-loss",
        "kl": "KL-divergence",
        "nelbo": "Negetive-ELBO",
    }

    current_metric_train_name = f"avg_{metric_name}_till_current_batch"
    current_metric_test_name = f"avg_{metric_name}_till_current_batch"

    ##########    Prepare metric data    ##############
    train_reports = [report[report["mode"] == "train"] for report in reports]
    test_reports = [report[report["mode"] == "test"] for report in reports]

    last_trains_batch_id = [
        train_report["batch_index"].max() for train_report in train_reports
    ]
    last_tests_batch_id = [
        test_report["batch_index"].max() for test_report in test_reports
    ]
    report_trains_last_index, report_tests_last_index = [], []
    current_metric_trains, current_metric_tests = [], []

    for index in range(len(reports)):
        report_trains_last_index.append(
            train_reports[index][
                train_reports[index].batch_index == last_trains_batch_id[index]
            ]
        )
        report_tests_last_index.append(
            test_reports[index][
                test_reports[index].batch_index == last_tests_batch_id[index]
            ]
        )
        current_metric_trains.append(
            report_trains_last_index[index][current_metric_train_name].values
        )
        current_metric_tests.append(
            report_tests_last_index[index][current_metric_test_name].values
        )

    ##############    Plot   ################
    fig, ax = plt.subplots(1, figsize=figsize)
    fig.suptitle(
        f"Compare {metric_names[metric_name]} of models", fontsize=title_fontsize
    )

    for model_index in range(len(models_name)):
        print(model_index)
        ax.plot(
            current_metric_trains[model_index],
            color=colors[model_index],
            linewidth=linewidth_plot,
            label=f"Train - {models_name[model_index]}",
        )
        ax.plot(
            current_metric_tests[model_index],
            color=colors[model_index],
            linewidth=linewidth_plot,
            label=f"Test - {models_name[model_index]}",
        )

    ##########    Change Properties    ##############
    ax.set_xlabel("Epoch", fontsize=xylabel_fontsize)
    ax.set_ylabel(metric_names[metric_name], fontsize=xylabel_fontsize)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(loc=0, prop={"size": legend_fontsize})
    ax.tick_params(axis="x", labelsize=xy_ticks_fontsize)
    ax.tick_params(axis="y", labelsize=xy_ticks_fontsize)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
