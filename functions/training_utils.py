import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from augmentation import apply_augmentation
from itertools import cycle

# optimizer selection flag
# options: 'adam', 'adamw', 'sgd', 'nadam'
optimizer_choice = "adamw"

# shared parameters
learning_rate = 0.001
weight_decay = 0.01  # for adomw only

# sgd specific
sgd_momentum = 0.9
sgd_nesterov = True  # nestrov momentum

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_alpha(current_step, total_steps):
    """
    computes alpha for gradient reversal layer where alpha increases from 0 to 1 as training progresses
    """
    progress = current_step / total_steps
    alpha = 2 / (1 + math.exp(-5 * progress)) - 1
    return alpha


def get_optimizer(model, choice=optimizer_choice):
    """
    returns optimizer with separate learning rates
    discriminator uses higher learning rate to maintain
    adversarial balance against gradient reversal
    """
    discriminator_params = list(model.domain_discriminator.parameters())
    discriminator_param_ids = set(id(p) for p in discriminator_params)
    other_params = [
        p for p in model.parameters() if id(p) not in discriminator_param_ids
    ]

    if choice == "adam":
        return optim.Adam(
            [
                {"params": other_params, "lr": learning_rate},
                {"params": discriminator_params, "lr": learning_rate * 10},
            ]
        )
    elif choice == "adamw":
        return optim.AdamW(
            [
                {
                    "params": other_params,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                },
                {
                    "params": discriminator_params,
                    "lr": learning_rate * 10,
                    "weight_decay": weight_decay,
                },
            ]
        )
    elif choice == "sgd":
        sgd_lr = 0.01  # determined via 20-subject LR probe in duplicate notebook where 0.001 too flat, 0.05/0.1 unstable
        return optim.SGD(
            [
                {
                    "params": other_params,
                    "lr": learning_rate,
                    "momentum": sgd_momentum,
                    "nesterov": sgd_nesterov,
                },
                {
                    "params": discriminator_params,
                    "lr": learning_rate * 10,
                    "momentum": sgd_momentum,
                    "nesterov": sgd_nesterov,
                },
            ]
        )
    elif choice == "nadam":
        return optim.NAdam(
            [
                {"params": other_params, "lr": learning_rate},
                {"params": discriminator_params, "lr": learning_rate * 10},
            ]
        )
    else:
        raise ValueError(f"Unknown optimizer: {choice}")


class EarlyStopping:
    """
    stops training when monitored metric stops improving
    how many epochs to wait before stopping
    min_delta: minimum change to count as improvement
    mode: max for accuracy and kappa, min for loss
    """

    def __init__(self, patience=10, min_delta=0.001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        if self.best_value is None:
            self.best_value = value
            return

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def reset(self):
        self.best_value = None
        self.counter = 0
        self.should_stop = False


class BestModelCheckpoint:
    """
    saves model weights whenever monitored metric improves
    loads best weights at end of training
    monitored metric: kappa by default
    mode: max for kappa and accuracy, min for loss
    """

    def __init__(self, mode="max"):
        self.mode = mode
        self.best_value = None
        self.best_weights = None
        self.best_epoch = None

    def step(self, model, value, epoch):
        if self.best_value is None:
            self.best_value = value
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return True

        if self.mode == "max":
            improved = value > self.best_value
        else:
            improved = value < self.best_value

        if improved:
            self.best_value = value
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return True

        return False

    def load_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

    def reset(self):
        self.best_value = None
        self.best_weights = None
        self.best_epoch = None


def initialise_log():
    """
    creates empty log dictionary for one training run
    stores per epoch metrics and overall training summary
    """
    return {
        # per epoch metrics
        "epoch": [],
        "class_loss": [],
        "domain_loss": [],
        "accuracy": [],
        "kappa": [],
        "auc": [],
        "alpha": [],
        "learning_rate": [],
        "best_weights_saved": [],
        "domain_accuracy": [],
        "source_domain_accuracy": [],
        "target_domain_accuracy": [],
        "per_class_accuracy": [],
        "confusion_matrix": [],
        # pre-adaptation distribution measures (band power, model independent)
        "pre_mmd_bandpower_multiscale": None,
        "pre_mmd_bandpower_optimal": None,
        "pre_mmd_bandpower_optimal_bandwidth": None,
        "pre_riemannian": None,
        # post-adaptation band power measures (directly comparable to pre)
        "post_mmd_bandpower_multiscale": None,
        "post_mmd_bandpower_optimal": None,
        "post_mmd_bandpower_optimal_bandwidth": None,
        "post_riemannian": None,
        # post-adaptation embedding measures (what model learned)
        "post_mmd_embedding_multiscale": None,
        "post_mmd_embedding_optimal": None,
        "post_mmd_embedding_optimal_bandwidth": None,
        # training summary
        "best_epoch": None,
        "best_kappa": None,
        "total_epochs_run": None,
        "early_stopping_triggered": False,
        "adaptation_quality": None,
        "initial_domain_loss": None,
        "final_domain_loss": None,
        "final_accuracy": None,
        "final_kappa": None,
        "final_auc": None,
        "final_per_class_accuracy": None,
        "final_confusion_matrix": None,
    }


def apply_adabn(model, target_loader, n_passes=10):
    """
    applies adaptive batch normalisation to model
    updates batch norm statistics using target subject data only
    called after training, before final evaluation
    no labels required, unsupervised post training step
    n_passes: number of times to pass through target data
    more passes gives more stable statistics estimate
    """
    # set to train mode so batch norm updates running statistics
    model.train()
    model.to(device)

    with torch.no_grad():
        for _ in range(n_passes):
            for target_batch in target_loader:
                target_x = target_batch[0].to(device)
                # forward pass through feature extractor only
                # updates batch norm running mean and variance
                model.feature_extractor(target_x)

    # back to eval mode for evaluation
    model.eval()
    return model


def train_epoch(
    model,
    source_loader,
    target_loader,
    optimizer,
    current_step,
    total_steps,
    lambda_domain=0.6,
    use_gradient_penalty=True,
    use_entropy_weighting=True,
    use_augmentation=False,
):
    """
    trains DANN for one epoch where source loader is labeled EEG from source subjects and target loader is unlabeled EEG from target subject and lambda_domain are the weight of domain loss relative to classification loss
    """
    model.train()
    model.to(device)

    classification_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    total_class_loss = 0
    total_domain_loss = 0
    n_batches = 0
    total_domain_accuracy = 0
    total_source_domain_accuracy = 0
    total_target_domain_accuracy = 0
    target_iter = cycle(target_loader)

    # zip source and target loaders together
    # shortest loader determines number of batches
    for source_x, source_y in source_loader:
        target_batch = next(target_iter)
        target_x = target_batch[0]

        if use_augmentation:
            source_x = apply_augmentation(source_x)

        # move to device
        source_x = source_x.to(device)
        source_y = source_y.to(device)
        target_x = target_x.to(device)

        # compute alpha for this step
        alpha = get_alpha(current_step, total_steps)
        current_step += 1

        # forward pass on source
        source_class_out, source_domain_out, _ = model(source_x, alpha)

        # forward pass on target
        _, target_domain_out, _ = model(target_x, alpha)

        # classification loss on source only
        class_loss = classification_criterion(source_class_out, source_y)

        # domain loss on both source and target
        # source domain label is 1, target domain label is 0
        source_domain_labels = torch.ones(source_x.size(0), 1).to(device)
        target_domain_labels = torch.zeros(target_x.size(0), 1).to(device)

        if use_entropy_weighting and model.use_conditional_discriminator:
            # entropy based weighting from CDAN paper
            # easy to transfer examples get higher weight
            with torch.no_grad():
                source_probs = torch.softmax(source_class_out, dim=1)
                target_probs = torch.softmax(
                    model.classifier(model.feature_extractor(target_x)), dim=1
                )

                # entropy H(p) = -sum(p * log(p))
                source_entropy = -(source_probs * torch.log(source_probs + 1e-6)).sum(
                    dim=1
                )
                target_entropy = -(target_probs * torch.log(target_probs + 1e-6)).sum(
                    dim=1
                )

                # weight w(H) = 1 + exp(-H)
                source_weights = (1 + torch.exp(-source_entropy)).unsqueeze(1)
                target_weights = (1 + torch.exp(-target_entropy)).unsqueeze(1)

            # weighted domain loss
            source_domain_loss = (
                source_weights
                * nn.functional.binary_cross_entropy_with_logits(
                    source_domain_out, source_domain_labels, reduction="none"
                )
            ).mean()
            target_domain_loss = (
                target_weights
                * nn.functional.binary_cross_entropy_with_logits(
                    target_domain_out, target_domain_labels, reduction="none"
                )
            ).mean()
            domain_loss = source_domain_loss + target_domain_loss
        else:
            domain_loss = domain_criterion(
                source_domain_out, source_domain_labels
            ) + domain_criterion(target_domain_out, target_domain_labels)

        # domain classifier accuracy
        with torch.no_grad():
            source_domain_pred = (source_domain_out > 0).float()
            target_domain_pred = (target_domain_out > 0).float()
            source_correct = (source_domain_pred == source_domain_labels).float().mean()
            target_correct = (target_domain_pred == target_domain_labels).float().mean()
            domain_accuracy = (source_correct + target_correct) / 2

        total_domain_accuracy += domain_accuracy.item()
        total_source_domain_accuracy += source_correct.item()
        total_target_domain_accuracy += target_correct.item()

        # gradient penalty on discriminator
        # bypasses GRL by calling network directly
        # keeps discriminator gradients within reasonable range
        if use_gradient_penalty:
            gp_lambda = 1.0
            source_features = model.feature_extractor(source_x)
            target_features = model.feature_extractor(target_x)

            # interpolate between source and target feature distributions
            epsilon = torch.rand(source_features.size(0), 1).to(device)
            epsilon = epsilon.expand_as(source_features)
            interpolated = (
                (epsilon * source_features + (1 - epsilon) * target_features)
                .detach()
                .requires_grad_(True)
            )

            # forward pass through discriminator network only, no GRL
            if model.use_conditional_discriminator:
                # for CDAN need to compute multilinear map for interpolated features
                with torch.no_grad():
                    interp_class_out = model.classifier(interpolated)
                    interp_probs = torch.softmax(interp_class_out, dim=1)
                interp_multilinear = torch.bmm(
                    interpolated.unsqueeze(2), interp_probs.unsqueeze(1)
                ).view(interpolated.size(0), -1)
                interp_domain_out = model.domain_discriminator.network(
                    interp_multilinear
                )
            else:
                interp_domain_out = model.domain_discriminator.network(interpolated)

            # compute gradient norm
            gradients = torch.autograd.grad(
                outputs=interp_domain_out,
                inputs=interpolated,
                grad_outputs=torch.ones_like(interp_domain_out),
                create_graph=True,
                retain_graph=True,
            )[0]

            gradient_norm = gradients.norm(2, dim=1)
            gradient_penalty = gp_lambda * ((gradient_norm - 1) ** 2).mean()

        # total loss with gradient penalty
        if use_gradient_penalty:
            total_loss = class_loss + lambda_domain * domain_loss + gradient_penalty
        else:
            total_loss = class_loss + lambda_domain * domain_loss

        # backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_class_loss += class_loss.item()
        total_domain_loss += domain_loss.item()
        n_batches += 1

    mean_class_loss = total_class_loss / n_batches
    mean_domain_loss = total_domain_loss / n_batches
    mean_domain_accuracy = total_domain_accuracy / n_batches
    mean_source_domain_accuracy = total_source_domain_accuracy / n_batches
    mean_target_domain_accuracy = total_target_domain_accuracy / n_batches

    return (
        mean_class_loss,
        mean_domain_loss,
        mean_domain_accuracy,
        mean_source_domain_accuracy,
        mean_target_domain_accuracy,
        current_step,
    )
