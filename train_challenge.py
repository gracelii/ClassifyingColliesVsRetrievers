"""
EECS 445 - Introduction to Machine Learning
Winter 2026 - Project 2

Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""

import torch
import matplotlib.pyplot as plt

from dataset_challenge import get_train_val_test_loaders
from model.source import Source 
from model.challenge import Challenge
from train_common import evaluate_epoch, early_stopping, restore_checkpoint, save_checkpoint, train_epoch
from utils import config, set_random_seed, make_training_plot


def main():
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("challenge.batch_size"),
    )
    
    # Model
    model = Challenge()
    #loading source model weights into the challenge model 
    print("Loading source model for transfer learning...")
    source_model = Source()
    source_model, _, _ = restore_checkpoint(
        source_model,
        config("source.checkpoint"),
        force=True,
        pretrain=True,
    )

    model.conv1.weight.data = source_model.conv1.weight.data.clone()
    model.conv1.bias.data = source_model.conv1.bias.data.clone()
    model.conv2.weight.data = source_model.conv2.weight.data.clone()
    model.conv2.bias.data = source_model.conv2.bias.data.clone()


    for name, param in model.named_parameters():
        #freeze conv1 and conv2 layers 
        if name.startswith("conv1") or name.startswith("conv2"):
            param.requires_grad = False



    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
    #filter(lambda p: p.requires_grad, model.parameters()),

    model.parameters(),
    #lr=config("challenge.learning_rate"),
    #lr = 1e-3,
    lr = 5e-4,
    weight_decay=0.05
    #weight_decay = 0.1 
)
    
      #learning rate scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    

    # Attempts to restore the latest checkpoint if exists
    print("Loading challenge...")
    model, start_epoch, stats = restore_checkpoint(model, config("challenge.checkpoint"))

    axes = make_training_plot()

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: define patience for early stopping
    patience = config("challenge.patience")
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("challenge.checkpoint"), stats)
        #evaluate the scheduler on the current validation loss 
        scheduler.step(stats[-1][1])

        # Updates early stopping parameters
        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)

        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    plt.savefig("challenge_training_plot.png", dpi=200)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
