"""
Contains functions fo training and testing a PyTorch model
"""

from typing import Tuple, Dict, List

import torch

from tqdm.auto import tqdm

from tqdm.auto import tqdm
from typing import Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward pass,
  loss calculations, optimizer step).

  Args:
    model: A PyTorch model to be trained
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function
    device: A target device to compute on.

  Returns:
    A tuple of training loss and training accuracy metrics
    in the form (train_loss, train_accuracy).
  """
  # Put the model in train mode.
  model.train()

  # Setup train_loss and train_accuracy values.
  train_loss, train_acc = 0, 0

  # Loop through each example in a batch.
  for batch, (X, y) in tqdm(enumerate(dataloader)):
    # send data to target device.
    X, y = X.to(device), y.to(device)

    # forward pass
    train_logit = model(X)
    pred = torch.softmax(train_logit, dim=1).argmax(dim=1)

    # loss calculation
    loss = loss_fn(train_logit, y.type(torch.float))

    # zero grad
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # gradient descent
    optimizer.step()

    train_loss += loss.item()
    train_acc += (pred==y).sum().item()/len(pred)

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y.type(torch.float))
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

from tqdm.auto import tqdm
from typing import Dict, List

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device)->Dict[str, List[float]]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
  model: A PyTorch model to be trained and tested.
  train_dataloader: A DataLoader instance for the model to be trained on.
  test_dataloader: A DataLoader instance for the model to be tested on.
  optimizer: A PyTorch optimizer to help minimize the loss function.
  loss_fn: A PyTorch loss function to calculate loss on both datasets.
  epochs: An integer indicating how many epochs to train for.
  device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
  A dictionary of training and testing loss as well as training and
  testing accuracy metrics. Each metric has a value in a list for
  each epoch.
  In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]}
  For example if training for epochs=2:
                {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]}
    """
  results = {"train_loss":[],
              "train_acc":[],
              "test_loss":[],
              "test_acc":[]}

  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f}"
    )

    results["train_loss"] = train_loss
    results["train_acc"] = train_acc
    results["test_loss"] = test_loss
    results["test_acc"] = test_acc

  return results
