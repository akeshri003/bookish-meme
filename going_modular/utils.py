"""
Contains various utility functions for PyTorch model training and saving.
"""
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision import transforms, datasets
from typing import Dict, Tuple, List
from PIL import Image

import torch
import torchvision

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
  model: A target PyTorch model to save.
  target_dir: A directory for saving the model to.
  model_name: A filename for the saved model. Should include
    either ".pth" or ".pt" as the file extension.

  Example usage:
  save_model(model=model_0,
              target_dir="models",
              model_name="05_going_modular_tingvgg_model.pth")
  """

  # Create target dir path
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  # Create model path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pth'mor '.pt'"
  model_save_path = target_dir_path / model_name

  # Saving the model
  print(f"[INFO] Saving the model to {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)


def plot_loss_curves(results: Dict):
  """Plots a graph of train and test loss and accuracy over the epochs, during training.
  """
  loss = results["train_loss"]
  test_loss = results["test_loss"]

  acc = results["train_acc"]
  test_acc = results["test_acc"]

  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15, 7))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="Train Loss")
  plt.plot(epochs, test_loss, label="Test loss")
  plt.title("Loss Curves")
  plt.xlabel("Epochs")
  plt.legend()

  plt.subplot(1,2,2)
  plt.plot(epochs, acc, label="Train accuracy")
  plt.plot(epochs, test_acc, label="Test accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()

def pred_plot_image(model: torch.nn.Module, 
                    image_path: str,
                    class_names: List[str],
                    image_size: Tuple[int, int]= (224, 224),
                    transform: torchvision.transforms = None,
                    device: torch.device="cpu"
                    ):
  
  img = Image.open(image_path)

  if transform is not None:
    image_transform = transform
  else:
    image_transform = transforms.Compose([
      transforms.Resize(size=image_size),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

  transformed_img = image_transform(img)

  model.to(device)

  model.eval()
  with torch.inference_mode():
    transformed_img = image_transform(img).unsqueeze(dim=0)
    logits = model(transformed_img.to(device))
    pred_probs = torch.softmax(logits, dim=1)
    pred_prob_class = torch.argmax(pred_probs, dim=1)

  plt.figure()
  plt.imshow(img)
  plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
  plt.axis(False)




