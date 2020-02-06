import os
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms.functional as VF
import PIL
from pathlib import Path


from torchvision import models
from torchvision import transforms
from fastai.torch_core import requires_grad, children

from models import FeatureLoss, Generator
from utils import load_image, paintx

def model_fn():
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    model_dir = Path(os.path.abspath(os.path.dirname(__file__))) / 'adversarial_models'
    print("model dir {}".format(model_dir))
    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = model_dir / 'brushstroke_gan_final_adversarial_generator_info.pth'
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(model_info['condition_dim'],
                      model_info['fc_dim'],
                      model_info['image_size'],
                      model_info['channels'],
                      model_info['n_extra_layers'])

    # Load the stored model parameters.

    model_path = model_dir / 'brushstroke_gan_final_adversarial_generator_param.pth'
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))

    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(image_bytes):
    input_data = PIL.Image.open(io.BytesIO(image_bytes))
    return input_data


def predict_fn(input_data, model, epochs):
    print('reading data. test')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device :{}'.format(device))
    
    print('printing data')
    print(input_data)
    


    print('input_data size: {}'.format(input_data.size))
    
    # load image and transform
    size = 64
    base_image = load_image(input_data, size=400)
    base_image = transforms.CenterCrop(256)(base_image).resize((size,size))
    print('base_image size: {}'.format(base_image.size))
    
    # move image to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_image = VF.to_tensor(base_image).to(device)
    print('base_image shape after move to device: {}'.format(base_image.shape))
    
    # parameters
    condition_dim = 12
    num_strokes = 64
    lr = 0.01
    
    # conditions to optimize w/ RMSprop
    conditions = torch.empty((num_strokes, condition_dim)).uniform_(0, 1).to(device).requires_grad_()
    print(f'conditions.shape: {conditions.shape}')
    optimizer = optim.RMSprop([conditions], lr=lr)
    
    
    # load feature extractor


    base_loss = F.mse_loss
    
    
    
    
    # training
    train_epoch = epochs
    losses = []
    channels = 3
    
    model.eval()
    
    for epoch in range(train_epoch):
        def closure():
            conditions.data.clamp_(0,1)

            optimizer.zero_grad()

            num_strokes, condition_dim = conditions.shape

            strokes = model(conditions.view(num_strokes, condition_dim))

            canvas = paintx(strokes)

            the_loss = base_loss(canvas, base_image.view(1, *base_image.shape))

            the_loss.backward()

            losses.append(the_loss.item())

            if epoch % 1 == 0:
                print('epoch {}/{}; loss: {}'.format(epoch, train_epoch, torch.mean(torch.tensor(losses)).item()))


            return the_loss.item()

        optimizer.step(closure)
        
    conditions.data.clamp_(0, 1)
    num_strokes, condition_dim = conditions.shape
    strokes = model(conditions.view(num_strokes, condition_dim))

    print('--final--')
    base_image = base_image.to('cpu')
    base_image = VF.to_pil_image(base_image)
    img = paintx(strokes).squeeze().cpu()
    img = VF.to_pil_image(img)
    print(img)
    print('img size {}'.format(img.size))

    return img, base_image, conditions
