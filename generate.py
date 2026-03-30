import torch
from torchvision.utils import save_image, make_grid

from src.flow import _generate
from src.model import FlowMatchingUNet

def generate(model_path, save_path, shape, nrow, device):
    print('Loading Model...')
    model = FlowMatchingUNet()
    
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    print('Loaded Model')

    print(f'Generating Samples...')
    generated_images = _generate(model, shape, device, leave_progress=True)

    save_image(make_grid(generated_images, nrow=nrow), save_path)
