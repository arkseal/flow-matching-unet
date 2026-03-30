from pathlib import Path

import torch
import torch.optim as optim
from torchvision.utils import save_image, make_grid

from tqdm.auto import tqdm, trange

from src.flow import compute_loss, _generate
from src.model import FlowMatchingUNet
from src.data import get_train_data

def train(batch_size=512, num_workers=8, lr=1e-4, epochs=25, device='xpu',
          checkpoint_path=Path('./checkpoints'), save_path=Path('./results')):
    dataloader = get_train_data(batch_size=batch_size, num_workers=num_workers)
    val_shape = (16, 1, 28, 28)
    
    
    model = FlowMatchingUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in trange(epochs):
        model.train()
        total_loss = 0
        
        for x1, _ in tqdm(dataloader, desc=f'Epoch {epoch}', leave=False):
            x1 = x1.to(device)
            optimizer.zero_grad()
            
            loss = compute_loss(model, x1, device=device)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.cpu().item()
        
        avg_loss = total_loss/len(dataloader)
        tqdm.write(f'Epoch: {epoch} Loss: {avg_loss:.4f}')
        
        if epoch % 5 == 0:
            model.eval()
            tqdm.write(f'Generating Samples for epoch {epoch}:')
            
            generated_images = _generate(model, val_shape, device)
            
            image_save_path = save_path / f'epoch_{epoch:03d}.png'
            save_image(make_grid(generated_images, nrow=4), image_save_path)
            
            torch.save(model.state_dict(), checkpoint_path / f'model_epoch_{epoch}.pth')
    
    print('Training complete')
    torch.save(model.state_dict(), checkpoint_path / 'model_final.pth')
