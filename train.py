from pathlib import Path

import torch
import torch.optim as optim
from torchvision.utils import save_image, make_grid

from tqdm.auto import tqdm, trange

from src.flow import compute_loss, _generate
from src.model import FlowMatchingUNet
from src.data import get_train_data

def train(dataset_name='MNIST', batch_size=512, num_workers=8, lr=1e-4, epochs=25, device='xpu',
          checkpoint_path=Path('./checkpoints'), save_path=Path('./results'), resume_path=None):
    start_epoch = 1
    val_shape = (16, 1, 28, 28)
    
    dataloader, num_channels = get_train_data(dataset_name=dataset_name, batch_size=batch_size, num_workers=num_workers)
    
    model = FlowMatchingUNet(num_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if resume_path:
        checkpoint = torch.load(resume_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epochs = epochs + start_epoch - 1
        loss = checkpoint['loss']
    
    for epoch in trange(start_epoch, epochs+1):
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
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            torch.save(checkpoint, checkpoint_path / f'model_epoch_{epoch}.pth')
    
    print('Training complete')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path / f'model_epoch_{epoch}.pth')
    
    final_model = {
        'model_state_dict': model.state_dict()
    }
    torch.save(final_model, checkpoint_path / 'model_final.pth')
