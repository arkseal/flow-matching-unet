import torch
import torch.optim as optim
from torchvision.utils import save_image, make_grid

from tqdm.auto import tqdm, trange

from src.flow import compute_loss, sample_ode
from src.model import FlowMatchingUNet
from src.data import get_train_data, inverse_normalization


def train(batch_size=512, num_workers=8, lr=1e-4, epochs=25, device='xpu'):
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
            
            generated_images = sample_ode(model, val_shape, device=device)
            
            generated_images = inverse_normalization(generated_images.cpu())
            save_path = f'results/epoch_{epoch:03d}.png'
            save_image(make_grid(generated_images, nrow=4), save_path)
            
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch}.pth')
    
    print('Training complete')
    torch.save(model.state_dict(), 'checkpoints/model_final.pth')