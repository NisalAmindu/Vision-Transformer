import torch 
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)


def evaluate(model,val_loader,device):
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  
            probs = torch.softmax(logits, dim=1)
            
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    
    
    # Top-1 Accuracy
    top1_acc = 100. * (all_preds == all_labels).mean()
    
    # Top-5 Accuracy
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_acc = 100. * np.mean([all_labels[i] in top5_preds[i] for i in range(len(all_labels))])
    
    
    # Results
    results = {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
    }
    
    return results
    
    
    
    
         