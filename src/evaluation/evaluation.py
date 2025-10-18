import torch 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)


def evaluate(model,val_loader,classes,device):
    
    os.makedirs('experiments/results', exist_ok=True)
    
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
    
    # Per class accuracy
    per_class_acc = {}
    for i, cls in enumerate(classes):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[cls] = 100. * (all_preds[mask] == all_labels[mask]).mean()
            
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                   target_names=classes, 
                                   output_dict=True)
    
    
    # Confusion matrix plot
    plt.figure(figsize=(100, 80))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('experiments/results/confusion_matrix.png', dpi=300)
    plt.close()
    print("Confusion matrix saved")
    
    # Per-class accuracy bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(per_class_acc.keys(), per_class_acc.values())
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig('experiments/results/per_class_accuracy.png', dpi=300)
    plt.close()
    print("Per-class accuracy saved")
    
    
    
    # Compare with paper
    paper_results = {
        'ViT-B/16': 98.13,
        'Our Implementation': top1_acc
    }
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(paper_results.keys(), paper_results.values())
    bars[0].set_color('skyblue')
    bars[1].set_color('lightcoral')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Comparison with Paper Results')
    plt.ylim([90, 100])
    
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('experiments/results/paper_comparison.png', dpi=300)
    plt.close()
    print("Paper comparison saved")
    
    
    # Results
    results = {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'per_class_accuracy': per_class_acc,
        'paper_accuracy': 98.13,
        'difference_from_paper': float(top1_acc - 98.13),
        'total_samples': len(all_labels),
        'classification_report': report
    }
    
    with open('experiments/results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results JSON saved")
    
    
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"\nPaper Result: 98.13%")
    print(f"Our Result: {top1_acc:.2f}%")
    print(f"Difference: {top1_acc - 98.13:+.2f}%")
    print("\nPer-Class Accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"  {cls:10s}: {acc:.2f}%")
    print("="*60)
    print("\nAll results saved to: experiments/results/")
    print("  - evaluation_results.json")
    print("  - confusion_matrix.png")
    print("  - per_class_accuracy.png")
    print("  - paper_comparison.png")
    
    return results
    
    
    
    
         