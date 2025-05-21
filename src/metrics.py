import torch as tr 

def compute_metrics(x_rec, x_true, mask, binary=False, threshold=0.5):
    """
    Calculates the F1 score, accuracy (sequence level and threshold),
    precision, recall, and perplexity.

    Args:
        x_rec (torch.Tensor): Reconstructed sequence [N, C, L].
        x_true (torch.Tensor): True sequence [N, C, L]. 
        mask (torch.Tensor): Mask to filter valid sequences [N, C, L].

    Returns:
        dict: Calculated metrics.
    """
    xt_rec = x_rec.permute(0,2,1)
    xt_true = x_true.permute(0,2,1) 

    total = mask.sum().item()
    if binary == False:
        # Obtener el índice de la clase predicha y verdadera
        pred_idx = xt_rec.argmax(dim=-1)   # [B, L]
        true_idx = xt_true.argmax(dim=-1)  # [B, L]     
        pred_flat = pred_idx[mask]    
        true_flat = true_idx[mask]
        
        TP    = (pred_flat == true_flat).sum().item() 
        FP = total - TP
        FN = FP
        TN=0. 
    else:
        pred_flat = (xt_rec[mask] > threshold).long()
        true_flat = xt_true[mask]
        
        TP = ((pred_flat == 1) & (true_flat == 1)).sum().item()
        FP = ((pred_flat == 1) & (true_flat == 0)).sum().item()
        FN = ((pred_flat == 0) & (true_flat == 1)).sum().item()
        TN = ((pred_flat == 0) & (true_flat == 0)).sum().item()
        
        
    accuracy = (TP+TN)/total if total>0 else 0.0
    precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
    recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

 
    seq_accuracy = (x_rec == x_true).all(dim=1).all(dim=1).float().mean().item()

    # Devolver todas las métricas
    return {"F1": f1, "Accuracy": accuracy, "Accuracy_seq": seq_accuracy}