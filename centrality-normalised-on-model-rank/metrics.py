import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def compute_ranking( compare_value, rank_type ):
  return rankdata( np.sum( -compare_value, axis = 0 ), method = rank_type )
#end

def compute_precision_at_k( predictions, labels, k = 10, rank_type = "min" ):
  predictions_binary = np.around( predictions )
  pred_ranks = compute_ranking( predictions_binary, rank_type = rank_type )
  label_ranks = compute_ranking( labels, rank_type = rank_type )
  pred_ranks_and_idx = list( filter( lambda x: x[1] <= k, sorted( enumerate( pred_ranks.tolist() ), key = lambda x: x[1] ) ) )
  label_ranks_and_idx = list( filter( lambda x: x[1] <= k, sorted( enumerate( label_ranks.tolist() ), key = lambda x: x[1] ) ) )
  labels = [ idx for idx, rank in label_ranks_and_idx ]
  prec = sum( [ 1 if idx in labels else 0 for idx, rank in pred_ranks_and_idx ] ) / len( pred_ranks_and_idx ) if len( pred_ranks_and_idx ) > 0 else 0
  return prec
#end compute_precision_at_k

def confusion_matrix( predictions, labels ):
  # Predicted true
  PT = np.around( predictions ).astype( np.bool )
  # Predicted false
  PF = np.logical_not( PT )
  # Label true
  LT = labels.astype( np.bool )
  # Label false
  LF = np.logical_not( LT )
  
  # True Positives
  TP = np.sum( np.logical_and( PT, LT ).astype( np.float ) )
  # False Positives
  FP = np.sum( np.logical_and( PT, LF ).astype( np.float ) )
  # True Negatives
  TN = np.sum( np.logical_and( PF, LF ).astype( np.float ) )
  # False Negatives
  FN = np.sum( np.logical_and( PF, LT ).astype( np.float ) )
  
  assert( TP+FP+TN+FN == predictions.shape[0] * predictions.shape[1] )
  assert( TP+FP+TN+FN == labels.shape[0] * labels.shape[1] )
  
  # True Positive Rate AKA Sensitivity, Recall, Hit Rate
  den = TP + FN
  TPR = TP / den if den != 0 else 0
  # True Negative Rate AKA Specificity
  den = TN + FP
  TNR = TN / den if den != 0 else 0
  # Positive Predictive Value AKA Precision
  den = TP + FP
  PPV = TP / den if den != 0 else 0
  # Negative Predictive Value
  den = TN + FN
  NPV = TN / den if den != 0 else 0
  
  # False Negative Rate AKA Miss Rate
  den = FN + TP
  FNR = FN / den if den != 0 else 0
  # False Positive Rate AKA Fall-out
  den = FP + TN
  FPR = FP / den if den != 0 else 0
  # False Discovery Rate
  den = FP + TP
  FDR = FP / den if den != 0 else 0
  # False Omission Rate
  den = FN + TN
  FOR = FN / den if den != 0 else 0
  
  # Accuracy
  den = TP + TN + FP + FN
  ACC = ( TP + TN ) / den if den != 0 else 0 
  
  # F1 score
  den = ( 2 * TP ) + FP + FN
  F1 = ( 2 * TP ) / den if den != 0 else 0
  # Matthews Correlation Coefficient
  den = np.sqrt( ( TP + FP ) * ( TP + FN ) * ( TN + FP ) * ( TN + FN ) )
  MCC = ( ( TP * TN ) - ( FP * FN ) ) / den if den != 0 else 0
  # Bookmaker Informedness AKA Informedness
  BM = TPR + TNR - 1
  # Markedness
  MK = PPV + NPV - 1
  
  #Area under ROC curve
  AUC = roc_auc_score( np.reshape(labels, -1), np.reshape(predictions, -1) )
  
  return TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK, AUC
#end confusion_matrix
