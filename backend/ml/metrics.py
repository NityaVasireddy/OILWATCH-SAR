import numpy as np

def binary_metrics(prob,truth,threshold=.5):
 p=np.asarray(prob)>=threshold; t=np.asarray(truth)>0
 tp=np.logical_and(p,t).sum(); tn=np.logical_and(~p,~t).sum(); fp=np.logical_and(p,~t).sum(); fn=np.logical_and(~p,t).sum(); eps=1e-9
 dice=2*tp/(2*tp+fp+fn+eps); iou=tp/(tp+fp+fn+eps); precision=tp/(tp+fp+eps); recall=tp/(tp+fn+eps); f1=2*precision*recall/(precision+recall+eps); acc=(tp+tn)/(tp+tn+fp+fn+eps); fpr=fp/(fp+tn+eps); fnr=fn/(fn+tp+eps)
 return {k:float(v) for k,v in locals().items() if k in ['dice','iou','precision','recall','f1','acc','fpr','fnr']}
