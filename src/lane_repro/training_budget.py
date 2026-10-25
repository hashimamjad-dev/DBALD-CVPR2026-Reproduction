"""Training-time extrapolation from measured iteration times; no framework imports."""
import math

def epoch_seconds(iteration_seconds,batch_size,images):
    if iteration_seconds<=0 or batch_size<1 or images<1: raise ValueError('Positive iteration time, batch size and image count required')
    return iteration_seconds*math.ceil(images/batch_size)

def budget(iteration_seconds,batch_size,images,epochs,runs=1):
    """Hours for epochs x runs at the measured speed; data loading, validation and cooling excluded."""
    per_epoch=epoch_seconds(iteration_seconds,batch_size,images)
    return dict(epoch_seconds=per_epoch,run_hours=per_epoch*epochs/3600,total_hours=per_epoch*epochs*runs/3600)
