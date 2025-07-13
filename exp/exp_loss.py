# coding : utf-8
# Author : Yuxiang Zeng


# 在这里加上每个Batch的loss，如果有其他的loss，请在这里添加，
def compute_loss(model, pred, label, config):
    loss = model.loss_function(pred, label)
    try:
        loss = loss
    except Exception as e:
        print(f"Error in computing loss: {e}")
        pass

    return loss
