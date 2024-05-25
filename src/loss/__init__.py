from .original_loss import DSLoss, MAELoss, MSELoss, BCELoss, DiceLoss, structure_loss, BCELogits, iou_loss
from .SI_loss import SILoss


def get_loss(config, epoch=0):
    # loss_function = DSLoss(criterion=[BCELoss, DiceLoss])
    criterion = []
    losses = config["losses"].split(',')

    for loss in losses:
        criterion.append(eval(loss))

    if config["SI"]:
        if epoch >= config["warm_up"]:
            loss_function = SILoss(criterion)
        else:
            loss_function = DSLoss(criterion)
    else:
        loss_function = DSLoss(criterion)

    return loss_function