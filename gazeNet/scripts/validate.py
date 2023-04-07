import torch
import torch.nn as nn
from torch.autograd import Variable




alpha = 1
softmax = nn.Softmax(dim=1).cuda()
idx_tensor = [idx for idx in range(90)]
idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda()
def valid_epoch(model, dataloader, criterion, reg_criterion, device, wandb):
    losses = []
    model.eval()
    for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(dataloader):
        sum_loss_pitch_gaze = 0
        sum_loss_yaw_gaze = 0
        images_gaze = Variable(images_gaze).to(device)
        
        # Binned labels
        label_pitch_gaze = Variable(labels_gaze[:, 0]).to(device)
        label_yaw_gaze = Variable(labels_gaze[:, 1]).to(device)

        # Continuous labels
        label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).to(device)
        label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).to(device)

        pitch, yaw = model(images_gaze)

        # Cross entropy loss
        loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
        loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

        # MSE loss
        pitch_predicted = softmax(pitch)
        yaw_predicted = softmax(yaw)

        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 4 - 180

        loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont_gaze)
        loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont_gaze)

        # Total loss
        loss_pitch_gaze += alpha * loss_reg_pitch
        loss_yaw_gaze += alpha * loss_reg_yaw

        sum_loss_pitch_gaze += loss_pitch_gaze
        sum_loss_yaw_gaze += loss_yaw_gaze


        final_loss = (sum_loss_pitch_gaze + sum_loss_yaw_gaze).item()
        losses.append(final_loss)

    return losses






       
    
       
    