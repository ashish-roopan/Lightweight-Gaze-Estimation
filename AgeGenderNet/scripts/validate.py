

def valid_epoch(model, dataloader, criterion, reg_criterion, device, wandb):
    losses = []
    model.eval()
    for i, (img_names, images, genders, ages) in enumerate(dataloader):
        images = images.to(device)
        gt_genders = genders.to(device)
        gt_ages = ages.to(device)
        pred_genders, pred_ages = model(images)
        
        gender_loss = criterion(pred_genders, gt_genders)
        age_loss = reg_criterion(pred_ages, gt_ages)
        loss = gender_loss + age_loss

        losses.append(loss.item())
    return losses