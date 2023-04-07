

def train_epoch(model, dataloader, criterion, reg_criterion, optimizer, scheduler, device, wandb):
    losses = []
    model.train()
    for i, (img_names, images, genders, ages) in enumerate(dataloader):
        images = images.to(device)
        gt_genders = genders.to(device)
        gt_ages = ages.to(device)
        pred_genders, pred_ages = model(images)

        gender_loss = criterion(pred_genders, gt_genders)
        age_loss = reg_criterion(pred_ages, gt_ages)
        loss = gender_loss + 2 * age_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        wandb.log({'lr': scheduler.get_lr()[0], 'loss': loss.item()})

    return losses