import time
import copy
import torch

from tqdm import tqdm

from src.attacks.classifier import pgd_classifier_attack


def generate_mixed_batch(fashionnet, inputs, category_labels, texture_labels,
                         category_criterion, texture_criterion, epsilon,
                         attack_strength):
    adv_inputs = torch.zeros(inputs.shape).to(inputs.device)
    partition_size = inputs.size(0) // (attack_strength + 1)

    # generate adversarial examples for k=0 .. attack_strength
    for k in range(attack_strength + 1):
        start_idx = partition_size * k
        end_idx = start_idx + partition_size
        if k != attack_strength:
            inputs_k = inputs[start_idx:end_idx]
            category_labels_k = category_labels[start_idx:end_idx]
            texture_labels_k = texture_labels[start_idx:end_idx]
        else:
            # this gets max share
            inputs_k = inputs[start_idx:]
            category_labels_k = category_labels[start_idx:]
            texture_labels_k = texture_labels[start_idx:]

        adv_inputs_k = pgd_classifier_attack(fashionnet,
                                             inputs_k,
                                             category_labels_k,
                                             texture_labels_k,
                                             category_criterion,
                                             texture_criterion,
                                             epsilon=epsilon,
                                             k=k)
        if k != attack_strength:
            adv_inputs[start_idx:end_idx] = adv_inputs_k
        else:
            adv_inputs[start_idx:] = adv_inputs_k
    return adv_inputs


def train_adversarial_epoch(fashionnet,
                            dataloader,
                            category_criterion,
                            texture_criterion,
                            optimizer,
                            writer,
                            global_step,
                            attack_strength=8,
                            epsilon=0.03,
                            batch_mixing=False):
    fashionnet.model.train()
    steps = 0
    running_loss = 0.0
    running_corrects = 0.0

    print(f'Training model using attack strength K={attack_strength}')

    # iterate over data
    for batch in tqdm(dataloader):
        inputs = batch['image'].to(fashionnet.device)
        category_labels = batch['category'].to(fashionnet.device)
        texture_labels = batch['textures'].to(fashionnet.device)

        # generate adversarial examples
        if batch_mixing:
            adv_inputs = generate_mixed_batch(fashionnet, inputs,
                                              category_labels, texture_labels,
                                              category_criterion,
                                              texture_criterion, epsilon,
                                              attack_strength)
        else:
            adv_inputs = pgd_classifier_attack(fashionnet,
                                               inputs,
                                               category_labels,
                                               texture_labels,
                                               category_criterion,
                                               texture_criterion,
                                               epsilon=epsilon,
                                               k=attack_strength)

        # train on adversarial examples
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            category_outputs, texture_outputs = fashionnet.model(adv_inputs)
            category_preds = torch.argmax(category_outputs, dim=1)
            category_loss = category_criterion(category_outputs,
                                               category_labels)
            texture_loss = texture_criterion(texture_outputs, texture_labels)
            loss = category_loss + texture_loss
            loss.backward()
            optimizer.step()
            steps += 1

        # statistics
        writer.add_scalar('loss/category', category_loss, global_step + steps)
        writer.add_scalar('loss/texture', texture_loss, global_step + steps)
        running_loss += loss.item() * adv_inputs.size(0)
        running_corrects += torch.sum(category_preds == category_labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc, steps


def test_adversarial_epoch(fashionnet,
                           dataloader,
                           category_criterion,
                           texture_criterion,
                           attack_strength=8,
                           epsilon=0.03):
    fashionnet.model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    steps = 0

    print(f'Testing model using attack strength K={attack_strength}')

    # iterate over data
    for batch in tqdm(dataloader):
        inputs = batch['image'].to(fashionnet.device)
        category_labels = batch['category'].to(fashionnet.device)
        texture_labels = batch['textures'].to(fashionnet.device)

        # generate adversarial examples
        adv_inputs = pgd_classifier_attack(fashionnet,
                                           inputs,
                                           category_labels,
                                           texture_labels,
                                           category_criterion,
                                           texture_criterion,
                                           epsilon=epsilon,
                                           k=attack_strength)

        # test on adversarial examples
        with torch.set_grad_enabled(False):
            category_outputs, texture_outputs = fashionnet.model(adv_inputs)
            category_loss = category_criterion(category_outputs,
                                               category_labels)
            texture_loss = texture_criterion(texture_outputs, texture_labels)
            loss = category_loss + texture_loss
            steps += 1

        # statistics
        category_preds = torch.argmax(category_outputs, dim=1)
        running_loss += loss.item() * adv_inputs.size(0)
        running_corrects += torch.sum(category_preds == category_labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def adversarial_train(fashionnet,
                      dataloaders,
                      category_criterion,
                      texture_criterion,
                      optimizer,
                      writer,
                      num_epochs=24,
                      attack_strength=8):
    since = time.time()
    global_step = 0

    best_acc = 0.0
    best_model_wts = copy.deepcopy(fashionnet.model.state_dict())

    total_epochs = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # train adversarial epoch
        adv_train_loss, adv_train_acc, steps = train_adversarial_epoch(
            fashionnet, dataloaders['train'], category_criterion,
            texture_criterion, optimizer, writer, global_step, attack_strength)
        global_step += steps

        # test on adversarial and clean examples
        adv_test_loss, adv_test_acc = test_adversarial_epoch(
            fashionnet, dataloaders['test'], category_criterion,
            texture_criterion, attack_strength)
        test_loss, test_acc = test_adversarial_epoch(fashionnet,
                                                     dataloaders['test'],
                                                     category_criterion,
                                                     texture_criterion, 0)
        print('Adv Loss: {:.4f} Category Adv Acc: {:.4f}'.format(
            adv_test_loss, adv_test_acc))
        print('Loss: {:.4f} Category Acc: {:.4f}'.format(test_loss, test_acc))

        # write epoch metrics to tensorboard
        writer.add_scalars(
            'loss/total', {
                'adv/train': adv_train_loss,
                'adv/test': adv_test_loss,
                'clean/test': test_loss
            }, global_step)
        writer.add_scalars(
            'accuracy/category', {
                'adv/train': adv_train_acc,
                'adv/test': adv_test_acc,
                'clean/test': test_acc
            }, global_step)

        if adv_test_acc > best_acc:
            best_acc = adv_test_acc
            best_model_wts = copy.deepcopy(fashionnet.model.state_dict())

        total_epochs += 1

    # reset to best model weights
    fashionnet.model.load_state_dict(best_model_wts)

    time_elapsed = time.time() - since
    print('Training complete after {:.0f}m {:.0f}s / {} epochs'.format(
        time_elapsed // 60, time_elapsed % 60, total_epochs))
    print('Best adversarial category accuracy: {:4f}'.format(best_acc))


def curriculum_adversarial_train(fashionnet,
                                 dataloaders,
                                 category_criterion,
                                 texture_criterion,
                                 optimizer,
                                 writer,
                                 max_k=8,
                                 epoch_overfit_threshold=10):
    since = time.time()
    global_step = 0

    best_acc = 0.0
    best_model_wts = copy.deepcopy(fashionnet.model.state_dict())

    epochs_since_acc_inc = 0
    total_epochs = 0

    # increase attack strength incrementally
    for attack_strength in range(max_k + 1):
        print('Increasing attack strength K={}/{}'.format(
            attack_strength, max_k))

        # train until overfitting for given attack strength
        while epochs_since_acc_inc < epoch_overfit_threshold:
            print('Epochs since accuracy increase {}/{}'.format(
                epochs_since_acc_inc + 1, epoch_overfit_threshold))

            # train adversarial epoch
            adv_train_loss, adv_train_acc, steps = train_adversarial_epoch(
                fashionnet,
                dataloaders['train'],
                category_criterion,
                texture_criterion,
                optimizer,
                writer,
                global_step,
                attack_strength,
                batch_mixing=True)
            global_step += steps

            # test on adversarial examples up to current attack strength
            adv_test_losses = {}
            adv_test_accs = {}
            for test_attack_strength in range(attack_strength + 1):
                test_loss, test_acc = test_adversarial_epoch(
                    fashionnet, dataloaders['test'], category_criterion,
                    texture_criterion, test_attack_strength)
                adv_test_losses['k' + str(test_attack_strength)] = test_loss
                adv_test_accs['k' + str(test_attack_strength)] = test_acc

            adv_test_loss = sum(
                adv_test_losses.values()) / len(adv_test_losses)
            adv_test_acc = sum(adv_test_accs.values()) / len(adv_test_accs)
            print('Adv Loss: {:.4f} Category Adv Acc: {:.4f}'.format(
                adv_test_loss, adv_test_acc))

            # write epoch metrics to tensorboard
            writer.add_scalars('loss/total', {
                'train': adv_train_loss,
                'test': adv_test_loss
            }, global_step)
            writer.add_scalars('accuracy/category', {
                'train': adv_train_acc,
                'test': adv_test_acc
            }, global_step)

            writer.add_scalar('attack-strength/k', attack_strength,
                              global_step)
            writer.add_scalars('attack-strength/loss', adv_test_losses,
                               global_step)
            writer.add_scalars('attack-strength/accuracy', adv_test_accs,
                               global_step)

            if adv_test_acc > best_acc:
                epochs_since_acc_inc = 0
                best_acc = adv_test_acc
                best_model_wts = copy.deepcopy(fashionnet.model.state_dict())
            else:
                epochs_since_acc_inc += 1

            total_epochs += 1

        # reset to best model weights
        fashionnet.model.load_state_dict(best_model_wts)
        # make sure to save the first epoch for next k
        best_acc = 0.0
        epochs_since_acc_inc = 0

    time_elapsed = time.time() - since
    print('Training complete after {:.0f}m {:.0f}s / {} epochs'.format(
        time_elapsed // 60, time_elapsed % 60, total_epochs))
    print('Best adversarial category accuracy: {:4f}'.format(best_acc))
