import time
import copy
import torch
import torch.nn as nn

from tqdm import tqdm
from torchvision import models


class PredictionHead(nn.Module):
    def __init__(self, in_features, num_categories, num_textures):
        super(PredictionHead, self).__init__()
        self.category = nn.Linear(in_features, num_categories)
        self.textures = nn.Linear(in_features, num_textures)

    def forward(self, x):
        category = self.category(x)
        textures = self.textures(x)
        return category, textures


class FashionNet():
    def __init__(self, device, num_categories, num_textures):
        self.model = models.mobilenet_v2(pretrained=True)
        self.num_features = self.model.classifier[-1].in_features
        self.num_categories = num_categories
        self.num_textures = num_textures
        self.model.classifier[-1] = PredictionHead(self.num_features,
                                                   self.num_categories,
                                                   self.num_textures)
        self.model.to(device)
        self.device = device

    def save(self, checkpoint_path):
        torch.save(self.model, checkpoint_path)

    def load(self, checkpoint_path):
        self.model = torch.load(checkpoint_path, map_location=self.device)

    def train_epoch(self, dataloader, category_criterion, texture_criterion,
                    optimizer, writer, global_step):
        self.model.train()
        steps = 0
        running_loss = 0.0
        running_corrects = 0.0

        print('Training model...')

        # iterate over data
        for batch in tqdm(dataloader):
            inputs = batch['image'].to(self.device)
            category_labels = batch['category'].to(self.device)
            texture_labels = batch['textures'].to(self.device)

            # train on adversarial examples
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                category_outputs, texture_outputs = self.model(inputs)
                category_preds = torch.argmax(category_outputs, dim=1)
                category_loss = category_criterion(category_outputs,
                                                   category_labels)
                texture_loss = texture_criterion(texture_outputs,
                                                 texture_labels)
                loss = category_loss + texture_loss
                loss.backward()
                optimizer.step()
                steps += 1

            # statistics
            writer.add_scalar('loss/category', category_loss,
                              global_step + steps)
            writer.add_scalar('loss/texture', texture_loss,
                              global_step + steps)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(
                category_preds == category_labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        return epoch_loss, epoch_acc, steps

    def test_epoch(self, dataloader, category_criterion, texture_criterion):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        steps = 0

        print('Testing model...')

        # iterate over data
        for batch in tqdm(dataloader):
            inputs = batch['image'].to(self.device)
            category_labels = batch['category'].to(self.device)
            texture_labels = batch['textures'].to(self.device)

            # test on adversarial examples
            with torch.set_grad_enabled(False):
                category_outputs, texture_outputs = self.model(inputs)
                category_loss = category_criterion(category_outputs,
                                                   category_labels)
                texture_loss = texture_criterion(texture_outputs,
                                                 texture_labels)
                loss = category_loss + texture_loss
                steps += 1

            # statistics
            category_preds = torch.argmax(category_outputs, dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(
                category_preds == category_labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        return epoch_loss, epoch_acc

    def train(self,
              dataloaders,
              category_criterion,
              texture_criterion,
              optimizer,
              writer,
              num_epochs=24):
        since = time.time()
        global_step = 0

        best_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())

        total_epochs = 0

        # increase attack strength incrementally
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # train adversarial epoch
            train_loss, train_acc, steps = self.train_epoch(
                dataloaders['train'], category_criterion, texture_criterion,
                optimizer, writer, global_step)
            global_step += steps

            # test on adversarial examples up to current attack strength
            test_loss, test_acc = self.test_epoch(dataloaders['test'],
                                                  category_criterion,
                                                  texture_criterion)
            print('Loss: {:.4f} Category Acc: {:.4f}'.format(
                test_loss, test_acc))

            # write epoch metrics to tensorboard
            writer.add_scalars('loss/total', {
                'train': train_loss,
                'test': test_loss,
            }, global_step)
            writer.add_scalars('accuracy/category', {
                'train': train_acc,
                'test': test_acc,
            }, global_step)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())

            total_epochs += 1

        # reset to best model weights
        self.model.load_state_dict(best_model_wts)

        time_elapsed = time.time() - since
        print('Training complete after {:.0f}m {:.0f}s / {} epochs'.format(
            time_elapsed // 60, time_elapsed % 60, total_epochs))
        print('Best category accuracy: {:4f}'.format(best_acc))
