import torch
import torch.nn as nn
from sklearn import metrics
from tqdm.auto import tqdm


class MyClassifier(nn.Module):
    def __init__(self, model, device, model_name):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name

    def train_model(self, dataloaders, image_datasets, criterion, optimizer, num_epochs=50):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            for phase in ['Training', 'Test']:
                if phase == 'Training':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                y_test = []
                y_pred = []
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'Training':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    y_test.extend(labels)
                    y_pred.extend(preds)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                f1_macro = metrics.f1_score(y_test, y_pred, average="macro")
                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])

                with open(self.model_name + '_epoch_res.txt', 'a') as f:
                    f.write('{} loss: {:.4f}, acc: {:.4f}, f1_macro: {:.4f}\n'.format(phase,
                                                                                    epoch_loss,
                                                                                    epoch_acc,
                                                                                    f1_macro))
                print('{} loss: {:.4f}, acc: {:.4f}, f1_macro: {:.4f}\n'.format(phase,
                                                                              epoch_loss,
                                                                              epoch_acc,
                                                                              f1_macro))
                if phase == 'Test' and f1_macro > 0.93:
                    return self.model
        return self.model
