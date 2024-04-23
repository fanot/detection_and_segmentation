import os
import torch
import torch.nn as nn
from time import time
import gc
from torch.autograd import Variable
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import DataLoader
from src.models.BaseModel import BaseModel
from src.models.UNet import UNet


class UnetModel(BaseModel):
    def __init__(self):
        self.estimator = None
        self.optimizer = None
        self.scheduler = None
        self.train_data = None
        self.val_data = None
        self.losses_history = None
        # self.n_epochs = None
        # self.batch_size = None
        self.device = None

    def train(self, data: str, imgsz: int, epochs: int, batch: int):
        """Train method"""

        self.__load_data(data)
        train_losses = []
        valid_losses = []
        metric_scores = []
        best_metric_score = 0

        for epoch in range(epochs):
            # train phase
            tic = time()
            print('* Epoch %d/%d' % (epoch + 1, epochs))
            self.estimator.train()
            running_train_losses = []
            for x_batch, y_batch in self.train_data:
                # data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # set parameter gradients to zero
                self.optimizer.zero_grad()
                # функции подсчета средних лоссов пришлось изменить,
                # потому что изначально они работали неправильно
                with torch.set_grad_enabled(True):
                    # forward
                    y_pred = self.estimator(x_batch)
                    loss = self.loss(y_pred, y_batch)  # forward-pass
                    running_train_losses.append(loss.detach().item())
                    loss.backward()  # backward-pass
                    self.optimizer.step()  # update weights
                avg_train_loss = np.mean(running_train_losses)
            # train loss
            epoch_train_loss = avg_train_loss
            train_losses.append(epoch_train_loss)
            toc = time()
            print('finished train phase;', end=' ')
            # show intermediate results
            # test phase
            self.estimator.eval()
            running_val_losses = []
            for x_val, y_val in self.val_data:
                with torch.no_grad():
                    y_hat = self.estimator(x_val.to(self.device)).detach().cpu()  # detach and put into cpu
                    loss = self.loss(y_hat, y_val)  # forward-pass
                    # print('loss: ', loss)
                    running_val_losses.append(loss.detach().item())
                avg_val_loss = np.mean(running_val_losses)
            # val loss
            epoch_val_loss = avg_val_loss
            valid_losses.append(epoch_val_loss)
            toc = time()
            print('finished valid phase;', end=' ')
            # metric score
            epoch_metric_score = self.__score_model_by_metric(self.estimator, self.__iou, self.train_data)
            metric_scores.append(epoch_metric_score)
            toc = time()
            print('finished metric score phase;', end=' ')
            if epoch_metric_score > best_metric_score:
                best_metric_score = epoch_metric_score
            if self.scheduler:
                self.scheduler.step(epoch_metric_score)
            res = {
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'metric_scores': metric_scores,
                'best_metric_score': best_metric_score
            }
            return res

    def __score_model_by_metric(self, model, metric, data):
        """Use metric method"""

        model.eval()  # testing mode
        scores = 0
        for x_batch, y_label in data:
            with torch.no_grad():
                threshold = 0.5
                y_pred = model(x_batch.to(self.device))
                y_pred = torch.sigmoid(y_pred)
                # Выходы данные  не приведены к 0 и 1 - введем threshold, значения больше которого будет относить к 1, меньше к 0
                y_pred = torch.where(y_pred > 0.5, 1, 0)
                scores += metric(y_pred, y_label.to(self.device)).mean().item()

        return scores / len(data)

    def val(self, data: str, imgsz: int):
        """Evaluate method"""

        pass

    def predict(self, source: str):
        """Predict method"""

        pass

    def __init_optimizer(self):
        """Init optimizer method"""

        self.optimizer = torch.optim.AdamW(self.estimator.parameters(), lr=0.001, weight_decay=0.05)

    def __init_scheduler(self):
        """Init scheduler method"""

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[25, 50, 75, 100], gamma=0.75)

    def loss(self, y_pred, y_real, eps=1e-8, gamma=2):
        """Loss function method"""

        # focal loss
        # https://arxiv.org/pdf/1708.02002.pdf
        sigmoid = nn.Sigmoid()
        y_real = y_real.view(-1)
        y_pred = y_pred.view(-1)
        y_pred = sigmoid(y_pred)
        loss = -(((1 - y_pred) ** gamma) * y_real * (y_pred + eps).log() + (1 - y_real) * (
                    1 - y_pred + eps).log()).mean()
        return loss

    def save_model(self, model_path: str):
        """Save model method"""

        torch.save(self.estimator, model_path)

    def __select_device(self):
        """Select device method"""

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.estimator is not None:
            self.estimator.to(self.device)

    def load(self, model_path: str):
        """Load model method"""

        if self.estimator is None:
            self.estimator = UNet()
        if model_path is not None:
            self.estimator.load_state_dict(torch.load(model_path))
        self.__init_optimizer()
        self.__init_scheduler()
        self.__select_device()

    def __iou(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Metric method"""

        outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
        labels = labels.squeeze(1).byte()
        smooth = 1e-8
        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0
        iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
        return iou

    def __load_data(self, path_to_dataset: str, batch_size: int = 25):
        """Load data to dataset method"""

        images = []
        masks = []

        images_in_folder = [f for f in os.listdir(os.path.join(path_to_dataset, 'train/images')) if
                           f.endswith(os.environ['FRAMES_EXTENSION'])]

        masks_in_folder = [f for f in os.listdir(os.path.join(path_to_dataset, 'train/masks')) if
                            f.endswith(os.environ['FRAMES_EXTENSION'])]

        for file in images_in_folder:
            images.append(imread(os.path.join(path_to_dataset, file)))

        for file in masks_in_folder:
            masks.append(imread(os.path.join(path_to_dataset, file)))

        size = (640, 640) # os.environ
        images = [resize(x, size, mode='constant', anti_aliasing=True) for x in images]
        masks = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in masks]

        images = np.array(images, np.float32)
        masks = np.array(masks, np.float32)

        ix = np.random.choice(len(images), len(images), False)
        tr, val, ts = np.split(ix, [100, 150])

        self.train_data = DataLoader(list(zip(np.rollaxis(images[tr], 3, 1), masks[tr, np.newaxis])),
                                     batch_size=batch_size, shuffle=True)
        self.val_data = DataLoader(list(zip(np.rollaxis(images[val], 3, 1), masks[val, np.newaxis])),
                                   batch_size=batch_size, shuffle=True)


