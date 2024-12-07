import pytorch_lightning as pl
import torch
from torchmetrics.classification import Accuracy
import argparse

from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR


all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default="resnet18", choices=list(all_classifiers.keys()), help="مدل مورد استفاده.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="میزان یادگیری")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="کاهش وزن")
    parser.add_argument("--max_epochs", type=int, default=50, help="تعداد اپوک‌ها")
    return parser.parse_args()


class CIFAR10Module(pl.LightningModule):
    def __init__(self, classifier, learning_rate, weight_decay, max_epochs):
        super().__init__()

        
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # بررسی وجود پارامتر 'classifier'
        if self.classifier not in all_classifiers:
            raise ValueError("مدل انتخاب‌شده در دیکشنری موجود نیست.")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=2)  # 10 کلاس برای CIFAR-10

        # بارگذاری مدل انتخاب‌شده
        self.model = all_classifiers[self.classifier]

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        print(f"Predictions shape: {predictions.shape}")
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]


if __name__ == "__main__":
    args = parse_args()
    model = CIFAR10Module(
        classifier=args.classifier,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs
    )
 
