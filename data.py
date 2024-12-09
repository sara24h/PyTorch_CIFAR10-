import os
import zipfile

import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from torchvision.datasets import ImageFolder


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def download_weights():
        url = (
            "https://drive.google.com/uc?id=1gg7ixCrUzc29FXZkmJzgt1e5kFWhyJXI"
        )

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open("state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = ImageFolder(root='/content/drive/MyDrive/real_and_fake_face/train', transform=transform)

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = ImageFolder(root='/content/drive/MyDrive/real_and_fake_face/test', transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,  # اندازه بچ
            num_workers=self.hparams.num_workers,  # تعداد پردازنده‌ها
            drop_last=True,  # حذف آخرین بچ در صورت عدم تکمیل
            pin_memory=True,  # فعال کردن pinned memory
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()



# مثال استفاده
args = {
    'data_dir': '/content/drive/MyDrive/real_and_fake_face',  # مسیر داده‌ها
    'batch_size': 64,
    'num_workers': 2,  # برای جلوگیری از هشدار
}


data_module = CIFAR10Data(args)

# استفاده از داده‌ها
train_loader = data_module.train_dataloader()
test_loader = data_module.test_dataloader()
