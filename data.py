
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)  # ذخیره پارامترهای ورودی به شیوه صحیح
        self.mean = (0.4914, 0.4822, 0.4465)  # میانگین برای نرمالیزه کردن داده‌ها
        self.std = (0.2471, 0.2435, 0.2616)   # انحراف معیار برای نرمالیزه کردن داده‌ها

    def train_dataloader(self):
        # ترنسفورم‌ها برای آموزش (افزایش داده‌ها و نرمالیزه کردن)
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),  # کراپ تصادفی تصاویر
                T.RandomHorizontalFlip(),     # چرخش افقی تصادفی
                T.ToTensor(),                 # تبدیل به تنسور
                T.Normalize(self.mean, self.std),  # نرمالیزه کردن تصاویر
            ]
        )
        # بارگذاری داده‌های آموزش با استفاده از ImageFolder
        dataset = ImageFolder(root='/content/drive/MyDrive/real_and_fake_face/train', transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,  # اندازه بچ
            num_workers=self.hparams.num_workers,  # تعداد پردازنده‌ها
            shuffle=True,  # شافل کردن داده‌ها
            drop_last=True,  # حذف آخرین بچ در صورت عدم تکمیل
            pin_memory=True,  # فعال کردن pinned memory
        )
        return dataloader

    def val_dataloader(self):
        # ترنسفورم‌ها برای تست (فقط نرمالیزه کردن)
        transform = T.Compose(
            [
                T.ToTensor(),  # تبدیل به تنسور
                T.Normalize(self.mean, self.std),  # نرمالیزه کردن تصاویر
            ]
        )
        # بارگذاری داده‌های تست با استفاده از ImageFolder
        dataset = ImageFolder(root='/content/drive/MyDrive/real_and_fake_face/train', transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,  # اندازه بچ
            num_workers=self.hparams.num_workers,  # تعداد پردازنده‌ها
            drop_last=True,  # حذف آخرین بچ در صورت عدم تکمیل
            pin_memory=True,  # فعال کردن pinned memory
        )
        return dataloader

    def test_dataloader(self):
        # متد تست مشابه متد اعتبارسنجی است
        return self.val_dataloader()  # استفاده از داده‌های تست برای ارزیابی

# اگر بخواهید این کلاس را در برنامه خود استفاده کنید، می‌توانید به این صورت عمل کنید:

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
