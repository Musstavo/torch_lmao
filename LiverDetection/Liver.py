import os
import glob
from natsort import natsorted
import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandAffined,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import tempfile
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "LiverDetection/data/liver-tumor-segmentation/volume"
label_dir = "LiverDetection/data/liver-tumor-segmentation/segmentations"

train_images = natsorted(glob.glob(os.path.join(train_dir, "*.nii")))
train_labels = natsorted(glob.glob(os.path.join(label_dir, "*.nii")))


data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

random.shuffle(data_dicts)
val_size = 20
train_files, val_files = data_dicts[:-val_size], data_dicts[-val_size:]
set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-90,
            a_max=170,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.7,
            spatial_size=(96, 96, 96),
            rotate_range=(0, 0, np.pi / 15),
            scale_range=(0.1, 0.1, 0.1),
        ),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-90,
            a_max=170,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
    ]
)

train_ds = Dataset(data=train_files, transform=train_transforms)
trian_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
dice_metric = DiceMetric(include_background=False)

epochs = 100
val_interval = 5
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

post_pred = AsDiscrete(argmax=True, to_onehot=3)
post_label = AsDiscrete(to_onehot=3)


for epoch in tqdm(range(epochs)):
    print(f"epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0
    batches_trained = 0

    for batch_data in train_loader:
        batches_trained += 1
        images, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(
            f"{batches_trained}/{len(train_loader)}, train_loss: {loss.item():.4f}",
            end="\r",
        )

    epoch_loss /= batches_trained
    epoch_loss_values.append(epoch_loss)
    print(f"\nepoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.inference_mode():
            for val_data in val_loader:
                val_images, val_labels = (
                    val_data["images"].to(device),
                    val_data["label"].to(device),
                )
            region_of_interest_size = (96, 96, 96)
            sliding_window_batch_size = 4

            val_preds = sliding_window_inference(
                val_images, region_of_interest_size, sliding_window_batch_size, model
            )
            val_preds = [post_pred(i) for i in decollate_batch(val_preds)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
