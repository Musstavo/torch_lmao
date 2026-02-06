import os
import glob
from natsort import natsorted

train_dir = "LiverDetection/data/liver-tumor-segmentation/volume"
label_dir = "LiverDetection/data/liver-tumor-segmentation/segmentations"

train_images = natsorted(glob.glob(os.path.join(train_dir, "*.nii")))
train_labels = natsorted(glob.glob(os.path.join(label_dir, "*.nii")))


data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

print(f"Found {len(data_dicts)} pairs")
print(f"First pair: {data_dicts[0]}")
print(f"Last pair: {data_dicts[-1]}")


print(f"Middle pair: {data_dicts[65]}")
