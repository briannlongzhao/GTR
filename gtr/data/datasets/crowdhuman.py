from detectron2.data.datasets.register_coco import register_coco_instances
import os
import platform
import re

categories = [
    {'id': 1, 'name': 'person'},
]

tmp_data_dir = ''
hostname = platform.node()
if "iGpu" in hostname or "iLab" in hostname:
    os.environ["TMPDIR"] = "/lab/tmpig8e/u/brian-data"
elif re.search("[a-z]\d\d-\d\d", hostname):
    os.environ["TMPDIR"] = "/scratch1/briannlz"
if "TMPDIR" in os.environ.keys():
    tmp_data_dir = os.path.join(os.environ["TMPDIR"], "GTR/datasets", '')
print(f"crowdhuman.py: HOSTNAME={hostname}")
print(f"crowdhuman.py: TMPDIR={os.environ['TMPDIR']}")

def _get_builtin_metadata():
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(sorted(categories, key=lambda x: x['id']))}
    thing_classes = [x['name'] for x in sorted(categories, key=lambda x: x['id'])]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS_CROWDHUMAN = {
    "crowdhuman_train":
        (os.path.join(tmp_data_dir, "crowdhuman/CrowdHuman_train/Images/"),
         os.path.join(tmp_data_dir,"crowdhuman/annotations/train_amodal.json")),
    "crowdhuman_val":
        (os.path.join(tmp_data_dir, "crowdhuman/CrowdHuman_val/Images/"),
         os.path.join(tmp_data_dir, "crowdhuman/annotations/val_amodal.json")),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_CROWDHUMAN.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
