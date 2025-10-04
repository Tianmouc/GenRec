# Author: YapengMeng
# Last modified: 2025-09-23

from .TMrec_trainer import TMPixelRec_Trainer

trainer_cls_name_dict = {
    "TMPixelRec_Trainer": TMPixelRec_Trainer
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
