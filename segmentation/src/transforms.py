import albumentations as A


def pre_transforms():
    return [
        A.Resize(448, 448),
    ]


def hard_augs():
    return [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.MultiplicativeNoise(),
    ]


def post_transforms():
    return [A.Normalize()]


def get_hard_augs():

    transforms = [pre_transforms(), hard_augs(), post_transforms()]
    return A.Compose([t for sublist in transforms for t in sublist])


def get_infer_augs():
    transforms = [pre_transforms(), post_transforms()]
    return A.Compose([t for sublist in transforms for t in sublist])
