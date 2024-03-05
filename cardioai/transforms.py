import os
import random
from typing import Tuple
from tqdm import tqdm


from torchio.transforms import (
    RescaleIntensity,
    RandomElasticDeformation,
    RandomFlip,
    RandomAffine,
    # intensity
    RandomMotion,
    RandomGhosting,
    RandomSpike,
    RandomBiasField,
    RandomBlur,
    RandomNoise,
    Resize,
    RandomSwap,
    RandomAnisotropy,
    #     RandomLabelsToImage,
    RandomGamma,
    OneOf,
    CropOrPad,
    Crop,
    ZNormalization,
    HistogramStandardization,
    Compose,
)
import torchio as tio
import torch
from torchvision.utils import save_image

import warnings
from typing import Optional
from typing import Sequence
from torchio.transforms import SpatialTransform
from torchio.data.image import ScalarImage, LabelMap
from torchio.data.subject import Subject

import cardioai.dataset as dataset


class SliceTime(SpatialTransform):
    """Time slice."""  # noqa: B950

    def __init__(
        self,
        timeframe: int = 0,
        labels: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.timeframe = timeframe

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            # print(f"Before time slice: {image.data.shape}")
            if isinstance(image, LabelMap):
                if image.data.shape[0] == 1:
                    continue
            slice = image.data[self.timeframe].unsqueeze(0)
            image.set_data(slice)
            # print(f"After time slice: {image.data.shape}")

        return subject


class Slice(SpatialTransform):
    """Slice data across depth and time"""  # noqa: B950

    def __init__(
        self,
        central_slice: bool = True,
        slice_min: int = None,
        slice_max: int = None,
        labels: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.central_slice = central_slice
        self.slice_min = slice_min
        self.slice_max = slice_max

    def apply_transform(self, subject: Subject) -> Subject:
        # print(f"Subject id: {subject['code']}, shape: {subject.mri.data.shape}")
        for image in self.get_images(subject):
            # print(f"Before depth slice: {image.data.shape}")
            # if isinstance(image, LabelMap):
            #     if image.data.shape[0] == 1:
            #         continue
            # print(f"Image {subject.code} with shape {image.data.shape}")
            if image.data.shape[3] == 50:
                image.set_data(image.data.permute(3, 0, 1, 2))

            print(f"Image {subject.code} with shape {image.data.shape}")
            if self.central_slice:
                slice_index = image.data.shape[3] // 2
            else:
                # get random slice
                slice_index = random.randint(self.slice_min, self.slice_max)
                if image.data.shape[3] <= slice_index:
                    # warnings.warn(f"Slice {slice_index} is larger than image depth {image.data.shape[3]}. Setting to central slice.")
                    slice_index = image.data.shape[3] // 2

            slice = image.data[:, :, :, slice_index].unsqueeze(-1)

            image.set_data(slice)
            # print(f"After depth slice: {image.data.shape}")

        return subject


class StackEDES(SpatialTransform):
    """Stack ED and ES frames into mulitchannel image"""  # noqa: B950

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def apply_transform(self, subject: Subject) -> Subject:

        image = subject.mri.data
        subject.mri.data = torch.stack([image[0], image[subject.es_index], image[0]])
        mask_ed = subject.gt.data
        mask_es = subject.gt_es.data
        subject.gt.data = torch.stack([mask_ed[0], mask_es[0], mask_ed[0]])

        return subject


class CropAdaptive(Crop):
    """Crop with the limits of target image size"""  # noqa: B950

    def __init__(
        self,
        cropping: Tuple[int, int, int, int, int, int],  # noqa: B950
        target_size=None,  # noqa: B950
        **kwargs,
    ):
        super().__init__(cropping, **kwargs)
        self.cropping = cropping
        self.target_size = target_size

    def apply_transform(self, subject: Subject) -> Subject:
        new_cropping = [0, 0, 0, 0, 0, 0]
        for axis in range(3):
            bound = self.cropping[axis * 2 + 1]
            if subject["mri"].shape[axis + 1] < self.target_size[axis] + bound:
                bound = subject.mri.data.shape[axis + 1] - self.target_size[axis]
            if bound < 0:
                bound = 0
            new_cropping[axis * 2 + 1] = bound

        crop = Crop(new_cropping)
        subject = crop(subject)

        return subject


class ReduceVideo(SpatialTransform):
    """Reduce number of video frames."""

    def __init__(
        self,
        timestep: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.timestep = timestep

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            # print(f"Before time slice: {image.data.shape}")
            if isinstance(image, LabelMap):
                if image.data.shape[0] == 1:
                    continue
            data = image.data[:: self.timestep, :, :, :]
            image.set_data(data)
            # print(f"After time slice: {image.data.shape}")

        return subject


class AnnotatePositives(SpatialTransform):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            # print(f"Before time slice: {image.data.shape}")
            if subject.target[1] == 1:
                black = torch.zeros_like(image.data)
                image.set_data(black)
            # print(f"After time slice: {image.data.shape}")

        return subject


def prepare_batch(batch, config):
    """Operates further curation and loads the label encoder to turn strings to encoded format

    Args:
        batch (next(iter(data_loader))): one batch loaded by your data loader
        device (str): 'cpu' or 'cuda'

    Returns:
        inputs, targets
    """
    inputs = batch["mri"]["data"]

    # print(f"Inputs shape: {inputs.shape}")
    if config["cache_data"]:
        inputs = inputs.squeeze(-1)
        masks = batch["gt"]["data"].squeeze(-1)

        if config["params"]["mask_images"]:
            multiply_mask = torch.zeros(masks.shape)
            multiply_mask[masks != 0] = 1
            inputs = inputs * multiply_mask
            
        # Get only ED frames
        # inputs = torch.stack([inputs[:,0], inputs[:,0], inputs[:,0]], axis=1)
        
        return {"image": inputs, "label": batch["target"], "mask": masks, "id": batch["code"], "meta_label": batch["meta_label"],
                "sample_weight": batch["sample_weight"]}


    if config["params"]["edes"]:
        edes = batch["es_index"]
        new_inputs = torch.zeros(
            inputs.shape[0], 3, inputs.shape[2], inputs.shape[3], inputs.shape[4]
        )
        for i in range(inputs.shape[0]):
            new_inputs[i] = inputs[i, [0, edes[i].item(), edes[i].item()], ...]
        inputs = new_inputs
        # inputs = torch.cat((inputs, edes), axis=1)
    if config["params"]["mask_images"]:
        mask = batch["gt"]["data"]
        if config["params"]["edes"]:
            mask = torch.concatenate(
                (batch["gt"]["data"], batch["gt_es"]["data"], batch["gt_es"]["data"]),
                axis=1,
            )
        multiply_mask = torch.zeros(mask.shape)
        multiply_mask[mask != 0] = 1
        inputs = inputs * multiply_mask
    if config["params"]["volume"]:
        # Select only the first timeframe (ED)
        # inputs = inputs[:, 0, ...].unsqueeze(1)
        # Make the right input ordering (batch, channel, depth, height, width)
        inputs = inputs.permute(0, 1, 4, 2, 3)
    if not config["params"]["volume"]:
        inputs = inputs.squeeze(4)
    if not config["params"]["video"]:
        inputs = inputs.squeeze(1)
    if config["params"]["gray2rgb"] and not config["params"]["edes"]:
        inputs = inputs.unsqueeze(1)
        inputs = torch.cat((inputs, inputs, inputs), axis=1)

    # print(f"Inputs shape: {inputs.shape}")
    return {"image": inputs, "label": batch["target"], "id": batch["code"]}


def get_transforms_torchio(config, test=True):
    INPUT_SIZE = config["params"]["input_size"]
    FINE_TUNING = config["params"]["fine_tuning"]
    GRAY2RGB = config["params"]["gray2rgb"]
    MASK_IMAGES = config["params"]["mask_images"]
    VIDEO = config["params"]["video"]
    VOLUME = config["params"]["volume"]
    TIME_STEP = config["params"]["time_step"]
    central_slice_only = config["params"]["central_slice_only"]
    VIEW = config["dataset"]["view"]
    EDES = config["params"]["edes"]
    STANDARDIZE_HISTOGRAMS = config["params"]["standardize_histograms"]

    # if VIEW != "sa":
    input_shape = (INPUT_SIZE, INPUT_SIZE, 1)

    if VOLUME:
        input_shape = (INPUT_SIZE, INPUT_SIZE, 10)

    if VOLUME and VIEW != "sa":
        raise ValueError(f"VOLUME and {VIEW} view are not compatible")

    if STANDARDIZE_HISTOGRAMS:
        landmarks = load_landmarks(config)

    # if MASK_IMAGES:
    #     mask_name = "gt"
    # else:
    #     mask_name = None
    mask_name = "gt"
    

    if config["params"]["edes"]:
        VIDEO = True
        TIME_STEP = 1

    preprocessing_list = []
    if VIDEO:
        if TIME_STEP != 1:
            preprocessing_list.append(ReduceVideo(timestep=TIME_STEP))
    else:
        preprocessing_list.append(SliceTime(timeframe=0))

    # if VIEW == "sa" and not VOLUME:

        # if test:
            # High slice index will force to calculate the center slice instead
            # SLICE_INDEX = 100
        # preprocessing_list.append(Slice(central_slice=central_slice_only, slice_min=3, slice_max=9))

    if STANDARDIZE_HISTOGRAMS:
        preprocessing_list.append(HistogramStandardization(landmarks=landmarks))
    # if VIEW == "sa":
        # preprocessing_list.append(CropAdaptive((0, 80, 0, 0, 0, 0), input_shape))
    # if EDES:
        # preprocessing_list.append(StackEDES())
    preprocessing_list.append(CropOrPad(input_shape, mask_name=mask_name))
    preprocessing_list.append(RescaleIntensity((0, 1)))
    preprocessing_list.append(Resize((224, 224, 1)))
    # preprocessing_list.append(AnnotatePositives())

    preprocessing_transforms = Compose(preprocessing_list)

    augmentation_transforms = OneOf(
        [
            RandomFlip(axes=("LR", "SI")),
            # RandomAffine(scales=(0.9, 1.2),degrees=10,isotropic=True,image_interpolation='nearest'),
            # RandomAffine(scales=(0.4, 0.4, 0), degrees=(50, 00, 0), image_interpolation='nearest'),
            # RandomAffine(scales=0, degrees=(90, 90, 0), image_interpolation='nearest'),
            # RandomAffine(scales=0, degrees=(-40, 40, 0,0,0,0), image_interpolation='nearest'),
            # RandomAffine(scales=0, degrees=(90), image_interpolation='nearest'),
            # RandomElasticDeformation(num_control_points=5, max_displacement=2),
            # RandomSpike(),
            # RandomBiasField(),
            RandomNoise(),
            RandomGamma(),
        ],
        p=0.5,
    )

    # if masks:
    #     train_transforms_list.insert(1, MaskIntensityd(["image"], mask_key="segmentation"))
    #     val_transforms_list.insert(1, MaskIntensityd(["image"], mask_key="segmentation"))

    transforms_list = []
    transforms_list.append(preprocessing_transforms)
    # if masks:
    #     transforms_list.append(MaskIntensityd(["image"], mask_key="segmentation"))
    if not test:
        transforms_list.append(augmentation_transforms)
    # transforms_list.append(ToTensord(keys=["image"]))

    transforms = Compose(transforms_list)
    # traind_transforms = Compose(train_transforms_list)
    # vald_transforms = Compose(val_transforms_list)

    return transforms


def get_dataset_torchio(datad_list, transforms, cache=True):
    subjects = []

    if cache:
        cache_dir = "cache/sa_edes"
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

        cacheable_transforms = (SliceTime,
                                Slice,
                                StackEDES,
                                CropAdaptive,
                                ReduceVideo)
        transforms_to_cache = []
        # Remove already cached transforms
        for index, transform in enumerate(transforms.transforms):
            if isinstance(transform, cacheable_transforms):
                transforms_to_cache.append(transform)
                transform.transforms[index] = None
            if isinstance(transform, Compose):
                for index, subtransform in enumerate(transform.transforms):
                    if isinstance(subtransform, cacheable_transforms):
                        transforms_to_cache.append(subtransform)
                        transform.transforms[index] = None
        
                transform.transforms = [x for x in transform.transforms if x is not None]
        transforms.transforms = [x for x in transforms.transforms if x is not None]
        transforms_to_cache = Compose(transforms_to_cache)
        # print(transforms_to_cache)

        print("Caching samples...")
        skipped_count = 0

        cached_datad_list = []

        for sample in tqdm(datad_list):
            if sample["subset"] == "synthetic":
                cached_datad_list.append(sample)
                continue
            cached_sample_name = cache_dir + "/" + sample["id"]
            # cached_image_path = cache_dir + "/" + sample["id"] + ".png"
            cached_mask_path = cache_dir + "/" + sample["id"] + "_mask.png"

            cached_sample_paths = []
            if sample["subset"] == "train":
                slice_neighbourhood = 1 # 1 to cache 3x3 neighbourhood
            else:
                slice_neighbourhood = 0

            indexes = []
            for i in range(-slice_neighbourhood, slice_neighbourhood+1):
                for j in range(-slice_neighbourhood, slice_neighbourhood+1):
                    indexes.append((i, j))
                    cached_sample = {}
                    cached_sample['image'] = cached_sample_name + f"_t_{i}_v_{j}" + ".png"
                    cached_sample['segmentation'] = cached_sample_name + f"_t_{i}_v_{j}" + "_mask.png"
                    cached_sample['segmentation_es'] = cached_sample_name + f"_t_{i}_v_{j}" + "_mask.png"
                    cached_sample_paths.append(cached_sample)

            all_cached = True
            for cached_sample in cached_sample_paths:
                if not os.path.isfile(cached_sample["image"]):
                    all_cached = False
                    break
            if not all_cached:
                skipped_count += 1
                continue # Allow missing samples

                # Cache sample
                image = tio.ScalarImage(sample["image"])
                if sample["segmentation"]:
                    mask = tio.LabelMap(sample["segmentation"])
                    mask_es = tio.LabelMap(sample["segmentation_es"])
                else:
                    mask = None
                    mask_es = None

                cached_subject = tio.Subject(
                    mri=image,
                    gt=mask,
                    gt_es=mask_es,
                    code=sample["id"],
                    target=sample["label"],
                    es_index=sample["es_index"],
                )
                cached_subject = transforms_to_cache(cached_subject)

                cached_image = cached_subject.mri.data
                print("ID: " + sample["id"] , "Image shape: ", cached_image.shape, "Mask shape: ", cached_subject.gt.data.shape)
                if cached_image.shape[3] == 50:
                    print("Permuting image")
                    cached_image = cached_image.permute(3, 0, 1, 2)
                
                if cached_image.shape[3] < 2 * slice_neighbourhood + 1:
                    print("Skipping sample due to insufficient number of slices")
                    continue

                cached_mask = cached_subject.gt.data

                es_index = sample["es_index"]
                central_slice = cached_image.shape[3] // 2

                for i, j in indexes:
                    # print(cached_image.shape)
                    cached_image_ed = cached_image[0+i, :, :, central_slice+j]
                    cached_image_es = cached_image[es_index+i, :, :, central_slice+j]
                    cached_image_edes = torch.stack([cached_image_ed, cached_image_es, cached_image_ed])
                    cached_image_edes = cached_image_edes.squeeze(-1)/255.0
                    cached_image_path = cached_sample_name + f"_t_{i}_v_{j}" + ".png"
                    save_image(cached_image_edes, cached_image_path)

                    if sample["segmentation"]:
                        cached_mask = cached_subject.gt.data
                        cached_mask_es = cached_subject.gt_es.data
                        cached_mask_ed = cached_mask[0, :, :, central_slice+j]
                        cached_mask_es = cached_mask_es[0, :, :, central_slice+j]
                        cached_mask = torch.stack([cached_mask_ed, cached_mask_es, cached_mask_ed])
                        cached_mask = cached_mask.squeeze(-1)/4.0
                        cached_mask_path = cached_sample_name + f"_t_{i}_v_{j}" + "_mask.png"
                        save_image(cached_mask, cached_mask_path)

            for cached_sample in cached_sample_paths:
                new_sample = sample.copy()
                new_sample["image"] = cached_sample["image"]
                new_sample["segmentation"] = cached_sample["segmentation"]
                new_sample["segmentation_es"] = cached_sample["segmentation_es"]
                cached_datad_list.append(new_sample)
                
        print(f"Skipped {skipped_count} samples")
        datad_list = cached_datad_list
            
    for sample in datad_list:
        if sample["label"][0] == 1:
            label = "NOR"
        else:
            label = "ABN"

        if not sample["segmentation"]:
            gt = 0
            gt_es = gt
        else:
            gt = tio.LabelMap(sample["segmentation"])
            gt_es = tio.LabelMap(sample["segmentation_es"])

        subject = tio.Subject(
            mri=tio.ScalarImage(sample["image"]),
            gt=gt,
            gt_es=gt_es,
            code=sample["id"],
            target=sample["label"],
            pathology=label,
            es_index=sample["es_index"],
            subset=sample["subset"],
            sample_weight=sample["sample_weight"],
            meta_label=sample["meta_label"],
            image_path=sample["image"],
        )
        subjects.append(subject)
    
    if subjects[0].subset == "train":
        print(f"Expanded train dataset size: {len(subjects)}")

    dataset = tio.SubjectsDataset(subjects, transform=transforms)

    return dataset


def load_landmarks(config):
    landmarks_dir = "landmarks/"
    name = config["dataset"]["name"]
    name += "_" + config["dataset"]["view"]
    name += "_VOLUME_" + str(config["params"]["volume"])
    name += "_VIDEO_" + str(config["params"]["video"])
    name += "_MASK_IMAGES_" + str(config["params"]["mask_images"])
    landmarks_file = f"landmarks_{name}.npy"
    landmarks_path = landmarks_dir + landmarks_file

    if os.path.isfile(landmarks_path):
        landmarks = {
            "mri": landmarks_path,
        }
        return landmarks
        # landmarks = torch.load(landmarks_path)
    else:
        print("Calculating landmarks for histogram standardization...")
        traind_list, _, _ = dataset.get_data_lists(config)
        traind_list = traind_list[:4]
        img_paths = [sample["image"] for sample in traind_list]
        if config["params"]["mask_images"]:
            mask_paths = [sample["segmentation"] for sample in traind_list]
        else:
            mask_paths = None

        tio.HistogramStandardization.train(
            img_paths, mask_path=mask_paths, output_path=landmarks_path
        )

        landmarks = {
            "mri": landmarks_path,
        }
    return landmarks
