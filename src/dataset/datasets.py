import os
from torchvision import transforms
from src.dataset.transforms import *
from src.dataset.masking_generator import TubeMaskingGenerator, TubeWindowMaskingGenerator
from src.dataset.kinetics import VideoClsDataset, VideoMAE, VideoClsDatasetFrame
from src.dataset.ssv2 import SSVideoClsDataset
from src.utils.config import get_cfg


class DataAugmentationForVideoMAE(object):
    def __init__(self):
        cfg = get_cfg()
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        # me: new added
        if not cfg.AUGMENTATION.NO_AUGMENTATION:
            self.train_augmentation = GroupMultiScaleCrop(cfg.MODEL.INPUT_SIZE, [1, .875, .75, .66])
        else:
            print(f"==> Note: do not use 'GroupMultiScaleCrop' augmentation during pre-training!!!")
            self.train_augmentation = IdentityTransform()
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if cfg.MODEL.MASK_TYPE == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                cfg.MODEL.WINDOW_SIZE, cfg.MODEL.MASK_RATIO
            )
        elif cfg.MODEL.MASK_TYPE == 'part_window':
            print(f"==> Note: use 'part_window' masking generator (window_size={cfg.MODEL.PART_WIN_SIZE[1:]}, apply_symmetry={cfg.MODEL.PART_APPLY_SYMMETRY})")
            self.masked_position_generator = TubeWindowMaskingGenerator(
                cfg.MODEL.WINDOW_SIZE, cfg.MODEL.MASK_RATIO, win_size=cfg.MODEL.PART_WIN_SIZE[1:], apply_symmetry=cfg.MODEL.PART_APPLY_SYMMETRY
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset():
    """Build pretraining dataset using global configuration."""
    cfg = get_cfg()
    transform = DataAugmentationForVideoMAE()
    dataset = VideoMAE(
        root=None,
        setting=cfg.DATA.DATA_PATH,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=cfg.DATA.NUM_FRAMES,
        new_step=cfg.DATA.SAMPLING_RATE,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        model=cfg.MODEL.NAME,
        num_segments=cfg.AUGMENTATION.NUM_SAMPLE,
    )
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode):
    """Build dataset using global configuration."""
    cfg = get_cfg()
    
    if cfg.DATA.DATASET_NAME == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            new_height=256,
            new_width=320,
            )
        nb_classes = 400
    
    elif cfg.DATA.DATASET_NAME == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=cfg.DATA.NUM_FRAMES,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            new_height=256,
            new_width=320,
            )
        nb_classes = 174

    elif cfg.DATA.DATASET_NAME == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            new_height=256,
            new_width=320,
            )
        nb_classes = 101
    
    elif cfg.DATA.DATASET_NAME == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            new_height=256,
            new_width=320,
            )
        nb_classes = 51

    # me: new added
    elif cfg.DATA.DATASET_NAME == 'DFEW':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'val.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            )
        nb_classes = 7

    elif cfg.DATA.DATASET_NAME == 'FERV39k':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            )
        nb_classes = 7

    elif cfg.DATA.DATASET_NAME == 'MAFW':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            file_ext='png' # me: new added for MAFW dataset
        )
        nb_classes = 11

    elif cfg.DATA.DATASET_NAME == 'RAVDESS':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            )
        nb_classes = 8


    elif cfg.DATA.DATASET_NAME == 'CREMA-D':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
        )
        nb_classes = 6


    elif cfg.DATA.DATASET_NAME == 'ENTERFACE':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
        )
        nb_classes = 6

    elif cfg.DATA.DATASET_NAME == 'Gaze360':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(cfg.DATA.DATA_PATH, 'val.csv')

        from src.dataset.kinetics import VideoClsDatasetGaze360
        dataset = VideoClsDatasetGaze360(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.DATA.TEST_NUM_SEGMENT,
            test_num_crop=cfg.DATA.TEST_NUM_CROP,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=cfg.MODEL.INPUT_SIZE,
            short_side_size=cfg.DATA.SHORT_SIDE_SIZE,
            file_ext='jpg',
            predict_last_frame=True,  # Predict the gaze of the last frame
        )
        nb_classes = 2

    else:
        raise NotImplementedError()
    assert nb_classes == cfg.DATA.NUM_CLASSES
    print("Number of the class = %d" % cfg.DATA.NUM_CLASSES)

    return dataset, nb_classes


# # 添加gaze360数据集构建函数

# def build_gaze360_dataset(args, is_train):
#     from gaze360.code.loader4finetune import ImagerLoader
#     import torchvision.transforms as transforms
    
#     # 数据变换
#     if is_train:
#         transform = transforms.Compose([
#             transforms.Resize((160, 160)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                std=[0.229, 0.224, 0.225])
#         ])
#     else:
#         transform = transforms.Compose([
#             transforms.Resize((160, 160)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                std=[0.229, 0.224, 0.225])
#         ])
    
#     # 数据集路径
#     if is_train:
#         file_name = os.path.join(args.data_path, 'train.txt')
#     else:
#         file_name = os.path.join(args.data_path, 'test.txt')
    
#     dataset = ImagerLoader(
#         source_path="../GazeCapture/Gaze360",
#         file_name=file_name,
#         transform=transform,
#         input_len=8
#     )
    
#     return dataset
