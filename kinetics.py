import os
import numpy as np
from numpy.lib.function_base import disp
import torch
import decord
from PIL import Image
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
import random
import glob


class VideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
            
            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                 / (self.test_num_crop - 1)
            temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                                / (self.test_num_segment - 1), 0)
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True ,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


"""
me: for frame-based datasets 
1. original min scale is too small (0.08) for faces, change it to 0.8
"""
class VideoClsDatasetFrame(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3, args=None,
                 file_ext='jpg', task='classification', gaze_frame_mode='last'):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args

        # me: new added (for MAFW with .png file)
        self.file_ext = file_ext
        self.task = task
        self.gaze_frame_mode = gaze_frame_mode  # 'last' or 'middle' for gaze prediction

        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        # print(f"==> Note: {self.anno_path} has {len(cleaned)} samples")
        # print(f"example: {cleaned.values[0]} - {cleaned.values[0, 1]}")
        self.dataset_samples = list(cleaned.values[:, 0])
        # me: support multi-outputs
        if task != 'classification': # regression
            # if task == 'gaze_regression':
            #     # Gaze360: each row has path and 3 angle values (yaw, pitch, roll)
            #     self.label_array = np.array(cleaned.values[:, 1:4], dtype=np.float32)
            # else:
            self.label_array = np.array(cleaned.values[:, 1:], dtype=np.float32)
        else: # classification
            self.label_array = list(cleaned.values[:, 1])

        if (mode == 'train'):
            # me: new added for less compute
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size, short_side_size), interpolation='bilinear')
                # me: old, may have bug (heigh != width)
                # video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size, short_side_size), interpolation='bilinear'),
                # me: old, may have bug (heigh != width)
                # video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size, short_side_size), interpolation='bilinear')
                # me: old, may have bug (heigh != width)
                # video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            # buffer = self.load_video(sample, sample_rate_scale=scale_t)  # T H W C
            try:
                buffer = self.load_video(sample, sample_rate_scale=scale_t)  # T H W C
            except Exception as e:
                print(f"==> Note: Error '{e}' occurred when load video of '{sample}'!!!")
                exit(-1)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_video(sample, sample_rate_scale=scale_t)

            # me: new added for less compute
            buffer = self.data_resize(buffer)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.load_video(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_video(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample
            # return buffer, self.label_array[index], sample.split("/")[-1] if self.args.data_set not in ['FERV39k', 'OULU-CASIA', 'AFEW', 'CAER'] else '_'.join(sample.split("/")[-3:])
            # me: .split(".")[0] will cause bug for chalearn dataset
            # return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0] if self.args.data_set not in ['FERV39k', 'OULU-CASIA', 'AFEW', 'CAER'] else '_'.join(sample.split("/")[-3:])
        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.load_video(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format( \
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.load_video(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                           / (self.test_num_crop - 1)
            temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                                / (self.test_num_segment - 1), 0)
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                         spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                         :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample, chunk_nb, split_nb
            # return buffer, self.test_label_array[index], sample.split("/")[-1]  if self.args.data_set not in ['FERV39k', 'OULU-CASIA', 'AFEW', 'CAER'] else '_'.join(sample.split("/")[-3:]), \
            #        chunk_nb, split_nb
            # return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0]  if self.args.data_set not in ['FERV39k', 'OULU-CASIA', 'AFEW', 'CAER'] else '_'.join(sample.split("/")[-3:]), \
            #        chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
            self,
            buffer,
            args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        # me: buffer is already a list of PIL Images (using VideoReaderFrame)
        # buffer = [
        #     transforms.ToPILImage()(frame) for frame in buffer
        # ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            # me: org min scale is too small
            # [0.08, 1.0],
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def load_video(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # me: support image dataset (by simply copy T times)
        if os.path.isfile(sample):
            with open(sample, "rb") as f:
                img = Image.open(f)
                return [img.convert("RGB")] * self.clip_len

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReaderFrame(fname, file_ext=self.file_ext)
            else:
                raise NotImplementedError
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            # me: buffer is a list of PIL Images returned by VideoReaderFrame, not numpy array in original implementation
            # vr.seek(0)
            # buffer = vr.get_batch(all_index).asnumpy()
            buffer = vr.load(all_index)
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        # me: buffer is a list of PIL Images returned by VideoReaderFrame, not numpy array in original implementation
        # vr.seek(0)
        # buffer = vr.get_batch(all_index).asnumpy()
        buffer = vr.load(all_index)
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


"""
me: apdot similar interfaces to VideoReader in decord 
"""
class VideoReaderFrame:
    def __init__(self, video_dir, file_ext='jpg'):
        self.video_dir = video_dir
        self.frames = sorted(glob.glob(os.path.join(video_dir, f'*.{file_ext}')))

    def __len__(self):
        return len(self.frames)

    # from: https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def load(self, idxs):
        return [self.pil_loader(self.frames[idx]) for idx in idxs]


class VideoReaderGaze360:
    """Specialized video reader for Gaze360 dataset with sequential frame prediction"""
    def __init__(self, base_frame, file_ext='jpg', clip_len=16):
        self.base_frame = base_frame.split('/')[-1].split('.')[0]  # get the base frame name without extension
        self.base_frame = int(self.base_frame)  # convert to integer if it's a number
        # print(f"==> Note: VideoReaderGaze360 initialized with base_frame: {self.base_frame}, file_ext: {file_ext}, clip_len: {clip_len}")
        self.video_dir = base_frame.split('/')[:-1]
        self.video_dir = os.path.join(*self.video_dir)  # join path components
        # print(f"==> Note: VideoReaderGaze360 video_dir: {self.video_dir}")
        self.file_ext = file_ext
        self.clip_len = clip_len
        self.frames = sorted(glob.glob(os.path.join(self.video_dir, f'*.{file_ext}')))


    def pil_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def load_clip(self):
       
        # end_idx = self.frames.index(self.base_frame)
        # return_frames = []
        # for i in range(end_idx, self.clip_len):
        #     return_frames.append(self.frames[0])
        # for i in range(max(0, end_idx - self.clip_len + 1), end_idx + 1):
        #     return_frames.append(self.frames[i])
        lst_available_frame = os.path.join(self.video_dir, f'{self.base_frame:06d}.{self.file_ext}')
        return_frames = []
        # print(f"==> Note[0]: Loading clip from {lst_available_frame} with clip_len {self.clip_len}")
        while os.path.exists(lst_available_frame) and len(return_frames) < self.clip_len:
            # append from front
            return_frames.append(lst_available_frame)
            self.base_frame -= 1
            lst_available_frame = os.path.join(self.video_dir, f'{self.base_frame:06d}.{self.file_ext}')
        # print(f"==> Note[1]: Loading clip from {lst_available_frame} with clip_len {self.clip_len}")
        if len(return_frames) < self.clip_len:
            # append from back
            self.base_frame += 1
            lst_available_frame = os.path.join(self.video_dir, f'{self.base_frame:06d}.{self.file_ext}')
        # print(f"==> Note[2]: Loading clip from {lst_available_frame} with clip_len {self.clip_len}")
        while len(return_frames) < self.clip_len:
            return_frames.append(lst_available_frame)
            
        # reverse the order to match the original sequence
        return_frames.reverse()
        
        # print(f"Loading clip from {self.base_frame} with length {self.clip_len} from {self.video_dir}")
        # print(f"end_idx: {self.base_frame}, clip_len: {self.clip_len}, frames: {self.frames}")
        # print(f"return_frames: {return_frames}")
        # print(f"return_frames: {len(return_frames)}")
            
        
        return [self.pil_loader(frame) for frame in return_frames]


class VideoClsDatasetGaze360(VideoClsDatasetFrame):
    """Specialized dataset for Gaze360 sequential frame prediction"""
    
    def __init__(self, anno_path, data_path, mode='train', clip_len=16,
                 frame_sample_rate=1, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3, args=None,
                 file_ext='jpg', predict_last_frame=True):
        # Initialize parent class with regression task
        super().__init__(
            anno_path=anno_path, data_path=data_path, mode=mode, clip_len=clip_len,
            frame_sample_rate=frame_sample_rate, crop_size=crop_size, short_side_size=short_side_size,
            new_height=new_height, new_width=new_width, keep_aspect_ratio=keep_aspect_ratio,
            num_segment=num_segment, num_crop=num_crop, test_num_segment=test_num_segment, 
            test_num_crop=test_num_crop, args=args, file_ext=file_ext, task='gaze_regression'
        )
        self.predict_last_frame = predict_last_frame

    def load_video(self, sample, sample_rate_scale=1):
        """Load sequential frames for Gaze360"""
        fname = sample
        
        if not os.path.exists(fname):
            print(f"file does not exist: {fname}")
            return []
        
        # Directory case - load sequential frames
        try:
            vr = VideoReaderGaze360(fname, file_ext=self.file_ext, clip_len=self.clip_len)

            buffer = vr.load_clip()
                
            # if self.mode == 'train':
            #     # Random sampling for training
            #     start_idx = np.random.randint(0, max(1, len(vr)))
            #     buffer = vr.load_clip(start_idx)
            # else:
            #     # Use middle clip for validation/test
            #     start_idx = len(vr) // 2 if len(vr) > 0 else 0
            #     buffer = vr.load_clip(start_idx)
                
                
            return buffer
            
        except Exception as e:
            print(f"Error loading Gaze360 video from {fname}: {e}")
            return []

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            sample = self.dataset_samples[index]
            # print(f"Loading training sample {index}: {sample}")
            
            try:
                buffer = self.load_video(sample, sample_rate_scale=1)
            except Exception as e:
                print(f"Error loading video {sample}: {e}")
                # Return a random sample instead
                index = np.random.randint(self.__len__())
                return self.__getitem__(index)
                
            if len(buffer) == 0:
                # Return a random sample instead
                index = np.random.randint(self.__len__())
                return self.__getitem__(index)

            # Apply resize transformation
            # print(f"Loaded {len(buffer)} frames for sample {sample}")
            buffer = self.data_resize(buffer)
            
            # # debug 
            # import matplotlib.pyplot as plt
            # show_img = np.array(buffer[-1])
            # show_lbl = self.label_array[index]
            # show_name = sample
            # plt.imshow(show_img)
            # # plt.text(0, 0, f"Label: {show_lbl}, Name: {show_name}", color='black', fontsize=12)
            # plt.title(f"Label: {show_lbl}, Name: {show_name}")
            # plt.axis('off')
            # # plt.show()
            # plt.savefig(f"debug/debug_{index}.png", bbox_inches='tight', pad_inches=0.1)
            # [checked, correct, BUT NOT NORMALIZED], done in _aug_frame
            
            # print(f"After resizing, buffer length: {len(buffer)}")
            
            # Apply data augmentation
            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            # Return 16 frames with last frame's gaze angles
            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.load_video(sample)
            
            if len(buffer) == 0:
                print(f"Warning: Empty buffer for validation sample {sample}")
                # Create dummy data
                dummy_img = Image.new('RGB', (self.crop_size, self.crop_size), color='black')
                buffer = [dummy_img] * self.clip_len
                
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.load_video(sample)

            if len(buffer) == 0:
                print(f"Warning: Empty buffer for test sample {sample}")
                dummy_img = Image.new('RGB', (self.crop_size, self.crop_size), color='black')
                buffer = [dummy_img] * self.clip_len

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            # For Gaze360, we don't need temporal and spatial cropping like action recognition
            # Just apply the standard transform
            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample, chunk_nb, split_nb
        else:
            raise NameError('mode {} unknown'.format(self.mode))


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


class VideoMAE(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """
    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False,
                 # me: new added for VoxCeleb2
                 model=None
                 ):

        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init


        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

        # me: new added for VoxCeleb2
        self.is_voxceleb2 = False
        self.crop_idxs = None
        if 'voxceleb2' in setting.lower():
            self.is_voxceleb2 = True
            image_size = int(model.split('_')[-1])
            if image_size == 192:
                self.crop_idxs = ((0, 192), (16, 208))
                print(f"==> Note: use crop_idxs={self.crop_idxs} for VoxCeleb2!!!")
            elif image_size <= 160: # me: old is == 160
                self.crop_idxs = ((0, 160), (32, 192))
                print(f"==> Note: use crop_idxs={self.crop_idxs} for VoxCeleb2!!!")


    def __getitem__(self, index):

        directory, target = self.clips[index]
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = '{}.{}'.format(directory, self.video_ext)
            try:
                decord_vr = decord.VideoReader(video_name, num_threads=1)
                duration = len(decord_vr)
            except Exception as e:
                next_idx = random.randint(0, self.__len__() - 1)
                print(f"==> Exception '{e}' occurred when processed '{directory}', move to random next one (idx={next_idx}).")
                return self.__getitem__(next_idx)

        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)

        process_data, mask = self.transform((images, None)) # T*C,H,W
        # process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        # me: for repeated sampling
        process_data = process_data.view((self.num_segments * self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W

        return (process_data, mask)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                item = (clip_path, target)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets


    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            if self.is_voxceleb2 and self.crop_idxs is not None:
                sampled_list = [Image.fromarray(video_data[vid, self.crop_idxs[0][0]:self.crop_idxs[0][1], self.crop_idxs[1][0]:self.crop_idxs[1][1], :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
            else:
                sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list
