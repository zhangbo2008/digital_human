#======所有训练代码只需要跑这一个py文件


#=====用这一个文件训练.后续再改多个文件.
data='video/test1.mp4'

import os.path
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, traceback
from tqdm import tqdm
import glob
import argparse
import math
from typing import List, Mapping, Optional, Tuple, Union
import cv2
import dataclasses
import numpy as np
from mediapipe.framework.formats import landmark_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--process_num', type=int, default=6) #number of process in ThreadPool to preprocess the dataset
parser.add_argument('--dataset_video_root', type=str, required=False)
parser.add_argument('--output_sketch_root', type=str, default='./lrs2_sketch128')
parser.add_argument('--output_face_root', type=str, default='./lrs2_face128')
parser.add_argument('--output_landmark_root', type=str, default='./lrs2_landmarks')

args = parser.parse_args()

input_mp4_root = 'video'
output_sketch_root = args.output_sketch_root
output_face_root=args.output_face_root
output_landmark_root=args.output_landmark_root



"""MediaPipe solution drawing utils."""
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                 (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])
# (10, 338), (338, 297), (297, 332), (332, 284),(284, 251), (251, 389) (162, 21), (21, 54),(54, 103), (103, 67), (67, 109), (109, 10)

FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4), \
                           (4, 45), (45, 220), (220, 115), (115, 48), \
                           (4, 275), (275, 440), (440, 344), (344, 278), ])
FACEMESH_FULL = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])
def summarize_landmarks(edge_set):
    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks

all_landmark_idx = summarize_landmarks(FACEMESH_FULL)
pose_landmark_idx = \
    summarize_landmarks(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE, \
                                              FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, ])).union(
        [162, 127, 234, 93, 389, 356, 454, 323])
content_landmark_idx = all_landmark_idx - pose_landmark_idx

def draw_landmarks(
        image: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_drawing_spec: Union[DrawingSpec,
        Mapping[int, DrawingSpec]] = DrawingSpec(
            color=RED_COLOR),
        connection_drawing_spec: Union[DrawingSpec,
        Mapping[Tuple[int, int],
        DrawingSpec]] = DrawingSpec()):
    """Draws the landmarks and the connections on the image.
  Args:
    image: A three channel BGR image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: Either a DrawingSpec object or a mapping from
      hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
      settings such as color, line thickness, and circle radius.
      If this argument is explicitly set to None, no landmarks will be drawn.
    connection_drawing_spec: Either a DrawingSpec object or a mapping from
      hand connections to the DrawingSpecs that specifies the
      connections' drawing settings such as color and line thickness.
      If this argument is explicitly set to None, no landmark connections will
      be drawn.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel BGR.
      b) If any connetions contain invalid landmark index.
  """
    if not landmark_list:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError('Input image must contain three channel bgr data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                # if start_idx in content_landmark and end_idx in content_landmark:
                cv2.line(image, idx_to_coordinates[start_idx],
                         idx_to_coordinates[end_idx], drawing_spec.color,
                         drawing_spec.thickness)
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# process_video_file 这个函数把一个视频每一帧的人脸识别出来, 往外扩5像素, 然后找到landmark记录成npy,然后画好变成128*128之后,保存图片. 3个保存地方是 lrs2_sketch128,  lrs2_landmarks   lrs2_face128
def process_video_file(mp4_path):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        video_stream = cv2.VideoCapture(mp4_path)
        fps = round(video_stream.get(cv2.CAP_PROP_FPS))
        if fps != 25:
            print(mp4_path, ' fps is not 25!!!')
            exit()
        frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)

        for frame_idx,full_frame in enumerate(frames):
            h, w = full_frame.shape[0], full_frame.shape[1]
            results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue  # not detect
            face_landmarks=results.multi_face_landmarks[0]

            #(1)normalize landmarks
            x_min=999
            x_max=-999
            y_min=999
            y_max=-999
            pose_landmarks, content_landmarks = [], []
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in all_landmark_idx:
                    if landmark.x<x_min:
                        x_min=landmark.x
                    if landmark.x>x_max:
                        x_max=landmark.x

                    if landmark.y<y_min:
                        y_min=landmark.y
                    if landmark.y>y_max:
                        y_max=landmark.y
                ######
                if idx in pose_landmark_idx:
                    pose_landmarks.append((idx,landmark.x,landmark.y))
                if idx in content_landmark_idx:
                    content_landmarks.append((idx,landmark.x,landmark.y))
            ##########plus 5 pixel to size###########但是推理时候为啥25???????
            x_min=max(x_min-5/w,0)
            x_max = min(x_max + 5 / w, 1)
            #
            y_min = max(y_min - 5 / h, 0)
            y_max = min(y_max + 5 / h, 1)
            face_frame=cv2.resize(full_frame[int(y_min*h):int(y_max*h),int(x_min*w):int(x_max*w)],(128,128))

            # update landmarks, landmarks变成0-1之间的数.归一化了.
            pose_landmarks=[ \
                (idx,(x-x_min)/(x_max-x_min),(y-y_min)/(y_max-y_min)) for idx,x,y in pose_landmarks]
            content_landmarks=[\
                (idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)) for idx, x, y in content_landmarks]
            # update drawed landmarks
            for idx,x,y in pose_landmarks + content_landmarks:
                face_landmarks.landmark[idx].x=x
                face_landmarks.landmark[idx].y=y
            #save landmarks
            result_dict={}
            result_dict['pose_landmarks']=pose_landmarks
            result_dict['content_landmarks']=content_landmarks
            out_dir = os.path.join(output_landmark_root, '/'.join(mp4_path[:-4].split('/')[-2:]))
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir,str(frame_idx)),result_dict)

            #save sketch
            h_new=(y_max-y_min)*h
            w_new = (x_max - x_min) * w
            annotated_image = np.zeros((int(h_new * 128 / min(h_new, w_new)), int(w_new * 128 / min(h_new, w_new)), 3))
            draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,  # FACEMESH_CONTOURS  FACEMESH_LIPS
                connections=FACEMESH_FULL,
                connection_drawing_spec=drawing_spec)  # landmark_drawing_spec=None,
            annotated_image = cv2.resize(annotated_image, (128, 128)) # 画完landmark之后变回128*128

            out_dir = os.path.join(output_sketch_root, '/'.join(mp4_path[:-4].split('/')[-2:]))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, str(frame_idx)+'.png'), annotated_image)

            #save face frame
            out_dir = os.path.join(output_face_root, '/'.join(mp4_path[:-4].split('/')[-2:]))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, str(frame_idx) + '.png'), face_frame)

def mp_handler(mp4_path):
    try:
        process_video_file(mp4_path)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main():
    print('looking up videos.... ')
    mp4_list = glob.glob(input_mp4_root + '/*/*.mp4')  #example: .../lrs2_video/5536038039829982468/00001.mp4
    print('total videos :', len(mp4_list))

    process_num = args.process_num
    print('process_num: ', process_num)
    p_frames = ThreadPoolExecutor(process_num)
    futures_frames = [p_frames.submit(mp_handler, mp4_path) for mp4_path in mp4_list]
    _ = [r.result() for r in tqdm(as_completed(futures_frames), total=len(futures_frames))]
    print("complete task!")

if 0:
    process_video_file(data)

    print('给的视频处理完毕了.')




#-========处理音频:
import sys
sys.path.append("..")
from  models import audio
from os import  path
from concurrent.futures import as_completed, ProcessPoolExecutor
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--process_num', type=int, default=6) #number of process to preprocess the audio
parser.add_argument("--data_root", type=str,help="Root folder of the LRS2 dataset", required=False)
parser.add_argument("--out_root", help="output audio root", required=False)
args = parser.parse_args()
args.out_root='lrs2_audio'

sample_rate=16000  # 16000Hz
template = 'ffmpeg -y -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}'
# ffmpeg -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}

def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.out_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)
    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile.replace(' ', r'\ '), wavpath.replace(' ', r'\ '))
    subprocess.run(command, shell=True)
    wav = audio.load_wav(wavpath, sample_rate)
    orig_mel = audio.melspectrogram(wav).T
    np.save(path.join(fulldir, 'audio'), orig_mel)
    print('保存音频地址为',path.join(fulldir, 'audio'))
if 0:
    process_audio_file(data,args)






# # train Landmark generator
# train the landmark generator network by running:

# CUDA_VISIBLE_DEVICES=0 python train_landmarks_generator.py --pre_audio_root ..../lrs2_audio --landmarks_root ..../lrs2_landmarks


# The models are trained until the eval_L1_loss no longer decreases (about 6e-3). Under the default batchsize setting on a single RTX 3090, our model stopped at epoch 1837(610k iteration) with eval_L1_loss 5.866 e-3, using no more than one day.





print('开始训练landmark生成器')

import os
from tqdm import tqdm
import torch
import numpy as np
from glob import glob
from os.path import join, isfile
import  random
# from tensorboardX import SummaryWriter
from models import Landmark_generator as Landmark_transformer
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--pre_audio_root',default='...../Dataset/lrs2_preprocessed_audio',
                    help='root path for preprocessed  audio')
parser.add_argument('--landmarks_root',default='...../Dataset/lrs2_landmarks',
                    help='root path for preprocessed  landmarks')
args=parser.parse_args()
args.pre_audio_root='lrs2_audio'
args.landmarks_root='lrs2_landmarks'
#network parameters
d_model=512
dim_feedforward=1024
nlayers=4
nhead=4
dropout=0.1 # 0.5
Nl=15
T = 5
Project_name = 'landmarkT5_d512_fe1024_lay4_head4'
print('Project_name:', Project_name)
finetune_path ='checkpoints/landmark_checkpoint.pth'
num_workers = 0  #=========windows禁止并发才行.
batch_size =128  # 512
batch_size_val =128  #512
evaluate_interval = 5000  #
checkpoint_interval = evaluate_interval
mel_step_size = 16
fps = 25
lr = 1e-4
global_step, global_epoch = 0, 0
landmark_root=args.landmarks_root
filelist_name = 'lrs2'
checkpoint_root = './checkpoints/landmark_generation/'
checkpoint_dir = os.path.join(checkpoint_root, 'Pro_' + Project_name)
reset_optimizer = False
save_optimizer_state = True
# writer = SummaryWriter('tensorboard_runs/Project_{}'.format(Project_name))
#we arrange the landmarks in some order
ori_sequence_idx=[162,127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,  #
    70,63,105,66,107,55,65,52,53,46,#
    336,296,334,293,300,276,283,282,295,285,#
    168,6,197,195,5,#
    48,115,220,45,4,275,440,344,278,#
     33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,#
    362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,#
    61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,#
    78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
full_face_sequence=[*list(range(0, 4)), *list(range(21, 25)), *list(range(25, 91)), *list(range(4, 21)), *list(range(91, 131))]
class LandmarkDict(dict):
    def __init__(self, idx, x, y):
        self['idx'] = idx
        self['x'] = x
        self['y'] = y

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class Dataset(object):
    def get_vidname_list(self, split):
        vid_name_list = []
        with open('filelists/{}/{}.txt'.format(filelist_name, split)) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                vid_name_list.append(line)
        return vid_name_list

    def __init__(self, split):
        min_len = 25  #filter videos that is too short
        vid_name_lists = self.get_vidname_list(split)
        self.all_video_names = []
        print("init dataset,filtering very short videos.....")
        #========================改成自己的单文件
        vid_name_lists=['video/test1'] #这里一个npy是一个帧的特征.
        for vid_name in tqdm(vid_name_lists, total=len(vid_name_lists)):
            pkl_paths = list(glob(join(landmark_root,vid_name, '*.npy')))
            vid_len=len(pkl_paths)
            if vid_len >= min_len:
                self.all_video_names.append((vid_name, vid_len))
        print("complete,with available vids: ", len(self.all_video_names), '\n')

    def __len__(self):
        return len(self.all_video_names)

    def __getitem__(self, idx):
        while 1:
            vid_idx = random.randint(0, len(self.all_video_names) - 1)
            vid_name = self.all_video_names[vid_idx][0]
            vid_len=self.all_video_names[vid_idx][1]
            # 00.randomly select a window of T video frames #=====随机选一个窗口5帧, 25帧一秒, 也就是窗口0.2秒.
            random_start_idx = random.randint(2, vid_len - T - 2) # 最小是2.
            T_idxs = list(range(random_start_idx, random_start_idx + T))
#====下面特征分2个不分一个是T_idxs表示当前窗口的特征, 一个是Nl_idxs表示非当前窗口的特征.
            # 01. get reference landmarks
            all_list=[i for i in range(vid_len) if i not in T_idxs]
            Nl_idxs = random.sample(all_list, Nl)       #======其他帧里面选landmark,因为我们最后的效果也是用其他音频匹配的视频, 来生成要的音频驱动的视频.
            Nl_landmarks_paths = [os.path.join(landmark_root, vid_name, str(idx) + '.npy') for idx in Nl_idxs]

            Nl_pose_landmarks,Nl_content_landmarks= [],[]
            for frame_landmark_path in Nl_landmarks_paths:
                if not os.path.exists(frame_landmark_path):
                    break
                landmarks=np.load(frame_landmark_path,allow_pickle=True).item()
                Nl_pose_landmarks.append(landmarks['pose_landmarks'])
                Nl_content_landmarks.append(landmarks['content_landmarks'])
            if len(Nl_pose_landmarks) != Nl:
                continue
            Nl_pose = torch.zeros((Nl, 2, 74))  # 74 landmark
            Nl_content = torch.zeros((Nl, 2, 57))  # 57 landmark
            for idx in range(Nl):
                Nl_pose_landmarks[idx] = sorted(Nl_pose_landmarks[idx],
                                               key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))  # 把pose_lm的索引按照 ori_sequence_idx 的索引进行排序.
                Nl_content_landmarks[idx] = sorted(Nl_content_landmarks[idx],
                                                  key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))

                Nl_pose[idx, 0, :] = torch.FloatTensor(
                    [Nl_pose_landmarks[idx][i][1] for i in range(len(Nl_pose_landmarks[idx]))])  # x
                Nl_pose[idx, 1, :] = torch.FloatTensor(
                    [Nl_pose_landmarks[idx][i][2] for i in range(len(Nl_pose_landmarks[idx]))])  # y

                Nl_content[idx, 0, :] = torch.FloatTensor(
                    [Nl_content_landmarks[idx][i][1] for i in range(len(Nl_content_landmarks[idx]))])  # x
                Nl_content[idx, 1, :] = torch.FloatTensor(
                    [Nl_content_landmarks[idx][i][2] for i in range(len(Nl_content_landmarks[idx]))])  # y
            # 02. get T pose landmark and content landmark
            T_ladnmark_paths = [os.path.join(landmark_root, vid_name, str(idx) + '.npy') for idx in T_idxs]
            T_pose_landmarks,T_content_landmarks=[],[]
            for frame_landmark_path in T_ladnmark_paths:
                if not os.path.exists(frame_landmark_path):
                    break
                landmarks=np.load(frame_landmark_path,allow_pickle=True).item()
                T_pose_landmarks.append(landmarks['pose_landmarks'])
                T_content_landmarks.append(landmarks['content_landmarks'])
            if len(T_pose_landmarks)!=T:
                continue
            T_pose=torch.zeros((T,2,74))   #74 landmark
            T_content=torch.zeros((T,2,57))  #57 landmark
            for idx in range(T):
                T_pose_landmarks[idx]=sorted(T_pose_landmarks[idx],key=lambda land_tuple:ori_sequence_idx.index(land_tuple[0]))
                T_content_landmarks[idx] = sorted(T_content_landmarks[idx],key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))

                T_pose[idx,0,:]=torch.FloatTensor([T_pose_landmarks[idx][i][1] for i in range(len(T_pose_landmarks[idx]))] ) #x
                T_pose[idx,1,:]=torch.FloatTensor([T_pose_landmarks[idx][i][2] for i in range(len(T_pose_landmarks[idx]))]) #y

                T_content[idx, 0, :] = torch.FloatTensor([T_content_landmarks[idx][i][1] for i in range(len(T_content_landmarks[idx]))])  # x
                T_content[idx, 1, :] = torch.FloatTensor([T_content_landmarks[idx][i][2] for i in range(len(T_content_landmarks[idx]))])  # y
            # 03. get T audio
            try:
                audio_mel = np.load(join(args.pre_audio_root,vid_name, "audio.npy"))
            except Exception as e:
                continue
            T_mels = []
            for frame_idx in T_idxs:
                mel_start_frame_idx = frame_idx - 2  ###around the frame, 因为窗口是5, 我们要的是, 每一个帧, 再对应他上2,到下2的长为5的窗口作为特征.
                if mel_start_frame_idx < 0:
                    break
                start_idx = int(80. * (mel_start_frame_idx / float(fps))) # mel里面一秒80个采样.  (mel_start_frame_idx / float(fps):是秒数, 乘以80,就是mel的索引.
                m = audio_mel[start_idx: start_idx + mel_step_size, :]  # get five frames around
                if m.shape[0] != mel_step_size:  # in the end of vid
                    break
                T_mels.append(m.T)  # transpose
            if len(T_mels) != T:
                continue
            T_mels = np.asarray(T_mels)  # (T,hv,wv)
            T_mels = torch.FloatTensor(T_mels).unsqueeze(1)  # (T,1,hv,wv)

            #  return value
            return T_mels,    T_pose,   T_content,Nl_pose,Nl_content
            #     (T,1,hv,wv) (T,2,74)  (T,2,57)
def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    # for k, v in s.items():
    #     new_s['module.'+k] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]
    return model


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}_epoch_{}_checkpoint_step{:09d}.pth".format(prefix, epoch, global_step))
    if isfile(checkpoint_path):
        os.remove(checkpoint_path)
    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

criterion_L1 = torch.nn.L1Loss()

def get_velocity_loss (pred, gt):  #(B*T,2,57) (B*T,2,57)

    pred=torch.stack(torch.split(pred,T,dim=0),dim=0)  #(B,T,2,57)
    gt = torch.stack(torch.split(gt, T, dim=0), dim=0)  # (B,T,2,57)

    pred=torch.cat([pred[:,:,:,i] for i in range(pred.size(3))],dim=2)  #(B,T,57*2)
    gt = torch.cat([gt[:, :, :, i] for i in range(gt.size(3))], dim=2)  # (B,T,57*2)

    b, t, c = pred.shape
    pred_spiky = pred[:, 1:, :] - pred[:, :-1, :]   #  计算帧与帧之间的差距. 让他小, 这样画面没有撕裂感.
    gt_spiky = gt[:, 1:, :] - gt[:, :-1, :] # 让撕裂感跟原撕裂感校准.

    pred_spiky = pred_spiky.view(b * (t - 1), c)
    gt_spiky = gt_spiky.view(b * (t - 1), c)
    pairwise_distance = torch.nn.functional.pairwise_distance(pred_spiky, gt_spiky)
    return torch.mean(pairwise_distance)

def evaluate(model, val_data_loader):
    global global_epoch, global_step
    eval_epochs = 25
    print('Evaluating model for {} epochs'.format(eval_epochs))
    eval_L1_loss = 0.
    eval_velocity_loss=0.
    count = 0
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder):
        os.mkdir(folder)
    for epoch in tqdm(range(eval_epochs),total=eval_epochs):
        prog_bar = enumerate(val_data_loader)
        for step, (T_mels,T_pose,T_content,Nl_pose,Nl_content) in prog_bar:
            model.eval()
            T_mels, T_pose, T_content,Nl_pose,Nl_content = T_mels.cuda(non_blocking=True), T_pose.cuda(non_blocking=True), T_content.cuda(non_blocking=True), \
                                                                 Nl_pose.cuda(non_blocking=True), Nl_content.cuda(non_blocking=True)
            # (B,T,1,hv,wv) (B,T,2,74)  (B,T,2,57)
            predict_content = model(T_mels, T_pose,Nl_pose,Nl_content)  # (B*T,2,57)
            T_content = torch.cat([T_content[i] for i in range(T_content.size(0))], dim=0)  # (B*T,2,57)

            eval_L1_loss += criterion_L1(predict_content, T_content).item()
            eval_velocity_loss +=get_velocity_loss(predict_content, T_content).item()
            count += 1
    # writer.add_scalar('eval_L1_loss', eval_L1_loss / count, global_step)
    print('eval_L1_loss', eval_L1_loss / count, 'global_step:', global_step)
    # writer.add_scalar('eval_velocity_loss', eval_velocity_loss / count, global_step)
    print('eval_velocity_loss', eval_velocity_loss / count, 'global_step:', global_step)
if 1:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
  
    # create a model and optimizer
    model = Landmark_transformer(T,d_model,nlayers,nhead,dim_feedforward,dropout)
    if finetune_path is not None:  ###fine tune # 加载官方权重.
        model_dict = model.state_dict()
        print('load module....from :', finetune_path)
        checkpoint = torch.load(finetune_path, map_location=device)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        state_dict_needed = {k: v for k, v in new_s.items() if k in model_dict.keys()}  # we need in model
        model_dict.update(state_dict_needed)
        model.load_state_dict(model_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    train_dataset = Dataset('train')
    val_dataset = Dataset('train')
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    while global_epoch < 9999999999: #=====你这不如直接死循环得了.....
        prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        running_L1_loss,running_velocity_loss=0.,0.
        for step, (T_mels, T_pose, T_content, Nl_pose, Nl_content) in enumerate(train_data_loader):
            if global_step and (global_step % checkpoint_interval) == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, prefix=Project_name)
            if 0:# 这里不进行eval了.
                if global_step % evaluate_interval == 0 or global_step == 100:
                    with torch.no_grad():
                        evaluate(model, val_data_loader)
            T_mels,T_pose,T_content,Nl_pose,Nl_content= T_mels.to(device), T_pose.to(device), T_content.to(device), \
                Nl_pose.to(device),Nl_content.to(device)
            #(B,T,1,hv,wv) (B,T,2,74)  (B,T,2,57)
            model.train() 
            optimizer.zero_grad() #======这个model, 输入音频 和 参考视频的landmark, 输出 音频驱动结果视频的  landmark ,   T_mel表示当前帧的mel, T_pose当前的pose, Nl_pose其他时间点的pose, Nl_content其他时间点的content   , 其中pose是身体特征, content是面部特征.
            predict_content=model(T_mels, T_pose, Nl_pose, Nl_content)  #(B*T,2,57)
            T_content=torch.cat([T_content[i] for i in range(T_content.size(0))],dim=0) #(B*T,2,57):ground truth lip and jaw landmarks   , T_content就是ground_true

            L1_loss=criterion_L1(predict_content,T_content)
            Velocity_loss=get_velocity_loss(predict_content, T_content)

            loss= L1_loss + Velocity_loss

            loss.backward()
            
            optimizer.step()
            print("当前loss",loss.item(),"step:",global_step)
            running_L1_loss+=L1_loss.item()
            running_velocity_loss+=Velocity_loss.item()


            prog_bar.set_description('epoch: %d step: %d running_L1_loss: %.4f  running_velocity_loss: %.4f '
                    % (global_epoch, global_step, running_L1_loss / (step + 1), running_velocity_loss / (step + 1)))
            # writer.add_scalar('running_L1_loss', running_L1_loss / (step + 1), global_step)
            # writer.add_scalar('running_velocity_loss', running_velocity_loss / (step + 1), global_step)
            global_step += 1
        global_epoch += 1
    print("end")
