#第一个训练脚本.
# 我们目标是, 客户给一个视频, 然后我们可以根据这个视频来优化我们的模型参数, 最后再通过这个视频,来生成任意语音的 音频驱动数字人.



import sys
sys.path.append("..")
from  models import audio
from os import  path
from concurrent.futures import as_completed, ProcessPoolExecutor
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob





class a():
    pass
args=a()
args.process_num=6
args.data_root='video'
args.out_root='lrs2_audio'



sample_rate=16000  # 16000Hz
template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

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


def mp_handler_audio(job):
    vfile, args = job
    try:
        process_audio_file(vfile, args)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main(args):
    print("looking up paths.... from", args.data_root)
    filelist = glob(path.join(args.data_root, '*/*.mp4'))

    jobs = [(vfile, args) for i, vfile in enumerate(filelist)]
    p_audio = ProcessPoolExecutor(args.process_num) # 进程数量是6个.
    futures_audio = [p_audio.submit(mp_handler_audio, j) for j in jobs]

    _ = [r.result() for r in tqdm(as_completed(futures_audio), total=len(futures_audio))]
    print("complete, output to",args.out_root)

if __name__ == '__main__':
    main(args)