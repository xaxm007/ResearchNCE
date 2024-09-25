import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import json
from PIL import Image, ImageFont, ImageDraw
import argparse
import os
from tqdm import tqdm
from deep_translator import GoogleTranslator

def get_frame_caption(frame_time, dense_captions):
    frame_captions = []
    print_captions = []
    idx_list = []
    for event in dense_captions:
        s, e = event['timestamp']
        if frame_time >= s and frame_time <= e:
            frame_captions.append(event)
            idx_list.append(event['original_id'])
    
    best_score = None
    for event in frame_captions:
        if best_score is None or event['sentence_score'] < best_score['sentence_score']:
            best_score = event
    
    if best_score:
        print_captions.append(best_score)

    return print_captions

def paint_text(im, text, font, pos, color, bg_color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_PIL)
    
    # Get text bounding box using getbbox
    text_bbox = font.getbbox(text)
    text_x, text_y = pos
    bg_x1, bg_y1 = text_x, text_y
    bg_x2, bg_y2 = text_x + (text_bbox[2] - text_bbox[0]), text_y + (text_bbox[3] - text_bbox[1])

    # Draw black rectangle for background
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)

    # Draw text on top of rectangle
    draw.text(pos, text, font=font, fill=color)
    
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img

def processImg(img, cur_time, title, dense_captions, output_language='en', n_caption=3):
    scale = 0.7
    basic_text_height = 50
    text_height = int(basic_text_height * scale)
    font_size = int(text_height * 0.8)

    h, w, c = img.shape
    last_time = cur_time
    cur_time = time.time()
    img_fps = 1. / (cur_time - last_time + 1e-8)

    img = img.astype('uint8')
    if output_language == 'en':
        font = ImageFont.truetype("visualization/Arial.ttf", font_size)
    else:
        font = ImageFont.truetype("visualization/NotoSansDevanagari.ttf", font_size)
    
    img = paint_text(img, title, font, (10, 0), color=(255, 255, 255), bg_color=(0, 0, 0))
    
    for i, proposal in enumerate(dense_captions[:n_caption]):  # Limit to n_caption
        caption, timestamp = proposal['sentence'], proposal['timestamp']
        caption = '{}'.format(caption)
        # to print the captions in center
        text_bbox = font.getbbox(caption)
        text_width = text_bbox[2] - text_bbox[0]
        ptText = ((w - text_width) // 2, h - text_height * (n_caption - i))
        img = paint_text(img, caption, font, ptText, color=(255, 255, 255), bg_color=(0, 0, 0))
        
    return img, cur_time, img_fps

def vid_show(vid_path, captions, save_mp4, save_mp4_path, output_language='en'):
    start_time = time.time()
    cur_time = time.time()
    video = cv2.VideoCapture(vid_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    img_fps = fps
    n = 0

    if save_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videoWriter = cv2.VideoWriter(save_mp4_path, fourcc, fps, (1280, 720))

    if output_language != 'en':
        for proposal in captions:
            caption = GoogleTranslator(source='en', target=output_language).translate(proposal['sentence'])
            proposal['sentence'] = caption

    for i, proposal in enumerate(captions):
        proposal['original_id'] = i
    captions = sorted(captions, key=lambda p: p['timestamp'])

    for frame_id in tqdm(range(int(frame_count))):
        ret, frame = video.read()
        if n >= int(fps / img_fps) or save_mp4:
            n = 0
        else:
            n += 1
            continue
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        frame_time = frame_id / fps
        frame_captions = get_frame_caption(frame_time, captions)
        title = ''.format(frame_time, duration)

        frame, cur_time, img_fps = processImg(frame, cur_time, title, frame_captions, output_language=output_language, n_caption=1)

        if not save_mp4:
            plt.axis('off')
            plt.imshow(frame[:, :, ::-1])
            plt.show()
        if save_mp4:
            videoWriter.write(frame)
    if save_mp4:
        videoWriter.release()
        print('Output video saved at {}, process time: {} s'.format(save_mp4_path, cur_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_language', type=str, default='en')
    parser.add_argument('--output_mp4_folder', type=str, default=None)
    parser.add_argument('--input_mp4_folder', type=str, required=True)
    parser.add_argument('--dvc_file', type=str, required=True)
    opt = parser.parse_args()
    d = json.load(open(opt.dvc_file))['results']
    for vid, dense_captions in d.items():
        if opt.output_mp4_folder is None:
            opt.output_mp4_folder = opt.input_mp4_folder + '_output'
        if not os.path.exists(opt.output_mp4_folder):
            os.mkdir(opt.output_mp4_folder)
        output_mp4_path = os.path.join(opt.output_mp4_folder, vid + '.mp4')

        input_mp4_path = os.path.join(opt.input_mp4_folder, vid + '.mp4')
        print('Processing video: {} --> Output: {}'.format(input_mp4_path, output_mp4_path))
        if not os.path.exists(input_mp4_path):
            print('Video {} does not exist, skipping.')
            continue
        vid_show(input_mp4_path, dense_captions, save_mp4=True, save_mp4_path=output_mp4_path, output_language=opt.output_language)
