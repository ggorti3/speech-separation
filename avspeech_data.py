import youtube_dl
import subprocess
import skvideo.io
import numpy as np
import face_recognition
import os
import os.path
import shutil
import time


def get_face_encoding_streams(video_frames, x0, y0):
    """
    given video frames array, return face encoding stream (we assume only one face per video)
    """
    ymax = video_frames.shape[1]
    xmax = video_frames.shape[2]

    t1 = time.time()
    detections = face_recognition.batch_face_locations(list(video_frames), batch_size=20, number_of_times_to_upsample=1)
    t2 = time.time()
    print("Time to get detections: {}".format(t2 - t1))

    encoding_stream = []
    for i, frame in enumerate(video_frames):
        # get the detection closest to x0, y0
        best_detection = None
        min_dist = float("inf")
        for d in detections[i]:
            x = d[1] - d[3]
            y = d[2] - d[0]
            x = x / xmax
            y = y / ymax
            dist = (x - x0)**2 + (y - y0)**2
            if dist < min_dist:
                best_detection = d
        frame_detection = []
        if best_detection is not None:
            frame_detection.append(best_detection)

        # retrieve encoding at given detection location
        encodings = face_recognition.face_encodings(
            frame, 
            known_face_locations=frame_detection,
            model="large"
        )

        if len(encodings) == 0:
            encoding_stream.append(np.zeros((128,)))
        else:
            encoding_stream.append(encodings[0])

    return encoding_stream



def generate_dataset(data_path, parent_save_path):
    """
    create dataset of face streams with raw sound audio
    """
    FPS = 25
    dummy_path = os.path.join(parent_save_path, "dummy.mp4")

    with open(data_path, "r") as f:
        # dont load all lines into memory
        line = f.readline()
        while line:
            # retrieve url, start time, end time, x0, y0
            line = line.strip()
            chunks = line.split(",")
            url = chunks[0]
            start = float(chunks[1])
            end = float(chunks[2])
            x0 = float(chunks[3])
            y0 = float(chunks[4])

            # retrieve true video url
            with youtube_dl.YoutubeDL({'format': 'best'}) as ydl:
                try:
                    result = ydl.extract_info(url, download=False)
                except:
                    print("Failed to load video information")
                    line = f.readline()
                    continue

            true_urls = [f['url'] for f in result['formats'] if f['acodec'] != 'none' and f['vcodec'] != 'none']
            formats = [f for f in result['formats'] if f['acodec'] != 'none' and f['vcodec'] != 'none']
            true_url = true_urls[-1]

            save_path = os.path.join(parent_save_path, "{}".format(url))
            if not os.path.exists(save_path):
                os.mkdir(save_path)

                # handle failure to process video
                try:
                    # download video clip into dummy file using ffmpeg
                    subprocess.run('ffmpeg -nostdin -ss {} -to {} -i "{}" -filter:v fps={} {}'.format(start, end, true_url, FPS, dummy_path), shell=True)
                    
                    # get video frames, process with face_recognition, save streams of face encodings as individual csv files
                    video_frames = skvideo.io.vread(dummy_path)
                    stream = get_face_encoding_streams(video_frames, x0, y0)
                    stream_path = os.path.join(save_path, "encoding_stream.csv")
                    np.savetxt(stream_path, stream, delimiter=",")

                    # save audio as seperate .wav file
                    audio_path = os.path.join(save_path, "audio.wav")
                    subprocess.run("ffmpeg -nostdin -i {} -vn {}".format(dummy_path, audio_path), shell=True)
                except:
                    # delete directory
                    print("failed to process video {}".format(url))
                    shutil.rmtree(save_path)

                # delete dummy file
                if os.path.exists(dummy_path):
                    os.remove(dummy_path)

            line = f.readline()

def clean_data(data_path):
    count = 0
    for root, dirs, _ in os.walk(data_path, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            stream_path = os.path.join(dir_path, "encoding_stream.csv")
            with open(stream_path, "r") as f:
                num_encodings = len(f.readlines())
                if num_encodings < 75:
                    count += 1
                    print("oof")
    print("number of files that are too short: {}".format(count))


if __name__ == "__main__":
    clean_data("data/train_dataset")