import youtube_dl
import subprocess
import skvideo.io
import numpy as np
import face_recognition
import datetime
import os
import os.path

from pprint import pprint, pformat


def get_face_encoding_streams(video_frames, similarity_tol=0.6):
    """
    given video frames array, return face encoding stream (we assume only one face per video)
    """
    detections = face_recognition.batch_face_locations(list(video_frames), batch_size=16, number_of_times_to_upsample=1)

    encoding_stream = []
    for i, frame in enumerate(video_frames):
        encodings = face_recognition.face_encodings(
            frame, 
            known_face_locations=detections[i],
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

    # get list of urls and start/stop times
    with open(data_path, "r") as f:
        lines = f.readlines()
    lines = lines[:4]

    vid_infos = []
    for line in lines:
        line = line.strip()
        chunks = line.split(",")
        vid_infos.append((chunks[0], float(chunks[1]), float(chunks[2])))
    
    for vid_info in vid_infos:
        url = "https://www.youtube.com/watch?v={}".format(vid_info[0])
        start = str(datetime.timedelta(seconds=vid_info[1]))
        end = str(datetime.timedelta(seconds=vid_info[2]))

        # find true video url
        with youtube_dl.YoutubeDL({'format': 'best'}) as ydl:
            try:
                result = ydl.extract_info(url, download=False)
            except:
                print("Failed to load video information")
                continue

        true_urls = [f['url'] for f in result['formats'] if f['acodec'] != 'none' and f['vcodec'] != 'none']
        formats = [f for f in result['formats'] if f['acodec'] != 'none' and f['vcodec'] != 'none']
        true_url = true_urls[-1]

        save_path = os.path.join(parent_save_path, "{}".format(vid_info[0]))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

            # download video clip into dummy file using ffmpeg
            subprocess.run('ffmpeg -nostdin -ss {} -to {} -i "{}" -filter:v fps={} {}'.format(start, end, true_url, FPS, dummy_path), shell=True)

            # save video clip information
            info = skvideo.io.ffprobe(dummy_path)
            info_str = pformat(info, indent=4)
            info_path = os.path.join(save_path, "info.txt")
            with open(info_path, 'w') as f:
                f.write(info_str)
            
            # get video frames, process with face_recognition, save streams of face encodings as individual csv files
            video_frames = skvideo.io.vread(dummy_path)
            stream = get_face_encoding_streams(video_frames)
            stream_path = os.path.join(save_path, "encoding_stream.csv")
            np.savetxt(stream_path, stream, delimiter=",")

            # save audio as seperate .wav file
            audio_path = os.path.join(save_path, "audio.wav")
            subprocess.run("ffmpeg -nostdin -i {} -vn {}".format(dummy_path, audio_path), shell=True)

            # delete dummy file
            os.remove(dummy_path)



if __name__ == "__main__":

    generate_dataset(
        data_path="../AVSpeech/avspeech_test.csv",
        parent_save_path="../avspeech_data"
    )

    pass