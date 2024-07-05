from pdf2image import convert_from_path
import ffmpeg
import subprocess
import os
import json

def probe_file(file_path):
    """ Use ffprobe to get the duration of the media file. """
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(result.stdout)['format']['duration']

def generate_video_from_pdf(pdf_file: str, audio_dir: str, output_video: str, resolution=(1920, 1080)):
    # Convert PDF pages to images
    images = convert_from_path(pdf_file)

    # Temporary directory for storing images
    temp_dir = 'temp_images'
    os.makedirs(temp_dir, exist_ok=True)

    # Prepare video and audio streams
    video_streams = []
    audio_streams = []

    for i, image in enumerate(images):
        image_path = os.path.join(temp_dir, f"slide_{i+1}.jpg")
        image.save(image_path, 'JPEG')

        # Find corresponding audio file
        audio_file = os.path.join(audio_dir, f"{str(i).zfill(3)}.mp3")
        if os.path.exists(audio_file):
            duration = float(probe_file(audio_file))
            video_input = ffmpeg.input(image_path, loop=1, t=duration).filter('scale', resolution[0], resolution[1])
            audio_input = ffmpeg.input(audio_file)
            video_streams.append(video_input)
            audio_streams.append(audio_input)

    # Concatenate video and audio streams separately and then combine
    if video_streams and audio_streams:
        combined_video = ffmpeg.concat(*video_streams, v=1, a=0)
        combined_audio = ffmpeg.concat(*audio_streams, v=0, a=1)
        # ffmpeg.output(combined_video, combined_audio, output_video, vcodec='libx264', acodec='aac').overwrite_output().run()
        ffmpeg.output(combined_video, combined_audio, output_video, vcodec='mpeg4', acodec='aac').overwrite_output().run()
    # Cleanup temporary files
    for item in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, item))
    os.rmdir(temp_dir)

    print("Video generation completed.")
