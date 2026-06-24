from moviepy import VideoFileClip, AudioFileClip


video = VideoFileClip("videos/connect_wars_gameplay_video_mp4.mp4")
ses = AudioFileClip("videos/connect_wars_gameplay_video_ogg.ogg")


sesli_video = video.with_audio(ses)


sesli_video.write_videofile("Connect4_Gameplay_Final.mp4", codec="libx264", audio_codec="aac")