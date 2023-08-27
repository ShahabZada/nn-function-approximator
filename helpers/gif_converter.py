import moviepy.editor as mp

# clip = mp.VideoFileClip("visualizations\prediction_animation_sin.gif")
# clip.write_videofile("visualizations\prediction_animation_sin.mp4")


import moviepy.editor as mp

gif_path = r"visualizations\prediction_animation_sin.gif"
mp4_output_path = r"visualizations\prediction_animation_sin_high_res.mp4"

clip = mp.VideoFileClip(gif_path)

# Set the desired frame rate (fps) and bitrate
fps = 30  # Adjust as needed
bitrate = "10000k"  # Adjust as needed

# Write the high-resolution MP4 video
clip.write_videofile(mp4_output_path, fps=fps, bitrate=bitrate)
