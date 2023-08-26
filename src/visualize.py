import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
import argparse
import os
from utils import read_data_from_file, make_dir
import matplotlib.pyplot as plt
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

def frame_args(duration): 
     return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
            }


def animate_2d(frames_list, frames_numbers, fixed_data, path,save_figure, fig_name):

    x_data_np, y_true_np = fixed_data
        
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=('Top view,(xy)', 'Side view(xz)'),
        horizontal_spacing=0.091
    )
    fig.add_trace(go.Scatter(x=[], y=[],mode="markers",name='',), row=1, col=1)   # trace of index 1
    fig.add_trace(go.Scatter(x=[], y=[],mode="markers",name='',), row=1, col=1)
    
    frames =[go.Frame(data=[
                            go.Scatter( x=x_data_np,
                                        y=frames_list[k],
                                        mode='markers',
                                        marker=dict( #size = list(abs(frames_list[k]['v'].values)), 
                                                    # color= frames[k]['m'].values,
                                                    # colorscale='Viridis',
                                                    # cmin = -1,
                                                    # cmax = 3,
                                                    size = 5
                                                    )
                                        ),
                            go.Scatter( x=x_data_np,
                                        y=y_true_np,
                                        mode='markers',
                                        marker=dict( #size = list(abs(frames_list[k]['v'].values)), 
                                                    # color= frames[k]['m'].values,
                                                    # colorscale='Viridis',
                                                    # cmin = -1,
                                                    # cmax = 3,
                                                    size = 5
                                                    )
                                        ),
                                   
                                        ],

                            name=f'frame{k}',
                    traces=[0,1]) for k in range(len(frames_list))]   

    fig.frames=frames
    sliders = [{"currentvalue":{"prefix": "Time: "},"pad": {"b": 10, "t": 60},"len": 0.9,"x": 0.1,"y": 0,"steps": [{"args": [[f.name], frame_args(1)],"label": frames_numbers[k],"method": "animate",} for k, f in enumerate(fig.frames)]}]
    fig.update_layout(updatemenus = [{"buttons":[{"args": [None, frame_args(1)],"label": "Play", "method": "animate",},{"args": [[None], frame_args(1)],"label": "Pause", "method": "animate",}],"direction": "left","pad": {"r": 10, "t": 70},"type": "buttons","x": 0.1,"y": 0,}],sliders=sliders)
    fig.update_layout(sliders=sliders)

    # fig.update_layout(xaxis1_range=[room_info['x_min']-0.2, room_info['x_max']+0.2], xaxis1_autorange=False)                              
    # fig.update_layout(yaxis1_range=[room_info['y_min']-0.3, room_info['y_max']+0.2], yaxis1_autorange=False)


    fig.update_xaxes(title_text="x", row = 1, col = 1)
    fig.update_yaxes(title_text="y", row = 1, col = 1)

 
    if save_figure ==1:
        make_dir(path +'figures/')
        fig.write_html(path +'figures/'+ fig_name+'.html')
    
    fig.show()

def animate_2d_video(frames_list, frames_numbers, fixed_data, path,save_figure, fig_name):

    x_data_np, y_true_np = fixed_data
    frames = []
    for i in range(len(frames_list)):
        fig = go.Figure(data=go.Scatter(x=x_data_np, y=frames_list[i], mode='markers', marker=dict(color='blue')))
        frames.append(fig.to_image(format="png"))

    # Save the frames as a video using imageio
    video_path = "sine_wave_video.mp4"
    imageio.mimsave(video_path, frames, fps=30)  # Adjust the frames per second as needed
    print(f"Video saved as {video_path}")    
    
    if save_figure ==1:
        make_dir(path +'figures/')
        
def animate_2d_video2(frames_list, frames_numbers, fixed_data, path,save_figure, fig_name):
    x_data_np, y_true_np = fixed_data
    # Create a figure and axis
    fig, ax = plt.subplots()

  
    # Get the number of frames
    num_frames = len(frames_list)

    # Function to update the plot for each animation frame
    def update(frame):
        ax.clear()  # Clear the previous plot
        ax.plot(x_data_np, y_true_np, label='True')  # Plot true data
        
        # Plot current predicted data
        y_pred = y_pred_list[frame]
        ax.plot(x_data_np, y_pred, label='Predicted (Frame {})'.format(frame))
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Frame {}'.format(frame))
        ax.legend()
        
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, blit=False)

    # Create a writer to save the animation as a GIF file
    writer = PillowWriter(fps=10)  # Adjust the frames per second as needed
    ani.save("prediction_animation.gif", writer=writer)

    # Show the animation (if needed)
    plt.show()
         

if __name__ == "__main__":

    x_data_np,y_true_np,y_pred_list = read_data_from_file('cache_data/cache_arrays.pkl')

    save_figure = 0
    fig_name = ''
    frames_list = []
    frames_numbers = []
   
  

 
    
    for frame_number, frame in enumerate(y_pred_list):
        
     
        
        frames_list.append(np.squeeze(frame))
        frames_numbers.append(frame_number)

    fixed_data = np.squeeze(x_data_np), np.squeeze(y_true_np)
    animate_2d_video2(frames_list, frames_numbers, fixed_data, 'visualizations', save_figure, fig_name)
 
