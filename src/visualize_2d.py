import numpy as np
from utils import read_data_from_file
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
        
def animate_2d_video(frames_list, fixed_data):
    x_data_np, y_true_np = fixed_data
    # Create a figure and axis
    fig, ax = plt.subplots()

  
    # Get the number of frames
    num_frames = len(frames_list)

    # Function to update the plot for each animation frame
    def update(frame):
        ax.clear()  # Clear the previous plot
        ax.scatter(x_data_np, y_true_np, label='True', color='blue', s=3)  # Scatter plot for true data
    
        # Plot current predicted data
        y_pred = y_pred_list[frame]
        ax.scatter(x_data_np, y_pred, label='Predicted (Frame {})'.format(frame), color='red', s=3)
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Frame {}'.format(frame))
        # ax.legend()
        
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, blit=False)

    # Create a writer to save the animation as a GIF file
    writer = PillowWriter(fps=30)  # Adjust the frames per second as needed
    ani.save("visualizations/prediction_animation.gif", writer=writer)

    # Show the animation (if needed)
    plt.show()
         

if __name__ == "__main__":

    x_data_np,y_true_np,y_pred_list = read_data_from_file('cache_data/cache_arrays.pkl')

    frames_list = []

    
    for  frame in (y_pred_list):
        frames_list.append(np.squeeze(frame))
        

    fixed_data = np.squeeze(x_data_np), np.squeeze(y_true_np)
    animate_2d_video(frames_list, fixed_data)
 
