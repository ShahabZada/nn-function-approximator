import numpy as np
from utils import read_data_from_file
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
        
def animate_2d_video(frames_list):
  
    # Create a figure and axis
    fig, ax = plt.subplots()

  
    # Get the number of frames
    num_frames = len(frames_list)

    # Function to update the plot for each animation frame
    def update(frame):
        ax.clear()  # Clear the previous plot
        if frame%100 ==0:
            print(frame)
        y_pred = y_pred_list[frame]
        # print(frame.shape)
        ax.imshow(y_pred, cmap='CMRmap')
    
        
        
       
        
        ax.axis('off')
        ax.set_title('Frame {}'.format(frame))
      
        
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, blit=False)

    # Create a writer to save the animation as a GIF file
    writer = PillowWriter(fps=30)  # Adjust the frames per second as needed
    ani.save("visualizations/prediction_animation_img2.gif", writer=writer)

    # Show the animation (if needed)
    plt.show()
         

if __name__ == "__main__":

    y_pred_list = read_data_from_file('cache_data/cache_arrays.pkl')


    
    animate_2d_video(y_pred_list)
 
