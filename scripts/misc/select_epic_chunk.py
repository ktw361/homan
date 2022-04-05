from pathlib import Path
import epylab as pylab

class EpicPicker(object):
    def __init__(self,
                annot_df='/home/skynet/Zhifan/data/epic/EPIC_100_train.csv', 
                root='~/data/epic_rgb_frames/'):
        self.annot_df = pd.read_csv(annot_df)
        self.root = Path(root)
        
    def get_chunk(self, entry):
        entry = self.annot_df
        st, ed = entry.start_frame, entry.stop_frame
        pid = entry.participand_id
        vid = entry.video_id
        frame_root = self.root/pid/vid
        
        
import matplotlib.animation as animation
import numpy as np
from pylab import *


dpi = 100

def ani_frame():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(rand(300,300),cmap='gray',interpolation='nearest')
    im.set_clim([0,1])
    fig.set_size_inches([5,5])


    tight_layout()


    def update_img(n):
        tmp = rand(300,300)
        im.set_data(tmp)
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,300,interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save('demo.mp4',writer=writer,dpi=dpi)
    return ani


if __name__ == '__main__':
    ani_frame()