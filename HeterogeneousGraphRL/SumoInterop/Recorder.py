import traci
import tempfile
import os.path
import numpy as np
from typing import Optional


class Recorder:
    def __init__(self, tracked_vehicle_id: str, width: int = -1, height: int = -1, name: str = 'video'):
        self.tracked_vehicle = tracked_vehicle_id
        self.frame_no = 0
        self.tracking_view = None
        self.width = width
        self.height = height
        self.tmpdir = None
        self.name = name
        self.__messages = {}
        self.__last_shot = None

    def prestep(self):
        if self.tracking_view is None:
            views = traci.gui.getIDList()

            if len(views) == 0:
                raise ValueError("TraCI has no views")

            traci.gui.setSchema(views[0], 'real world')

            traci.gui.setBoundary(views[0], -50, -50, 50, 50)
            traci.gui.setZoom(views[0], 350)
            traci.gui.trackVehicle(views[0], self.tracked_vehicle)
            self.tracking_view = views[0]
            self.tmpdir = tempfile.mkdtemp(prefix='sumo_recorder')

        self.__last_shot = '{:04d}.png'.format(self.frame_no)
        traci.gui.screenshot(self.tracking_view, os.path.join(self.tmpdir, self.__last_shot),
                             width=self.width, height=self.height)
        self.frame_no += 1

    def poststep(self, msg):
        self.__messages[self.__last_shot] = msg

    def close(self):
        import shutil

        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)
            self.tmpdir = None

    def make_ndarray(self, file_paths):
        from PIL import Image, ImageDraw

        imgs = []

        for path in file_paths:
            with open(path, 'rb') as img_fp:
                img = Image.open(img_fp)
                file = os.path.basename(path)

                if file in self.__messages:
                    draw = ImageDraw.Draw(img)
                    draw.text((5, 5), self.__messages[file])

                imgs.append(np.array(img.convert('RGB')))

        return np.stack(imgs, axis=0)

    def get_result(self, output_format='ndarray') -> Optional[np.ndarray]:
        """
        Create a numpy array from the recording.
        :return: Numpy array or none if no images
        """
        from logging import debug

        # Flush last shot
        traci.simulationStep()

        # Skip first frame which is sometimes garbage
        os.unlink(os.path.join(self.tmpdir, '{:04d}.png'.format(0)))
        paths = [os.path.join(self.tmpdir, '{:04d}.png'.format(i)) for i in range(1, self.frame_no)]

        debug("Recorder: Read {} images", len(paths))

        # If empty after clipping, return None
        if len(paths) <= 1:
            return None

        if output_format == 'ndarray':
            return self.make_ndarray(paths)
        elif output_format == 'ffmpeg':
            return self.tmpdir + '/%04d.png'
        else:
            raise ValueError(output_format)
