import numpy as np
import time
from multiprocessing import Process, Queue
from pyrep import PyRep
from pyrep.objects import Shape
from pyrep.objects.vision_sensor import VisionSensor


class PyrepRenderer:
    def __init__(self, cfg):
        """

        :param cfg: (dotmap)
        """
        self.scene_name = cfg.render_scene_name
        self.food_name_list = cfg.obj_name_list[-cfg.num_dynamic_obj:]
        self.trj_queue, self.term_queue, self.img_queue = Queue(), Queue(), Queue()

        # Initialize CoppeliaSim environment
        self.p = Process(target=self.trj_to_img, args=(self.img_queue, self.trj_queue, self.term_queue, cfg.debug))
        print(f'Simulator process is starting now...')
        self.p.start()
        # To avoid delay at the start of the simulator
        self.trj_queue.put(np.tile(np.array([0, 0, 0, 0, 0, 0, 1]), (1, cfg.num_dynamic_obj)))
        self.img_queue.get()
        print('Simulator processes have been started!!!')

    def terminate(self):
        self.term_queue.put(True)
        self.p.join()

    def gen_images(self, trj):
        self.trj_queue.put(trj)
        img_series = self.img_queue.get()
        return img_series.copy()

    def trj_to_img(self, img_queue, trj_queue, term_queue, debug_fl):
        pr = PyRep()
        pr.launch(self.scene_name, headless=not debug_fl)

        food_dict = {name: Shape(name) for name in self.food_name_list}
        cam = VisionSensor('vs')

        term_sig = False
        pr.start()

        [food_dict[name].set_dynamic(False) for name in self.food_name_list]

        while True:
            while True:
                time.sleep(0.1)
                if not trj_queue.empty():
                    break
                if not term_queue.empty():
                    term_sig = term_queue.get()
                    break

            if term_sig:
                print('Renderer is terminated')
                break
            trj_list = trj_queue.get()
            for trj in trj_list:
                [food_dict[name].set_pose(trj[j * 7:(j + 1) * 7]) for j, name in enumerate(self.food_name_list)]
                pr.step()
                img_queue.put(cam.capture_rgb())


if __name__ == '__main__':
    import argparse
    import sys
    import matplotlib.pyplot as plt
    from config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='deep-fried_shrimp')
    parser = parser.parse_args()
    parser.pref_weight = [0.2, 0.3]

    cfg = load_config(parser)

    try:
        renderer = PyrepRenderer(cfg)
        img = renderer.gen_images(np.tile(np.array([0, 0, 0.2, 0, 0, 0, 1])[None, :], (1, 2)))
        plt.imshow(img)
        plt.savefig(f'{cfg.logdir}/images/test.png')
        renderer.terminate()

    except Exception as e:
        print(e)
        sys.exit()
