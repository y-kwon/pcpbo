import numpy as np
import quaternion
from isaacgym import gymapi
from isaacgym import gymtorch
import torch
from math import ceil, sqrt
from simulators.BaseEnv import BaseEnv


class IsaacBaseEnv(BaseEnv):
    def __init__(self, cfg):
        super(IsaacBaseEnv, self).__init__(cfg)
        self.compute_device_id = 0
        self.graphics_device_id = 0
        self.physics_engine = gymapi.SIM_PHYSX
        self.debug_fl = cfg.debug
        self.num_dyn_obj = cfg.num_dynamic_obj
        self.num_envs = cfg.num_path

        # for artificial rater (todo: Normalize the rewards to remove these)
        self.less_unc_th = None
        self.more_unc_th = None

        self.zero_array = np.zeros((self.num_envs, 3), dtype=np.float32)

        self.pos_tens = None
        self.fixed_handle = None
        self.num_handle = None

    @staticmethod
    def __load_sim_params():
        sim_params = gymapi.SimParams()
        sim_params.substeps = 1
        sim_params.physx.solver_type = 1
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        return sim_params

    def init_sim_env(self):
        self.gym = gymapi.acquire_gym()
        sim_params = self.__load_sim_params()
        self.sim = self.gym.create_sim(self.compute_device_id, self.graphics_device_id, self.physics_engine, sim_params)
        self.gym.prepare_sim(self.sim)

        if not self.sim:
            raise Exception("*** Failed to create sim")

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        if self.debug_fl:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if not self.viewer:
                raise Exception("*** Failed to create viewer")
            self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(8, 0, 0.5), gymapi.Vec3(8, 12, 0))
            # subscribe to spacebar event for reset
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

        self.envs, self.handle_dict = self.set_assets()
        self.initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))

    def load_assets(self):
        raise NotImplementedError()

    def set_assets(self):
        env_list = []
        handle_dict = {}

        # set up the env grid
        num_per_row = int(ceil(sqrt(self.num_envs)))
        env_spacing = 0.6
        env_lower = gymapi.Vec3(-env_spacing / 2, -env_spacing / 2, 0.0)
        env_upper = gymapi.Vec3(env_spacing / 2, env_spacing / 2, 0.5)

        assets_dict = self.load_assets()

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            env_list.append(env)
            for assets_name, v in assets_dict.items():
                if 'cam_' in assets_name:
                    handle = self.gym.create_camera_sensor(env, v['props'])
                    self.gym.set_camera_transform(handle, env, v['transform'])
                else:
                    handle = self.gym.create_actor(env, v['asset'], v['pose'], f'assets_name_{i}', i, 0)
                    if 'color' in v.keys():
                        self.gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, v['color'])
                    # self.set_rigid_shape_properties(env, handle)
                if not i:
                    handle_dict[assets_name] = handle

        return env_list, handle_dict

    def step(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def reset(self):
        self.gym.set_sim_rigid_body_states(self.sim, self.initial_state, gymapi.STATE_ALL)

    def terminate(self):
        if self.debug_fl:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def get_clone_tensor(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        return self.pos_tens.clone()

    def swap_fix_free(self, i_food, handle):
        cl_pos_tens = self.get_clone_tensor()
        for i_fix in range(i_food):
            self.pos_tens[handle - 2 * (i_food - i_fix)::self.num_handle], \
            self.pos_tens[self.fixed_handle[i_fix]::self.num_handle] = \
                cl_pos_tens[self.fixed_handle[i_fix]::self.num_handle], \
                cl_pos_tens[handle - 2 * (i_food - i_fix)::self.num_handle]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_tens))

    def place_food(self, cl_pos_tens, handle, ctrl_xy, h_list, quat_zw):

        cl_pos_tens[handle::self.num_handle, :2] = ctrl_xy
        cl_pos_tens[handle::self.num_handle, 2] = h_list
        cl_pos_tens[handle::self.num_handle, 3:5] = 0
        cl_pos_tens[handle::self.num_handle, 5:7] = quat_zw
        cl_pos_tens[handle::self.num_handle, 7:] = 0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(cl_pos_tens))

    def ctrl_to_trj(self, ctrl):
        raise NotImplementedError()

    def clip_ctrl(self, ctrl):
        raise NotImplementedError()

    def sum_cost_fcn(self, trj, weight):
        raise NotImplementedError()

    def gen_near_optimal_ctrl_series(self, weight):
        raise NotImplementedError()

    def step_visual(self):
        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        self.gym.sync_frame_time(self.sim)

        # Get input actions from the self.viewer and handle them appropriately
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "reset" and evt.value > 0:
                self.reset()

        # update the viewer
        if self.debug_fl:
            self.gym.draw_viewer(self.viewer, self.sim, True)


class IsaacShrimpEnv(IsaacBaseEnv):
    def __init__(self, cfg):
        super(IsaacShrimpEnv, self).__init__(cfg)

        self.shrimp_size = np.array([0.12, 0.031])
        self.len_shrimp = self.shrimp_size[0]
        self.dish_height = 0.08
        self.dh = 0.004
        self.h_start = [0.118, 0.2, 0.2]  # start point of dropped height calculation
        self.weight_time = [100, 200, 200]
        self.shrimp_vec = np.array([self.len_shrimp / 2, 0, 0])[None, :, None]

        # for artificial rater
        self.less_unc_th = [20, 50]
        self.more_unc_th = [20, 100]

        self.init_sim_env()
        self.fixed_handle = [self.handle_dict[f'fixed_S{i}'] for i in range(3)]
        self.free_handle = [self.handle_dict[f'S{i}'] for i in range(3)]

        _pos_tens = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.pos_tens = gymtorch.wrap_tensor(_pos_tens)
        self.num_handle = len(self.handle_dict)

    def load_assets(self):
        def fixed_asset_option():
            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = True
            asset_options.fix_base_link = True
            return asset_options

        asset_dict = {}
        asset_root = "simulators/assets/urdf/"

        # load dish
        asset_dict['T'] = {
            'asset': self.gym.load_asset(self.sim, asset_root, 'T.urdf', fixed_asset_option()),
            'pose': gymapi.Transform(p=gymapi.Vec3(0, 0, 0), r=gymapi.Quat(0, 0, 0, 1)),
            'color': gymapi.Vec3(0.95, 0.95, 0.95)}
        # load fixed bowl
        asset_dict['L'] = {
            'asset': self.gym.load_asset(self.sim, asset_root, 'L.urdf', fixed_asset_option()),
            'pose': gymapi.Transform(p=gymapi.Vec3(+7.00e-02, -4e-03, +8.7000e-02), r=gymapi.Quat(0, 0, 0, 1)),
            'color': gymapi.Vec3(0.3, 0.3, 0.3)}
        # load fixed cabbage
        for k, v in {'CA0': 8.5000e-02, 'CA1': .9000e-02}.items():
            asset_dict[k] = {
                'asset': self.gym.load_asset(self.sim, asset_root, f'{k}.urdf', fixed_asset_option()),
                'pose': gymapi.Transform(p=gymapi.Vec3(-1.3000e-02, +3.7000e-02, v), r=gymapi.Quat(0, 0, 0, 1)),
                'color': gymapi.Vec3(0.59, 0.85, 0.32)}

        # load shrimps (fixed and unfixed)
        asset_options = gymapi.AssetOptions()
        asset_options.linear_damping, asset_options.angular_damping = 5, 5
        for i in range(3):
            color = gymapi.Vec3(0.9, 0.5, 0.3) if i else gymapi.Vec3(1.0, 1.0, 0.0)

            asset_dict[f'S{i}'] = {
                'asset': self.gym.load_asset(self.sim, asset_root, 'S.urdf', asset_options),
                'pose': gymapi.Transform(p=gymapi.Vec3(0.2, 0.05 * (i - 1), 0.0225), r=gymapi.Quat(0, 0, 0, 1)),
                'color': color}
            asset_dict[f'fixed_S{i}'] = {
                'asset': self.gym.load_asset(self.sim, asset_root, 'S.urdf', fixed_asset_option()),
                'pose': gymapi.Transform(p=gymapi.Vec3(-0.35, 0.05 * (i - 1), 0.0225), r=gymapi.Quat(0, 0, 0, 1)),
                'color': color}

        return asset_dict

    def sum_cost_fcn(self, trj, weight):

        rs_state = trj.reshape(len(trj), self.num_dyn_obj, 7)

        cost = 0
        tgt_list = [-130 / 180 * np.pi, (weight[0] * 50 - 100) / 180 * np.pi, (weight[1] * 50 - 100) / 180 * np.pi]

        xi_list = []
        for i, tgt in enumerate(tgt_list):
            quat = trj[:, 7 * i + 3:7 * (i + 1)]
            rot_vec = quaternion.as_rotation_vector(quaternion.as_quat_array(np.roll(quat, 1, axis=-1)))
            rot_mac = quaternion.as_rotation_matrix(quaternion.as_quat_array(np.roll(quat, 1, axis=-1)))
            xi_list.append(rot_vec[:, -1].copy())
            cost += 1e3 * (rot_vec[:, -1] - tgt) ** 2
            half_c2e = (rot_mac @ self.shrimp_vec)[:, 2, 0]
            cost += 1e2 * (np.abs(self.dish_height - (rs_state[:, i, 2] + half_c2e)) ** 2
                           + np.abs((self.dish_height + self.len_shrimp * np.sin(np.pi / 3))
                                    - (rs_state[:, i, 2] - half_c2e)) ** 2)

        fl = ((((xi_list[-2] < xi_list[-1]) * (rs_state[:, 1, 0] > rs_state[:, 2, 0]))
               + ((xi_list[-2] > xi_list[-1]) * (rs_state[:, 1, 0] < rs_state[:, 2, 0])))
              * np.abs(xi_list[-2] - xi_list[-1]) > (5 * np.pi / 180)) > 0

        cost[fl] += 1000
        return cost[:, None]

    def ctrl_to_trj(self, ctrl_series):
        ctrl_series = torch.from_numpy(ctrl_series)
        self.reset()
        self.step()
        for i_food, handle in enumerate(self.free_handle):
            h_list = self.h_start[i_food]
            torch_quat = torch.from_numpy(self.rz_to_quat(ctrl_series.cpu().detach().numpy()[:, i_food, -1]))[:, -2:]

            # Calculate the height at which the food is dropped.
            self.swap_fix_free(i_food, handle)
            cl_pos_tens = self.get_clone_tensor()

            contact_env_num = True
            while contact_env_num:
                self.place_food(cl_pos_tens, handle, ctrl_series[:, i_food, :2], h_list, torch_quat)
                self.step()
                if self.debug_fl:
                    self.step_visual()
                self.gym.refresh_actor_root_state_tensor(self.sim)
                contact_idx = self.pos_tens[handle::self.num_handle, 3] == 0
                h_list -= (contact_idx * self.dh)
                contact_env_num = contact_idx.sum()

            self.place_food(cl_pos_tens, handle, ctrl_series[:, i_food, :2], h_list, torch_quat)
            self.swap_fix_free(i_food, handle)

            # For stabilization
            for step in range(self.weight_time[i_food]):
                self.step()
                if self.debug_fl:
                    self.step_visual()

        cl_pos_tens = self.get_clone_tensor()
        idx = np.array(
            [np.arange(handle, len(cl_pos_tens), len(self.handle_dict)) for handle in self.free_handle]).flatten()
        idx.sort()
        trj = cl_pos_tens[idx, :7].reshape(self.num_envs, -1).detach().numpy()

        return trj

    def rz_to_quat(self, rz):
        self.zero_array[:, -1] = rz
        return np.roll(quaternion.as_float_array(quaternion.from_rotation_vector(self.zero_array)), -1, axis=-1)

    def gen_near_optimal_ctrl_series(self, weight):
        if weight.shape != (2,):
            raise NotImplementedError(f'weight shape is {weight.shape} != (2,)')

        r = np.sum((self.shrimp_size / 2) ** 2) ** 0.5
        coord = [[-0.03, 0.006, -130 * np.pi / 180]]  # for S0

        for i, each_weight in enumerate(weight):
            cos_val = 0.3 * np.cos((each_weight * 50 - 100 + 20) / 180 * np.pi)
            sin_val = 0.75 * np.sin((each_weight * 50 - 100 - 10) / 180 * np.pi)
            x = r * cos_val + 0.005 if weight[i] > weight[i - 1] else r * cos_val - 0.03
            y = r * sin_val + 0.035
            xi = (each_weight * 50 - 100) * np.pi / 180
            coord.append([x, y, xi])
        return np.array(coord)[None, ...]

    def clip_ctrl(self, ctrl):
        """
        :param ctrl: [K, num_food, 3]
        :return:
        """

        ctrl[:, 0] = np.array([-3.5e-02, +7.0e-03, -130 / 180 * np.pi])
        ctrl[:, 1:, 0] = np.clip(ctrl[:, 1:, 0], -0.035, 0.035)
        ctrl[:, 1:, 1] = np.clip(ctrl[:, 1:, 1], -0.035, -0.005)
        ctrl[:, 1:, -1] = np.clip(ctrl[:, 1:, -1], -100 / 180 * np.pi, -50 / 180 * np.pi)
        return ctrl
