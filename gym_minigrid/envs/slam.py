from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import csv
import glob
import scipy.ndimage.morphology
from dc2g.util import get_traversable_colors, get_goal_colors, find_traversable_inds, find_goal_inds, inflate, wrap, get_colormap_dict

class SLAMEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    Agent observes image of *entire* world, but only can "see" cells it has
    seen at some point ==> we maintain which cells have been seen.
    """

    def __init__(self, size=8):
        MiniGridEnv.__init__(
            self,
            grid_size=size,
            max_steps=500,
            # max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            reset_on_init=False,
            remember_seen_cells=True
        )

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.grid_size, self.grid_size, 3),
            dtype='uint8'
        )
        position_observation_space = spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype='uint8')
        theta_observation_space = spaces.Box(low=0, high=3, shape=(1,), dtype='uint8')
        goal_observation_space = spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype='uint8')
        # self.observation_space = spaces.Tuple([position_observation_space, goal_observation_space])
        # self.observation_space = image_observation_space
        # self.observation_space = spaces.Box(low=0,high=self.grid_size, shape=(6,), dtype='uint8')
        self.observation_space = spaces.Dict({
            'semantic_gridmap': image_observation_space,
            'pos': position_observation_space,
            'theta_ind': theta_observation_space,
            'goal_pos': goal_observation_space,
        })

        # Actions are discrete integer values: only allow forward/left/right/stop for now (no picking up/done)
        self.action_space = spaces.Discrete(3)

        # self.world_id = 30
        # self.world_image_filename = '/home/mfe/code/dc2g/training_data/driveways_icra19/full_semantic/test/world' + str(self.world_id).zfill(3) + '.png'
        # self.world_image_filename = '/home/mfe/code/dc2g/training_data/driveways_icra19/full_semantic/test/world' + str(self.world_id).zfill(3) + '.png'

        # Probably don't need to call self.reset(), since the process running
        # the env will need to capture the 1st obs anyway.
        reset_on_init = False
        if reset_on_init:
            self.reset()

    def set_difficulty_level(self, difficulty_level):
        dataset = "driveways_bing_iros19"
        image_type = "full_semantic"
        image_filename = "/home/mfe/code/dc2g_new/data/datasets/{dataset}/{image_type}/{mode}/{world_id}{goal}.png"
        worlds = {'training': {'mode': 'train', 'worlds': "world*"},
                  'same_neighborhood': {'mode': 'test', 'worlds': "worldn001*"},
                  'new_neighborhood': {'mode': 'test', 'worlds': "worldn002*"},
                  'urban': {'mode': 'test', 'worlds': "worldn004*"},
                  'test_scenario': {'mode': 'test', 'worlds': 'worldn002m002h011'}}
        if difficulty_level == 'easy':
            if dataset == "driveways_bing_iros19":
                category = 'training'
            # self.world_id = self._rand_int(25, 31)
        elif difficulty_level == 'medium':
            if dataset == "driveways_bing_iros19":
                category = 'same_neighborhood'
            # self.world_id = self._rand_int(31, 34)
        elif difficulty_level == 'hard':
            if dataset == "driveways_bing_iros19":
                category = 'new_neighborhood'
            # self.world_id = self._rand_int(34, 41)
        elif difficulty_level == 'very_hard':
            if dataset == "driveways_bing_iros19":
                category = 'urban'
        elif difficulty_level == 'test_scenario':
            if dataset == "driveways_bing_iros19":
                category = 'test_scenario'

        world_id_filenames = glob.glob(image_filename.format(dataset=dataset, image_type="full_semantic", goal="", world_id=worlds[category]['worlds'], mode=worlds[category]['mode']))

        self.world_id = None
        while self.world_id in [None, "worldn001m001h002", "worldn002m001h003"]:
            self.world_image_filename = self._rand_choice(world_id_filenames)
            self.world_id = self.world_image_filename.split('/')[-1].split('.')[0]

        self.difficulty_level = difficulty_level
        
        # world_id = "n001m003h000"
        # self.world_image_filename = '/home/mfe/code/dc2g/training_data/driveways_bing_iros19/full_semantic/test/world' + world_id + '.png'
        # self.world_image_filename = '/home/mfe/code/dc2g/training_data/driveways_icra19/full_semantic/test/world' + str(self.world_id).zfill(3) + '.png'

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height, remember_seen_cells=self.remember_seen_cells)

        self.orig_world_array = resize(plt.imread(self.world_image_filename), (height,width,3), order=0)
        assert height == self.orig_world_array.shape[0], "height of loaded world doesn't match gridworld size"
        assert width == self.orig_world_array.shape[1], "width of loaded world doesn't match gridworld size"

        # Place a goal square
        dataset = "driveways_bing_iros19"
        goal_color_dict = get_colormap_dict(dataset)

        gym_terrain_class_name_dict = {
            # labels_colors name: regex for airsim object
            "road": Road,
            "driveway": Driveway,
            "path": Path,
            "sidewalk": Sidewalk,
            "front_door": Goal,
            "house": House,
            "grass": Grass,
            "car": Car,
            # "__ignore__": "(?:fence|garden_chair|street_sign|stop_sign)[\w]*",
        }

        self.world_array = self.orig_world_array.copy()
        for key in gym_terrain_class_name_dict:
            try:
                color = goal_color_dict[key]
            except:
                print("{key}'s color isn't defined in the labels_colors.csv file for this dataset.".format(key=key))
                continue
            obj_name = gym_terrain_class_name_dict[key]
            goal_array, inds, goal_inds_arr = find_goal_inds(self.world_array, color, "object")
            for i in range(len(inds[0])):
                self.grid.set(inds[1][i], inds[0][i], obj_name(255.*np.array(color)))
            if obj_name == Path:
                struct2 = scipy.ndimage.generate_binary_structure(2, 2)
                goal_array = scipy.ndimage.morphology.binary_dilation(goal_array, struct2).astype(goal_array.dtype)
                self.world_array[goal_array == 1] = color
                inflated_inds = np.where(goal_array == 1)
                for i in range(len(inflated_inds[0])):
                    self.grid.set(inflated_inds[1][i], inflated_inds[0][i], obj_name(255.*np.array(color)))

        key = "front_door"
        color = goal_color_dict[key]
        obj_name = gym_terrain_class_name_dict[key]
        goal_array, inds, goal_inds_arr = find_goal_inds(self.orig_world_array, color, "object")
        self.world_array[goal_array == 1] = color
        # for i in range(len(inds[0])):
        i = 0
        self.grid.set(inds[1][i], inds[0][i], obj_name(255.*np.array(color)))
        self.goal_pos = [inds[1][i], inds[0][i]]

        # self.start_pos = np.array([inds[1][i]+1, inds[0][i]+1])
        # self.start_dir = 0


        self.place_agent(max_tries=500, reject_fn=reject_untraversable)
        
        # self.place_agent(top=[9, 4], size=[5, 5], max_tries=500, reject_fn=reject_untraversable)
        # print("setting agent at {}, {}".format(self.start_pos, self.start_dir))
        # self.start_pos = (9, 10)
        # self.start_pos = (np.random.randint(7, 10), np.random.randint(5, 10))
        # self.start_pos = (np.random.randint(2,15), np.random.randint(5, 15))
        # self.start_dir = 0
        # self.grid.wall_rect(0, 0, width, height)

        self.mission = "get to the yellow goal square"

    def gen_obs_grid_square_fov(self): # Square observation grid
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, AGENT_VIEW_SIZE, AGENT_VIEW_SIZE)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(AGENT_VIEW_SIZE // 2 , AGENT_VIEW_SIZE - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            agent_i, agent_j = agent_pos
            grid.set(agent_i, agent_j, self.carrying)
        else:
            agent_i, agent_j = agent_pos
            grid.set(agent_i, agent_j, None)

        return grid, vis_mask

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        self.camera_fov = np.pi/2 # full FOV in radians
        camera_range_meters = 10. # range of sensor horizon in meters
        self.world_size_m = np.array([100., 100.]) # meters of true world
        self.grid_resolution_array = self.world_size_m / np.array([self.grid.width, self.grid.height]) # meters/gridcell
        self.grid_resolution_array = self.world_size_m / np.array([self.grid.width, self.grid.height]) # meters/gridcell
        self.camera_range_x = int(camera_range_meters / self.grid_resolution_array[0])  # range of sensor horizon in cells
        self.camera_range_y = int(camera_range_meters / self.grid_resolution_array[1]) # range of sensor horizon in cells

        self.grid_resolution = 1

        grid_inds = np.indices((self.grid.height,self.grid.width))
        grid_array = np.dstack([grid_inds[1], grid_inds[0]])
        rel_pos = self.agent_pos.astype(np.float32) - grid_array
        ellipse_r = rel_pos**2 / np.array([self.camera_range_x, self.camera_range_y])**2
        r_inds = np.where(np.sum(ellipse_r, axis=2) <= 1)
        angle_offset = wrap(np.arctan2(rel_pos[:,:,1], -rel_pos[:,:,0]) + wrap(np.pi * self.agent_dir / 2.0))
        angle_offset_inds = np.where(abs(angle_offset) < (self.camera_fov/2))
        r_inds_arr = np.dstack([r_inds[1], r_inds[0]])[0]
        angle_offset_inds_arr = np.dstack([angle_offset_inds[1], angle_offset_inds[0]])[0]
        observable_inds_arr = np.array([x for x in set(tuple(x) for x in r_inds_arr) & set(tuple(x) for x in angle_offset_inds_arr)])

        for i in range(observable_inds_arr.shape[0]):
            cell = self.grid.get(observable_inds_arr[i, 0], observable_inds_arr[i, 1])
            if cell:
                cell.has_been_seen = True
        self.grid.get(*self.agent_pos).has_been_seen = True

        return None, None

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        # grid: the small region the agent can currently see
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = np.transpose(self.grid.encode(), axes=(1, 0, 2)) / 255.0

        # print(image[:,:,0])

        # plt.imshow(image, interpolation='none', cmap=plt.cm.gray)
        # plt.pause(2)

        # print(image[:,:,1])
        # image = grid.encode()

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        # obs = [self.agent_pos[0], self.agent_pos[1], self.agent_dir, self.goal_pos[0], self.goal_pos[1], self.max_steps - self.step_count]
        # obs = image
        obs = {
            'semantic_gridmap': image,
            'pos': self.agent_pos,
            # 'goal_pos': self.goal_pos,
            'theta': wrap(np.pi * self.agent_dir / 2.0),
            'theta_ind': self.agent_dir,
            'direction': self.agent_dir,
            'mission': self.mission
        }

        return obs

    def to_grid(self, x, y):
        return x, y
        # """
        # Convert continuous coordinate to grid location
        # """
        # # print("x,y: ({},{})".format(x,y))
        # # print("self.lower_grid_x_min: {}".format(self.lower_grid_x_min))
        # # print("self.lower_grid_y_min: {}".format(self.lower_grid_y_min))
        # # print("self.grid_resolution: {}".format(self.grid_resolution))
        # gx = np.floor((x - self.lower_grid_x_min) / self.grid_resolution).astype(int)
        # gy = np.floor((y - self.lower_grid_y_min) / self.grid_resolution).astype(int)
        # return gx, gy
        # # print("gx,gy: ({},{})".format(gx,gy))
        # # return int(gx), int(gy)

    def to_coor(self, x, y):
        return x, y
        # """
        # Convert grid location to continuous coordinate
        # """
        # wx = x * self.grid_resolution + self.lower_grid_x_min
        # wy = y * self.grid_resolution + self.lower_grid_y_min
        # return wx, wy

    def rescale(self,x1,y1,x2,y2,n_row=None):
        return x1, y1, x2, y2
        # """
        # convert the continuous rectangle region in the SUNCG dataset to the grid region in the house
        # """
        # gx1, gy1 = self.to_grid(x1, y1)
        # gx2, gy2 = self.to_grid(x2, y2)
        # return gx1, gy1, gx2, gy2
        # # if n_row is None: n_row = self.n_row
        # # tiny = 1e-9
        # # tx1 = np.floor((x1 - self.L_lo) / self.L_det * n_row+tiny)
        # # ty1 = np.floor((y1 - self.L_lo) / self.L_det * n_row+tiny)
        # # tx2 = np.floor((x2 - self.L_lo) / self.L_det * n_row+tiny)
        # # ty2 = np.floor((y2 - self.L_lo) / self.L_det * n_row+tiny)
        # # return int(tx1),int(ty1),int(tx2),int(ty2)

    def next_coords(self, start_x, start_y, start_theta_ind):
        # action_dict = { 0: (1, 0),
        #                 1: (0, np.pi/2),
        #                 2: (0, -np.pi/2)}
        action_dict = { 0: (1, 0),
                        1: (0, -np.pi/2),
                        2: (0, np.pi/2)}
        start_theta = theta_ind_to_theta(start_theta_ind)
        num_actions = len(action_dict.keys())
        state_dim = 3 # (x,y,theta)
        next_states = np.empty((num_actions, state_dim))
        actions = [None for i in range(num_actions)]
        for i in range(num_actions):
            action = action_dict.keys()[i]
            actions[i] = action
            cmd_vel = action_dict[action]

            # Compute displacement
            num_steps = 10
            dt = 0.1 * num_steps
            dx = dt * cmd_vel[0]
            dy = 0
            dtheta = dt * cmd_vel[1]

            x = start_x + dx * np.cos(start_theta) - dy * np.sin(start_theta)
            y = start_y + dx * np.sin(start_theta) + dy * np.cos(start_theta)
            theta = start_theta + dtheta

            next_states[i,0] = x
            next_states[i,1] = y
            next_states[i,2] = theta_to_theta_ind(theta)
        return next_states, actions

# keep angle between [-pi, pi]
def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def theta_to_theta_ind(yaw):
    return int(round(yaw / (np.pi/2))) % 4

def theta_ind_to_theta(theta_ind):
    return wrap(theta_ind * np.pi/2)

def reject_untraversable(env, pos):
    pi, pj = pos
    # return env.grid.get(pi, pj).type not in ['driveway']
    return env.grid.get(pi, pj).type not in ['road']
    # return env.grid.get(pi, pj).type not in ['path']

class SLAMEnv6x6(SLAMEnv):
    def __init__(self):
        SLAMEnv.__init__(self, size=6)

class SLAMEnv16x16(SLAMEnv):
    def __init__(self):
        SLAMEnv.__init__(self, size=16)

class SLAMEnv32x32(SLAMEnv):
    def __init__(self):
        SLAMEnv.__init__(self, size=50)

register(
    id='MiniGrid-EmptySLAM-6x6-v0',
    entry_point='gym_minigrid.envs:SLAMEnv6x6'
)

register(
    id='MiniGrid-EmptySLAM-8x8-v0',
    entry_point='gym_minigrid.envs:SLAMEnv'
)

register(
    id='MiniGrid-EmptySLAM-16x16-v0',
    entry_point='gym_minigrid.envs:SLAMEnv16x16'
)

register(
    id='MiniGrid-EmptySLAM-32x32-v0',
    entry_point='gym_minigrid.envs:SLAMEnv32x32'
)
