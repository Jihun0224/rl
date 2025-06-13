from legged_gym.envs.base.manipulated_robot_config import ManipulatedRobotCfg, ManipulatedRobotCfgPPO
# READY_POSE = [[0.0, 0.632, -1.322, 0.692, 0.0, 0.0], [-0.477, -0.185, -0.128, -2.261, -0.182, 1.702, -0.0], [-0.477, 0.185, 0.128, -2.261, 0.182, 1.702, 0.0]]
# [-0.0, 0.632, -1.322, 0.692, 0.0, -0.0], [0.061, -0.639, -1.636, -1.832, 0.121, 1.76, -1.506], [0.061, 0.639, 1.636, -1.832, -0.121, 1.76, 1.506]


import pdb

class RBY1RoughCfg( ManipulatedRobotCfg ):
    class init_state( ManipulatedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.050 ] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # rainbow init point
           #'right_arm_0' : 0.2059488517353309,
           #'right_arm_1' : -0.1832595714594046,
           #'right_arm_2' : 0.2897246558310587,
           #'right_arm_3' : -0.6318091892219474,
           #'right_arm_4' : -0.05759586531581287,
           #'right_arm_5' : -0.48171087355043496,
           #'right_arm_6' : 1.6057029118347832,
           #'left_arm_0' : 0.2059488517353309,
           #'left_arm_1' : 0.1832595714594046,
           #'left_arm_2' : -0.2897246558310587,
           #'left_arm_3' : -0.6318091892219474,
           #'left_arm_4' : 0.05759586531581287,
           #'left_arm_5' : -0.48171087355043496,
           #'left_arm_6' : -1.6057029118347832

           # home pose
           #'right_arm_0' : -0.477, 
           #'right_arm_1' : -0.185,  
           #'right_arm_2' : -0.128,                                        
           #'right_arm_3' : -2.261,                                              
           #'right_arm_4' : -0.182,                               
           #'right_arm_5' : 1.702,       
           #'right_arm_6' : 0.,
           #'left_arm_0' : -0.477,  
           #'left_arm_1' : 0.185, 
           #'left_arm_2' : 0.128,                                        
           #'left_arm_3' : -2.261,                                              
           #'left_arm_4' : 0.182,                               
           #'left_arm_5' : 1.702,       
           #'left_arm_6' : 0.,

           # init pose
           #'right_arm_0' : 0.2059488517353309,
           #'right_arm_1' : -0.1832595714594046,
           #'right_arm_2' : 0.2897246558310587,
           #'right_arm_3' : -1.9233528356977512,
           #'right_arm_4' : -0.05759586531581287,
           #'right_arm_5' : -0.5864306286700948,
           #'right_arm_6' : 1.6057029118347832,
           #'left_arm_0' : 0.2059488517353309,
           #'left_arm_1' : 0.1832595714594046,
           #'left_arm_2' : -0.2897246558310587,
           #'left_arm_3' : -1.9233528356977512,
           #'left_arm_4' : 0.05759586531581287,
           #'left_arm_5' : -0.5864306286700948,
           #'left_arm_6' : -1.6057029118347832 

           # panel init pose
           'right_arm_0' : 0.061,
           'right_arm_1' : -0.639,
           'right_arm_2' : -1.636,
           'right_arm_3' : -1.832,
           'right_arm_4' : 0.121,
           'right_arm_5' : 1.76,
           'right_arm_6' : -1.506,
           'left_arm_0' : 0.061,
           'left_arm_1' : 0.639,
           'left_arm_2' : 1.636,
           'left_arm_3' : -1.832,
           'left_arm_4' : -0.121,
           'left_arm_5' : 1.76,
           'left_arm_6' : 1.506 

           # target 0
           #'right_arm_0' : 0.2059488517353309,
           #'right_arm_1' : -0.1832595714594046,
           #'right_arm_2' : 0.2897246558310587,
           #'right_arm_3' : -1.9233528356977512,
           #'right_arm_4' : -0.05759586531581287,
           #'right_arm_5' : -0.5864306286700948,
           #'right_arm_6' : 1.6057029118347832,
           #'left_arm_0' :  0.26354471705114374,
           #'left_arm_1' :  1.9373154697137058,
           #'left_arm_2' :  1.0943214410004447,
           #'left_arm_3' :  -0.3752457891787809,
           #'left_arm_4' :  0.022689280275926284,
           #'left_arm_5' :  -0.4066617157146788,
           #'left_arm_6' :  -0.5323254218582705

           # target 1
           #'right_arm_0' : 0.26354471705114374,
           #'right_arm_1' : -1.9373154697137058,
           #'right_arm_2' : -1.0943214410004447,
           #'right_arm_3' : -0.3752457891787809,
           #'right_arm_4' : -0.022689280275926284,
           #'right_arm_5' : -0.4066617157146788,
           #'right_arm_6' : 0.5323254218582705,
           #'left_arm_0' :  0.26354471705114374,
           #'left_arm_1' :  1.9373154697137058,
           #'left_arm_2' :  1.0943214410004447,
           #'left_arm_3' :  -0.3752457891787809,
           #'left_arm_4' :  0.022689280275926284,
           #'left_arm_5' :  -0.4066617157146788,
           #'left_arm_6' :  -0.5323254218582705
        }
        default_hand_angles = [ # = target angles [rad] when action = 0.0
            # rainbow init point
           #'right_arm_0' : 0.2059488517353309,
           #'right_arm_1' : -0.1832595714594046,
           #'right_arm_2' : 0.2897246558310587,
           #'right_arm_3' : -0.6318091892219474,
           #'right_arm_4' : -0.05759586531581287,
           #'right_arm_5' : -0.48171087355043496,
           #'right_arm_6' : 1.6057029118347832,
           #'left_arm_0' : 0.2059488517353309,
           #'left_arm_1' : 0.1832595714594046,
           #'left_arm_2' : -0.2897246558310587,
           #'left_arm_3' : -0.6318091892219474,
           #'left_arm_4' : 0.05759586531581287,
           #'left_arm_5' : -0.48171087355043496,
           #'left_arm_6' : -1.6057029118347832

           # home pose
           #'right_arm_0' : -0.477, 
           #'right_arm_1' : -0.185,  
           #'right_arm_2' : -0.128,                                        
           #'right_arm_3' : -2.261,                                              
           #'right_arm_4' : -0.182,                               
           #'right_arm_5' : 1.702,       
           #'right_arm_6' : 0.,
           #'left_arm_0' : -0.477,  
           #'left_arm_1' : 0.185, 
           #'left_arm_2' : 0.128,                                        
           #'left_arm_3' : -2.261,                                              
           #'left_arm_4' : 0.182,                               
           #'left_arm_5' : 1.702,       
           #'left_arm_6' : 0.,

           # init pose
           # 0.2230,  0.1972,  1.1453,  0.2231, -0.1971,  1.1451

           # panel init pose
            -0.0294,  0.5916,  1.3982, -0.0295, -0.5920,  1.3983

           # target 0
           # 0.0667,  0.6858,  1.6734,  0.2229, -0.1971,  1.1453

           # target 1
           #'right_arm_0' : 0.26354471705114374,
           #'right_arm_1' : -1.9373154697137058,
           #'right_arm_2' : -1.0943214410004447,
           #'right_arm_3' : -0.3752457891787809,
           #'right_arm_4' : -0.022689280275926284,
           #'right_arm_5' : -0.4066617157146788,
           #'right_arm_6' : 0.5323254218582705,
           #'left_arm_0' :  0.26354471705114374,
           #'left_arm_1' :  1.9373154697137058,
           #'left_arm_2' :  1.0943214410004447,
           #'left_arm_3' :  -0.3752457891787809,
           #'left_arm_4' :  0.022689280275926284,
           #'left_arm_5' :  -0.4066617157146788,
           #'left_arm_6' :  -0.5323254218582705
        ]
    
    class env(ManipulatedRobotCfg.env):
        #num_envs = 8192
        num_envs = 4096
        #num_envs = 2048
        #num_envs = 1024
        num_observations = 48
        num_privileged_obs = 48     # + hand_pos 6
        #num_observations = 62
        #num_privileged_obs = 62     # + hand_pos 6
        #num_observations = 68
        #num_privileged_obs = 68     # + hand_pos&target 12
        #num_privileged_obs = 42     #
        num_actions = 14    # upper : 14, rby1 : 24
        episode_length_s = 1.0 # episode length in seconds
        num_hand_dof=6

    class normalization(ManipulatedRobotCfg.normalization):
        class obs_scales(ManipulatedRobotCfg.normalization.obs_scales):
            hand_pos = 1.0
            hand_target = 1.0
            dof_target = 1.0

    class domain_rand(ManipulatedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.1, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 3.]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.5
        randomize_dof = False
        randomize_dof_wide = True  # learning: True
        serialize_dof = False

    class noise (ManipulatedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales(ManipulatedRobotCfg.noise.noise_scales):
            hand_pos = 0.0
            dof_target=0.0

    class commands(ManipulatedRobotCfg.commands):  
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.0, 0.0] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-0.0, 0.0]

    class control( ManipulatedRobotCfg.control ):
        # PD Drive parameters:
        #control_type = 'P'
        control_type = 'P_target'
          # PD Drive parameters:
        stiffness = {'torso_0': 100,
                     'torso_1': 1000,
                     'torso_2': 1000,
                     'torso_3': 220,
                     'torso_4': 50,
                     'torso_5': 220,
                     
                     'right_arm_0': 80,
                     'right_arm_1': 80,
                     'right_arm_2': 80,
                     'right_arm_3': 35,
                     'right_arm_4': 30,
                     'right_arm_5': 30,
                     'right_arm_6': 100,

                     'left_arm_0': 80,
                     'left_arm_1': 80,
                     'left_arm_2': 80,
                     'left_arm_3': 35,
                     'left_arm_4': 30,
                     'left_arm_5': 30,
                     'left_arm_6': 100,
                     } # [N*m/rad]
        damping = {  'torso_0': 900,
                     'torso_1': 900,
                     'torso_2': 900,
                     'torso_3': 400,
                     'torso_4': 400,
                     'torso_5': 400,
                     
                     'right_arm_0': 200,
                     'right_arm_1': 200,
                     'right_arm_2': 200,
                     'right_arm_3': 80,
                     'right_arm_4': 70,
                     'right_arm_5': 70,
                     'right_arm_6': 150,

                     'left_arm_0': 200,
                     'left_arm_1': 200,
                     'left_arm_2': 200,
                     'left_arm_3': 80,
                     'left_arm_4': 70,
                     'left_arm_5': 70,
                     'left_arm_6': 150,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class viewer (ManipulatedRobotCfg.viewer):
        ref_env = 0
        pos = [31, 11, 3]  # [m]
        lookat = [-0., 11, -5.]  # [m]
        #ref_env = 0
        #pos = [41, 31, 2]  # [m]
        #lookat = [-0., 31, -5.]  # [m]

    class terrain (ManipulatedRobotCfg.terrain):
        measure_heights = False

    class asset( ManipulatedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/rby1/rby1_upper.urdf'
        #file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/rby1/rby1.urdf'
        #file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/rby1/rby1_col_v0.urdf'
        #file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/rby1/original.urdf'
        name = "rby1"
        foot_name = "wheel"
        hand_name = "arm_6"
        #penalize_contacts_on = [""]
        terminate_after_contacts_on = ["torso_5", "arm_6"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
        fix_base_link = True # fixe the base of the robot

    class rewards( ManipulatedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        goal_reach_threshold = 0.01
        h_obstacle = 1.2
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        approaching_sigma=0.20
        class scales:
            torques = -0.1e-7
            dof_vel = -2e-4
            dof_acc = -2.0e-7
            target_pos_dist=-0.1
            target_hand_dist=-5.0
            action_rate = -0.0025
            wrist = -0.1
            #obstacle=-10.0
            #approaching=1.0
            #reaching_vel=-0.1

class RBY1RoughCfgPPO( ManipulatedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class runner( ManipulatedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        save_interval = 250 # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'rby1'

  
