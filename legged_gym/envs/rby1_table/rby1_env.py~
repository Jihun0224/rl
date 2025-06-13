
from legged_gym.envs.base.manipulated_robot import ManipulatedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import legged_gym.envs.rby1.target.target_both as target_data
#import legged_gym.envs.rby1.target.target_left as target_data

#from legged_gym.utils.draw import sphere

import pdb

class RBY1Robot(ManipulatedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[self.num_actions:2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        #noise_vec[2*self.num_actions:3*self.num_actions] = noise_scales.dof_target * noise_level * self.obs_scales.dof_target
        noise_vec[2*self.num_actions:2*self.num_actions+6] = noise_scales.hand_pos * noise_level * self.obs_scales.hand_pos
        noise_vec[2*self.num_actions+6:2*self.num_actions+12] = noise_scales.hand_pos * noise_level * self.obs_scales.hand_pos # previous actions
        noise_vec[2*self.num_actions+12:3*self.num_actions+12] = 0. # previous actions
        #noise_vec[3*self.num_actions+6:3*self.num_actions+12] = 0. #noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        #noise_vec[3*self.num_actions+12:4*self.num_actions+12] = 0. # previous actions

        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _init_target(self):
        # load target sets
        self.dof_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_target_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.hand_target = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.hand_target_pos = torch.zeros(6, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(7):
            angle = target_data.mot_target[i+6]    # +6 omitting the torso joints
            angle2 = target_data.mot_target[i+6+7]    # +6 omitting the torso joints, + 7 starting from the right arm
            self.dof_target[:,i] = angle2
            self.dof_target_pos[i] = angle2
            self.dof_target[:,i+7] = angle
            self.dof_target_pos[i+7] = angle
        self.dof_target_pos=self.dof_target_pos.unsqueeze(0)

        for i in range(6):
            angle = target_data.hand_target[i]    # hand target
            self.hand_target[:,i] = angle
            self.hand_target_pos[i] = angle

        self.hand_target_pos=self.hand_target_pos.unsqueeze(0)

    def _init_hand(self):
        self.hand_num = len(self.hand_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.hand_state = self.rigid_body_states_view[:, self.hand_indices, :]
        self.hand_pos = self._global2local(self.hand_state[:, :, :3],6)  # dim=6 for two hands
        self.hand_vel = self.hand_state[:, :, 7:10]


    def _global2local(self,body_state,dim):
        state=body_state.clone().detach()
        for i in range(self.num_envs):
            state[i,0,:] -= self.env_origins[i]
            state[i,1,:] -= self.env_origins[i]
        return state.reshape(self.num_envs,dim) 
        
    def _init_buffers(self):
        self._init_hand()
        super()._init_buffers()
        self._init_foot()
        self._init_target()
        #self._init_serial()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def update_hand_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.hand_state = self.rigid_body_states_view[:, self.hand_indices, :]
        self.hand_pos = self._global2local(self.hand_state[:, :, :3],6)
        self.hand_vel = self.hand_state[:, :, 7:10]
        #print('hand_pos : ', torch.mean(self.hand_pos,dim=0))  # used to get default_hand_pos
        #pdb.set_trace()

    def check_termination(self):
        # [early termination] the case when the current pose is similar to the target
        dist_to_target=torch.sum(torch.square(self.dof_pos[:, :] - self.dof_target_pos), dim=1)
        #dist_to_target_hand = torch.sum(torch.square(self.hand_pos[:, :] - self.hand_target_pos), dim=1)
        self.reset_buf |= dist_to_target[:] < 1e-3
        
        return super().check_termination()

        
    def _post_physics_step_callback(self):
        self.update_feet_state()
        self.update_hand_state()

        #sph=Sphere(0.5,[1.0,0,1.0])
        #sph.draw()

        #jperiod = 0.8
        #joffset = 0.5
        #jself.phase = (self.episode_length_buf * self.dt) % period / period
        #jself.phase_left = self.phase
        #jself.phase_right = (self.phase + offset) % 1
        #jself.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    (self.hand_pos) * self.obs_scales.hand_pos,
                                    (self.hand_target) * self.obs_scales.hand_pos,
                                    self.actions,
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    (self.hand_pos) * self.obs_scales.hand_pos,
                                    (self.hand_target) * self.obs_scales.hand_pos,
                                    self.actions,
                                    ),dim=-1)
        #print('test pose:', (self.dof_pos - self.default_dof_pos))
        #pdb.set_trace()
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P_none":
            torques = self.p_gains*(actions_scaled - self.dof_pos - self.dof_target_pos) - self.d_gains*self.dof_vel
        elif control_type=="P_target":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos - self.dof_target_pos) - self.d_gains*self.dof_vel
        elif control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reward_target_pos_dist(self):
        pos_error = torch.square(self.dof_pos[:, :] - self.dof_target_pos)
        return torch.sum(pos_error, dim=(1))

    def _reward_wrist(self):
        pos_error = torch.square(self.dof_pos[:, 5] - self.dof_target[:,5])
        pos_error += torch.square(self.dof_pos[:, 5+7] - self.dof_target[:,5+7])
        return pos_error

    def _reward_approaching(self):
        pos_error = torch.square(self.dof_pos[:, :] - self.dof_target)
        last_pos_error = torch.square(self.last_dof_pos[:, :] - self.dof_target)
        return torch.sum(last_pos_error[:]-pos_error[:],dim=1)

    def _reward_target_hand_dist(self):
        hand_error = torch.square(self.hand_pos[:, :] - self.hand_target_pos)
        return torch.sum(hand_error, dim=(1))

    def _reward_hand_pos_left(self):
        hand_error = torch.square(self.hand_pos[:, :3] - self.hand_target[:,:3])
        return torch.sum(hand_error, dim=(1))

    def _reward_hand_pos_right(self):
        hand_error = torch.square(self.hand_pos[:, 3:6] - self.hand_target[:,3:6])
        return torch.sum(hand_error, dim=(1))

    def _reward_obstacle(self):
        obs_height=torch.sum(torch.square(self.hand_pos[:, 2] - self.cfg.rewards.h_obstacle), dim=1)
        return 100.0 * obs_height[:] < 0

    def _reward_reaching_vel(self):
        pos_error = torch.square(self.dof_pos[:, :] - self.dof_target)
        print('close enough : ',torch.sum(pos_error, dim=(1)) < 0.01)
        return self.dof_vel * torch.sum(pos_error, dim=(1)) < self.cfg.rewards.goal_reach_threshold
        
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    
    def _reward_goal(self):
        hand_error = torch.square(self.hand_pos[:, :] - self.hand_target)
        return torch.sum(hand_error, dim=(1))
