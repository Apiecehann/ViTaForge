from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.utils.data import DataLoader

from .bc import load_bc_checkpoint
from .dataset import ActionPhaseDataset, split_episode_paths


class SFTRegularizedSAC(SAC):
    def __init__(
        self,
        *args,
        bc_checkpoint: str,
        bc_dataset_root: str,
        online_bc_regularization: float = 10.0,
        offline_bc_regularization: float = 100.0,
        bc_image_size: int = 128,
        **kwargs,
    ):
        self.bc_checkpoint = str(bc_checkpoint)
        self.bc_dataset_root = str(bc_dataset_root)
        self.online_bc_regularization = float(online_bc_regularization)
        self.offline_bc_regularization = float(offline_bc_regularization)
        self.bc_image_size = int(bc_image_size)
        super().__init__(*args, **kwargs)
        self.sft_teacher, _ = load_bc_checkpoint(
            self.bc_checkpoint,
            device=self.device,
        )
        self.sft_teacher.requires_grad_(False)
        self.sft_teacher.eval()
        self._bc_dataset = None
        self._bc_loader = None
        self._bc_iterator = None

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "sft_teacher",
            "_bc_dataset",
            "_bc_loader",
            "_bc_iterator",
        ]

    def _next_bc_observations(self, batch_size: int):
        if self._bc_loader is None:
            training_paths, validation_paths = split_episode_paths(
                self.bc_dataset_root
            )
            self._bc_dataset = ActionPhaseDataset(
                training_paths + validation_paths,
                image_size=self.bc_image_size,
            )
            self._bc_loader = DataLoader(
                self._bc_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
            self._bc_iterator = iter(self._bc_loader)
        try:
            observations, _ = next(self._bc_iterator)
        except StopIteration:
            self._bc_iterator = iter(self._bc_loader)
            observations, _ = next(self._bc_iterator)
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in observations.items()
        }

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses = []
        ent_coefs = []
        actor_losses = []
        critic_losses = []
        online_bc_losses = []
        offline_bc_losses = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(
                        replay_data.next_observations,
                        next_actions,
                    ),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(
                    -1,
                    1,
                )
                target_q_values = replay_data.rewards + (
                    (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations,
                replay_data.actions,
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi),
                dim=1,
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            with th.no_grad():
                online_teacher_actions = self.sft_teacher.forward_policy_action(
                    replay_data.observations
                )
            online_bc_loss = F.mse_loss(actions_pi, online_teacher_actions)

            offline_observations = self._next_bc_observations(batch_size)
            offline_actions = self.actor(
                offline_observations,
                deterministic=True,
            )
            with th.no_grad():
                offline_teacher_actions = self.sft_teacher.forward_policy_action(
                    offline_observations
                )
            offline_bc_loss = F.mse_loss(
                offline_actions,
                offline_teacher_actions,
            )
            actor_loss = (
                (ent_coef * log_prob - min_qf_pi).mean()
                + self.online_bc_regularization * online_bc_loss
                + self.offline_bc_regularization * offline_bc_loss
            )
            actor_losses.append(actor_loss.item())
            online_bc_losses.append(online_bc_loss.item())
            offline_bc_losses.append(offline_bc_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0,
                )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/online_bc_loss", np.mean(online_bc_losses))
        self.logger.record("train/offline_bc_loss", np.mean(offline_bc_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
