from ._base_task import Action
from .insert_USB import (
    PRE_INSERT_CLEARANCE,
    USB_PLUG_HEIGHT,
    Task as InsertUSBTask,
    TaskCfg,
)
import numpy as np
import transforms3d as t3d


RECOVERY_RETRACT = 0.004
ANGLE_STEP = np.deg2rad(1.0)
ANGLE_THRESHOLD = np.deg2rad(2.0)
POS_STEP = 0.0002
POS_THRESHOLD = 0.0004
MAX_ALIGN_ITERS = 30
INITIAL_PROBE_DISTANCE = 0.012
RECOVERY_PROBE_DISTANCE = 0.004
STUCK_PROGRESS_TOLERANCE = 0.0005
MAX_RECOVERY_ATTEMPTS = 5


class Task(InsertUSBTask):
    def _get_actor_axis(self, local_axis):
        axis = self.prism.get_pose().to_transformation_matrix()[:3, :3] @ np.array(local_axis).reshape(3)
        return axis / np.linalg.norm(axis)

    def _get_slot_axis(self, local_axis):
        axis = self.slot.get_pose().to_transformation_matrix()[:3, :3] @ np.array(local_axis).reshape(3)
        return axis / np.linalg.norm(axis)

    def _rotate_ee_in_world(self, axis, angle):
        ee_pose = self._robot_manager.get_ee_pose()
        target_pose = ee_pose.clone()
        delta_q = t3d.quaternions.mat2quat(t3d.axangles.axangle2mat(axis, angle))
        target_pose.q = t3d.quaternions.qmult(delta_q, ee_pose.q)
        return self.move(
            [Action(action='move', target_pose=target_pose)],
            tag='realign_rotate',
            delay=False,
            time_dilation_factor=0.5,
            constraint_pose=[0, 0, 0, 1, 1, 1]
        )

    def _realign_orientation(self, angle_step=ANGLE_STEP, angle_threshold=ANGLE_THRESHOLD, max_iters=MAX_ALIGN_ITERS):
        rotate_iters = 0
        for _ in range(max_iters):
            curr_z = self._get_actor_axis([0, 0, 1])
            target_z = self._get_slot_axis([0, 0, 1])
            rot_axis = np.cross(curr_z, target_z)
            axis_norm = np.linalg.norm(rot_axis)
            angle = float(np.arctan2(axis_norm, np.dot(curr_z, target_z)))
            if angle < angle_threshold or axis_norm < 1e-8:
                break

            rot_axis = rot_axis / axis_norm
            step = min(angle, angle_step)
            if not self._rotate_ee_in_world(rot_axis, step):
                break
            rotate_iters += 1

        self.metadata['realign_rotate_iters'] = rotate_iters

    def _realign_orientation_once(self, angle_threshold=ANGLE_THRESHOLD):
        curr_z = self._get_actor_axis([0, 0, 1])
        target_z = self._get_slot_axis([0, 0, 1])
        rot_axis = np.cross(curr_z, target_z)
        axis_norm = np.linalg.norm(rot_axis)
        angle = float(np.arctan2(axis_norm, np.dot(curr_z, target_z)))

        if angle < angle_threshold or axis_norm < 1e-8:
            self.metadata['realign_rotate_once_angle'] = angle
            self.metadata['realign_rotate_once_success'] = True
            return True

        rot_axis = rot_axis / axis_norm
        success = self._rotate_ee_in_world(rot_axis, angle)
        self.metadata['realign_rotate_once_angle'] = angle
        self.metadata['realign_rotate_once_success'] = success
        return success

    def _realign_position(self, target_pose, pos_step=POS_STEP, pos_threshold=POS_THRESHOLD, max_iters=MAX_ALIGN_ITERS):
        position_iters = 0
        target_mat = target_pose.to_transformation_matrix()
        target_rot = target_mat[:3, :3]

        for _ in range(max_iters):
            rel_pose = self.prism.get_pose().rebase(target_pose)
            rel_xyz = rel_pose.p
            if np.linalg.norm(rel_xyz) < pos_threshold:
                break

            target_frame_delta = np.clip(-rel_xyz, -pos_step, pos_step)
            world_delta = target_rot @ target_frame_delta.reshape(3, 1)
            world_delta = world_delta.reshape(3)
            if not self.move(self.atom.move_by_displacement(
                x=world_delta[0],
                y=world_delta[1],
                z=world_delta[2],
                xyz_coord='world'
            ), tag='realign_position', delay=False, time_dilation_factor=0.5):
                break
            position_iters += 1

        self.metadata['realign_position_iters'] = position_iters

    def _realign_position_once(self, target_pose, pos_threshold=POS_THRESHOLD):
        target_mat = target_pose.to_transformation_matrix()
        target_rot = target_mat[:3, :3]
        rel_pose = self.prism.get_pose().rebase(target_pose)
        rel_xyz = rel_pose.p
        rel_norm = float(np.linalg.norm(rel_xyz))

        if rel_norm < pos_threshold:
            self.metadata['realign_position_once_delta'] = [0.0, 0.0, 0.0]
            self.metadata['realign_position_once_error'] = rel_norm
            self.metadata['realign_position_once_success'] = True
            return True

        target_frame_delta = -rel_xyz
        world_delta = target_rot @ target_frame_delta.reshape(3, 1)
        world_delta = world_delta.reshape(3)
        success = self.move(self.atom.move_by_displacement(
            x=world_delta[0],
            y=world_delta[1],
            z=world_delta[2],
            xyz_coord='world'
        ), tag='realign_position_once', delay=False, time_dilation_factor=0.5)
        self.metadata['realign_position_once_delta'] = world_delta.tolist()
        self.metadata['realign_position_once_error'] = rel_norm
        self.metadata['realign_position_once_success'] = success
        return success

    def _get_recovery_pose(self):
        recovery_pose = self.pre_insert_pose.clone()
        recovery_pose.p[2] = self.prism.get_pose().p[2]
        return recovery_pose

    def _recover_from_stuck(self):
        if not self.move(self.atom.move_by_displacement(
            z=RECOVERY_RETRACT,
            xyz_coord='world'
        ), tag='recover_retract', delay=False, time_dilation_factor=0.5,
                constraint_pose=[1, 1, 1, 1, 1, 0]):
            return False
        if not self._realign_orientation_once():
            return False
        return self._realign_position_once(self._get_recovery_pose())

    def _move_down_and_measure(self, delta, tag):
        before_pose = self.prism.get_pose()
        success = self.move(self.atom.move_by_displacement(
            z=-delta,
            xyz_coord='world'
        ), tag=tag, delay=False, time_dilation_factor=0.5,
                constraint_pose=[1, 1, 1, 1, 1, 0])
        after_pose = self.prism.get_pose()
        actual_progress = max(0.0, float(before_pose.p[2] - after_pose.p[2]))
        return success, before_pose, after_pose, actual_progress

    def _is_insert_stuck(self, commanded_delta, actual_progress, stuck_tolerance=STUCK_PROGRESS_TOLERANCE):
        return commanded_delta - actual_progress > stuck_tolerance

    def _get_remaining_insert_distance(self, start_z, distance):
        current_z = float(self.prism.get_pose().p[2])
        inserted_depth = max(0.0, start_z - current_z)
        return max(0.0, distance - inserted_depth)

    def _direct_finish_insert(self, start_z, distance):
        remaining = self._get_remaining_insert_distance(start_z, distance)
        if remaining <= 1e-8:
            return True, 0.0

        success = self.move(self.atom.move_by_displacement(
            z=-remaining,
            xyz_coord='world'
        ), tag='realign_direct_finish', delay=False, time_dilation_factor=0.5,
                constraint_pose=[1, 1, 1, 1, 1, 0])
        return success, remaining

    def _insert_in_steps(
        self,
        distance=PRE_INSERT_CLEARANCE + USB_PLUG_HEIGHT,
        initial_probe_distance=INITIAL_PROBE_DISTANCE,
        recovery_probe_distance=RECOVERY_PROBE_DISTANCE,
        stuck_tolerance=STUCK_PROGRESS_TOLERANCE,
        max_recovery_attempts=MAX_RECOVERY_ATTEMPTS
    ):
        insert_iters = 0
        recovery_attempts = 0
        stuck_events = []
        start_z = float(self.prism.get_pose().p[2])
        initial_probe_delta = min(initial_probe_distance, distance)
        initial_probe_progress = 0.0
        initial_probe_success = True
        initial_probe_stuck = False
        direct_finish_delta = 0.0
        direct_finish_success = False

        if initial_probe_delta > 1e-8:
            initial_probe_success, before_pose, after_pose, initial_probe_progress = \
                self._move_down_and_measure(initial_probe_delta, tag='realign_initial_probe')
            insert_iters += 1
            initial_probe_stuck = self._is_insert_stuck(
                initial_probe_delta, initial_probe_progress, stuck_tolerance)

            if initial_probe_stuck:
                stuck_events.append({
                    'phase': 'initial_probe',
                    'insert_iter': insert_iters,
                    'attempt': recovery_attempts + 1,
                    'commanded_delta': float(initial_probe_delta),
                    'actual_progress': initial_probe_progress,
                    'progress_error': float(initial_probe_delta - initial_probe_progress),
                    'before_pose': before_pose.tolist(),
                    'after_pose': after_pose.tolist(),
                })

        if self.plan_success and initial_probe_success and not initial_probe_stuck:
            direct_finish_success, direct_finish_delta = self._direct_finish_insert(start_z, distance)

        while self.plan_success and initial_probe_success and initial_probe_stuck \
                and recovery_attempts < max_recovery_attempts:
            recovery_attempts += 1
            if not self._recover_from_stuck():
                break

            remaining = self._get_remaining_insert_distance(start_z, distance)
            if remaining <= 1e-8:
                direct_finish_success = True
                break

            delta = min(recovery_probe_distance, remaining)
            probe_success, before_pose, after_pose, actual_progress = \
                self._move_down_and_measure(delta, tag='realign_recovery_probe')
            insert_iters += 1
            if not probe_success:
                break

            probe_stuck = self._is_insert_stuck(delta, actual_progress, stuck_tolerance)
            if not probe_stuck:
                direct_finish_success, direct_finish_delta = self._direct_finish_insert(start_z, distance)
                break

            stuck_event = {
                'phase': 'recovery_probe',
                'insert_iter': insert_iters,
                'attempt': recovery_attempts,
                'commanded_delta': float(delta),
                'actual_progress': actual_progress,
                'progress_error': float(delta - actual_progress),
                'before_pose': before_pose.tolist(),
                'after_pose': after_pose.tolist(),
            }
            stuck_events.append(stuck_event)

        self.metadata['realign_insert_iters'] = insert_iters
        self.metadata['realign_recovery_attempts'] = recovery_attempts
        self.metadata['realign_stuck_events'] = stuck_events
        self.metadata['realign_initial_probe_delta'] = float(initial_probe_delta)
        self.metadata['realign_initial_probe_progress'] = initial_probe_progress
        self.metadata['realign_initial_probe_success'] = initial_probe_success
        self.metadata['realign_initial_probe_stuck'] = initial_probe_stuck
        self.metadata['realign_recovery_probe_delta'] = float(recovery_probe_distance)
        self.metadata['realign_stuck_tolerance'] = float(stuck_tolerance)
        self.metadata['realign_direct_finish_delta'] = float(direct_finish_delta)
        self.metadata['realign_direct_finish_success'] = direct_finish_success

    def _play_once(self):
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=self.pre_insert_pose,
            pre_dis=0.01,
            dis=0.002,
            is_open=False
        ), time_dilation_factor=0.5)
        self._insert_in_steps()
        self.delay(20, is_save=True)
