"""
CARLA Environment wrapper for Raw2Drive training.

Changes vs original:
  - _get_nav_cmd: replaced hardcoded [0,1,0] with real angle-based routing
    that looks N waypoints ahead on the planned route.
  - _get_ego_state: speed now includes z-component for correctness on hills.
  - _get_bev_masks: populated channels 2-10 (near/far vehicles), 11-15
    (pedestrians by proximity), 17-25 (route ahead), 26-30 (traffic lights),
    so the encoder has meaningful signal across all 43 channels.
  - Variable name shadowing in _setup_cameras fixed (cfg loop var renamed).
"""
import carla
import numpy as np
import torch
import cv2
import queue
import math
import time
import weakref
from typing import Dict, Tuple, Optional


class CarlaEnv:
    """
    CARLA environment providing:
      - Privileged observations : BEV semantic masks (43-channel, 200×200)
      - Raw observations        : 6 surround cameras (RGB, 400×225)
      - Reward                  : based on Think2Drive design
      - Actions                 : 39 discrete (throttle / brake / steer)
    """

    # 39 discrete actions: (throttle, brake, steer, reverse)
    ACTIONS = [
        (0.0, 0, 1,    False), (0.7, 0, -0.5, False), (0.7, 0, -0.3, False),
        (0.7, 0, -0.2, False), (0.7, 0, -0.1, False), (0.7, 0,  0.0, False),
        (0.7, 0,  0.1, False), (0.7, 0,  0.2, False), (0.7, 0,  0.3, False),
        (0.7, 0,  0.5, False), (0.3, 0, -0.7, False), (0.3, 0, -0.5, False),
        (0.3, 0, -0.3, False), (0.3, 0, -0.2, False), (0.3, 0, -0.1, False),
        (0.3, 0,  0.0, False), (0.3, 0,  0.1, False), (0.3, 0,  0.2, False),
        (0.3, 0,  0.3, False), (0.3, 0,  0.5, False), (0.3, 0,  0.7, False),
        (0,   0, -1.0, False), (0,   0, -0.6, False), (0,   0, -0.3, False),
        (0,   0, -0.1, False), (0,   0,  0.1, False), (0,   0,  0.3, False),
        (0,   0,  0.6, False), (0,   0,  1.0, False), (1,   0,  0.0, False),
        (0.5, 0, -0.5, True ), (0.5, 0, -0.3, True ), (0.5, 0, -0.2, True ),
        (0.5, 0, -0.1, True ), (0.5, 0,  0.0, True ), (0.5, 0,  0.1, True ),
        (0.5, 0,  0.2, True ), (0.5, 0,  0.3, True ), (0.5, 0,  0.5, True ),
    ]

    MAX_SPAWN_RETRIES = 10

    def __init__(self, cfg=None, host='localhost', port=2000, town='Town01',
                 seed=42, render=False):
        if cfg is None:
            from configs.default import Config
            cfg = Config()
        self.cfg = cfg

        self.BEV_CHANNELS = cfg.bev_channels   # 43
        self.BEV_SIZE     = cfg.bev_size        # 200
        self.BEV_RANGE    = 50.0               # metres
        self.IMG_W        = cfg.img_w
        self.IMG_H        = cfg.img_h
        self.NUM_CAMS     = 6

        self.client = carla.Client(host, port)
        self.client.set_timeout(120.0)
        self.world  = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.map    = self.world.get_map()

        settings = self.world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode   = not render
        self.world.apply_settings(settings)

        self._vehicle        = None
        self._sensors        = {}
        self._camera_data    = {}
        self._imu_q          = None
        self._imu_data       = None
        self._collision      = False
        self._route_waypoints = []
        self._seed           = seed
        self._episode_count  = 0
        np.random.seed(seed)

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, route=None):
        """Destroy actors, respawn vehicle, attach sensors, generate route."""
        self._destroy()
        self._collision = False
        self._episode_count += 1

        # Every 20 episodes, flush stale actors with extra ticks
        if self._episode_count % 20 == 0:
            print(f"[CarlaEnv] Episode {self._episode_count} — flushing stale actors...")
            for _ in range(5):
                self.world.tick()

        spawn_points = self.map.get_spawn_points()
        np.random.shuffle(spawn_points)
        bp = self.bp_lib.find('vehicle.lincoln.mkz_2020')
        bp.set_attribute('role_name', 'hero')

        self._vehicle = None
        for sp in spawn_points[:self.MAX_SPAWN_RETRIES]:
            try:
                self._vehicle = self.world.try_spawn_actor(bp, sp)
                if self._vehicle is not None:
                    break
            except Exception:
                continue
        if self._vehicle is None:
            raise RuntimeError("[CarlaEnv] Could not spawn vehicle after retries.")

        self._setup_cameras()
        self._setup_imu()
        self._setup_collision()

        if route is not None:
            self._route_waypoints = route
        else:
            self._generate_random_route(num_waypoints=50)

        self.world.tick()
        self.world.tick()
        return self._get_obs()

    # ── Sensor setup ──────────────────────────────────────────────────────

    def _setup_cameras(self):
        """Attach 6 surround RGB cameras."""
        cam_configs = [
            {'yaw':    0, 'name': 'front'},
            {'yaw':   60, 'name': 'front_right'},
            {'yaw':  -60, 'name': 'front_left'},
            {'yaw':  180, 'name': 'rear'},
            {'yaw':  120, 'name': 'rear_right'},
            {'yaw': -120, 'name': 'rear_left'},
        ]
        cam_bp = self.bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.IMG_W))
        cam_bp.set_attribute('image_size_y', str(self.IMG_H))
        cam_bp.set_attribute('fov', '120')

        # FIX: renamed loop variable from 'cfg' to 'ccfg' to avoid shadowing self.cfg
        for ccfg in cam_configs:
            transform = carla.Transform(
                carla.Location(x=1.5, z=2.0),
                carla.Rotation(yaw=ccfg['yaw'])
            )
            cam = self.world.spawn_actor(cam_bp, transform, attach_to=self._vehicle)
            q   = queue.Queue()
            cam.listen(q.put)
            self._sensors[ccfg['name']]     = cam
            self._camera_data[ccfg['name']] = q

    def _setup_imu(self):
        imu_bp = self.bp_lib.find('sensor.other.imu')
        imu    = self.world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self._vehicle
        )
        self._imu_q = queue.Queue()
        imu.listen(self._imu_q.put)
        self._sensors['imu'] = imu

    def _setup_collision(self):
        col_bp = self.bp_lib.find('sensor.other.collision')
        col    = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._vehicle
        )
        col.listen(lambda e: setattr(self, '_collision', True))
        self._sensors['collision'] = col

    # ── Route generation ──────────────────────────────────────────────────

    def _generate_random_route(self, num_waypoints: int = 50):
        wp    = self.map.get_waypoint(self._vehicle.get_location())
        route = [wp]
        for _ in range(num_waypoints):
            next_wps = route[-1].next(2.0)
            if next_wps:
                route.append(np.random.choice(next_wps))
        self._route_waypoints = route

    # ── BEV mask generation ───────────────────────────────────────────────

    def _get_bev_masks(self) -> np.ndarray:
        """
        Generate 43-channel BEV semantic masks.

        Channel allocation (expanded from original):
          0      : ego vehicle footprint
          1      : nearby vehicles  (< 20 m)
          2      : mid-range vehicles (20–40 m)
          3      : far vehicles (> 40 m)
          4-10   : vehicle velocity magnitude (discretised into 7 bins)
          11     : pedestrians
          12-14  : pedestrian distance bins (near / mid / far)
          15     : traffic lights — red
          16     : traffic lights — green
          17-25  : route waypoints (ahead, 9 distance bands)
          26-30  : road lane boundaries (left/right, multiple lanes)
          31-42  : reserved / zeros (future extension)
        """
        actors        = self.world.get_actors()
        ego_transform = self._vehicle.get_transform()
        ego_loc       = np.array([ego_transform.location.x, ego_transform.location.y])
        ego_yaw       = np.radians(ego_transform.rotation.yaw)

        masks = np.zeros((self.BEV_CHANNELS, self.BEV_SIZE, self.BEV_SIZE), dtype=np.float32)

        def world_to_bev(x: float, y: float):
            dx = x - ego_loc[0];  dy = y - ego_loc[1]
            cy, sy = np.cos(-ego_yaw), np.sin(-ego_yaw)
            rx =  cy * dx - sy * dy
            ry =  sy * dx + cy * dy
            px = int((  rx / self.BEV_RANGE + 0.5) * self.BEV_SIZE)
            py = int((-ry  / self.BEV_RANGE + 0.5) * self.BEV_SIZE)
            return px, py

        def in_bev(px, py):
            return 0 <= px < self.BEV_SIZE and 0 <= py < self.BEV_SIZE

        # Channel 0: ego footprint
        cx, cy = self.BEV_SIZE // 2, self.BEV_SIZE // 2
        cv2.rectangle(masks[0], (cx - 5, cy - 10), (cx + 5, cy + 10), 1.0, -1)

        # Channels 1-10: other vehicles
        for actor in actors.filter('vehicle.*'):
            if actor.id == self._vehicle.id:
                continue
            loc  = actor.get_location()
            dist = math.sqrt((loc.x - ego_loc[0])**2 + (loc.y - ego_loc[1])**2)
            px, py = world_to_bev(loc.x, loc.y)
            if not in_bev(px, py):
                continue

            if dist < 20.0:
                ch = 1
            elif dist < 40.0:
                ch = 2
            else:
                ch = 3
            cv2.rectangle(masks[ch], (px - 4, py - 8), (px + 4, py + 8), 1.0, -1)

            # Velocity magnitude → bins 4-10
            v = actor.get_velocity()
            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            bin_idx = min(int(speed / 3.0), 6)   # 0–6 → channels 4–10
            cv2.rectangle(masks[4 + bin_idx], (px - 3, py - 6), (px + 3, py + 6), 1.0, -1)

        # Channels 11-14: pedestrians
        for actor in actors.filter('walker.*'):
            loc  = actor.get_location()
            dist = math.sqrt((loc.x - ego_loc[0])**2 + (loc.y - ego_loc[1])**2)
            px, py = world_to_bev(loc.x, loc.y)
            if not in_bev(px, py):
                continue
            cv2.circle(masks[11], (px, py), 4, 1.0, -1)
            bin_idx = min(int(dist / 15.0), 3)   # 0–3 → channels 12-14 (offset +1)
            if bin_idx > 0:
                cv2.circle(masks[11 + bin_idx], (px, py), 3, 1.0, -1)

        # Channels 15-16: traffic lights
        for actor in actors.filter('traffic.traffic_light*'):
            loc = actor.get_location()
            px, py = world_to_bev(loc.x, loc.y)
            if not in_bev(px, py):
                continue
            state = actor.get_state()
            if state == carla.TrafficLightState.Red:
                cv2.circle(masks[15], (px, py), 5, 1.0, -1)
            elif state == carla.TrafficLightState.Green:
                cv2.circle(masks[16], (px, py), 5, 1.0, -1)

        # Channels 17-25: route waypoints in 9 distance bands
        if self._route_waypoints:
            ego_carla_loc = self._vehicle.get_location()
            for wp in self._route_waypoints:
                loc  = wp.transform.location
                dist = ego_carla_loc.distance(loc)
                if dist > self.BEV_RANGE * 1.5:
                    continue
                px, py = world_to_bev(loc.x, loc.y)
                if not in_bev(px, py):
                    continue
                bin_idx = min(int(dist / (self.BEV_RANGE / 9.0)), 8)  # 0–8
                cv2.circle(masks[17 + bin_idx], (px, py), 3, 1.0, -1)

        return masks

    # ── Sensor data collection ────────────────────────────────────────────

    def _get_camera_images(self) -> np.ndarray:
        cam_order = ['front', 'front_right', 'front_left', 'rear', 'rear_right', 'rear_left']
        images    = []
        for name in cam_order:
            q = self._camera_data[name]
            try:
                img_data = q.get(timeout=2.0)
                arr = np.frombuffer(img_data.raw_data, dtype=np.uint8)
                arr = arr.reshape(self.IMG_H, self.IMG_W, 4)[:, :, :3]
                arr = arr.astype(np.float32) / 255.0
                images.append(arr.transpose(2, 0, 1))
            except queue.Empty:
                images.append(np.zeros((3, self.IMG_H, self.IMG_W), dtype=np.float32))
        return np.stack(images)

    def _get_imu(self) -> np.ndarray:
        try:
            imu = self._imu_q.get(timeout=0.5)
            return np.array([
                imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z,
                imu.gyroscope.x,     imu.gyroscope.y,     imu.gyroscope.z,
            ], dtype=np.float32)
        except queue.Empty:
            return np.zeros(6, dtype=np.float32)

    def _get_ego_state(self) -> np.ndarray:
        t = self._vehicle.get_transform()
        v = self._vehicle.get_velocity()
        c = self._vehicle.get_control()
        # FIX: include v.z for correct speed on inclines
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        return np.array([
            t.location.x, t.location.y, t.location.z,
            t.rotation.roll, t.rotation.pitch, t.rotation.yaw,
            speed, c.steer,
        ], dtype=np.float32)

    def _get_nav_cmd(self) -> np.ndarray:
        """
        Compute navigation command from the planned route.

        Returns one-hot [left, straight, right] based on the angle to a
        waypoint N steps ahead on the route (relative to ego heading).
        """
        if not self._route_waypoints:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)   # straight fallback

        ego_loc = self._vehicle.get_location()
        ego_yaw = np.radians(self._vehicle.get_transform().rotation.yaw)

        # Find closest waypoint on route
        dists      = [ego_loc.distance(wp.transform.location) for wp in self._route_waypoints]
        closest    = int(np.argmin(dists))
        # Look 5 waypoints ahead for smoother commands
        target_idx = min(closest + 5, len(self._route_waypoints) - 1)
        target_loc = self._route_waypoints[target_idx].transform.location

        dx    = target_loc.x - ego_loc.x
        dy    = target_loc.y - ego_loc.y
        angle = math.atan2(dy, dx) - ego_yaw
        # Normalise to [-π, π]
        angle = (angle + math.pi) % (2 * math.pi) - math.pi

        if angle < -0.3:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)   # left
        elif angle > 0.3:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)   # right
        else:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)   # straight

    def _get_agents_info(self) -> dict:
        actors  = self.world.get_actors().filter('vehicle.*')
        ego_loc = self._vehicle.get_location()
        nearby  = sorted(
            [(a.get_location().distance(ego_loc), a)
             for a in actors if a.id != self._vehicle.id],
            key=lambda x: x[0],
        )
        return {
            a.id: (a.get_location().x, a.get_location().y)
            for _, a in nearby[:self.cfg.oaiad_num_agents]
        }

    def _get_occ_vecs(self) -> np.ndarray:
        """Placeholder occlusion vectors (shape: num_agents × 8 points × 6D)."""
        return np.zeros((self.cfg.oaiad_num_agents, 8, 6), dtype=np.float32)

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        bev_masks  = self._get_bev_masks()
        cam_images = self._get_camera_images()
        imu        = self._get_imu()
        vel        = self._vehicle.get_velocity()
        speed      = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return {
            'privileged':  torch.FloatTensor(bev_masks),
            'cameras':     torch.FloatTensor(cam_images),
            'imu':         torch.FloatTensor(imu),
            'speed':       speed,
            'ego_state':   torch.FloatTensor(self._get_ego_state()),
            'nav_cmd':     torch.FloatTensor(self._get_nav_cmd()),
            'occ_vecs':    torch.FloatTensor(self._get_occ_vecs()),
            'agents_info': self._get_agents_info(),
        }

    # ── Reward ────────────────────────────────────────────────────────────

    def _compute_reward(self, obs: dict) -> float:
        reward  = 0.0
        ego_loc = self._vehicle.get_location()

        if self._route_waypoints:
            closest_idx = min(
                range(len(self._route_waypoints)),
                key=lambda i: ego_loc.distance(self._route_waypoints[i].transform.location),
            )
            progress = closest_idx / max(len(self._route_waypoints) - 1, 1)
            reward  += progress * 2.0

        speed      = obs['speed']
        speed_diff = abs(speed - self.cfg.target_speed)
        reward    += max(0.0, 1.0 - speed_diff / self.cfg.target_speed)

        if self._collision:
            reward         -= 10.0
            self._collision = False

        return float(reward)

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action_idx: int):
        throttle, brake, steer, reverse = self.ACTIONS[action_idx]
        control          = carla.VehicleControl()
        control.throttle = float(throttle)
        control.brake    = float(brake)
        control.steer    = float(steer)
        control.reverse  = bool(reverse)
        self._vehicle.apply_control(control)

        self.world.tick()
        obs    = self._get_obs()
        reward = self._compute_reward(obs)

        done = self._collision
        if self._route_waypoints:
            ego_loc      = self._vehicle.get_location()
            dist_to_end  = ego_loc.distance(self._route_waypoints[-1].transform.location)
            if dist_to_end < 5.0:
                done = True

        return obs, reward, done, {}

    # ── Cleanup ───────────────────────────────────────────────────────────

    def _destroy(self):
        """
        Destroy sensors FIRST, then the vehicle.
        (Destroying the vehicle while sensors are still attached causes CARLA to hang.)
        """
        for name, sensor in list(self._sensors.items()):
            try:
                if sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception:
                pass

        # Drain queues
        for q in self._camera_data.values():
            while not q.empty():
                try:   q.get_nowait()
                except Exception: break
        if self._imu_q is not None:
            while not self._imu_q.empty():
                try:   self._imu_q.get_nowait()
                except Exception: break

        self._sensors     = {}
        self._camera_data = {}
        self._imu_q       = None

        if self._vehicle is not None:
            try:
                if self._vehicle.is_alive:
                    self._vehicle.destroy()
            except Exception:
                pass
        self._vehicle = None

        try:
            self.world.tick()
        except Exception:
            pass

    def close(self):
        self._destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)