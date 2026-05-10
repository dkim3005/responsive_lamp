from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

from config import (
    ENGAGEMENT_OFF_HOLD_SEC,
    ENGAGEMENT_ON_HOLD_SEC,
    IDLE_AFTER_NO_FACE_SEC,
    SEEK_DELAY_SEC,
)


BASE_JOINTS = [0.0, -30.0, 60.0, 0.0, -25.0, 0.0]
HEAD_PITCH_NEUTRAL_DEG = -25.0

# Arm pose presets [lower_pitch, upper_pitch]
_ARM_NEUTRAL = (-30.0, 60.0)
_ARM_SLOUCH  = (-33.0, 52.0)   # disengaged — slightly folded
_ARM_EXTEND  = (-12.0, 38.0)   # object found — arm reaches forward


@dataclass
class LampCommand:
    state: str
    joints: list[float]
    light: dict
    sound: str | None = None

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "joints": self.joints,
            "light": self.light,
            "sound": self.sound,
        }


class LampFSM:
    def __init__(self) -> None:
        self.state = "IDLE"
        self.state_enter_t = 0.0
        self.raw_true_since: float | None = None
        self.raw_false_since: float | None = None
        self.no_face_since: float | None = None
        self.disengaged_since: float | None = None
        self.last_seek_level = 0
        self.demo_until = 0.0
        self.object_until = 0.0
        self.object_xy: list[float] | None = None

    def trigger_demo(self, t: float) -> None:
        self.demo_until = t + 2.5
        self._set_state("DEMO_WAVE", t)

    def trigger_object_found(self, t: float, bbox: list[float]) -> None:
        x1, y1, x2, y2 = bbox
        self.object_xy = [((x1 + x2) / 2.0 - 0.5) * 2.0, ((y1 + y2) / 2.0 - 0.5) * 2.0]
        self.object_until = t + 2.2
        self._set_state("OBJECT_FOUND", t)

    def tick(self, t: float, engaged_raw: bool, face_xy: list[float] | None) -> dict:
        if self.state not in {"DEMO_WAVE", "OBJECT_FOUND"}:
            self._update_hysteresis(t, engaged_raw, face_xy)
        elapsed = t - self.state_enter_t

        if self.state == "SEEKING_1" and elapsed >= 3.0:
            self._set_state("DISENGAGED", t)
        elif self.state == "SEEKING_2" and elapsed >= 4.0:
            self._set_state("DISENGAGED", t)
        elif self.state == "SEEKING_3" and elapsed >= 3.0:
            self._set_state("DISENGAGED", t)
        elif self.state == "DEMO_WAVE" and t >= self.demo_until:
            self._set_state("DISENGAGED", t)
            self.disengaged_since = t
        elif self.state == "OBJECT_FOUND" and t >= self.object_until:
            self._set_state("ENGAGED" if engaged_raw else "DISENGAGED", t)
            self.disengaged_since = None if engaged_raw else t

        if self.state == "DISENGAGED" and self.disengaged_since is not None:
            disengaged_for = t - self.disengaged_since
            if disengaged_for >= SEEK_DELAY_SEC[2] and self.last_seek_level < 3:
                self.last_seek_level = 3
                self._set_state("SEEKING_3", t)
            elif disengaged_for >= SEEK_DELAY_SEC[1] and self.last_seek_level < 2:
                self.last_seek_level = 2
                self._set_state("SEEKING_2", t)
            elif disengaged_for >= SEEK_DELAY_SEC[0] and self.last_seek_level < 1:
                self.last_seek_level = 1
                self._set_state("SEEKING_1", t)

        cmd = self._command(t, face_xy).as_dict()
        cmd["disengaged_for"] = round(t - self.disengaged_since, 1) if self.disengaged_since is not None else None
        return cmd

    def _update_hysteresis(self, t: float, engaged_raw: bool, face_xy: list[float] | None) -> None:
        if face_xy is None:
            self.no_face_since = self.no_face_since if self.no_face_since is not None else t
        else:
            self.no_face_since = None

        if self.no_face_since is not None and t - self.no_face_since >= IDLE_AFTER_NO_FACE_SEC:
            self._set_state("IDLE", t)
            self.disengaged_since = None
            return

        if engaged_raw:
            self.raw_true_since = self.raw_true_since if self.raw_true_since is not None else t
            self.raw_false_since = None
            if t - self.raw_true_since >= ENGAGEMENT_ON_HOLD_SEC:
                self._set_state("ENGAGED", t)
                self.disengaged_since = None
                self.last_seek_level = 0
        else:
            self.raw_false_since = self.raw_false_since if self.raw_false_since is not None else t
            self.raw_true_since = None
            if self.state == "ENGAGED" and t - self.raw_false_since >= ENGAGEMENT_OFF_HOLD_SEC:
                self._set_state("DISENGAGED", t)
                self.disengaged_since = t
                self.last_seek_level = 0
            elif face_xy is not None and self.state == "IDLE":
                self._set_state("DISENGAGED", t)
                self.disengaged_since = self.disengaged_since if self.disengaged_since is not None else t

    def _set_state(self, state: str, t: float) -> None:
        if self.state != state:
            self.state = state
            self.state_enter_t = t

    def _command(self, t: float, face_xy: list[float] | None) -> LampCommand:
        elapsed = t - self.state_enter_t
        joints = BASE_JOINTS.copy()
        light = {"intensity": 0.5, "color": "#ffffff"}
        sound = None

        if self.state == "IDLE":
            # Slow breathing on upper arm — feels alive but resting
            joints[2] = 60.0 + math.sin(t * 0.38) * 2.5
            joints[4] = HEAD_PITCH_NEUTRAL_DEG
            light = {"intensity": 0.3, "color": "#d0d8ff"}

        elif self.state == "ENGAGED":
            if face_xy:
                joints = _track_face_joints(joints, face_xy, engaged=True)
            # Subtle breathing so it never feels frozen
            joints[2] += math.sin(t * 0.38) * 1.5
            light = {"intensity": 1.0, "color": "#ffd28a"}

        elif self.state == "DISENGAGED":
            if face_xy:
                joints = _track_face_joints(joints, face_xy, engaged=False)
            else:
                joints[4] = HEAD_PITCH_NEUTRAL_DEG - 5.0
            # Slouched arm + slow idle roll — "I see you but you're ignoring me"
            joints[1] = _ARM_SLOUCH[0]
            joints[2] = _ARM_SLOUCH[1]
            joints[5] += math.sin(t * 0.72) * 5.0
            light = {"intensity": 0.55, "color": "#8fb7ff"}

        elif self.state == "SEEKING_1":
            # "hey..." — noticeable nod + head sweep
            joints[0] = math.sin(elapsed * 1.1) * 22.0
            joints[1] = -30.0 + math.sin(elapsed * 1.6) * 14.0
            joints[2] = 60.0 + math.sin(elapsed * 1.3) * 14.0
            joints[3] = math.sin(elapsed * 1.8) * 28.0
            joints[4] = HEAD_PITCH_NEUTRAL_DEG + math.sin(elapsed * 1.4) * 16.0
            joints[5] = math.sin(elapsed * 0.9) * 18.0
            pulse = 0.65 + 0.3 * abs(math.sin(elapsed * 1.5))
            light = {"intensity": pulse, "color": "#ffe6a7"}

        elif self.state == "SEEKING_2":
            # "HEY!!" — full arm wave + big puppy tilt
            joints[0] = math.sin(elapsed * 2.2) * 32.0
            joints[1] = -28.0 + math.sin(elapsed * 2.5) * 24.0
            joints[2] = 62.0 + math.sin(elapsed * 2.0) * 30.0
            joints[3] = math.sin(elapsed * 3.1) * 38.0
            joints[4] = HEAD_PITCH_NEUTRAL_DEG + math.sin(elapsed * 2.8) * 18.0
            joints[5] = math.sin(elapsed * 1.7) * 38.0
            intensity = 0.75 + 0.25 * abs(math.sin(elapsed * math.pi * 3))
            hue = (elapsed * 120.0) % 360.0
            light = {"intensity": intensity, "color": _hsl_to_hex(hue, 1.0, 0.55)}

        elif self.state == "SEEKING_3":
            # "PLEASE" — maximum chaos, all joints different coprime freqs
            joints[0] = math.sin(elapsed * 2.3) * 45.0
            joints[1] = -24.0 + math.sin(elapsed * 3.7) * 28.0
            joints[2] = 62.0 + math.sin(elapsed * 2.9) * 32.0
            joints[3] = math.sin(elapsed * 4.3) * 45.0
            joints[4] = HEAD_PITCH_NEUTRAL_DEG + math.sin(elapsed * 3.1) * 22.0
            joints[5] = math.sin(elapsed * 5.1) * 38.0
            light = {"intensity": 1.0, "color": "#ffb347"}
            sound = "chirp" if elapsed < 0.25 else None

        elif self.state == "DEMO_WAVE":
            joints[0] = math.sin(5.0 * elapsed) * 42.0
            joints[1] = -26.0 + math.sin(7.0 * elapsed) * 20.0
            joints[2] = 60.0 + math.sin(6.0 * elapsed) * 24.0
            joints[3] = math.sin(11.0 * elapsed) * 42.0
            joints[4] = HEAD_PITCH_NEUTRAL_DEG + math.sin(9.0 * elapsed) * 18.0
            joints[5] = math.sin(7.3 * elapsed) * 28.0
            light = {"intensity": 1.0, "color": "#ffcc58"}
            sound = "chirp" if elapsed < 0.35 else None

        elif self.state == "OBJECT_FOUND":
            ox, oy = self.object_xy or [0.0, 0.0]
            # Always point head/base toward the object
            joints[0] = ox * 14.0
            joints[3] = ox * 28.0
            joints[4] = HEAD_PITCH_NEUTRAL_DEG + oy * 18.0
            joints[5] = -ox * 6.0
            if elapsed < 0.5:
                # Smoothly extend arm toward object
                p = elapsed / 0.5
                joints[1] = _ARM_NEUTRAL[0] + (_ARM_EXTEND[0] - _ARM_NEUTRAL[0]) * p
                joints[2] = _ARM_NEUTRAL[1] + (_ARM_EXTEND[1] - _ARM_NEUTRAL[1]) * p
            else:
                # Hold extended + tiny excited tremor on arm and head
                joints[1] = _ARM_EXTEND[0] + math.sin(elapsed * 8.0) * 2.0
                joints[2] = _ARM_EXTEND[1] + math.sin(elapsed * 7.3) * 2.0
                joints[5] += math.sin(elapsed * 6.1) * 8.0
            blink = 0.4 + 0.6 * abs(math.sin(elapsed * math.pi * 8))
            light = {"intensity": blink, "color": "#8fffd2"}
            sound = "chirp" if elapsed < 0.22 else None

        return LampCommand(self.state, [round(v, 3) for v in joints], light, sound)


def _track_face_joints(joints: list[float], face_xy: list[float], engaged: bool) -> list[float]:
    x, y = face_xy
    scale = 1.0 if engaged else 0.7
    joints[0] = x * 22.0 * scale
    joints[1] = -27.0
    joints[2] = 58.0
    joints[3] = x * 38.0 * scale
    joints[4] = HEAD_PITCH_NEUTRAL_DEG - y * 22.0
    joints[5] = -x * 14.0 * scale
    return joints


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
