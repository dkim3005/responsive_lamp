from __future__ import annotations

from dataclasses import dataclass

from config import (
    ENGAGEMENT_OFF_HOLD_SEC,
    ENGAGEMENT_ON_HOLD_SEC,
    IDLE_AFTER_NO_FACE_SEC,
    SEEK_DELAY_SEC,
)
from behavior.motions import add_joints, base_scan, head_wiggle


BASE_JOINTS = [0.0, -30.0, 60.0, 20.0, 0.0, 0.0]


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

    def trigger_demo(self, t: float) -> None:
        self.demo_until = t + 2.5
        self._set_state("DEMO_WAVE", t)

    def tick(self, t: float, engaged_raw: bool, face_xy: list[float] | None) -> dict:
        self._update_hysteresis(t, engaged_raw, face_xy)
        elapsed = t - self.state_enter_t

        if self.state == "SEEKING_1" and elapsed >= 2.0:
            self._set_state("DISENGAGED", t)
        elif self.state == "SEEKING_2" and elapsed >= 3.0:
            self._set_state("DISENGAGED", t)
        elif self.state == "SEEKING_3" and elapsed >= 1.2:
            self._set_state("DISENGAGED", t)
        elif self.state == "DEMO_WAVE" and t >= self.demo_until:
            self._set_state("DISENGAGED", t)
            self.disengaged_since = t

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

        return self._command(t, face_xy).as_dict()

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
            elif self.state == "IDLE" and face_xy is not None:
                self._set_state("DISENGAGED", t)
                self.disengaged_since = t

    def _set_state(self, state: str, t: float) -> None:
        if self.state != state:
            self.state = state
            self.state_enter_t = t

    def _command(self, t: float, face_xy: list[float] | None) -> LampCommand:
        elapsed = t - self.state_enter_t
        joints = BASE_JOINTS.copy()
        light = {"intensity": 0.5, "color": "#ffffff"}
        sound = None

        if self.state == "ENGAGED":
            if face_xy:
                joints[3] = -face_xy[0] * 30.0
                joints[4] = 20.0 - face_xy[1] * 20.0
            light = {"intensity": 1.0, "color": "#ffd28a"}
        elif self.state == "DISENGAGED":
            joints[4] = 12.0
            light = {"intensity": 0.4, "color": "#b8c7ff"}
        elif self.state == "SEEKING_1":
            joints = add_joints(joints, head_wiggle(elapsed))
            light = {"intensity": 0.65, "color": "#ffe6a7"}
        elif self.state == "SEEKING_2":
            intensity = 0.7 + 0.3 * abs(__import__("math").sin(3.0 * elapsed))
            joints[4] = 16.0
            light = {"intensity": intensity, "color": "#ffcf70"}
        elif self.state == "SEEKING_3":
            joints = add_joints(joints, base_scan(elapsed))
            joints[1] = -22.0 + __import__("math").sin(8.0 * elapsed) * 10.0
            joints[3] = __import__("math").sin(10.0 * elapsed) * 28.0
            light = {"intensity": 1.0, "color": "#ffb347"}
            sound = "chirp" if elapsed < 0.25 else None
        elif self.state == "DEMO_WAVE":
            math = __import__("math")
            joints[0] = math.sin(5.0 * elapsed) * 35.0
            joints[1] = -28.0 + math.sin(7.0 * elapsed) * 14.0
            joints[2] = 58.0 + math.sin(6.0 * elapsed) * 16.0
            joints[3] = math.sin(11.0 * elapsed) * 35.0
            joints[4] = 18.0 + math.sin(9.0 * elapsed) * 12.0
            light = {"intensity": 1.0, "color": "#ffcc58"}
            sound = "chirp" if elapsed < 0.35 else None

        if self.state not in {"SEEKING_1", "SEEKING_2", "SEEKING_3", "DEMO_WAVE"} and int(t) % 30 in {0, 1}:
            joints[0] += 6.0

        return LampCommand(self.state, [round(v, 3) for v in joints], light, sound)
