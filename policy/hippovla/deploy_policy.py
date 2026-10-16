"""StarVLA <-> RoboSynChallenge evaluation adapter.

RoboSynChallenge loads policies through ``scripts/eval_policy.py`` with the
``get_model`` / ``encode_obs`` / ``encode_action`` / ``eval`` / ``reset_model``
interface documented in ``tutorials/policy/your_own_policy.html``.

This adapter keeps the StarVLA model in a separate WebSocket policy server and
runs only a lightweight client inside the RoboSynChallenge/EmbodiChain
environment.
"""

from __future__ import annotations

import base64
import atexit
import functools
import hashlib
import json
import logging
import os
import socket
import ssl
import struct
import subprocess
import time
import urllib.parse
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2 as cv
import msgpack
import numpy as np
import torch

try:
    import yaml
except ImportError:
    yaml = None

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


TRAIN_TO_ROBOSYN_ACTION_ORDER = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
ROBOSYN_TO_TRAIN_ACTION_ORDER = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
DEFAULT_BINARY_ACTION_INDICES = [12, 13]
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _pack_array(obj: object) -> object:
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj: dict) -> object:
    ndarray_key = b"__ndarray__" if b"__ndarray__" in obj else "__ndarray__"
    npgeneric_key = b"__npgeneric__" if b"__npgeneric__" in obj else "__npgeneric__"
    if ndarray_key in obj:
        data = obj.get(b"data", obj.get("data"))
        dtype = obj.get(b"dtype", obj.get("dtype"))
        shape = obj.get(b"shape", obj.get("shape"))
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=shape)

    if npgeneric_key in obj:
        data = obj.get(b"data", obj.get("data"))
        dtype = obj.get(b"dtype", obj.get("dtype"))
        return np.dtype(dtype).type(data)

    return {
        key.decode("utf-8") if isinstance(key, bytes) else key: value
        for key, value in obj.items()
    }


MsgpackPacker = functools.partial(msgpack.Packer, default=_pack_array)
msgpack_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array, raw=False)


class _StdlibWebSocketConnection:
    """Small ws:// client used inside the simulation-only RoboSyn container."""

    def __init__(self, uri: str, headers: Optional[dict[str, str]] = None, timeout: float = 150) -> None:
        self.uri = uri
        self.headers = headers or {}
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._buffer = b""
        self._connect()

    def _connect(self) -> None:
        parsed = urllib.parse.urlparse(self.uri)
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError(f"Unsupported WebSocket URI scheme: {parsed.scheme!r}")
        host = parsed.hostname
        if host is None:
            raise ValueError(f"WebSocket URI has no host: {self.uri}")
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        raw_sock = socket.create_connection((host, port), timeout=self.timeout)
        if parsed.scheme == "wss":
            raw_sock = ssl.create_default_context().wrap_socket(raw_sock, server_hostname=host)
        raw_sock.settimeout(self.timeout)
        self.sock = raw_sock

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        host_header = f"{host}:{port}" if parsed.port is not None else host
        request_lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {host_header}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        request_lines.extend(f"{name}: {value}" for name, value in self.headers.items())
        request = "\r\n".join(request_lines) + "\r\n\r\n"
        raw_sock.sendall(request.encode("ascii"))

        response = b""
        while b"\r\n\r\n" not in response:
            chunk = raw_sock.recv(4096)
            if not chunk:
                raise ConnectionError("WebSocket server closed during handshake.")
            response += chunk

        header_bytes, self._buffer = response.split(b"\r\n\r\n", 1)
        header_text = header_bytes.decode("iso-8859-1")
        status_line, *header_lines = header_text.split("\r\n")
        if " 101 " not in status_line:
            raise ConnectionError(f"WebSocket handshake failed: {status_line}")

        response_headers = {}
        for line in header_lines:
            if ":" in line:
                name, value = line.split(":", 1)
                response_headers[name.strip().lower()] = value.strip()
        expected_accept = base64.b64encode(
            hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if response_headers.get("sec-websocket-accept") != expected_accept:
            raise ConnectionError("WebSocket handshake returned an invalid Sec-WebSocket-Accept header.")

        raw_sock.settimeout(None)

    def _read_exact(self, size: int) -> bytes:
        while len(self._buffer) < size:
            if self.sock is None:
                raise ConnectionError("WebSocket is closed.")
            chunk = self.sock.recv(max(4096, size - len(self._buffer)))
            if not chunk:
                raise ConnectionError("WebSocket connection closed.")
            self._buffer += chunk
        data = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return data

    def _send_frame(self, payload: bytes, opcode: int = 0x2) -> None:
        if self.sock is None:
            raise ConnectionError("WebSocket is closed.")
        payload_len = len(payload)
        if payload_len < 126:
            header = struct.pack("!BB", 0x80 | opcode, 0x80 | payload_len)
        elif payload_len < (1 << 16):
            header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, payload_len)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, payload_len)
        mask = os.urandom(4)
        masked_payload = bytes(byte ^ mask[idx % 4] for idx, byte in enumerate(payload))
        self.sock.sendall(header + mask + masked_payload)

    def send(self, payload: bytes) -> None:
        self._send_frame(payload, opcode=0x2)

    def recv(self) -> bytes | str:
        fragments: list[bytes] = []
        message_opcode: Optional[int] = None

        while True:
            first_byte, second_byte = self._read_exact(2)
            fin = bool(first_byte & 0x80)
            opcode = first_byte & 0x0F
            masked = bool(second_byte & 0x80)
            payload_len = second_byte & 0x7F

            if payload_len == 126:
                payload_len = struct.unpack("!H", self._read_exact(2))[0]
            elif payload_len == 127:
                payload_len = struct.unpack("!Q", self._read_exact(8))[0]

            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(payload_len) if payload_len else b""
            if masked:
                payload = bytes(byte ^ mask[idx % 4] for idx, byte in enumerate(payload))

            if opcode == 0x8:
                self.close()
                raise ConnectionError("WebSocket server closed the connection.")
            if opcode == 0x9:
                self._send_frame(payload, opcode=0xA)
                continue
            if opcode == 0xA:
                continue
            if opcode in {0x1, 0x2}:
                message_opcode = opcode
                fragments = [payload]
            elif opcode == 0x0:
                if message_opcode is None:
                    raise ConnectionError("Received WebSocket continuation without a message start.")
                fragments.append(payload)
            else:
                raise ConnectionError(f"Unsupported WebSocket opcode: {opcode}")

            if fin:
                data = b"".join(fragments)
                return data.decode("utf-8") if message_opcode == 0x1 else data

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self._send_frame(b"", opcode=0x8)
        except Exception:
            pass
        try:
            self.sock.close()
        finally:
            self.sock = None


class WebsocketClientPolicy:
    """StarVLA WebSocket client without importing the host StarVLA package."""

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = MsgpackPacker()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float = 300) -> tuple[_StdlibWebSocketConnection, Dict]:
        logging.info("Waiting for StarVLA policy server at %s...", self._uri)
        start_time = time.time()
        last_error: Optional[BaseException] = None

        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)

        while time.time() - start_time <= timeout:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = _StdlibWebSocketConnection(self._uri, headers=headers, timeout=150)
                metadata = msgpack_unpackb(conn.recv())
                return conn, metadata
            except OSError as exc:
                last_error = exc
                logging.info("Still waiting for server %s ...", self._uri)
                time.sleep(2)

        raise TimeoutError(f"Failed to connect to policy server at {self._uri} within {timeout} seconds.") from last_error

    def close(self) -> None:
        self._ws.close()

    def predict_action(self, query_info: Dict) -> Dict:
        self._ws.send(self._packer.pack(query_info))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_unpackb(response)


def read_mode_config(pretrained_checkpoint: str) -> tuple[dict, dict]:
    """Load StarVLA config/statistics from a checkpoint run directory.

    In Docker eval mode, ``eval.sh`` can create a small metadata-only shadow
    checkpoint path containing ``config.yaml`` and ``dataset_statistics.json``.
    """
    checkpoint_pt = Path(pretrained_checkpoint)
    if checkpoint_pt.suffix == ".pt" and len(checkpoint_pt.parents) >= 2:
        run_dir = checkpoint_pt.parents[1]
    elif checkpoint_pt.is_dir():
        run_dir = checkpoint_pt
    else:
        run_dir = checkpoint_pt.parent

    config_yaml = run_dir / "config.yaml"
    dataset_statistics_json = run_dir / "dataset_statistics.json"
    if not dataset_statistics_json.exists():
        raise FileNotFoundError(
            f"Missing dataset_statistics.json under {run_dir}. "
            "When running RoboSynChallenge in Docker, use eval.sh/start_eval.sh so the "
            "checkpoint metadata is copied into the container, or set ROBOSYN_DOCKER_CKPT_PATH "
            "to a container-visible metadata path."
        )

    if config_yaml.exists():
        if yaml is None:
            raise ImportError("PyYAML is required to load StarVLA config.yaml inside the eval environment.")
        with config_yaml.open("r") as f:
            model_config = yaml.safe_load(f) or {}
    else:
        model_config = {}

    with dataset_statistics_json.open("r") as f:
        norm_stats = json.load(f)
    return model_config, norm_stats


def _load_history_utils():
    """Import history helpers without forcing one package layout."""
    try:
        from history_flow_utils import (
            compute_flow_rgb_farneback,
            parse_hw_size,
            sample_history_offsets,
        )
    except ImportError:
        from .history_flow_utils import (
            compute_flow_rgb_farneback,
            parse_hw_size,
            sample_history_offsets,
        )
    return compute_flow_rgb_farneback, parse_hw_size, sample_history_offsets


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().any().item())
    arr = np.asarray(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(arr.any())


def _first_scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    arr = _as_numpy(value).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def _ensure_rgb_uint8(image: Any) -> np.ndarray:
    image = _as_numpy(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape [H, W, C], got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] > 3:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            max_value = float(np.nanmax(image)) if image.size else 1.0
            if max_value <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def _extract_first_env_vector(value: Any, env: Any = None) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim > 1:
        array = array[0]
    array = np.asarray(array, dtype=np.float32).reshape(-1)

    unwrapped = getattr(env, "unwrapped", env)
    active_joint_ids = getattr(unwrapped, "active_joint_ids", None)
    if active_joint_ids is not None:
        env_action_dim = int(np.prod(unwrapped.single_action_space.shape))
        if array.shape[0] > env_action_dim:
            active_joint_ids = list(active_joint_ids)
            if len(active_joint_ids) == env_action_dim and max(active_joint_ids) < array.shape[0]:
                array = array[active_joint_ids]
    return np.ascontiguousarray(array)


def _get_env_instruction(env: Any) -> str:
    instruction = getattr(env, "_current_instruction", None)
    if instruction is None and hasattr(env, "unwrapped"):
        instruction = getattr(env.unwrapped, "_current_instruction", None)
    return "" if instruction is None else str(instruction)


def _env_is_success(env: Any) -> bool:
    try:
        return bool(env.get_wrapper_attr("is_task_success")())
    except Exception:
        return False


def _parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _parse_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _parse_indices(value: object, default: Sequence[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return list(default)
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [int(item) for item in value]
    raise ValueError(f"Cannot parse index list: {value!r}")


class ModelClient:
    """WebSocket client for StarVLA inference during RoboSynChallenge eval."""

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robosynchallenge",
        horizon: int = 0,
        action_ensemble: bool = False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] | None = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        normalize_all_action_dims: bool = False,
        actions_are_normalized: bool = True,
        model_action_order: str = "train",
        send_state: bool = False,
        binary_action_indices: Optional[Sequence[int]] = None,
        binary_threshold: float = 0.5,
        eval_action_steps: Optional[int] = None,
        history_k: int = 0,
        history_stride: int = 1,
        history_mode: str = "flow",
        history_image_size: Optional[list[int]] = None,
        history_flow_compute_size: Optional[list[int]] = None,
        memory_enabled: bool = True,
        memory_max_step: int = 5,
        memory_interval: int = 10,
        starvla_eval_video_log: bool = False,
        starvla_video_dir: Optional[str] = None,
        eval_video_obs_key: str = "cam_high",
        starvla_video_fps: int = 10,
        starvla_video_format: str = "jpg",
        success_min_steps: int = 0,
        task_name: str = "",
        task_success_mode: str = "official",
        water_pouring_success_min_steps: int = 80,
        water_pouring_success_max_xy: float = 0.30,
        water_pouring_success_z_min: float = -0.10,
        water_pouring_success_z_max: float = 0.15,
        water_pouring_success_min_tilt_deg: float = 45.0,
        water_pouring_success_cup_upright_max_deg: float = 25.0,
        camera_overrides: Optional[Dict[str, Any]] = None,
        action_debug_interval: int = 0,
        replan_action_steps: Optional[int] = None,
    ) -> None:
        if image_size is None:
            image_size = [224, 224]

        self.policy_ckpt_path = policy_ckpt_path
        self.model_config, self.norm_stats = read_mode_config(policy_ckpt_path)
        self.unnorm_key = self._resolve_unnorm_key(self.norm_stats, unnorm_key)
        self.action_norm_stats = self._get_action_stats(self.norm_stats, self.unnorm_key)

        self.client = WebsocketClientPolicy(host, port)
        server_meta = self.client.get_server_metadata()

        self.policy_setup = policy_setup
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = [int(image_size[0]), int(image_size[1])]
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode
        self.normalize_all_action_dims = bool(normalize_all_action_dims)
        self.actions_are_normalized = bool(actions_are_normalized)
        self.model_action_order = self._normalize_action_order(model_action_order)
        self.send_state = bool(send_state)
        self.binary_action_indices = _parse_indices(
            binary_action_indices, default=DEFAULT_BINARY_ACTION_INDICES
        )
        self.binary_threshold = float(binary_threshold)

        self.action_mode = str(action_mode).strip().lower()
        if self.action_mode not in {"abs", "delta", "rel"}:
            raise ValueError("action_mode must be one of: abs, delta, rel")

        self.initial_state = None
        self.prev_action = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon,
                self.adaptive_ensemble_alpha,
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.raw_actions = None
        self.action_chunk_index = 0
        self.env_step = 0

        self.action_chunk_size = int(
            server_meta.get("action_chunk_size")
            or self._get_action_chunk_size_from_config(self.model_config)
        )
        self.eval_action_steps = _parse_optional_int(eval_action_steps) or self.action_chunk_size
        self.eval_action_steps = max(1, int(self.eval_action_steps))
        requested_replan_steps = _parse_optional_int(replan_action_steps)
        self.replan_action_steps = (
            self.action_chunk_size if requested_replan_steps is None else requested_replan_steps
        )
        if not 1 <= self.replan_action_steps <= self.action_chunk_size:
            raise ValueError(
                f"replan_action_steps must be between 1 and action_chunk_size={self.action_chunk_size}"
            )

        self.history_k = max(0, int(history_k))
        self.history_stride = max(1, int(history_stride))
        self.history_mode = self._normalize_history_mode(history_mode)
        _, parse_hw_size, _ = _load_history_utils()
        self.history_enabled = self.history_k > 0 and self.history_mode != "none"
        self.history_image_size: Tuple[int, int] = parse_hw_size(
            history_image_size, default_size=tuple(self.image_size)
        )
        self.history_flow_compute_size: Tuple[int, int] = parse_hw_size(
            history_flow_compute_size, default_size=(128, 128)
        )
        if self.history_enabled:
            self.history_frame_buffer: deque[np.ndarray] = deque(
                maxlen=self.history_k * self.history_stride
            )
        else:
            self.history_frame_buffer = deque(maxlen=0)

        self.memory_enabled = bool(memory_enabled)
        self.memory_max_step = max(0, int(memory_max_step))
        self.memory_interval = max(1, int(memory_interval))
        self.memory_frame_history: Dict[int, List[np.ndarray]] = {}
        self.video_enabled = bool(starvla_eval_video_log) and bool(starvla_video_dir)
        self.video_dir = Path(starvla_video_dir) if starvla_video_dir else None
        self.video_obs_key = str(eval_video_obs_key or "cam_high")
        self.video_fps = max(1, int(starvla_video_fps))
        self.video_format = str(starvla_video_format or "jpg").strip().lower()
        if self.video_format not in {"jpg", "jpeg", "png"}:
            raise ValueError("starvla_video_format must be one of: jpg, jpeg, png")
        if self.video_format == "jpeg":
            self.video_format = "jpg"
        self.video_episode_idx = -1
        self.video_frame_idx = 0
        self.video_tmp_dir: Optional[Path] = None
        self.video_episode_dir: Optional[Path] = None
        self.video_closed = True
        if self.video_enabled and self.video_dir is not None:
            self.video_dir.mkdir(parents=True, exist_ok=True)
            atexit.register(self.close_video_episode, False)
        self.success_min_steps = max(0, int(success_min_steps))
        self._success_gate_installed = False
        self.task_name = str(task_name or "")
        self.task_success_mode = str(task_success_mode or "official").strip().lower()
        self.task_success_latched = False
        self.last_success_metrics: Dict[str, float] = {}
        self.water_pouring_success_min_steps = max(0, int(water_pouring_success_min_steps))
        self.water_pouring_success_max_xy = float(water_pouring_success_max_xy)
        self.water_pouring_success_z_min = float(water_pouring_success_z_min)
        self.water_pouring_success_z_max = float(water_pouring_success_z_max)
        self.water_pouring_success_min_tilt_deg = float(water_pouring_success_min_tilt_deg)
        self.water_pouring_success_cup_upright_max_deg = float(water_pouring_success_cup_upright_max_deg)
        self.camera_overrides = camera_overrides or {}
        self._camera_overrides_applied = False
        self.action_debug_interval = max(0, int(action_debug_interval))

        print(
            "*** StarVLA RoboSynChallenge client: "
            f"host={host}, port={port}, unnorm_key={self.unnorm_key}, "
            f"action_mode={self.action_mode}, normalization={self.normalization_mode}, "
            f"chunk={self.action_chunk_size}, eval_steps={self.eval_action_steps}, "
            f"replan_steps={self.replan_action_steps}, "
            f"model_action_order={self.model_action_order}, send_state={self.send_state}, "
            f"server_meta={server_meta} ***"
        )
        if self.history_enabled:
            print(
                "*** history enabled: "
                f"k={self.history_k}, stride={self.history_stride}, "
                f"mode={self.history_mode}, out_size={self.history_image_size}, "
                f"flow_compute_size={self.history_flow_compute_size} ***"
            )
        if self.memory_enabled:
            print(
                "*** memory enabled: "
                f"max_step={self.memory_max_step}, interval={self.memory_interval} ***"
            )
        if self.video_enabled:
            print(
                "*** StarVLA eval video enabled: "
                f"dir={self.video_dir}, obs_key={self.video_obs_key}, "
                f"fps={self.video_fps}, format={self.video_format} ***"
            )
        if self.success_min_steps > 0:
            print(f"*** success gate enabled: min_env_steps={self.success_min_steps} ***")
        if self.task_success_mode != "official":
            print(f"*** task success mode: {self.task_success_mode} ***")
        if self.camera_overrides:
            print(f"*** camera overrides enabled: {list(self.camera_overrides.keys())} ***")
        if self.action_debug_interval > 0:
            print(f"*** action debug enabled: interval={self.action_debug_interval} ***")

    @staticmethod
    def _config_get(config: object, keys: Tuple[str, ...], default: object = None) -> object:
        current = config
        for key in keys:
            if isinstance(current, dict):
                if key not in current:
                    return default
                current = current[key]
            else:
                current = getattr(current, key, default)
                if current is default:
                    return default
        return current

    @classmethod
    def _get_action_chunk_size_from_config(cls, model_config: object) -> int:
        action_cfg = cls._config_get(model_config, ("framework", "action_model"), {})
        # The action head emits the current action plus its temporal windows.
        # ``num_actions_chunk`` can be larger than the actual inference output.
        past_value = cls._config_get(action_cfg, ("past_action_window_size",), None)
        future_value = cls._config_get(action_cfg, ("future_action_window_size",), None)
        if past_value is not None or future_value is not None:
            return int(past_value or 0) + 1 + int(future_value or 0)
        explicit = cls._config_get(action_cfg, ("num_actions_chunk",), None)
        if explicit is not None:
            return int(explicit)
        return 1

    @staticmethod
    def _resolve_unnorm_key(norm_stats: dict, unnorm_key: Optional[str]) -> Optional[str]:
        if not norm_stats:
            return unnorm_key
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    "Checkpoint has multiple dataset statistics. Set unnorm_key to one of: "
                    f"{list(norm_stats.keys())}"
                )
            return next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise ValueError(
                f"unnorm_key={unnorm_key!r} not found in checkpoint statistics. "
                f"Available keys: {list(norm_stats.keys())}"
            )
        return unnorm_key

    @staticmethod
    def _get_action_stats(norm_stats: dict, unnorm_key: Optional[str]) -> Optional[dict]:
        if not norm_stats or unnorm_key is None:
            return None
        stats = norm_stats[unnorm_key]
        if "action" not in stats:
            raise KeyError(f"Dataset statistics for {unnorm_key!r} do not contain an action block.")
        return stats["action"]

    @staticmethod
    def _normalize_action_order(action_order: str) -> str:
        order = str(action_order).strip().lower()
        aliases = {
            "train": "train",
            "starvla": "train",
            "model": "train",
            "robosyn": "robosyn",
            "robosynchallenge": "robosyn",
            "env": "robosyn",
        }
        if order not in aliases:
            raise ValueError("model_action_order must be 'train' or 'robosyn'.")
        return aliases[order]

    def reset(self, task_description: str = "") -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.history_frame_buffer.clear()
        self.memory_frame_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self.action_chunk_index = 0
        self.initial_state = None
        self.prev_action = None
        self.env_step = 0
        self._camera_overrides_applied = False
        self.task_success_latched = False
        self.last_success_metrics = {}

    @staticmethod
    def _normalize_history_mode(history_mode: Optional[str]) -> str:
        mode = "flow" if history_mode is None else str(history_mode).strip().lower()
        if mode in {"off", "none", "disabled"}:
            return "none"
        if mode not in {"flow", "frames"}:
            raise ValueError(
                f"Unknown history_mode: {history_mode!r}. Expected one of ['flow', 'frames', 'none']."
            )
        return mode

    def _push_history_frame(self, image: np.ndarray) -> None:
        if not self.history_enabled:
            return
        h, w = self.history_flow_compute_size
        resized = cv.resize(image, (w, h), interpolation=cv.INTER_AREA)
        self.history_frame_buffer.append(resized)

    def _build_history_context(self, current_image: np.ndarray) -> Optional[List[np.ndarray]]:
        if not self.history_enabled:
            return None
        if self.history_mode == "flow":
            return self._build_history_flow(current_image)
        if self.history_mode == "frames":
            return self._build_history_frames(current_image)
        return None

    def _build_history_payload(self, current_image: np.ndarray) -> Dict[str, object]:
        history_images = self._build_history_context(current_image)
        if history_images is None:
            return {}
        return {"history_images": history_images}

    def _build_history_flow(self, current_image: np.ndarray) -> List[np.ndarray]:
        compute_flow_rgb_farneback, _, sample_history_offsets = _load_history_utils()
        h, w = self.history_flow_compute_size
        current_small = cv.resize(current_image, (w, h), interpolation=cv.INTER_AREA)
        offsets = sample_history_offsets(self.history_k, self.history_stride)

        sampled: List[np.ndarray] = []
        buf_len = len(self.history_frame_buffer)
        for offset in offsets:
            idx = max(0, min(buf_len + offset, buf_len - 1)) if buf_len > 0 else -1
            sampled.append(self.history_frame_buffer[idx] if idx >= 0 else current_small)
        sampled.append(current_small)

        out_h, out_w = self.history_image_size
        flow_images: List[np.ndarray] = []
        for i in range(len(sampled) - 1):
            flow_rgb = compute_flow_rgb_farneback(
                sampled[i],
                sampled[i + 1],
                compute_size=self.history_flow_compute_size,
            )
            flow_images.append(cv.resize(flow_rgb, (out_w, out_h), interpolation=cv.INTER_AREA))
        return flow_images

    def _build_history_frames(self, current_image: np.ndarray) -> List[np.ndarray]:
        _, _, sample_history_offsets = _load_history_utils()
        h, w = self.history_flow_compute_size
        current_small = cv.resize(current_image, (w, h), interpolation=cv.INTER_AREA)
        offsets = sample_history_offsets(self.history_k, self.history_stride)

        frames: List[np.ndarray] = []
        buf_len = len(self.history_frame_buffer)
        out_h, out_w = self.history_image_size
        for offset in offsets:
            idx = max(0, min(buf_len + offset, buf_len - 1)) if buf_len > 0 else -1
            frame = self.history_frame_buffer[idx] if idx >= 0 else current_small
            frames.append(cv.resize(frame, (out_w, out_h), interpolation=cv.INTER_AREA))
        return frames

    def _build_memory(self, step: int, images: list[np.ndarray]) -> list[list[np.ndarray]]:
        current_step = int(step)
        current_images = [np.array(image, copy=True) for image in images]
        self.memory_frame_history[current_step] = current_images

        min_keep_step = current_step - (self.memory_max_step + 1) * self.memory_interval
        for old_step in list(self.memory_frame_history.keys()):
            if old_step < min_keep_step:
                self.memory_frame_history.pop(old_step, None)

        zero_views = [np.zeros_like(image, dtype=np.uint8) for image in current_images]
        memory = []
        # Match the training dataset: chronological order from the oldest
        # sparse frame to the newest one, e.g. [t-50, ..., t-10].
        for i in range(self.memory_max_step, 0, -1):
            hist_step = current_step - i * self.memory_interval
            hist_images = self.memory_frame_history.get(hist_step, zero_views)
            memory.append([np.array(image, copy=True) for image in hist_images])
        return memory

    def step(self, example: dict, step: int = 0) -> np.ndarray:
        state = example.get("state", None)
        state = None if state is None else np.asarray(state, dtype=np.float32).reshape(-1)
        state_in_model_order = self._robosyn_to_model_action_order(state) if state is not None else None

        if self.action_mode in {"delta", "rel"} and self.initial_state is None:
            if state_in_model_order is None:
                raise ValueError(f"action_mode={self.action_mode!r} requires state in the observation.")
            self.initial_state = state_in_model_order.copy()

        task_description = example.get("lang", "")
        if task_description != self.task_description:
            self.reset(task_description)
            if self.action_mode in {"delta", "rel"} and state_in_model_order is not None:
                self.initial_state = state_in_model_order.copy()

        images = [self._resize_image(_ensure_rgb_uint8(image)) for image in example["image"]]
        example["image"] = images
        if self.memory_enabled:
            example["memory"] = self._build_memory(step, images)
            example["step"] = int(step)

        example_copy = example.copy()
        if self.send_state and state_in_model_order is not None:
            example_copy["state"] = state_in_model_order.reshape(1, -1)
        else:
            example_copy.pop("state", None)
        example_copy.update(self._build_history_payload(images[0]))

        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "unnorm_key": self.unnorm_key,
        }

        # Execute only the configured prefix, then replan from the latest observation.
        # A local cursor also handles episode resets and shorter server responses.
        if self.raw_actions is None or self.action_chunk_index >= min(
            self.replan_action_steps, len(self.raw_actions)
        ):
            response = self.client.predict_action(vla_input)
            if response.get("ok") is False:
                raise RuntimeError(f"Policy server returned an error: {response.get('error', response)}")
            data = response["data"]
            if "actions" in data:
                raw_actions = np.asarray(data["actions"][0], dtype=np.float32)
            elif "normalized_actions" in data:
                raw_actions = np.asarray(data["normalized_actions"][0], dtype=np.float32)
                if self.actions_are_normalized:
                    raw_actions = self.unnormalize_actions(raw_actions)
            else:
                raise KeyError(f"Response data has no actions field: {data.keys()}")

            if raw_actions.ndim != 2 or len(raw_actions) == 0:
                raise ValueError("Policy server must return a non-empty (steps, action_dim) chunk")

            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state_in_model_order)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions
            self.action_chunk_index = 0

        self._push_history_frame(images[0])

        current_action = np.asarray(self.raw_actions[self.action_chunk_index], dtype=np.float32)
        self.action_chunk_index += 1
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        return self._model_to_robosyn_action_order(current_action)

    def unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        if self.action_norm_stats is None:
            raise ValueError("Cannot unnormalize actions because checkpoint action statistics are missing.")

        action_high, action_low = self._get_normalization_bounds(
            self.action_norm_stats,
            normalization_mode=self.normalization_mode,
        )
        action_high = np.asarray(action_high, dtype=np.float32)
        action_low = np.asarray(action_low, dtype=np.float32)
        mask = np.asarray(
            self.action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool)),
            dtype=bool,
        )

        # HIPPO Plus trains all dimensions with continuous min-max normalization.
        # The exported name-based gripper mask does not describe that transform.
        if self.normalize_all_action_dims:
            mask = np.ones_like(action_low, dtype=bool)

        actions = np.asarray(normalized_actions, dtype=np.float32).copy()
        clipped = np.clip(actions, -1.0, 1.0)
        if actions.shape[-1] != action_low.shape[0]:
            raise ValueError(
                f"Predicted action dim {actions.shape[-1]} does not match stats dim {action_low.shape[0]}."
            )
        actions[..., mask] = (
            0.5 * (clipped[..., mask] + 1.0) * (action_high[mask] - action_low[mask])
            + action_low[mask]
        )

        binary_indices = set(np.flatnonzero(~mask).tolist())
        binary_indices.update(idx for idx in self.binary_action_indices if 0 <= idx < actions.shape[-1])
        for idx in sorted(binary_indices):
            actions[..., idx] = (actions[..., idx] > self.binary_threshold).astype(actions.dtype)
        return actions

    @staticmethod
    def _get_normalization_bounds(
        norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> tuple[np.ndarray, np.ndarray]:
        if normalization_mode == "q99":
            if "q01" not in norm_stats or "q99" not in norm_stats:
                raise KeyError("Normalization mode 'q99' requires stats keys 'q01' and 'q99'.")
            return np.asarray(norm_stats["q99"]), np.asarray(norm_stats["q01"])
        if normalization_mode == "min_max":
            if "min" not in norm_stats or "max" not in norm_stats:
                raise KeyError("Normalization mode 'min_max' requires stats keys 'min' and 'max'.")
            return np.asarray(norm_stats["max"]), np.asarray(norm_stats["min"])
        raise ValueError("normalization_mode must be one of: min_max, q99")

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        if current_state is None:
            raise ValueError("Delta decoding requires measured joint state")
        return np.asarray(current_state, dtype=np.float32)[None, :] + np.cumsum(
            np.asarray(delta_actions, dtype=np.float32), axis=0
        )

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        return rel_actions + self.initial_state

    def _model_to_robosyn_action_order(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.model_action_order == "train" and action.shape[0] >= len(TRAIN_TO_ROBOSYN_ACTION_ORDER):
            return action[TRAIN_TO_ROBOSYN_ACTION_ORDER]
        return action

    def _robosyn_to_model_action_order(self, action: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if action is None:
            return None
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.model_action_order == "train" and action.shape[0] >= len(ROBOSYN_TO_TRAIN_ACTION_ORDER):
            return action[ROBOSYN_TO_TRAIN_ACTION_ORDER]
        return action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        h, w = self.image_size
        return cv.resize(image, (w, h), interpolation=cv.INTER_AREA)

    def close(self) -> None:
        self.close_video_episode(success=False)
        self.client.close()

    def start_video_episode(self) -> None:
        if not self.video_enabled or self.video_dir is None:
            return
        self.close_video_episode(success=False)
        self.video_episode_idx += 1
        self.video_frame_idx = 0
        self.video_closed = False
        self.video_tmp_dir = self.video_dir / f"episode_{self.video_episode_idx:03d}_pending_frames"
        self.video_episode_dir = self.video_tmp_dir
        self.video_episode_dir.mkdir(parents=True, exist_ok=True)

    def record_video_obs(self, observation: dict) -> None:
        if not self.video_enabled or self.video_episode_dir is None or self.video_closed:
            return
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            frame = _ensure_rgb_uint8(observation["sensor"][key]["color"])
            folder = self.video_episode_dir / key
            folder.mkdir(exist_ok=True)
            output_path = folder / f"frame_{self.video_frame_idx:06d}.{self.video_format}"
            options = [int(cv.IMWRITE_JPEG_QUALITY), 92] if self.video_format == "jpg" else []
            if not cv.imwrite(str(output_path), cv.cvtColor(frame, cv.COLOR_RGB2BGR), options):
                raise RuntimeError(f"Failed to write eval video frame: {output_path}")
        self.video_frame_idx += 1

    def close_video_episode(self, success: bool = False) -> None:
        if self.video_closed or self.video_episode_dir is None:
            return

        episode_dir = self.video_episode_dir
        if self.video_frame_idx > 0 and episode_dir.exists():
            status = "success" if success else "fail"
            final_dir = self.video_dir / f"episode_{self.video_episode_idx:03d}_{status}_frames"
            if final_dir.exists():
                raise FileExistsError(f"Refusing to overwrite video frames: {final_dir}")
            os.replace(episode_dir, final_dir)
            print(f"  Video frames saved: {final_dir}")
        elif episode_dir.exists():
            episode_dir.rmdir()

        self.video_episode_dir = None
        self.video_tmp_dir = None
        self.video_closed = True
        if self.video_frame_idx > 0:
            for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                target = self.video_dir / f"episode_{self.video_episode_idx:03d}_{status}_{key}.mp4"
                subprocess.run([
                    "ffmpeg", "-nostdin", "-n", "-loglevel", "error",
                    "-framerate", str(self.video_fps), "-i",
                    str(final_dir / key / f"frame_%06d.{self.video_format}"),
                    "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(target),
                ], check=True)

    def evaluate_task_success(self, observation: dict, env: Any) -> tuple[bool, bool]:
        """Return (success, authoritative) for task-specific eval metrics."""
        if self.task_success_latched:
            return True, self.task_success_mode != "official" or self.task_name == "water_pouring"

        if self.task_success_mode == "official" and self.task_name == "water_pouring":
            return self._official_water_pouring_success(env)

        if self.task_success_mode in {"water_pouring", "water_pouring_geometry"}:
            success = self._water_pouring_geometry_success(observation, env)
            if success:
                self.task_success_latched = True
                print(
                    "  water_pouring geometry success: "
                    + ", ".join(
                        f"{key}={value:.4f}" for key, value in self.last_success_metrics.items()
                    )
                )
            return self.task_success_latched, True

        return _env_is_success(env), False

    def _official_water_pouring_success(self, env: Any) -> tuple[bool, bool]:
        base_env = getattr(env, "unwrapped", env)
        evaluate_task_state = getattr(base_env, "_evaluate_task_state", None)
        if not callable(evaluate_task_state):
            return _env_is_success(env), False

        try:
            success, fail, detail = evaluate_task_state()
        except Exception as exc:
            if self.action_debug_interval > 0 and (
                self.env_step < 3 or (self.env_step + 1) % self.action_debug_interval == 0
            ):
                print(f"  water_pouring official success debug unavailable: {exc}")
            return _env_is_success(env), False

        detail = detail or {}
        success_bool = _as_bool(success)
        fail_bool = _as_bool(fail)
        self.last_success_metrics = {
            "step": float(self.env_step),
            "elapsed_steps": _first_scalar(getattr(base_env, "_elapsed_steps", None), self.env_step),
            "success": float(success_bool),
            "fail": float(fail_bool),
            "bottle_angle_deg": _first_scalar(detail.get("bottle_angle_deg")),
            "bottle_in_pour_range": float(_as_bool(detail.get("bottle_in_pour_range", False))),
            "pouring_started": float(_as_bool(detail.get("pouring_started", False))),
            "pouring_completed": float(_as_bool(detail.get("pouring_completed", False))),
            "bottle_fall": float(_as_bool(detail.get("bottle_fall", False))),
            "cup_fall": float(_as_bool(detail.get("cup_fall", False))),
            "schedule_ready": float(_as_bool(detail.get("schedule_ready", False))),
            "stand_end": _first_scalar(detail.get("stand_end"), -1.0),
            "stand_reached": float(_as_bool(detail.get("stand_reached", False))),
        }

        if self.action_debug_interval > 0 and (
            success_bool or self.env_step < 3 or (self.env_step + 1) % self.action_debug_interval == 0
        ):
            print(
                "  water_pouring official success debug: "
                f"step={self.env_step}, "
                f"elapsed={self.last_success_metrics['elapsed_steps']:.0f}, "
                f"success={int(success_bool)}, fail={int(fail_bool)}, "
                f"bottle_angle_deg={self.last_success_metrics['bottle_angle_deg']:.2f}, "
                f"in_pour_range={int(self.last_success_metrics['bottle_in_pour_range'])}, "
                f"pour_started={int(self.last_success_metrics['pouring_started'])}, "
                f"pour_completed={int(self.last_success_metrics['pouring_completed'])}, "
                f"bottle_fall={int(self.last_success_metrics['bottle_fall'])}, "
                f"cup_fall={int(self.last_success_metrics['cup_fall'])}, "
                f"schedule_ready={int(self.last_success_metrics['schedule_ready'])}, "
                f"stand_end={self.last_success_metrics['stand_end']:.0f}, "
                f"stand_reached={int(self.last_success_metrics['stand_reached'])}"
            )

        if success_bool:
            self.task_success_latched = True
        return success_bool, True

    def _water_pouring_geometry_success(self, observation: dict, env: Any) -> bool:
        bottle_pose = self._get_object_pose_matrix(observation, env, "bottle", "bottle_pose")
        cup_pose = self._get_object_pose_matrix(observation, env, "cup", "cup_pose")

        bottle_pos = bottle_pose[:3, 3]
        cup_pos = cup_pose[:3, 3]
        rel = bottle_pos - cup_pos
        xy_dist = float(np.linalg.norm(rel[:2]))
        dz = float(rel[2])

        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        bottle_y_zdot = float(np.dot(bottle_pose[:3, 1], world_z))
        cup_z_zdot = float(np.dot(cup_pose[:3, 2], world_z))
        bottle_tilt_deg = float(
            np.degrees(np.arccos(np.clip(bottle_y_zdot, -1.0, 1.0)))
        )
        cup_tilt_deg = float(
            np.degrees(np.arccos(np.clip(cup_z_zdot, -1.0, 1.0)))
        )

        enough_steps = self.env_step >= self.water_pouring_success_min_steps
        close_xy = xy_dist <= self.water_pouring_success_max_xy
        close_z = self.water_pouring_success_z_min <= dz <= self.water_pouring_success_z_max
        bottle_tilted = bottle_tilt_deg >= self.water_pouring_success_min_tilt_deg
        cup_upright = cup_tilt_deg <= self.water_pouring_success_cup_upright_max_deg

        self.last_success_metrics = {
            "step": float(self.env_step),
            "xy": xy_dist,
            "dz": dz,
            "bottle_tilt_deg": bottle_tilt_deg,
            "cup_tilt_deg": cup_tilt_deg,
        }
        if self.action_debug_interval > 0 and (
            self.env_step < 3 or (self.env_step + 1) % self.action_debug_interval == 0
        ):
            print(
                "  water_pouring success debug: "
                f"step={self.env_step}, xy={xy_dist:.4f}, dz={dz:.4f}, "
                f"bottle_tilt_deg={bottle_tilt_deg:.2f}, "
                f"cup_tilt_deg={cup_tilt_deg:.2f}, "
                f"checks=steps:{int(enough_steps)} xy:{int(close_xy)} "
                f"z:{int(close_z)} tilt:{int(bottle_tilted)} cup:{int(cup_upright)}"
            )

        return enough_steps and close_xy and close_z and bottle_tilted and cup_upright

    def _get_object_pose_matrix(
        self,
        observation: dict,
        env: Any,
        object_uid: str,
        obs_key: str,
    ) -> np.ndarray:
        pose = observation.get(obs_key)
        if pose is None:
            sim = getattr(getattr(env, "unwrapped", env), "sim", None)
            if sim is None:
                raise KeyError(f"Observation is missing {obs_key!r} and env.sim is unavailable.")
            pose = sim.get_rigid_object(object_uid).get_local_pose(to_matrix=True)

        pose_np = _as_numpy(pose)
        if pose_np.ndim == 3:
            pose_np = pose_np[0]
        if pose_np.shape != (4, 4):
            raise ValueError(f"Expected {obs_key} pose matrix with shape (4, 4), got {pose_np.shape}.")
        return np.asarray(pose_np, dtype=np.float32)

    def install_success_gate(self, env: Any) -> None:
        if self.success_min_steps <= 0 or self._success_gate_installed:
            return

        original_get_wrapper_attr = env.get_wrapper_attr
        model = self

        def gated_get_wrapper_attr(name: str):
            attr = original_get_wrapper_attr(name)
            if name != "is_task_success":
                return attr

            def gated_is_task_success(*args, **kwargs):
                value = attr(*args, **kwargs)
                if model.env_step < model.success_min_steps:
                    if isinstance(value, torch.Tensor):
                        return torch.zeros_like(value, dtype=torch.bool)
                    return False
                return value

            return gated_is_task_success

        env.get_wrapper_attr = gated_get_wrapper_attr
        self._success_gate_installed = True

    def apply_camera_overrides(self, env: Any) -> bool:
        if not self.camera_overrides or self._camera_overrides_applied:
            return False

        unwrapped = getattr(env, "unwrapped", env)
        sim = getattr(unwrapped, "sim", None)
        device = getattr(unwrapped, "device", torch.device("cpu"))
        if sim is None:
            raise RuntimeError("Cannot apply camera overrides because env.unwrapped.sim is missing.")

        for camera_uid, override in self.camera_overrides.items():
            if not isinstance(override, dict):
                raise ValueError(f"Camera override for {camera_uid!r} must be a mapping.")
            try:
                camera = sim.get_sensor(camera_uid)
            except Exception as exc:
                raise KeyError(f"Camera override references unknown sensor {camera_uid!r}.") from exc

            eye = override.get("eye")
            target = override.get("target")
            up = override.get("up", [0.0, 0.0, 1.0])
            if eye is None or target is None:
                raise ValueError(f"Camera override for {camera_uid!r} requires eye and target.")
            camera.look_at(
                torch.tensor([eye], dtype=torch.float32, device=device),
                torch.tensor([target], dtype=torch.float32, device=device),
                torch.tensor([up], dtype=torch.float32, device=device),
            )

        self._camera_overrides_applied = True
        print(f"*** camera overrides applied: {list(self.camera_overrides.keys())} ***")
        return True

    def log_action_debug(
        self,
        action: np.ndarray,
        state_before: Optional[np.ndarray],
        state_after: Optional[np.ndarray],
    ) -> None:
        if self.action_debug_interval <= 0:
            return
        if self.env_step >= 3 and (self.env_step + 1) % self.action_debug_interval != 0:
            return

        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        finite_action = action_array[np.isfinite(action_array)]
        if finite_action.size == 0:
            action_min = action_max = float("nan")
        else:
            action_min = float(np.min(finite_action))
            action_max = float(np.max(finite_action))

        delta_msg = "nan"
        if state_before is not None and state_after is not None:
            before = np.asarray(state_before, dtype=np.float32).reshape(-1)
            after = np.asarray(state_after, dtype=np.float32).reshape(-1)
            if before.shape == after.shape:
                delta = after - before
                finite_delta = delta[np.isfinite(delta)]
                if finite_delta.size > 0:
                    delta_msg = f"{float(np.max(np.abs(finite_delta))):.6f}"

        print(
            "  action debug: "
            f"env_step={self.env_step + 1}, "
            f"action_dim={action_array.size}, "
            f"action_min={action_min:.6f}, action_max={action_max:.6f}, "
            f"qpos_delta_linf={delta_msg}"
        )


def encode_obs(observation: dict, env: Any = None) -> tuple[list[np.ndarray], np.ndarray]:
    """Convert a RoboSynChallenge Gymnasium observation to StarVLA inputs."""
    try:
        img_front_raw = observation["sensor"]["cam_high"]["color"]
        img_left_raw = observation["sensor"]["cam_left_wrist"]["color"]
        img_right_raw = observation["sensor"]["cam_right_wrist"]["color"]
        state_raw = observation["robot"]["qpos"]
    except KeyError as exc:
        raise KeyError(
            "Expected RoboSynChallenge observation keys: "
            "sensor/cam_high/color, sensor/cam_left_wrist/color, "
            "sensor/cam_right_wrist/color, robot/qpos"
        ) from exc

    img_front = _ensure_rgb_uint8(img_front_raw)
    img_left = _ensure_rgb_uint8(img_left_raw)
    img_right = _ensure_rgb_uint8(img_right_raw)
    state = _extract_first_env_vector(state_raw, env=env)

    return [img_front, img_left, img_right], state


def map_gripper_to_physical(action_tensor, env, mode):
    """Map reconstructed absolute gripper targets, never normalized model deltas."""
    if mode == "physical":
        return action_tensor
    if mode != "ratio":
        raise ValueError(f"Unsupported gripper action units: {mode}")
    unwrapped = env.unwrapped
    active = list(unwrapped.active_joint_ids)
    result = action_tensor.clone()
    mapped = []
    for part in ("left_eef", "right_eef"):
        joints = unwrapped.robot.get_joint_ids(name=part, remove_mimic=True)
        for joint_id in joints:
            if joint_id not in active:
                continue
            index = active.index(joint_id)
            limits = unwrapped.robot.body_data.qpos_limits[0, joint_id].to(result)
            low, high = limits[0], limits[1]
            if not bool(torch.isfinite(limits).all()) or not bool(high > low):
                raise ValueError(f"Invalid gripper joint limits: {part} {joint_id}")
            result[..., index] = low + result[..., index].clamp(0.0, 1.0) * (high-low)
            mapped.append(index)
    if len(mapped) != 2:
        raise ValueError(f"Expected two active CobotMagic grippers, got {mapped}")
    return result


def encode_action(action: np.ndarray, env: Any, gripper_action_units: str = "physical") -> torch.Tensor:
    """Convert a StarVLA action vector to EmbodiChain's action tensor."""
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    unwrapped = env.unwrapped
    env_action_dim = int(np.prod(unwrapped.single_action_space.shape))

    if action_array.shape[0] < env_action_dim:
        raise ValueError(
            f"Policy action has dim {action_array.shape[0]}, but env expects {env_action_dim}."
        )
    if action_array.shape[0] > env_action_dim:
        active_joint_ids = getattr(unwrapped, "active_joint_ids", None)
        if active_joint_ids is not None:
            active_joint_ids = list(active_joint_ids)
            if len(active_joint_ids) == env_action_dim and max(active_joint_ids) < action_array.shape[0]:
                action_array = action_array[active_joint_ids]
            else:
                action_array = action_array[:env_action_dim]
        else:
            action_array = action_array[:env_action_dim]

    action_tensor = torch.as_tensor(
        action_array,
        dtype=torch.float32,
        device=unwrapped.device,
    )
    return map_gripper_to_physical(action_tensor.unsqueeze(0), env, gripper_action_units)


def get_model(usr_args: dict) -> ModelClient:
    """Construct the StarVLA client from ``deploy_policy.yml`` and CLI overrides."""
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config or eval.sh arguments.")

    task_name = str(usr_args.get("task_name", ""))
    success_min_steps = usr_args.get("success_min_steps", 0)
    success_min_steps_by_task = usr_args.get("success_min_steps_by_task", {}) or {}
    if isinstance(success_min_steps_by_task, dict) and task_name in success_min_steps_by_task:
        success_min_steps = success_min_steps_by_task[task_name]

    camera_overrides = usr_args.get("camera_overrides", {}) or {}
    camera_overrides_by_task = usr_args.get("camera_overrides_by_task", {}) or {}
    if isinstance(camera_overrides_by_task, dict) and task_name in camera_overrides_by_task:
        camera_overrides = camera_overrides_by_task[task_name] or {}

    task_success_mode = usr_args.get("task_success_mode", "official")
    task_success_mode_by_task = usr_args.get("task_success_mode_by_task", {}) or {}
    if isinstance(task_success_mode_by_task, dict) and task_name in task_success_mode_by_task:
        task_success_mode = task_success_mode_by_task[task_name]

    model = ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 5694)),
        unnorm_key=usr_args.get("unnorm_key", None),
        policy_setup=usr_args.get("policy_setup", "robosynchallenge"),
        action_mode=usr_args.get("action_mode", "abs"),
        normalization_mode=usr_args.get(
            "action_normalization_mode",
            usr_args.get("normalization_mode", "min_max"),
        ),
        normalize_all_action_dims=_parse_bool(usr_args.get("normalize_all_action_dims", False), default=False),
        actions_are_normalized=_parse_bool(
            usr_args.get("actions_are_normalized", True),
            default=True,
        ),
        model_action_order=usr_args.get("model_action_order", "train"),
        send_state=_parse_bool(usr_args.get("send_state", False), default=False),
        binary_action_indices=usr_args.get("binary_action_indices", DEFAULT_BINARY_ACTION_INDICES),
        binary_threshold=float(usr_args.get("binary_threshold", 0.5)),
        eval_action_steps=_parse_optional_int(usr_args.get("eval_action_steps", None)),
        replan_action_steps=_parse_optional_int(usr_args.get("replan_action_steps", None)),
        image_size=usr_args.get("image_size", [224, 224]),
        use_ddim=_parse_bool(usr_args.get("use_ddim", True), default=True),
        num_ddim_steps=int(usr_args.get("num_ddim_steps", 10)),
        history_k=int(usr_args.get("history_k", 0)),
        history_stride=int(usr_args.get("history_stride", 1)),
        history_mode=usr_args.get("history_mode", "flow"),
        history_image_size=usr_args.get("history_image_size", None),
        history_flow_compute_size=usr_args.get("history_flow_compute_size", None),
        memory_enabled=_parse_bool(usr_args.get("memory_enabled", True), default=True),
        memory_max_step=int(usr_args.get("memory_max_step", usr_args.get("max_memory_step", 5))),
        memory_interval=int(usr_args.get("memory_interval", 10)),
        starvla_eval_video_log=_parse_bool(
            usr_args.get("starvla_eval_video_log", False),
            default=False,
        ),
        starvla_video_dir=usr_args.get("starvla_video_dir", None),
        eval_video_obs_key=usr_args.get("eval_video_obs_key", "cam_high"),
        starvla_video_fps=int(usr_args.get("starvla_video_fps", 10)),
        starvla_video_format=usr_args.get("starvla_video_format", "jpg"),
        success_min_steps=int(success_min_steps),
        task_name=task_name,
        task_success_mode=task_success_mode,
        water_pouring_success_min_steps=int(usr_args.get("water_pouring_success_min_steps", 80)),
        water_pouring_success_max_xy=float(usr_args.get("water_pouring_success_max_xy", 0.30)),
        water_pouring_success_z_min=float(usr_args.get("water_pouring_success_z_min", -0.10)),
        water_pouring_success_z_max=float(usr_args.get("water_pouring_success_z_max", 0.15)),
        water_pouring_success_min_tilt_deg=float(usr_args.get("water_pouring_success_min_tilt_deg", 45.0)),
        water_pouring_success_cup_upright_max_deg=float(
            usr_args.get("water_pouring_success_cup_upright_max_deg", 25.0)
        ),
        camera_overrides=camera_overrides,
        action_debug_interval=int(usr_args.get("action_debug_interval", 0)),
    )

    task_units = usr_args.get("gripper_action_units_by_task", {}) or {}
    model.gripper_action_units = task_units.get(task_name, usr_args.get("gripper_action_units", "physical"))
    if model.gripper_action_units not in {"physical", "ratio"}:
        raise ValueError("gripper_action_units must be physical or ratio")
    print(f"Gripper output mapping: task={task_name}, units={model.gripper_action_units}")
    return model


def reset_model(model: ModelClient) -> None:
    model.reset(task_description="")
    model.start_video_episode()


def eval(env: Any, model: ModelClient, observation: dict):
    """Run one StarVLA action chunk in RoboSynChallenge and return the final observation."""
    model.install_success_gate(env)
    camera_changed = model.apply_camera_overrides(env)
    instruction = _get_env_instruction(env)
    final_obs = env.unwrapped.get_obs() if camera_changed else observation
    info: dict[str, Any] = {}
    truncated: Any = False
    success_now = False
    success_authoritative = False

    if model.video_enabled and model.video_frame_idx == 0:
        model.record_video_obs(final_obs)

    for _ in range(model.eval_action_steps):
        images, state = encode_obs(final_obs, env=env)
        example = {
            "lang": instruction,
            "image": images,
            "state": state,
        }
        action = model.step(example, step=model.env_step)
        action_tensor = encode_action(action, env, getattr(model, "gripper_action_units", "physical"))
        state_before = state
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        try:
            state_after = _extract_first_env_vector(final_obs["robot"]["qpos"], env=env)
        except Exception:
            state_after = None
        model.record_video_obs(final_obs)
        model.log_action_debug(action, state_before, state_after)
        model.env_step += 1

        success_now, success_authoritative = model.evaluate_task_success(final_obs, env)
        if _as_bool(truncated) or _as_bool(terminated) or success_now:
            model.close_video_episode(success=success_now)
            break

    info = dict(info)
    info["starvla_env_steps"] = int(model.env_step)
    info["starvla_success"] = bool(success_now)
    info["starvla_success_authoritative"] = bool(success_authoritative)
    info["starvla_success_metrics"] = dict(model.last_success_metrics)
    return final_obs, info, truncated
