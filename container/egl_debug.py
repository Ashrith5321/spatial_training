"""EGL + habitat_sim init debug for the Great Lakes SIGSEGV at sim init.

Run inside longnav-rl.sif on a GPU node, in the `vln` conda env:

    MAGNUM_LOG=verbose HABITAT_SIM_LOG=debug python egl_debug.py

Stage 1 enumerates EGL devices straight through libEGL (glvnd), so we can
see whether the NVIDIA vendor ICD is being picked up at all.
Stage 2 creates a minimal habitat_sim.Simulator (NONE stage + one 64x64 RGB
sensor) -- the exact headless-EGL-context step that segfaults in
HabitatEnvActor, isolated from Ray/longnav.
"""

import ctypes
import os
import sys

# EGL constants
EGL_EXTENSIONS = 0x3055
EGL_VENDOR = 0x3053
EGL_VERSION = 0x3054
EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_CUDA_DEVICE_NV = 0x323A
EGL_PLATFORM_DEVICE_EXT = 0x313F


def stage1_enumerate_egl():
    print("=== stage 1: raw EGL device enumeration ===", flush=True)
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "__EGL_VENDOR_LIBRARY_DIRS",
        "LD_LIBRARY_PATH",
    ):
        print(f"  env {var}={os.environ.get(var)}", flush=True)

    try:
        egl = ctypes.CDLL("libEGL.so.1")
    except OSError as e:
        print(f"FAIL: cannot dlopen libEGL.so.1: {e}", flush=True)
        return False

    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglQueryString.restype = ctypes.c_char_p
    egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]

    client_ext = egl.eglQueryString(None, EGL_EXTENSIONS)
    print(f"  client extensions: {client_ext}", flush=True)

    addr = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
    if not addr:
        print(
            "FAIL: eglQueryDevicesEXT missing -> no vendor ICD with device "
            "enumeration loaded (NVIDIA ICD json almost certainly absent)",
            flush=True,
        )
        return False
    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )(addr)
    query_dev_string = ctypes.CFUNCTYPE(
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int
    )(egl.eglGetProcAddress(b"eglQueryDeviceStringEXT"))
    get_platform_display = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p
    )(egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT"))

    devices = (ctypes.c_void_p * 16)()
    num = ctypes.c_int(0)
    if not query_devices(16, devices, ctypes.byref(num)):
        print("FAIL: eglQueryDevicesEXT call failed", flush=True)
        return False
    print(f"  {num.value} EGL device(s):", flush=True)

    nvidia_seen = False
    for i in range(num.value):
        dev = devices[i]
        ext = query_dev_string(dev, EGL_EXTENSIONS) or b"?"
        drm = query_dev_string(dev, EGL_DRM_DEVICE_FILE_EXT)
        print(f"  device {i}: extensions={ext.decode()} drm={drm}", flush=True)
        if b"NV" in ext or b"nvidia" in ext.lower():
            nvidia_seen = True

        disp = get_platform_display(EGL_PLATFORM_DEVICE_EXT, dev, None)
        if not disp:
            print(f"    -> no display from this device", flush=True)
            continue
        major, minor = ctypes.c_int(0), ctypes.c_int(0)
        egl.eglInitialize.restype = ctypes.c_uint
        egl.eglInitialize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        if egl.eglInitialize(disp, ctypes.byref(major), ctypes.byref(minor)):
            vendor = egl.eglQueryString(disp, EGL_VENDOR)
            version = egl.eglQueryString(disp, EGL_VERSION)
            print(
                f"    -> initialized EGL {major.value}.{minor.value}, "
                f"vendor={vendor}, version={version}",
                flush=True,
            )
            egl.eglTerminate.argtypes = [ctypes.c_void_p]
            egl.eglTerminate(disp)
        else:
            print(f"    -> eglInitialize FAILED on this device", flush=True)

    if not nvidia_seen:
        print(
            "RESULT: no NVIDIA EGL device visible -> vendor ICD not loaded; "
            "habitat will land on mesa/swrast and crash",
            flush=True,
        )
    else:
        print("RESULT: NVIDIA EGL device present", flush=True)
    return nvidia_seen


def stage2_habitat_sim():
    print("=== stage 2: minimal habitat_sim.Simulator init ===", flush=True)
    import habitat_sim

    print(
        f"  habitat_sim {habitat_sim.__version__} from {habitat_sim.__file__}",
        flush=True,
    )
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.gpu_device_id = 0

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [64, 64]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    print("  creating Simulator (this is the step that segfaults)...", flush=True)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    obs = sim.get_sensor_observations()
    print(f"  OK: EGL context created, rgb obs shape={obs['rgb'].shape}", flush=True)
    sim.close()
    return True


if __name__ == "__main__":
    ok = stage1_enumerate_egl()
    print(flush=True)
    stage2_habitat_sim()
    print("ALL STAGES PASSED", flush=True)
    sys.exit(0 if ok else 1)
