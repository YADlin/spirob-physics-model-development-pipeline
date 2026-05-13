import mujoco as mj
from mujoco.viewer import launch_passive
import numpy as np
import time
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import defaultdict
from pynput import keyboard

XML_PATH = r'./scene/spirob_physics_model_with_touch_sensor.xml'

# ====================== USER CONFIG ======================
SELECTED_LINKS = [6, 7]   # [] = all links
WINDOW = 250

PLOT_EVERY = 20           # redraw plot every N sim steps
SAMPLE_EVERY = 3          # sample sensor every N sim steps
# =========================================================

# ---------------- Keyboard ----------------
pressed_keys = set()

def is_ctrl_pressed():

    return (
        keyboard.Key.ctrl in pressed_keys
        or keyboard.Key.ctrl_l in pressed_keys
        or keyboard.Key.ctrl_r in pressed_keys
    )

def on_press(key):

    try:
        pressed_keys.add(key.char.lower())

    except AttributeError:
        pressed_keys.add(key)

def on_release(key):

    try:
        pressed_keys.discard(key.char.lower())

    except AttributeError:
        pressed_keys.discard(key)

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)

listener.start()


# -------- Load model --------
model = mj.MjModel.from_xml_path(XML_PATH)
data = mj.MjData(model)

# -------- Parse XML --------
tree = ET.parse(XML_PATH)

# grouped_sensors[link_id] = [(sensor_name, contact_name)]
grouped_sensors = defaultdict(list)

for sensor in tree.findall(".//sensor/touch"):

    name = sensor.get("name")

    if not name:
        continue

    try:
        # Example:
        # touch_006_c0
        parts = name.split("_")

        link_id = int(parts[1])
        contact_name = parts[2]

        if not SELECTED_LINKS or link_id in SELECTED_LINKS:
            grouped_sensors[link_id].append((name, contact_name))

    except Exception:
        pass

print("\nDetected sensors:")
for link_id, sensors in grouped_sensors.items():
    print(f"Link {link_id}: {[s[1] for s in sensors]}")

# -------- Cache sensor IDs --------
sensor_ids = {}

for link_id, sensors in grouped_sensors.items():

    for sensor_name, contact_name in sensors:

        sid = mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_SENSOR,
            sensor_name
        )

        sensor_ids[sensor_name] = sid

# -------- Plot setup --------
plt.ion()

n_links = len(grouped_sensors)

if n_links == 1:

    fig, ax = plt.subplots(figsize=(10, 5))
    axes = [ax]

else:

    n_cols = 2
    n_rows = (n_links + 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 4 * n_rows),
        constrained_layout=True
    )

    axes = np.array(axes).flatten()

colors = [
    'red',
    'cyan',
    'lime',
    'orange',
    'magenta',
    'yellow'
]

xdata = np.arange(WINDOW)

history = {}
lines = {}

# -------- One subplot per link --------
for ax_idx, (link_id, sensors) in enumerate(grouped_sensors.items()):

    ax = axes[ax_idx]

    for i, (sensor_name, contact_name) in enumerate(sensors):

        # Fixed-size rolling buffer
        history[sensor_name] = np.zeros(WINDOW)

        line, = ax.plot(
            xdata,
            history[sensor_name],
            label=contact_name,
            color=colors[i % len(colors)],
            linewidth=2
        )

        lines[sensor_name] = line

    ax.set_title(f"Link {link_id:03d}")

    # IMPORTANT:
    # set only ONCE
    ax.set_xlim(0, WINDOW)
    ax.set_ylim(0, 6)

    ax.grid(True, alpha=0.3)
    ax.legend()

# Hide unused axes
for i in range(n_links, len(axes)):
    axes[i].axis('off')

fig.suptitle(
    "SpiRob Touch Sensor Monitor",
    fontsize=14
)


# ---------------- Controls ----------------
nominal_length = 0.22

ctrl_min = model.actuator_ctrlrange[:, 0]
ctrl_max = model.actuator_ctrlrange[:, 1]

slider = delta0 = delta1 = 0.0
step_size = 0.0005

data.ctrl[0] = nominal_length
data.ctrl[1] = nominal_length

for _ in range(1000):
    mj.mj_step(model, data)

# ---------------- Main Loop ----------------
with launch_passive(model, data) as viewer:

    step = 0

    while viewer.is_running():

        # -------- Controls --------
        if keyboard.Key.left in pressed_keys:
            slider -= step_size

        if keyboard.Key.right in pressed_keys:
            slider += step_size

        if (
            keyboard.Key.up in pressed_keys
            and not is_ctrl_pressed()
        ):
            delta0 += step_size

        if (
            keyboard.Key.down in pressed_keys
            and not is_ctrl_pressed()
        ):
            delta0 -= step_size

        if (
            is_ctrl_pressed()
            and keyboard.Key.up in pressed_keys
        ):
            delta1 += step_size

        if (
            is_ctrl_pressed()
            and keyboard.Key.down in pressed_keys
        ):
            delta1 -= step_size

        if 'r' in pressed_keys:
            slider = delta0 = delta1 = 0.0

        ctrl0 = nominal_length + slider + delta0
        ctrl1 = nominal_length - slider + delta1

        data.ctrl[0] = np.clip(
            ctrl0,
            ctrl_min[0],
            ctrl_max[0]
        )

        data.ctrl[1] = np.clip(
            ctrl1,
            ctrl_min[1],
            ctrl_max[1]
        )

        # -------- Physics Step --------
        physics_steps_per_frame = 50

        for _ in range(physics_steps_per_frame):
            mj.mj_step(model, data)

        # ==================================================
        # FAST SENSOR UPDATE
        # ==================================================
        if step % SAMPLE_EVERY == 0:

            for link_id, sensors in grouped_sensors.items():

                for sensor_name, contact_name in sensors:

                    sid = sensor_ids[sensor_name]

                    if sid < 0:
                        continue

                    val = data.sensordata[sid]

                    # Fast rolling buffer
                    history[sensor_name] = np.roll(
                        history[sensor_name],
                        -1
                    )

                    history[sensor_name][-1] = val

                    line = lines[sensor_name]

                    # MUCH faster than set_data()
                    line.set_ydata(
                        history[sensor_name]
                    )

                    # Contact highlight
                    if val > 0.01:
                        line.set_linewidth(4)
                    else:
                        line.set_linewidth(2)

        # ==================================================
        # FAST PLOT REFRESH
        # ==================================================
        if step % PLOT_EVERY == 0:

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        viewer.sync()

        time.sleep(1/60)

        step += 1

plt.close(fig)

print("Session ended.")
