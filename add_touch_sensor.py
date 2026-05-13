# Adds touch sensor of adaptive size (sphere) along the tendon location (middle) + some offset
# given by OFFSET_MARGIN. Also allows you to select the link number on which to add the sensors.
import xml.etree.ElementTree as ET
import numpy as np

INPUT_FILE = r".\scene\spirob_scene_base.xml"
OUTPUT_FILE = r".\scene\spirob_physics_model_with_touch_sensor.xml"

# ====================== CONFIGURATION ======================
OFFSET_MARGIN = 0.0018          # Radial outward margin in meters

# Select which links to add sensors to:
# - Use [] for ALL links
# # - Or specify list like [1, 2, 3, 10, 15, 21]
# SELECTED_LINKS = [1, 5, 10, 15, 21]     # Only these links
# SELECTED_LINKS = list(range(1, 11))     # Links 1 to 10
# SELECTED_LINKS = [21]                   # Only the tip link
SELECTED_LINKS = []             # Empty = All links

# =========================================================

def get_link_index(name):
    try:
        return int(name.split("_")[1])
    except:
        return -1


def parse_site_pos(site_elem):
    pos_str = site_elem.get("pos", "0 0 0")
    return np.array([float(x) for x in pos_str.split()])

def indent_xml(elem, level=0):

    indent = "\n" + level * "    "

    if len(elem):

        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "

        for child in elem:
            indent_xml(child, level + 1)

        if not child.tail or not child.tail.strip():
            child.tail = indent

    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent

def main():
    tree = ET.parse(INPUT_FILE)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    link_bodies = [body for body in worldbody.iter("body") 
                   if body.get("name", "").startswith("link_")]
    link_bodies.sort(key=lambda b: get_link_index(b.get("name")))

    sensor_elem = ET.Element("sensor")
    total_sensors = 0
    processed_links = 0

    for body in link_bodies:
        link_name = body.get("name")
        link_id = get_link_index(link_name)

        # Skip links not in SELECTED_LINKS (if list is not empty)
        if SELECTED_LINKS and link_id not in SELECTED_LINKS:
            continue

        processed_links += 1

        cable_groups = {}
        for site in body.findall("site"):
            sname = site.get("name", "").strip()
            if sname.startswith("c") and f"_{link_id:03d}_s" in sname:
                try:
                    cable_id = int(sname.split("_")[0][1:])
                    cable_groups.setdefault(cable_id, []).append((sname, site))
                except:
                    continue

        for cable_id, site_list in sorted(cable_groups.items()):
            s1 = s2 = None
            for name, site in site_list:
                if name.endswith("_s1"):
                    s1 = site
                elif name.endswith("_s2"):
                    s2 = site

            if s1 is None or s2 is None:
                continue

            pos1 = parse_site_pos(s1)
            pos2 = parse_site_pos(s2)
            
            midpoint = (pos1 + pos2) / 2.0

            # Radial outward offset
            radius = np.linalg.norm(midpoint[:2])
            if radius > 1e-6:
                radial_unit = midpoint[:2] / radius
                new_radius = radius + OFFSET_MARGIN
                new_xy = radial_unit * new_radius
            else:
                new_xy = midpoint[:2]

            final_pos = np.array([new_xy[0], new_xy[1], midpoint[2]])

            # Dynamic size
            half_distance = float(np.linalg.norm(pos1 - pos2) / 2.0)

            site_name = f"cs_{link_id:03d}_c{cable_id}"
            pos_str = f"{final_pos[0]:.8f} {final_pos[1]:.8f} {final_pos[2]:.8f}"

            touch_site = ET.Element("site", {
                "name": site_name,
                "pos": pos_str,
                "size": f"{half_distance:.6f}",
                "rgba": "1 0 0 0.3",
                "group": "3"
            })
            body.append(touch_site)

            sensor_name = f"touch_{link_id:03d}_c{cable_id}"
            touch = ET.Element("touch", {"name": sensor_name, "site": site_name})
            sensor_elem.append(touch)

            total_sensors += 1

    # root.append(sensor_elem)
    # tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
    
    # Remove old sensor block if present
    old_sensor = root.find("sensor")

    if old_sensor is not None:
        root.remove(old_sensor)
        
    # Add sensor block to root
    root.append(sensor_elem)

    # Apply indentation to entire XML tree
    indent_xml(root)

    # Write formatted XML
    tree.write(
        OUTPUT_FILE,
        encoding="utf-8",
        xml_declaration=True
    )


    print(f"✅ Done!")
    print(f"Links processed     : {processed_links}")
    print(f"Total touch sensors : {total_sensors}")
    print(f"Input  : {INPUT_FILE}")
    print(f"Output : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
