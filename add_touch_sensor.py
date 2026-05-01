import xml.etree.ElementTree as ET

INPUT_FILE = "spirob_physics_model.xml"
OUTPUT_FILE = "spirob_physics_model_with_touch_sensor.xml"


def get_link_index(name):
    """Extract numeric index from 'link_XXX'."""
    return int(name.split("_")[1])


def main():
    tree = ET.parse(INPUT_FILE)
    root = tree.getroot()

    # Find worldbody
    worldbody = root.find("worldbody")

    link_bodies = []

    # Recursively find all bodies with name link_XXX
    for body in worldbody.iter("body"):
        name = body.get("name")
        if name and name.startswith("link_"):
            link_bodies.append(body)

    # Sort links numerically
    link_bodies.sort(key=lambda b: get_link_index(b.get("name")))

    # Create sensor element
    sensor_elem = ET.Element("sensor")

    for i, body in enumerate(link_bodies, start=1):
        link_id = get_link_index(body.get("name"))

        # Find geom inside body
        geom = body.find("geom")
        if geom is None:
            continue

        # Create site at geom local origin
        site_name = f"cs_{link_id:03d}"
        site = ET.Element("site", {
            "name": site_name,
            "pos": "0 0 0.002",
            "size": "0.002",
            "rgba": "1 0 0 0.5",
            "group" : "3"
        })

        body.append(site)

        # Create touch sensor
        touch = ET.Element("touch", {
            "name": f"touch_{link_id:03d}",
            "site": site_name
        })

        sensor_elem.append(touch)

    # Append sensor block at root level (after actuator ideally)
    root.append(sensor_elem)

    # Write output
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

    print(f"Processed {len(link_bodies)} links.")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
