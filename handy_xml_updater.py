import xml.etree.ElementTree as ET

# ------------------------------------------------------------
# USER PARAMETERS (YOU ONLY MODIFY THESE)
# ------------------------------------------------------------

# Separate manually assigned worldbody->link_001 joint values
first_joint_stiffness = 100
first_joint_damping   = 50

# Beta rule parameters (for subsequent joints)
beta = 1.03
base_stiffness = 0.2     # stiffness of joint_002
base_damping   = 0.01     # damping   of joint_002

# Body transform modification
new_link_pos  = "0 0 0.22"  # "0 0 0.22"
new_link_quat = "0 0 -1 0"

# Site positions
target_site_pos = "0.02 0.0 0.04"
tip_site_pos    = "0 0 0.005"

# ------------------------------------------------------------
# Load XML
# ------------------------------------------------------------

tree = ET.parse("spirob_physics_model_new.xml")
root = tree.getroot()

# ------------------------------------------------------------
# 1. Update top-level default joint (optional)
# ------------------------------------------------------------

default_joint = root.find(".//default/joint")
if default_joint is not None:
    # You can choose to override or keep — here we set to first joint values
    default_joint.set("damping", str(base_damping))
    default_joint.set("stiffness", str(base_stiffness))

# ------------------------------------------------------------
# 2. Update joints:
#    - Joint 0 → manual values
#    - Joint 1..n → beta-based sequence
# ------------------------------------------------------------

joints = root.findall(".//joint")

if len(joints) > 0:
    # Apply beta rule to remaining joints
    k = base_stiffness
    d = base_damping

    for j in joints[0:]:
        j.set("stiffness", str(k))
        j.set("damping", str(d))

        # Update for next joint
        k = k / (beta**3)
        d = d / (beta**3)

    # First joint gets manually assigned stiffness & damping
    first_joint = joints[0]
    first_joint.set("stiffness", str(first_joint_stiffness))
    first_joint.set("damping", str(first_joint_damping))

# ------------------------------------------------------------
# 3. Edit body pos/quat for link_001
# ------------------------------------------------------------

body = root.find(".//body[@name='link_001']")
if body is not None:
    body.set("pos", new_link_pos)
    body.set("quat", new_link_quat)

# ------------------------------------------------------------
# 4. Add 'target' site in <worldbody>
# ------------------------------------------------------------

worldbody = root.find(".//worldbody")
if worldbody is not None:
    ET.SubElement(worldbody, "site",
                  name="target",
                  pos=target_site_pos,
                  size="0.005",
                  rgba="1 0 0 1",
                  type="sphere")

# ------------------------------------------------------------
# 5. Add 'tip_site' to the LAST body
# ------------------------------------------------------------

all_bodies = root.findall(".//body")
if len(all_bodies) > 0:
    last_body = all_bodies[-1]

    ET.SubElement(last_body, "site",
                  name="tip_site",
                  pos=tip_site_pos,
                  size="0.001",
                  rgba="0 1 0 1",
                  type="sphere")

# ------------------------------------------------------------
# 6. Save updated XML
# ------------------------------------------------------------

tree.write("spirob_physics_model.xml")
print("Saved → spirob_physics_model.xml")
