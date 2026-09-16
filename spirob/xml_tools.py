"""Small, atomic edits of standalone MJCF without losing relative assets."""
from pathlib import Path
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco


def read_standalone(path):
    tree = ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    if any(tree.getroot().find('.//'+tag) is not None for tag in ('include', 'attach')):
        raise ValueError('Use a standalone, expanded MJCF; include/attach editing is not supported')
    return tree


def set_arena_memory(root, mib):
    if isinstance(mib, bool) or not isinstance(mib, int) or not 1 <= mib <= 4096:
        raise ValueError('arena memory must be an integer from 1 to 4096 MiB')
    size = root.find('size')
    if size is None:
        size = ET.SubElement(root, 'size')
    for legacy in ('njmax', 'nstack'):
        size.attrib.pop(legacy, None)
    size.set('memory', f'{mib}M')


def write_validated(tree, source, destination, verify=None, overwrite=False):
    """Rebase asset paths, compile a temporary file, then publish atomically."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('Write a separate output XML so the original is retained')
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} exists; use --overwrite to replace this output')
    root = tree.getroot()
    compiler = root.find('compiler')
    dirs = {} if compiler is None else dict(compiler.attrib)
    for asset in root.findall('./asset/*'):
        filename = asset.get('file')
        if filename is None:
            continue
        kind = 'meshdir' if asset.tag == 'mesh' else 'texturedir' if asset.tag == 'texture' else None
        directory = dirs.get(kind, dirs.get('assetdir', '')) if kind else ''
        absolute = (source.parent / directory / filename).resolve()
        asset.set('file', os.path.relpath(absolute, destination.parent))
    if compiler is not None:
        for attr in ('meshdir', 'texturedir', 'assetdir'):
            compiler.attrib.pop(attr, None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix='.xml', prefix='.spirob-check-', dir=destination.parent)
    os.close(fd)
    try:
        ET.indent(tree, space='  ')
        tree.write(temporary, encoding='unicode')
        compiled = mujoco.MjModel.from_xml_path(temporary)
        if verify is not None:
            verify(compiled)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return compiled
