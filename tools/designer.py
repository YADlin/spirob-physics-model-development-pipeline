"""Serve the SpiRob designer locally, with exact Python/CAD build downloads.

The static site also runs on GitHub Pages. This optional server binds only to
loopback and accepts same-origin JSON build requests, one at a time.
"""
from __future__ import annotations
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import uuid
import webbrowser
import zipfile
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'


def validate_build_request(params):
    from spirob.parameters import normalize_params
    from spirob.pipeline.spirob_csv_generator import validate_params
    from spirob.geometry import b_from_phi, a_from_tip_width, q_from_arc_length
    p = normalize_params(params)
    validate_params(p)
    if p['n_cables'] > 32:
        raise ValueError('The browser builder supports up to 32 cables; use the CLI for larger models.')
    b = b_from_phi(phi=math.radians(p['phi_deg']))
    a = a_from_tip_width(p['d_tip'], b)
    count = q_from_arc_length(p['L'], a, b) / math.radians(p['Delta_theta_deg'])
    if not math.isfinite(count) or count > 400:
        raise ValueError('The browser builder supports up to 400 links; increase Delta_theta_deg.')
    return p


class BuilderServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, output_dir):
        super().__init__(address, Handler)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.guard = threading.Lock()
        self.busy = False

    def start_build(self, params):
        params = validate_build_request(params)
        with self.guard:
            if self.busy:
                raise ValueError('A build is already running. Wait for it to finish.')
            if len(self.jobs) >= 30:
                raise ValueError('This session has completed 30 builds; restart the local builder to continue.')
            self.busy = True
            identifier = uuid.uuid4().hex
            folder = self.output_dir / identifier
            folder.mkdir()
            self.jobs[identifier] = dict(status='running', folder=folder)

        def worker():
            status = 'failed'
            log_path = folder / 'build.log'
            try:
                path = folder / 'params.json'
                path.write_text(json.dumps(params, indent=2)+'\n')
                with log_path.open('w') as log:
                    result = subprocess.run([sys.executable, str(ROOT/'build.py'), '--params', str(path),
                                             '--no-preview', '--output-dir', str(folder/'model')],
                                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                if result.returncode == 0:
                    with zipfile.ZipFile(folder/'model.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                        for path in sorted((folder/'model').rglob('*')):
                            if path.is_file():
                                archive.write(path, path.relative_to(folder/'model'))
                        archive.write(log_path, 'build.log')
                    status = 'completed'
            except Exception as exc:
                with log_path.open('a') as log:
                    log.write(f'\nBuild failed: {exc}\n')
            finally:
                with self.guard:
                    self.jobs[identifier]['status'] = status
                    self.busy = False
        threading.Thread(target=worker, daemon=True).start()
        return identifier


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.mjs':'text/javascript'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE), **kwargs)

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def json_response(self, data, status=200):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def permitted_host(self):
        port = self.server.server_port
        return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

    def do_GET(self):
        if not self.permitted_host():
            return self.json_response({'error':'Use the loopback URL printed by the server.'},403)
        path=urlsplit(self.path).path
        if path == '/api/status':
            return self.json_response({'builder':True})
        if path.startswith('/api/jobs/'):
            identifier = path.removeprefix('/api/jobs/')
            with self.server.guard:
                job = self.server.jobs.get(identifier)
                job = dict(job) if job else None
            if job is None:
                return self.json_response({'error':'Unknown build'},404)
            log = job['folder']/'build.log'
            return self.json_response(dict(status=job['status'],
                log=log.read_text(errors='replace')[-16000:] if log.exists() else '',
                download=f'/downloads/{identifier}.zip' if job['status']=='completed' else None))
        if path.startswith('/downloads/'):
            name=path.removeprefix('/downloads/')
            identifier=name.removesuffix('.zip')
            with self.server.guard:
                job=self.server.jobs.get(identifier)
            if not job or job['status']!='completed' or name!=identifier+'.zip':
                return self.json_response({'error':'Build download unavailable'},404)
            payload=(job['folder']/'model.zip').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type','application/zip')
            self.send_header('Content-Disposition','attachment; filename="spirob-model.zip"')
            self.send_header('Content-Length',str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
            return
        resolved=(SITE/unquote(path).lstrip('/')).resolve()
        if not resolved.is_relative_to(SITE) or (resolved.is_dir() and not (resolved/'index.html').exists()):
            return self.json_response({'error':'Not found'},404)
        super().do_GET()

    def do_POST(self):
        origin=self.headers.get('Origin')
        allowed=f'http://{self.headers.get("Host")}'
        if not self.permitted_host() or (origin is not None and origin != allowed):
            return self.json_response({'error':'Build requests must come from this local designer.'},403)
        if self.path!='/api/build':
            return self.json_response({'error':'Not found'},404)
        try:
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                raise ValueError('Expected application/json')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=100_000:
                raise ValueError('Request must be between 1 and 100000 bytes')
            params=json.loads(self.rfile.read(size))
            identifier=self.server.start_build(params)
            self.json_response({'id':identifier},202)
        except (ValueError,TypeError,OverflowError) as exc:
            self.json_response({'error':str(exc)},400)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--port',type=int,default=8765,help='Local HTTP port; the server binds only to 127.0.0.1')
    p.add_argument('--output-dir',default='build/designer',help='Keep generated model ZIPs and logs in per-build folders')
    p.add_argument('--no-browser',action='store_true',help='Print the URL without opening your browser automatically')
    args=p.parse_args()
    with BuilderServer(('127.0.0.1',args.port),args.output_dir) as server:
        url=f'http://127.0.0.1:{server.server_port}'
        print(f'SpiRob designer: {url}\nGenerated models: {server.output_dir}\nCtrl+C to stop.',flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__=='__main__':
    main()
