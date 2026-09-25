"""Build a pinned, simulation-only PX4 executable; never starts an aircraft."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.cache/PX4-Autopilot-v1.16.2'
COMMIT = '54f0455ffcd755534539a7cf33a09a20bf71d29d'
TRUTH_HEADER = 'src/modules/mavlink/streams/HIL_STATE_QUATERNION.hpp'
BOARD = 'boards/px4/sitl/mantis.px4board'

# Default-off diagnostic: keep the random draws and all estimation/control/
# dynamics, but reduce generated measurement noise to 1% when explicitly enabled.
# Zero noise trips PX4's valid repeated-value sensor checks while stationary.
NOISE_HELPER = b'''\n#include <cstdlib>
static float mantis_sensor_noise_scale()
{
\tstatic const float scale = []() {
\t\tconst char *flag = std::getenv("MANTIS_SIH_LOW_NOISE_SENSORS");
\t\treturn flag && flag[0] == '1' && flag[1] == '\\0' ? 0.01f : 1.f;
\t}();
\treturn scale;
}
'''


def patch_sensor(relative, original):
    definitions = {
        'sensor_gps_sim/SensorGpsSim.cpp': ('SensorGpsSim.hpp', [
            ('(double)generate_wgn() * 0.2', '(double)generate_wgn() * (double)mantis_sensor_noise_scale() * 0.2', 2),
            ('generate_wgn() * 0.5f', 'generate_wgn() * mantis_sensor_noise_scale() * 0.5f', 1),
            ('noiseGauss3f(0.06f, 0.077f, 0.158f)',
             'noiseGauss3f(0.06f, 0.077f, 0.158f) * mantis_sensor_noise_scale()', 1),
            ('sensor_gps.s_variance_m_s = 0.4f;',
             'sensor_gps.s_variance_m_s = std::getenv("MANTIS_SIH_CALIBRATED_GPS") ? '
             '(std::getenv("MANTIS_SIH_CALIBRATED_GPS")[0] == \'2\' ? 0.11f : 0.158f) : 0.4f;', 1),
            ('sensor_gps.eph = 0.9f;',
             'sensor_gps.eph = std::getenv("MANTIS_SIH_CALIBRATED_GPS") ? 0.2f : 0.9f;', 1),
            ('sensor_gps.epv = 1.78f;',
             'sensor_gps.epv = std::getenv("MANTIS_SIH_CALIBRATED_GPS") ? 0.5f : 1.78f;', 1)]),
        'sensor_baro_sim/SensorBaroSim.cpp': ('SensorBaroSim.hpp', [
            ('1.f * (float)y1', 'mantis_sensor_noise_scale() * (float)y1', 1)]),
        'sensor_mag_sim/SensorMagSim.cpp': ('SensorMagSim.hpp', [
            ('noiseGauss3f(0.02f, 0.02f, 0.03f)',
             'noiseGauss3f(0.02f, 0.02f, 0.03f) * mantis_sensor_noise_scale()', 1)]),
        'simulator_sih/sih.cpp': ('sih.hpp', [
            ('Vector3f accel = specific_force_B + accel_noise;',
             'Vector3f accel = specific_force_B + accel_noise * mantis_sensor_noise_scale();', 1),
            ('Vector3f gyro = _w_B + earth_spin_rate_B + gyro_noise;',
             'Vector3f gyro = _w_B + earth_spin_rate_B + gyro_noise * mantis_sensor_noise_scale();', 1)]),
    }
    include, replacements = definitions[relative]
    patched = original
    for before, after, count in [(f'#include "{include}"',
            f'#include "{include}"'+NOISE_HELPER.decode(), 1), *replacements]:
        if patched.count(before.encode()) != count:
            raise RuntimeError(f'Unexpected sensor patch source: {relative}')
        patched = patched.replace(before.encode(), after.encode())
    return patched


SENSOR_SOURCES = ['src/modules/simulation/'+name for name in (
    'sensor_gps_sim/SensorGpsSim.cpp', 'sensor_baro_sim/SensorBaroSim.cpp',
    'sensor_mag_sim/SensorMagSim.cpp', 'simulator_sih/sih.cpp')]


def write_expected(path, original, patched):
    current = path.read_bytes()
    if current not in (original, patched):
        raise RuntimeError(f'Preserving unexpected local edits: {path}')
    if current != patched:
        path.write_bytes(patched)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args, cwd=SOURCE):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    SOURCE.parent.mkdir(exist_ok=True)
    if not SOURCE.exists():
        run('git', 'clone', '--depth', '1', '--branch', 'v1.16.2',
            'https://github.com/PX4/PX4-Autopilot.git', str(SOURCE), cwd=ROOT)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    if head != COMMIT:
        raise RuntimeError('Unexpected PX4 checkout; refusing to patch or build')
    run('git', 'submodule', 'update', '--init', '--recursive', '--depth', '1',
        'src/modules/mavlink/mavlink', 'src/lib/events/libevents', 'src/lib/heatshrink/heatshrink')
    # Upstream main sets this field; v1.16.2 sends zero. Source freshness must
    # include time spent waiting in sockets and rendering, not just arrival.
    original = subprocess.check_output(['git', 'show', f'HEAD:{TRUTH_HEADER}'], cwd=SOURCE)
    needle = b'mavlink_hil_state_quaternion_t msg{};'
    if original.count(needle) != 1:
        raise RuntimeError('Unexpected truth stream source')
    timestamp_line = (b'msg.time_usec = math::min(math::min(att.timestamp_sample, '
        b'gpos.timestamp_sample), math::min(lpos.timestamp_sample, angular_velocity.timestamp_sample));')
    patched = original.replace(needle, needle + b'\n\t\t\t' + timestamp_line)
    header = SOURCE / TRUTH_HEADER
    write_expected(header, original, patched)
    sensor_patches = {}
    for relative in SENSOR_SOURCES:
        original = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=SOURCE)
        patched = patch_sensor(relative.removeprefix('src/modules/simulation/'), original)
        write_expected(SOURCE/relative, original, patched)
        sensor_patches[relative] = dict(original_sha256=hashlib.sha256(original).hexdigest(),
                                       patched_sha256=sha(SOURCE/relative))
    overlay = (ROOT / 'integrations/px4/sih.px4board').read_bytes()
    board = SOURCE / BOARD
    if board.exists() and board.read_bytes() != overlay:
        raise RuntimeError('Preserving unexpected local board overlay')
    if not board.exists():
        board.write_bytes(overlay)
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=SOURCE,
                                      text=True).splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'],
                                       cwd=SOURCE, text=True).splitlines()
    if set(changed)-{TRUTH_HEADER, *SENSOR_SOURCES} or set(untracked)-{BOARD}:
        raise RuntimeError(f'Unexpected source modifications: {changed}, {untracked}')
    submodules = subprocess.check_output(['git', 'submodule', 'status', '--recursive',
        'src/modules/mavlink/mavlink', 'src/lib/events/libevents', 'src/lib/heatshrink/heatshrink'],
        cwd=SOURCE, text=True).splitlines()
    if any(not line.startswith(' ') for line in submodules):
        raise RuntimeError('Selected submodules do not match pinned source')
    run('git', 'submodule', 'foreach', '--recursive',
        'test -z "$(git status --porcelain --untracked-files=normal)"')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    build = SOURCE / 'build/px4_sitl_mantis'
    compiler_flags = []
    if sys.platform == 'darwin':
        sdk = Path(subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip())
        headers = sdk / 'usr/include/c++/v1'
        if not (headers / 'cstdlib').is_file():
            raise RuntimeError('Apple SDK C++ headers unavailable')
        # Some CLT installs contain an incomplete compiler-local header tree.
        # Point this build at the matching SDK; never modify system headers.
        compiler_flags = [f'-DCMAKE_CXX_FLAGS=-isystem {headers} -Wno-error=vla-cxx-extension']
    run('cmake', '-S', str(SOURCE), '-B', str(build), '-G', 'Ninja',
        '-DCONFIG=px4_sitl_mantis', f'-DPYTHON_EXECUTABLE={sys.executable}', '-DBUILD_TESTING=OFF',
        *compiler_flags)
    run('cmake', '--build', str(build), '--target', 'px4', '--parallel', '2')
    sys.path.insert(0, str(ROOT))
    from flybrain_sim.px4_sih import tree_sha
    receipt = dict(px4_commit=COMMIT, profile='px4_sitl_mantis',
        build_recipe_sha256=sha(Path(__file__).resolve()),
        binary_sha256=sha(build / 'bin/px4'),
        startup_tree_sha256=tree_sha(build / 'etc'),
        overlay_sha256=sha(board), truth_header_sha256=sha(header),
        timestamp_patch=timestamp_line.decode(),
        sensor_noise_patches=sensor_patches,
        sensor_mode_switch='MANTIS_SIH_LOW_NOISE_SENSORS=1 gives 1% noise; absent means stock noise',
        gps_accuracy_switch='MANTIS_SIH_CALIBRATED_GPS reports the actual stock SIH noise '
            '(eph .2m, epv .5m); value1 uses velocity .158m/s, value2 uses .11m/s '
            'accounting for EKF vertical x1.5. All samples are unchanged',
        submodules=subprocess.check_output(['git', 'submodule', 'status', '--recursive'],
                                          cwd=SOURCE, text=True).splitlines())
    (build / 'mantis-build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
