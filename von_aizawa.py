#!/usr/bin/env python3
"""
VON NEUMANN CHAOTIC ANALYZER
"""

import subprocess
import numpy as np
import io
import base64
import time
import os
import math
import hashlib
import random
import signal
import sys
from collections import Counter
from datetime import datetime
from flask import Flask, render_template_string, jsonify

# ============================================
# CONFIGURACION GRAFICAS
# ============================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_OK = True
except:
    MATPLOTLIB_OK = False

try:
    from scipy.integrate import odeint
    SCIPY_OK = True
except:
    SCIPY_OK = False

app = Flask(__name__)

# ============================================
# CARPETA PARA GUARDAR IMAGENES
# ============================================
class ImageSaver:
    def __init__(self):
        self.base_path = "/storage/emulated/0/Download/has1py"
        self.current_session = None
        self.image_count = 0
        self._create_session()

    def _create_session(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = os.path.join(self.base_path, timestamp)
        os.makedirs(self.current_session, exist_ok=True)
        print(f"\n📁 Imagenes: {self.current_session}")

    def save_plot(self, plt_figure, name):
        if not plt_figure:
            return None
        filename = f"{name}_{datetime.now().strftime('%H%M%S')}.png"
        filepath = os.path.join(self.current_session, filename)
        try:
            plt_figure.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(plt_figure)
            self.image_count += 1
            print(f"   {filename}")
            return filepath
        except:
            return None

image_saver = ImageSaver()

# ============================================
# HTML TEMPLATE - SIN EMOJIS NI VALIDADORES
# ============================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Von Neumann Chaotic Analyzer</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0a0f1e;
            color: #e0e0e0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #00ffaa;
            border-bottom: 2px solid #00ffaa;
            padding-bottom: 10px;
        }
        h2 {
            color: #66ccff;
            margin-top: 30px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }
        .card {
            background: #1a1f30;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #2a2f40;
        }
        .card h3 {
            margin: 0 0 8px 0;
            color: #00ffaa;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: white;
        }
        .card .unit {
            font-size: 0.8em;
            color: #888;
        }
        .plot-container {
            background: #1a1f30;
            border-radius: 10px;
            padding: 20px;
            margin: 30px 0;
            border: 1px solid #2a2f40;
        }
        .plot-container img {
            width: 100%;
            height: auto;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background: #2a2f40;
            color: #00ffaa;
            padding: 10px;
            text-align: left;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #2a2f40;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            color: #666;
            font-size: 0.8em;
        }
        .refresh {
            background: #00ffaa;
            color: #0a0f1e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin: 20px 0;
        }
        .refresh:hover {
            background: #66ffcc;
        }
        .hash-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .hash-item {
            background: #0f1422;
            border: 1px solid #2a2f40;
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            font-size: 0.7em;
            word-break: break-all;
        }
        .hash-item .label {
            color: #00ffaa;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .folder-info {
            background: #1a1f30;
            border-left: 4px solid #00ffaa;
            padding: 10px;
            margin: 20px 0;
            font-family: monospace;
            word-break: break-all;
        }
        .ultra-badge {
            display: inline-block;
            background: #00ffaa;
            color: #0a0f1e;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 20px;
            font-size: 0.8em;
        }
        .debug-info {
            background: #2a2f40;
            color: #ffaa00;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 0.9em;
        }
    </style>
    <script>
        function refreshPage() {
            location.reload();
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>Von Neumann Chaotic Analyzer <span class="ultra-badge">v3 (doble von Neumann)</span></h1>

        <div class="folder-info">
            Imagenes: {{ image_folder }}
        </div>

        <div class="debug-info">
            Muestras: RAW={{ raw_samples }} bytes | Post={{ post_samples }} bytes
        </div>

        <div class="metrics">
            <div class="card">
                <h3>CPU Entropy</h3>
                <div class="value">{{ "%.3f"|format(cpu_entropy) }}</div>
                <div class="unit">bits</div>
            </div>
            <div class="card">
                <h3>PMIC Entropy</h3>
                <div class="value">{{ "%.3f"|format(pmic_entropy) }}</div>
                <div class="unit">bits</div>
            </div>
            <div class="card">
                <h3>Lyapunov λ</h3>
                <div class="value">{{ "%.4f"|format(lyapunov) }}</div>
                <div class="unit">{{ "%.2f"|format(dup_time) }} steps</div>
            </div>
            <div class="card">
                <h3>Autocorr R(1)</h3>
                <div class="value">{{ "%.4f"|format(autocorr_r1_post) }}</div>
                <div class="unit">post-proc</div>
            </div>
            <div class="card">
                <h3>Mejora</h3>
                <div class="value">{{ "%.1f"|format(improvement) }}%</div>
                <div class="unit">reduccion</div>
            </div>
        </div>

        <button class="refresh" onclick="refreshPage()">Refresh</button>

        <div class="plot-container">
            <h2>Atractor de Aizawa (3D)</h2>
            <img src="data:image/png;base64,{{ aizawa_plot }}" alt="Aizawa">
        </div>

        <div class="plot-container">
            <h2>Temperatura</h2>
            <img src="data:image/png;base64,{{ temp_plot }}" alt="Temp">
        </div>

        <div class="plot-container">
            <h2>Autocorrelacion: RAW vs Von Neumann</h2>
            <img src="data:image/png;base64,{{ autocorr_plot }}" alt="Autocorr">
        </div>

        <h2>Hashes</h2>
        <div class="hash-container">
            <div class="hash-item">
                <div class="label">SHA-256</div>
                <div>{{ hash_raw[:16] }}...{{ hash_raw[-16:] }}</div>
            </div>
            <div class="hash-item">
                <div class="label">SHA-384</div>
                <div>{{ hash_384[:16] }}...{{ hash_384[-16:] }}</div>
            </div>
            <div class="hash-item">
                <div class="label">SHA-512</div>
                <div>{{ hash_512[:16] }}...{{ hash_512[-16:] }}</div>
            </div>
            <div class="hash-item">
                <div class="label">Hash Final</div>
                <div>{{ hash_final[:16] }}...{{ hash_final[-16:] }}</div>
            </div>
        </div>

        <h2>Analisis</h2>
        <table>
            <tr>
                <th>Metrica</th>
                <th>Valor</th>
            </tr>
            <tr>
                <td>Min-Entropy (1-byte) CPU</td>
                <td>{{ "%.3f"|format(cpu_entropy_1) }} bits</td>
            </tr>
            <tr>
                <td>Min-Entropy (1-byte) PMIC</td>
                <td>{{ "%.3f"|format(pmic_entropy_1) }} bits</td>
            </tr>
            <tr>
                <td>Min-Entropy (2-byte) CPU</td>
                <td>{{ "%.3f"|format(cpu_entropy_2) }} bits</td>
            </tr>
            <tr>
                <td>Min-Entropy (2-byte) PMIC</td>
                <td>{{ "%.3f"|format(pmic_entropy_2) }} bits</td>
            </tr>
            <tr>
                <td>Lyapunov</td>
                <td>λ = {{ "%.4f"|format(lyapunov) }}</td>
            </tr>
            <tr>
                <td>Autocorr RAW R(1)</td>
                <td>{{ "%.4f"|format(autocorr_r1) }}</td>
            </tr>
            <tr>
                <td>Autocorr Von Neumann R(1)</td>
                <td>{{ "%.4f"|format(autocorr_r1_post) }}</td>
            </tr>
            <tr>
                <td>Mejora</td>
                <td>{{ "%.1f"|format(improvement) }}%</td>
            </tr>
            <tr>
                <td>Muestras CPU</td>
                <td>{{ cpu_samples }} ({{ cpu_unique }} unicos)</td>
            </tr>
            <tr>
                <td>Muestras PMIC</td>
                <td>{{ pmic_samples }} ({{ pmic_unique }} unicos)</td>
            </tr>
            <tr>
                <td>Post-procesamiento</td>
                <td>XOR + Von Neumann + XOR + Von Neumann + XOR</td>
            </tr>
        </table>

        <div class="footer">
            Von Neumann Analyzer | Imagenes: {{ image_folder }} | {{ timestamp }}
        </div>
    </div>
</body>
</html>
"""

# ============================================
# POST-PROCESADOR VON NEUMANN MEJORADO (v3)
# ============================================
class VonNeumannProcessor:
    @staticmethod
    def xor_whitening(data_bytes):
        if len(data_bytes) < 2:
            return data_bytes
        result = bytearray()
        prev = data_bytes[0]
        for b in data_bytes[1:]:
            result.append(prev ^ b)
            prev = b
        return bytes(result)

    @staticmethod
    def von_neumann(bits):
        output = []
        i = 0
        while i < len(bits) - 1:
            if bits[i] != bits[i+1]:
                output.append(bits[i])
            i += 2
        return output

    @staticmethod
    def bytes_to_bits(data_bytes):
        bits = []
        for b in data_bytes:
            for i in range(8):
                bits.append((b >> i) & 1)
        return bits

    @staticmethod
    def bits_to_bytes(bits):
        bytes_out = bytearray()
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte |= (bits[i+j] << j)
            bytes_out.append(byte)
        return bytes(bytes_out)

    @classmethod
    def process_v3(cls, data_bytes):
        """Version 3: Doble Von Neumann - ajustada para mantener suficientes muestras"""
        if len(data_bytes) < 40:
            return data_bytes

        # Primera pasada - mantener mas muestras
        xored = cls.xor_whitening(data_bytes)
        bits1 = cls.bytes_to_bits(xored)
        vn_bits1 = cls.von_neumann(bits1)
        if len(vn_bits1) < 32:  # Aumentado para mantener muestra
            return xored[:32]
        vn_bytes1 = cls.bits_to_bytes(vn_bits1)

        # Segunda pasada - mantener mas muestras
        xored2 = cls.xor_whitening(vn_bytes1)
        bits2 = cls.bytes_to_bits(xored2)
        vn_bits2 = cls.von_neumann(bits2)
        if len(vn_bits2) < 24:  # Aumentado para mantener muestra
            return xored2[:24]
        vn_bytes2 = cls.bits_to_bytes(vn_bits2)

        # XOR final
        final = cls.xor_whitening(vn_bytes2)
        return final

# ============================================
# GENERADOR DE HASHES
# ============================================
class HashGenerator:
    @staticmethod
    def generate_from_bytes(data_bytes):
        if not data_bytes:
            data_bytes = os.urandom(64)
        if len(data_bytes) < 64:
            data_bytes = data_bytes + os.urandom(64 - len(data_bytes))

        h256 = hashlib.sha256(data_bytes).hexdigest()
        h384 = hashlib.sha384(data_bytes).hexdigest()
        h512 = hashlib.sha512(data_bytes).hexdigest()
        final_data = data_bytes + str(time.time_ns()).encode() + os.urandom(16)
        final = hashlib.sha512(final_data).hexdigest()

        return {
            'sha256': h256,
            'sha384': h384,
            'sha512': h512,
            'final': final
        }

# ============================================
# ANALIZADOR CIENTIFICO
# ============================================
class ChaoticAnalyzer:
    def __init__(self):
        self.aizawa_params = [0.95, 0.7, 0.6, 3.5, 0.25, 0.1]
        self.has_root = self._check_root()
        self.processor = VonNeumannProcessor()
        self.hash_gen = HashGenerator()

    def _check_root(self):
        try:
            subprocess.run("su -c 'echo test'", shell=True, timeout=1, capture_output=True)
            return True
        except:
            return False

    def read_sensor_bulk(self, sensor, n_samples=40):  # Aumentado muestras
        values = []
        if not self.has_root:
            base = 35000 + sensor * 100
            for i in range(n_samples):
                values.append(str(base + random.randint(0, 50)))
                time.sleep(0.001)
            return values

        for _ in range(n_samples):
            try:
                cmd = f"su -c 'cat /sys/class/thermal/thermal_zone{sensor}/temp 2>/dev/null || echo 35000'"
                res = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=0.2)
                values.append(res.stdout.strip() or "35000")
            except:
                values.append("35000")
            time.sleep(0.01)
        return values

    def estimate_min_entropy(self, samples, byte_width=1):
        if len(samples) < 2:
            return 0
        diffs = []
        for i in range(1, len(samples)):
            try:
                if len(samples[i]) >= 4 and len(samples[i-1]) >= 4:
                    diff = abs(int(samples[i]) - int(samples[i-1]))
                    mask = (1 << (8 * byte_width)) - 1
                    diffs.append(diff & mask)
            except:
                continue
        if not diffs:
            return 0
        try:
            freq = Counter(diffs)
            max_prob = max(freq.values()) / len(diffs)
            return -math.log2(max_prob)
        except:
            return 0

    def aizawa_system(self, state, t, params):
        if not SCIPY_OK:
            return [0, 0, 0]
        x, y, z = state
        a, b, c, d, e, f = params
        dx = (z - b) * x - d * y
        dy = d * x + (z - b) * y
        dz = c + a*z - (z**3)/3 - (x**2 + y**2)*(1 + e*z) + f*z*x**3
        return [dx, dy, dz]

    def lyapunov_analysis(self, base_temp, steps=300, dt=0.01):
        if not SCIPY_OK:
            return [], [], 0.1
        try:
            def run(temp):
                params = self.aizawa_params.copy()
                params[0] += (temp / 1000) / 1e5
                t = np.linspace(0, steps*dt, steps)
                return odeint(self.aizawa_system, [0.1, 0.1, 0.1], t, args=(params,))

            traj1 = run(base_temp)
            traj2 = run(base_temp + 1)
            delta_0 = np.linalg.norm(traj1[0] - traj2[0]) or 1e-12

            log_sep, times = [], []
            for i in range(0, steps, 20):
                delta_t = np.linalg.norm(traj1[i] - traj2[i])
                if delta_t > 0:
                    log_sep.append(math.log(delta_t / delta_0))
                    times.append(i * dt)

            if len(times) > 1:
                coeffs = np.polyfit(times, log_sep, 1)
                return times, log_sep, coeffs[0]
            return times, log_sep, 0.1
        except:
            return [], [], 0.1

    def autocorrelation(self, samples, max_lag=20):
        try:
            values = []
            for s in samples[:200]:
                if isinstance(s, int):
                    values.append(s)
                elif isinstance(s, bytes):
                    for b in s[:50]:
                        values.append(b)
                elif isinstance(s, str) and len(s) >= 4:
                    try:
                        values.append(int(s[-4:]))
                    except:
                        pass

            if len(values) < 10:
                return [0] * max_lag

            values = np.array(values[:100])
            values = (values - np.mean(values)) / (np.std(values) + 1e-10)

            autocorr = []
            for lag in range(max_lag):
                if len(values) > lag:
                    try:
                        corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                        autocorr.append(corr if not np.isnan(corr) else 0)
                    except:
                        autocorr.append(0)
                else:
                    autocorr.append(0)
            return autocorr
        except:
            return [0] * max_lag

    def generate_raw_bytes(self, cpu_samples, pmic_samples):
        raw = bytearray()
        for c, p in zip(cpu_samples[:40], pmic_samples[:40]):  # Mas muestras
            try:
                cv = int(c[-4:]) if len(c) >= 4 else 0
                pv = int(p[-4:]) if len(p) >= 4 else 0
                raw.append(cv & 0xFF)
                raw.append((cv >> 8) & 0xFF)
                raw.append(pv & 0xFF)
                raw.append((pv >> 8) & 0xFF)
            except:
                pass
        return bytes(raw)

    def generate_aizawa_plot(self, cpu_samples):
        if not MATPLOTLIB_OK or not SCIPY_OK or not cpu_samples:
            return ""
        try:
            base_temp = int(cpu_samples[0]) if cpu_samples else 35000
            params = self.aizawa_params.copy()
            params[0] += (base_temp / 1000) / 1e5

            t = np.linspace(0, 20, 5000)
            sol = odeint(self.aizawa_system, [0.1, 0.1, 0.1], t, args=(params,))

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], 'b-', alpha=0.6, linewidth=0.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Atractor de Aizawa')

            image_saver.save_plot(fig, "aizawa_3d")

            img = io.BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight', dpi=80)
            img.seek(0)
            plt.close(fig)
            return base64.b64encode(img.getvalue()).decode()
        except:
            return ""

    def generate_plots(self, cpu_samples, pmic_samples):
        plots = {}
        if not MATPLOTLIB_OK:
            return plots

        # Temperatura
        try:
            fig_temp = plt.figure(figsize=(8, 3))
            cpu_vals = []
            for s in cpu_samples[:50]:
                try:
                    cpu_vals.append(int(s) / 1000)
                except:
                    cpu_vals.append(35.0)
            plt.plot(cpu_vals, 'r-', alpha=0.7, label='CPU')
            plt.xlabel('Muestra')
            plt.ylabel('°C')
            plt.title('Temperatura')
            plt.legend()
            plt.grid(True, alpha=0.3)

            image_saver.save_plot(fig_temp, "temperatura")

            img = io.BytesIO()
            fig_temp.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plots['temp'] = base64.b64encode(img.getvalue()).decode()
            plt.close(fig_temp)
        except:
            pass

        # Autocorrelacion
        try:
            raw_bytes = self.generate_raw_bytes(cpu_samples, pmic_samples)
            post_bytes = self.processor.process_v3(raw_bytes)

            raw_corr = self.autocorrelation(raw_bytes)[:15]
            post_corr = self.autocorrelation(post_bytes)[:15]

            fig_ac = plt.figure(figsize=(8, 4))
            plt.plot(raw_corr, 'r-', label='RAW', marker='o', markersize=4)
            plt.plot(post_corr, 'g-', label='Von Neumann v3', marker='s', markersize=4)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelacion')
            plt.title('RAW vs Von Neumann v3')
            plt.legend()
            plt.grid(True, alpha=0.3)

            image_saver.save_plot(fig_ac, "autocorrelacion_v3")

            img = io.BytesIO()
            fig_ac.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plots['autocorr'] = base64.b64encode(img.getvalue()).decode()
            plt.close(fig_ac)
        except:
            pass

        return plots

# ============================================
# FLASK ROUTES
# ============================================
analyzer = ChaoticAnalyzer()

@app.route('/')
def index():
    cpu_samples = analyzer.read_sensor_bulk(7, 40)  # 40 muestras
    pmic_samples = analyzer.read_sensor_bulk(9, 40)

    cpu_entropy_1 = analyzer.estimate_min_entropy(cpu_samples, 1)
    cpu_entropy_2 = analyzer.estimate_min_entropy(cpu_samples, 2)
    pmic_entropy_1 = analyzer.estimate_min_entropy(pmic_samples, 1)
    pmic_entropy_2 = analyzer.estimate_min_entropy(pmic_samples, 2)

    try:
        base_temp = int(cpu_samples[0]) if cpu_samples else 35000
        _, _, lyap = analyzer.lyapunov_analysis(base_temp, steps=300)
    except:
        lyap = 0.1
    dup_time = math.log(2) / lyap if lyap > 0 else 999

    raw_bytes = analyzer.generate_raw_bytes(cpu_samples, pmic_samples)
    post_bytes = analyzer.processor.process_v3(raw_bytes)

    raw_corr = analyzer.autocorrelation(raw_bytes)
    post_corr = analyzer.autocorrelation(post_bytes)
    autocorr_r1 = raw_corr[1] if len(raw_corr) > 1 else 0
    autocorr_r1_post = post_corr[1] if len(post_corr) > 1 else 0

    improvement = 0
    if autocorr_r1 != 0:
        improvement = (abs(autocorr_r1) - abs(autocorr_r1_post)) / abs(autocorr_r1) * 100

    hashes = analyzer.hash_gen.generate_from_bytes(post_bytes)
    plots = analyzer.generate_plots(cpu_samples, pmic_samples)
    aizawa_plot = analyzer.generate_aizawa_plot(cpu_samples)

    return render_template_string(
        HTML_TEMPLATE,
        cpu_entropy=cpu_entropy_1,
        pmic_entropy=pmic_entropy_1,
        cpu_entropy_1=cpu_entropy_1,
        cpu_entropy_2=cpu_entropy_2,
        pmic_entropy_1=pmic_entropy_1,
        pmic_entropy_2=pmic_entropy_2,
        lyapunov=lyap,
        dup_time=dup_time,
        autocorr_r1=autocorr_r1,
        autocorr_r1_post=autocorr_r1_post,
        improvement=improvement,
        cpu_samples=len(cpu_samples),
        pmic_samples=len(pmic_samples),
        cpu_unique=len(set(cpu_samples)),
        pmic_unique=len(set(pmic_samples)),
        raw_samples=len(raw_bytes),
        post_samples=len(post_bytes),
        hash_raw=hashes['sha256'],
        hash_384=hashes['sha384'],
        hash_512=hashes['sha512'],
        hash_final=hashes['final'],
        temp_plot=plots.get('temp', ''),
        autocorr_plot=plots.get('autocorr', ''),
        aizawa_plot=aizawa_plot,
        image_folder=image_saver.current_session,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
    )

@app.route('/api/ping')
def ping():
    return jsonify({'alive': True, 'time': time.time()})

# ============================================
# MANEJADOR DE SEÑAL
# ============================================
def signal_handler(sig, frame):
    print("\n\nDeteniendo servidor...")
    print(f"Imagenes: {image_saver.current_session}")
    print(f"Total: {image_saver.image_count}")
    sys.stdout.flush()
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print(" VON NEUMANN CHAOTIC ANALYZER v3".center(70))
    print("="*70)
    print(f"\nDashboard: http://localhost:5000")
    print(f"Imagenes: {image_saver.current_session}")
    print(f"\nmatplotlib: {'OK' if MATPLOTLIB_OK else 'NO'}")
    print(f"scipy: {'OK' if SCIPY_OK else 'NO'}")
    print(f"root: {'OK' if analyzer.has_root else 'NO'}")
    print("\nModo: Von Neumann v3 (doble) - CORREGIDO")
    print("   • 40 muestras por sensor")
    print("   • Umbrales aumentados para mantener datos")
    print("   • Muestras post-proc visibles en interfaz")
    print("\nPresiona Ctrl+C para detener")
    print("="*70)

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nDeteniendo servidor...")
        print(f"Imagenes: {image_saver.current_session}")
        print(f"Total: {image_saver.image_count}")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        os._exit(0)
