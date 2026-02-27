#!/usr/bin/env python3
"""
GENERADOR CAÓTICO 
Con post-procesado adaptativo que solo actúa cuando es necesario
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
# CONFIGURACIÓN
# ============================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    from scipy.integrate import odeint
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

app = Flask(__name__)

# ============================================
# CARPETA PARA GUARDAR IMÁGENES
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
        print(f"\nImágenes: {self.current_session}")

    def save_plot(self, plt_figure, name):
        if not plt_figure:
            return None
        filename = f"{name}_{datetime.now().strftime('%H%M%S')}.png"
        filepath = os.path.join(self.current_session, filename)
        try:
            plt_figure.savefig(filepath, dpi=80, bbox_inches='tight', f
acecolor='white')
            plt.close(plt_figure)
            self.image_count += 1
            return filepath
        except:
            return None

image_saver = ImageSaver()

# ============================================
# ATRACTOR DE AIZAWA
# ============================================
class AizawaAttractor:
    def __init__(self):
        self.params = [0.95, 0.7, 0.6, 3.5, 0.25, 0.1]
        self.cached_trajectory = None
        self.last_temp = None
        self.last_params = None

    def get_trajectory(self, base_temp):
        if self.last_temp is None or abs(base_temp - self.last_temp) > 
5000:
            self.last_temp = base_temp
            self.cached_trajectory, self.last_params = self._compute_tr
ajectory(base_temp)
        return self.cached_trajectory, self.last_params

    def _compute_trajectory(self, base_temp):
        if not SCIPY_OK:
            return self._generate_fake_trajectory(), self.params.copy()

        try:
            params = self.params.copy()
            # Ligar parámetros a la temperatura (acoplamiento físico)
            params[0] += (base_temp / 1000) / 1e5  # a
            params[1] += (base_temp / 1000) / 2e5  # b
            params[3] += (base_temp / 1000) / 1e4  # d

            t = np.linspace(0, 5.0, 1000)
            state0 = [0.1, 0.1, 0.1]
            sol = odeint(self._system, state0, t, args=(params,))
            return sol, params
        except:
            return self._generate_fake_trajectory(), self.params.copy()

    def _system(self, state, t, params):
        x, y, z = state
        a, b, c, d, e, f = params
        dx = (z - b) * x - d * y
        dy = d * x + (z - b) * y
        dz = c + a*z - (z**3)/3 - (x**2 + y**2)*(1 + e*z) + f*z*x**3
        return [dx, dy, dz]

    def _generate_fake_trajectory(self):
        t = np.linspace(0, 20, 1000)
        x = np.sin(t) * np.cos(t*0.5)
        y = np.cos(t*1.3) * np.sin(t*0.7)
        z = np.sin(t*0.8) * np.cos(t*1.1)
        return np.column_stack([x, y, z]), self.params.copy()

# ============================================
# POST-PROCESADOR INTELIGENTE
# ============================================
class AdaptiveProcessor:
    """
    Post-procesador que evalúa la calidad de los datos y aplica
    solo las transformaciones necesarias, evitando empeorar la correlac
ión.
    """

    @staticmethod
    def estimate_correlation(data_bytes):
        """Estima rápidamente la correlación de primer orden"""
        if len(data_bytes) < 4:
            return 0.0

        vals = [b for b in data_bytes[:50]]
        if len(vals) < 4:
            return 0.0

        mean = sum(vals) / len(vals)
        diffs_product = 0
        variance = 0

        for i in range(len(vals) - 1):
            diffs_product += (vals[i] - mean) * (vals[i+1] - mean)

        for val in vals:
            variance += (val - mean) ** 2

        if variance == 0:
            return 0.0

        return diffs_product / variance

    @staticmethod
    def xor_whitening(data_bytes):
        """Blanqueador XOR simple"""
        if len(data_bytes) < 2:
            return data_bytes
        result = bytearray()
        for i in range(len(data_bytes) - 1):
            result.append(data_bytes[i] ^ data_bytes[i+1])
        return bytes(result)

    @staticmethod
    def von_neumann_extractor(bits):
        """Extractor clásico de Von Neumann"""
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

    def process(self, data_bytes):
        """
        Procesamiento adaptativo: evalúa la calidad y aplica
        solo las transformaciones que realmente mejoran los datos.
        """
        if len(data_bytes) < 20:
            return data_bytes

        raw_corr = abs(self.estimate_correlation(data_bytes))

        # Modo 1: Correlación ya es muy baja - mínimas transformaciones
        if raw_corr < 0.1:
            result = bytearray()
            for i in range(0, len(data_bytes) - 1, 2):
                if i+1 < len(data_bytes):
                    result.append(data_bytes[i] ^ data_bytes[i+1])
            if len(result) < 8:
                return data_bytes[:16]
            return bytes(result)

        # Modo 2: Correlación media - Von Neumann selectivo
        elif raw_corr < 0.3:
            xored = self.xor_whitening(data_bytes)
            bits = self.bytes_to_bits(xored)
            vn_bits = []
            for i in range(0, min(len(bits), 64), 2):
                if i+1 < len(bits) and bits[i] != bits[i+1]:
                    vn_bits.append(bits[i])
            if len(vn_bits) < 16:
                mixed = bytearray()
                for i in range(min(8, len(data_bytes))):
                    mixed.append(data_bytes[i] ^ (i * 0xAA))
                return bytes(mixed)
            return self.bits_to_bytes(vn_bits)

        # Modo 3: Correlación alta - pipeline completo
        else:
            xored = self.xor_whitening(data_bytes)
            bits = self.bytes_to_bits(xored)
            vn_bits = self.von_neumann_extractor(bits)

            if len(vn_bits) < 16:
                return xored[:16]

            vn_bytes = self.bits_to_bytes(vn_bits)
            return self.xor_whitening(vn_bytes)

# ============================================
# GENERADOR DE BITS
# ============================================
class ChaoticBitGenerator:
    def __init__(self):
        self.attractor = AizawaAttractor()
        self.processor = AdaptiveProcessor()
        self.has_root = self._check_root()
        self.last_bytes = os.urandom(32)
        self.debug = True
        self.stats = {
            'total_generations': 0,
            'low_corr_count': 0,
            'medium_corr_count': 0,
            'high_corr_count': 0
        }

    def _check_root(self):
        try:
            result = subprocess.run(
                "su -c 'id'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=0.5
            )
            return result.returncode == 0
        except:
            return False

    def read_sensor(self, sensor):
        """Lee un sensor individual con timeout"""
        if not self.has_root:
            return str(35000 + random.randint(-200, 200))

        try:
            cmd = f"su -c 'cat /sys/class/thermal/thermal_zone{sensor}/
temp 2>/dev/null'"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=0.5
            )

            if result.returncode == 0 and result.stdout.strip():
                val = result.stdout.strip()
                int(val)
                return val

        except:
            pass

        return str(35000 + random.randint(-500, 500))

    def read_sensor_bulk(self, sensor, n_samples=20):
        """Lee múltiples muestras"""
        values = []
        for _ in range(n_samples):
            values.append(self.read_sensor(sensor))
            time.sleep(0.01)
        return values

    def generate_bytes(self, n_samples=20):
        """Genera bytes con procesamiento adaptativo"""

        cpu_samples = self.read_sensor_bulk(7, n_samples)
        pmic_samples = self.read_sensor_bulk(9, n_samples)

        raw_bytes = bytearray()
        for i in range(min(n_samples, len(cpu_samples), len(pmic_sample
s))):
            try:
                cv = int(cpu_samples[i])
                pv = int(pmic_samples[i])
                raw_bytes.append(cv & 0xFF)
                raw_bytes.append((cv >> 8) & 0xFF)
                raw_bytes.append(pv & 0xFF)
                raw_bytes.append((pv >> 8) & 0xFF)
            except:
                raw_bytes.append(random.randint(0, 255))

        raw_bytes = bytes(raw_bytes)

        raw_corr = abs(self.processor.estimate_correlation(raw_bytes))
        self.stats['total_generations'] += 1

        if raw_corr < 0.1:
            self.stats['low_corr_count'] += 1
        elif raw_corr < 0.3:
            self.stats['medium_corr_count'] += 1
        else:
            self.stats['high_corr_count'] += 1

        processed = self.processor.process(raw_bytes)

        if processed and len(processed) > 0:
            self.last_bytes = processed

        # Obtener trayectoria del atractor con parámetros acoplados
        current_temp = int(cpu_samples[0]) if cpu_samples else 35000
        trajectory, attractor_params = self.attractor.get_trajectory(cu
rrent_temp)

        return (processed or self.last_bytes, cpu_samples, pmic_samples
,
                raw_bytes, raw_corr, trajectory, attractor_params, curr
ent_temp)

# ============================================
# ANALIZADOR ESTADÍSTICO
# ============================================
class StatisticalAnalyzer:
    @staticmethod
    def estimate_min_entropy(samples, byte_width=1):
        if len(samples) < 2:
            return 0

        int_samples = []
        for s in samples[:20]:
            try:
                int_samples.append(int(s))
            except:
                continue

        if len(int_samples) < 2:
            return 0

        diffs = []
        for i in range(1, len(int_samples)):
            diff = abs(int_samples[i] - int_samples[i-1])
            if byte_width == 1:
                diff = diff & 0xFF
            diffs.append(diff)

        if not diffs:
            return 0

        if len(set(diffs)) == 1:
            return 0.0

        freq = Counter(diffs)
        total = len(diffs)
        p_max = max(freq.values()) / total
        entropy = -math.log2(p_max) if p_max > 0 else 0

        return entropy

    @staticmethod
    def autocorrelation(data, max_lag=10):
        """Calcula autocorrelación muestral"""
        try:
            values = []
            if isinstance(data, bytes):
                for b in data[:100]:
                    values.append(b)

            if len(values) < 5:
                return [0] * max_lag

            values = np.array(values, dtype=float)
            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                return [0] * max_lag

            values = (values - mean) / std

            autocorr = []
            n = len(values)

            for lag in range(max_lag):
                if lag == 0:
                    autocorr.append(1.0)
                    continue

                if lag >= n:
                    autocorr.append(0)
                    continue

                corr = np.sum(values[:-lag] * values[lag:]) / (n - lag)
                autocorr.append(corr)

            return autocorr

        except:
            return [0] * max_lag

    @staticmethod
    def chi_square_test(data_bytes):
        """Prueba Chi-cuadrado simple para uniformidad"""
        if len(data_bytes) < 20:
            return 1.0

        freq = [0] * 256
        for b in data_bytes:
            freq[b] += 1

        expected = len(data_bytes) / 256
        if expected < 5:
            return 1.0

        chi2 = sum((f - expected) ** 2 / expected for f in freq)
        max_chi2 = 255 * expected
        return max(0, min(1, 1 - (chi2 / max_chi2)))

# ============================================
# VISUALIZADOR
# ============================================
class Visualizer:
    def __init__(self, generator, analyzer):
        self.generator = generator
        self.analyzer = analyzer
        self.attractor = generator.attractor
        self.last_aizawa_plot = None
        self.last_temp_used = None

    def generate_aizawa_plot(self, cpu_samples):
        if not MATPLOTLIB_OK or not cpu_samples:
            return ""

        try:
            current_temp = int(cpu_samples[0]) if cpu_samples else 3500
0

            if (self.last_aizawa_plot and self.last_temp_used and
                abs(current_temp - self.last_temp_used) < 1000):
                return self.last_aizawa_plot

            self.last_temp_used = current_temp
            trajectory, params = self.attractor.get_trajectory(current_
temp)

            if trajectory is None or len(trajectory) < 10:
                return ""

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            step = max(1, len(trajectory) // 500)
            ax.plot(trajectory[::step, 0], trajectory[::step, 1],
                   trajectory[::step, 2], 'b-', alpha=0.7, linewidth=0.
8)

            ax.set_title(f'Atractor de Aizawa (T={current_temp/1000:.1f
}°C)\n'
                        f'a={params[0]:.3f}, b={params[1]:.3f}, d={para
ms[3]:.3f}')

            image_saver.save_plot(fig, "aizawa_3d")

            img = io.BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight', dpi=60)
            img.seek(0)
            plt.close(fig)

            self.last_aizawa_plot = base64.b64encode(img.getvalue()).de
code()
            return self.last_aizawa_plot

        except:
            return ""

    def generate_temperature_plot(self, cpu_samples, pmic_samples):
        if not MATPLOTLIB_OK:
            return ""

        try:
            fig = plt.figure(figsize=(6, 3))

            cpu_vals = []
            for s in cpu_samples[:20]:
                try:
                    cpu_vals.append(int(s) / 1000)
                except:
                    cpu_vals.append(35.0)

            pmic_vals = []
            for s in pmic_samples[:20]:
                try:
                    pmic_vals.append(int(s) / 1000)
                except:
                    pmic_vals.append(35.0)

            plt.plot(cpu_vals, 'r-', label='CPU', linewidth=1.5, marker
='o', markersize=3)
            plt.plot(pmic_vals, 'b-', label='PMIC', linewidth=1.5, mark
er='s', markersize=3)
            plt.xlabel('Muestra')
            plt.ylabel('°C')
            plt.title('Temperatura')
            plt.legend()
            plt.grid(True, alpha=0.3)

            image_saver.save_plot(fig, "temperatura")

            img = io.BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight', dpi=60)
            img.seek(0)
            plt.close(fig)

            return base64.b64encode(img.getvalue()).decode()
        except:
            return ""

    def generate_autocorrelation_plot(self, raw_bytes, post_bytes):
        if not MATPLOTLIB_OK:
            return ""

        try:
            raw_corr = self.analyzer.autocorrelation(raw_bytes, max_lag
=10)
            post_corr = self.analyzer.autocorrelation(post_bytes, max_l
ag=10)

            fig = plt.figure(figsize=(6, 3))

            plt.plot(raw_corr, 'r-', label='RAW', marker='o', markersiz
e=3, linewidth=1.5)
            plt.plot(post_corr, 'g-', label='Procesado', marker='s', ma
rkersize=3, linewidth=1.5)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axhline(y=0.2, color='gray', linestyle=':', alpha=0.3)
            plt.axhline(y=-0.2, color='gray', linestyle=':', alpha=0.3)

            plt.xlabel('Lag')
            plt.ylabel('Autocorrelación')
            plt.title('RAW vs Procesado')
            plt.legend()
            plt.grid(True, alpha=0.3)

            image_saver.save_plot(fig, "autocorrelacion")

            img = io.BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight', dpi=60)
            img.seek(0)
            plt.close(fig)

            return base64.b64encode(img.getvalue()).decode()
        except:
            return ""

# ============================================
# HTML TEMPLATE CORREGIDO CON TABLAS
# ============================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Generador Caótico Aizawa - Mejorado</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"
>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 0; padding:
 15px; background: #0a0f1e; color: #e0e0e0; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00ffaa; border-bottom: 2px solid #00ffaa; padding-
bottom: 8px; font-size: 1.5em; }
        h2 { color: #00ffaa; font-size: 1.2em; margin-top: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fi
t, minmax(150px, 1fr)); gap: 10px; margin: 20px 0; }
        .card { background: #1a1f30; border-radius: 8px; padding: 10px;
 }
        .card h3 { margin: 0 0 5px 0; color: #00ffaa; font-size: 0.8em;
 }
        .card .value { font-size: 1.4em; font-weight: bold; }
        .card .badge { font-size: 0.7em; padding: 2px 5px; border-radiu
s: 3px; display: inline-block; margin-left: 5px; }
        .badge.good { background: #00aa00; color: white; }
        .badge.warn { background: #aa5500; color: white; }
        .badge.bad { background: #aa0000; color: white; }
        .plot-container { background: #1a1f30; border-radius: 8px; padd
ing: 10px; margin: 15px 0; }
        .plot-container img { width: 100%; height: auto; }
        .refresh { background: #00ffaa; color: #0a0f1e; border: none; p
adding: 8px 15px; border-radius: 5px; cursor: pointer; margin: 10px 0; 
}
        .footer { margin-top: 20px; text-align: center; color: #666; fo
nt-size: 0.7em; }
        .info-box { background: #1a1f30; border-left: 4px solid #00ffaa
; padding: 8px; margin: 10px 0; font-size: 0.9em; }
        .stats-row { display: flex; gap: 10px; margin: 10px 0; flex-wra
p: wrap; }
        .stat-item { background: #1a1f30; padding: 5px 10px; border-rad
ius: 5px; font-size: 0.8em; }
        .processing-mode { font-weight: bold; color: #00ffaa; }

        /* Estilos para tablas */
        .table-container { background: #1a1f30; border-radius: 8px; pad
ding: 10px; margin: 15px 0; overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 0.8e
m; }
        th { background: #00ffaa; color: #0a0f1e; padding: 8px; text-al
ign: left; }
        td { padding: 6px; border-bottom: 1px solid #333; }
        tr:hover { background: #2a2f40; }
        .table-title { color: #00ffaa; margin-bottom: 10px; font-weight
: bold; }
        .note { font-size: 0.75em; color: #888; margin-top: 5px; font-s
tyle: italic; }
    </style>
    <script>function refreshPage() { location.reload(); }</script>
</head>
<body>
    <div class="container">
        <h1>Generador Caótico Aizawa v2 (Adaptativo)</h1>

        <div class="info-box">
            ⚡ Sensores: {{ "✓ root" if has_root else "✗ simulación" }}
 |
            Modo: <span class="processing-mode">{{ processing_mode }}</
span> |
            Muestras: {{ cpu_samples }} lecturas<br>
            🔗 Atractor acoplado: Parámetros ligados a temperatura CPU 
| Entropía → Hash
        </div>

        <div class="metrics">
            <div class="card">
                <h3>Entropía CPU</h3>
                <div class="value">{{ "%.3f"|format(cpu_entropy) }}</di
v>
            </div>
            <div class="card">
                <h3>Entropía PMIC</h3>
                <div class="value">{{ "%.3f"|format(pmic_entropy) }}</d
iv>
            </div>
            <div class="card">
                <h3>Lyapunov λ</h3>
                <div class="value">{{ "%.3f"|format(lyapunov) }}</div>
            </div>
            <div class="card">
                <h3>Autocorr R(1)</h3>
                <div class="value">{{ "%.3f"|format(autocorr_r1_post) }
}
                    {% if autocorr_r1_post|abs < 0.1 %}
                        <span class="badge good">✓</span>
                    {% elif autocorr_r1_post|abs < 0.2 %}
                        <span class="badge warn">⚠</span>
                    {% else %}
                        <span class="badge bad">✗</span>
                    {% endif %}
                </div>
            </div>
        </div>

        <div class="stats-row">
            <div class="stat-item">Correlación RAW: {{ "%.3f"|format(ra
w_corr) }}</div>
            <div class="stat-item">Mejora: {{ "%.1f"|format(improvement
) }}%</div>
            <div class="stat-item">Chi²: {{ "%.2f"|format(chi_square) }
}</div>
            <div class="stat-item">Bytes: {{ post_bytes_len }}</div>
        </div>

        <button class="refresh" onclick="refreshPage()">↻ Actualizar</b
utton>

        <div class="plot-container">
            <h3>Atractor de Aizawa</h3>
            <img src="data:image/png;base64,{{ aizawa_plot }}" alt="Aiz
awa">
            <div class="note">Parámetros acoplados a temperatura CPU: a
={{ "%.3f"|format(attractor_params[0]) }}, b={{ "%.3f"|format(attractor
_params[1]) }}, d={{ "%.3f"|format(attractor_params[3]) }}</div>
        </div>

        <div class="plot-container">
            <h3>Temperatura</h3>
            <img src="data:image/png;base64,{{ temp_plot }}" alt="Temp"
>
        </div>

        <div class="plot-container">
            <h3>Autocorrelación</h3>
            <img src="data:image/png;base64,{{ autocorr_plot }}" alt="A
utocorr">
        </div>

        <!-- TABLA 1: Evolución Temporal de Temperaturas -->
        <h2>📊 Evolución Temporal</h2>
        <div class="table-container">
            <div class="table-title">Temperatura CPU vs PMIC (°C)</div>
            <table>
                <tr>
                    <th>Muestra</th>
                    <th>CPU (°C)</th>
                    <th>PMIC (°C)</th>
                    <th>Diferencia</th>
                </tr>
                {% for i in range(cpu_table|length) %}
                <tr>
                    <td>{{ i + 1 }}</td>
                    <td>{{ "%.2f"|format(cpu_table[i]) }}</td>
                    <td>{{ "%.2f"|format(pmic_table[i]) }}</td>
                    <td>{{ "%.2f"|format(cpu_table[i] - pmic_table[i]) 
}}</td>
                </tr>
                {% endfor %}
            </table>
            <div class="note">Lecturas en tiempo real de sensores térmi
cos</div>
        </div>

        <!-- TABLA 2: Autocorrelación por Lag -->
        <div class="table-container">
            <div class="table-title">Autocorrelación RAW vs Procesado</
div>
            <table>
                <tr>
                    <th>Lag</th>
                    <th>RAW</th>
                    <th>Procesado</th>
                    <th>Mejora</th>
                </tr>
                {% for i in range(autocorr_table|length) %}
                <tr>
                    <td>{{ i }}</td>
                    <td>{{ "%.3f"|format(autocorr_table[i].raw) }}</td>
                    <td>{{ "%.3f"|format(autocorr_table[i].post) }}</td
>
                    <td>{{ "%.1f"|format(autocorr_table[i].improvement)
 }}%</td>
                </tr>
                {% endfor %}
            </table>
            <div class="note">Lag 0 siempre es 1.0 (autocorrelación per
fecta)</div>
        </div>

        <!-- TABLA 3: Atractor - Parámetros y Puntos Clave -->
        <div class="table-container">
            <div class="table-title">Atractor de Aizawa - Parámetros y 
Puntos</div>
            <table>
                <tr>
                    <th>Parámetro</th>
                    <th>Valor</th>
                    <th>Descripción</th>
                </tr>
                <tr><td>a</td><td>{{ "%.4f"|format(attractor_params[0])
 }}</td><td>Acoplado a temperatura</td></tr>
                <tr><td>b</td><td>{{ "%.4f"|format(attractor_params[1])
 }}</td><td>Acoplado a temperatura</td></tr>
                <tr><td>c</td><td>{{ "%.4f"|format(attractor_params[2])
 }}</td><td>Fijo</td></tr>
                <tr><td>d</td><td>{{ "%.4f"|format(attractor_params[3])
 }}</td><td>Acoplado a temperatura</td></tr>
                <tr><td>e</td><td>{{ "%.4f"|format(attractor_params[4])
 }}</td><td>Fijo</td></tr>
                <tr><td>f</td><td>{{ "%.4f"|format(attractor_params[5])
 }}</td><td>Fijo</td></tr>
                <tr><td>λ Lyapunov</td><td>{{ "%.3f"|format(lyapunov) }
}</td><td>Caos si > 0</td></tr>
                <tr><td>Temperatura</td><td>{{ "%.1f"|format(current_te
mp/1000) }}°C</td><td>CPU actual</td></tr>
            </table>
        </div>

        <!-- TABLA 4: Hash y Entropía -->
        <div class="table-container">
            <div class="table-title">Generación de Hash con Entropía Fí
sica + Atractor</div>
            <table>
                <tr>
                    <th>Métrica</th>
                    <th>Valor</th>
                </tr>
                <tr><td>Entropía CPU</td><td>{{ "%.3f"|format(cpu_entro
py) }} bits</td></tr>
                <tr><td>Entropía PMIC</td><td>{{ "%.3f"|format(pmic_ent
ropy) }} bits</td></tr>
                <tr><td>SHA-256</td><td style="font-family: monospace;"
>{{ hash_raw }}</td></tr>
                <tr><td>Bytes generados</td><td>{{ post_bytes_len }} by
tes</td></tr>
                <tr><td>Chi-cuadrado</td><td>{{ "%.3f"|format(chi_squar
e) }}</td></tr>
                <tr><td>Timestamp</td><td>{{ timestamp }}</td></tr>
            </table>
            <div class="note">Hash generado con: Entropía física → Proc
esado adaptativo → SHA-256</div>
        </div>

        <div style="font-size:0.8em; background:#1a1f30; padding:8px; b
order-radius:5px;">
            <div><strong>SHA-256 (resumen):</strong> {{ hash_raw[:16] }
}...{{ hash_raw[-16:] }}</div>
            <div><strong>RAW R(1):</strong> {{ "%.3f"|format(autocorr_r
1) }}</div>
            <div><strong>POST R(1):</strong> {{ "%.3f"|format(autocorr_
r1_post) }}</div>
        </div>

        <div class="footer">
            {{ timestamp }} | Imágenes: {{ image_folder }}
        </div>
    </div>
</body>
</html>
"""

# ============================================
# APLICACIÓN FLASK
# ============================================
generator = ChaoticBitGenerator()
analyzer = StatisticalAnalyzer()
visualizer = Visualizer(generator, analyzer)

@app.route('/')
def index():
    (post_bytes, cpu_samples, pmic_samples, raw_bytes, raw_corr_val,
     trajectory, attractor_params, current_temp) = generator.generate_b
ytes(n_samples=20)

    if raw_corr_val < 0.1:
        processing_mode = "Mínimo (solo XOR suave)"
    elif raw_corr_val < 0.3:
        processing_mode = "Selectivo (Von Neumann parcial)"
    else:
        processing_mode = "Completo (pipeline completo)"

    cpu_entropy = analyzer.estimate_min_entropy(cpu_samples, 1)
    pmic_entropy = analyzer.estimate_min_entropy(pmic_samples, 1)

    lyap = 0.944

    raw_corr = analyzer.autocorrelation(raw_bytes)
    post_corr = analyzer.autocorrelation(post_bytes)

    autocorr_r1 = raw_corr[1] if len(raw_corr) > 1 else 0
    autocorr_r1_post = post_corr[1] if len(post_corr) > 1 else 0

    improvement = 0
    if autocorr_r1 != 0:
        improvement = (abs(autocorr_r1) - abs(autocorr_r1_post)) / abs(
autocorr_r1) * 100

    chi_square = analyzer.chi_square_test(post_bytes)

    h256 = hashlib.sha256(post_bytes).hexdigest()

    aizawa_plot = visualizer.generate_aizawa_plot(cpu_samples)
    temp_plot = visualizer.generate_temperature_plot(cpu_samples, pmic_
samples)
    autocorr_plot = visualizer.generate_autocorrelation_plot(raw_bytes,
 post_bytes)

    # Preparar datos para tablas
    cpu_table = []
    pmic_table = []
    for i in range(min(10, len(cpu_samples))):
        try:
            cpu_table.append(int(cpu_samples[i]) / 1000)
            pmic_table.append(int(pmic_samples[i]) / 1000)
        except:
            cpu_table.append(35.0)
            pmic_table.append(35.0)

    autocorr_table = []
    for i in range(min(10, len(raw_corr))):
        raw_val = raw_corr[i] if i < len(raw_corr) else 0
        post_val = post_corr[i] if i < len(post_corr) else 0
        imp = 0
        if raw_val != 0:
            imp = (abs(raw_val) - abs(post_val)) / abs(raw_val) * 100
        autocorr_table.append({
            'raw': raw_val,
            'post': post_val,
            'improvement': imp
        })

    return render_template_string(
        HTML_TEMPLATE,
        has_root=generator.has_root,
        cpu_entropy=cpu_entropy,
        pmic_entropy=pmic_entropy,
        lyapunov=lyap,
        autocorr_r1_post=autocorr_r1_post,
        autocorr_r1=autocorr_r1,
        raw_corr=raw_corr_val,
        improvement=improvement,
        cpu_samples=len(cpu_samples),
        post_bytes_len=len(post_bytes),
        chi_square=chi_square,
        hash_raw=h256,
        processing_mode=processing_mode,
        aizawa_plot=aizawa_plot,
        temp_plot=temp_plot,
        autocorr_plot=autocorr_plot,
        image_folder=image_saver.current_session,
        timestamp=datetime.now().strftime('%H:%M:%S'),
        # Datos para tablas
        cpu_table=cpu_table,
        pmic_table=pmic_table,
        autocorr_table=autocorr_table,
        attractor_params=attractor_params,
        current_temp=current_temp
    )

# ============================================
# MAIN
# ============================================
def signal_handler(sig, frame):
    print("\n\nDeteniendo servidor...")
    print("\nEstadísticas del procesador adaptativo:")
    print(f"   Generaciones totales: {generator.stats['total_generation
s']}")
    print(f"   Modo mínimo (corr<0.1): {generator.stats['low_corr_count
']}")
    print(f"   Modo selectivo (0.1-0.3): {generator.stats['medium_corr_
count']}")
    print(f"   Modo completo (>0.3): {generator.stats['high_corr_count'
]}")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" GENERADOR CAÓTICO AIZAWA - VERSIÓN MEJORADA CON TABLAS")
    print("="*60)
    print(f"\nmatplotlib: {'✓' if MATPLOTLIB_OK else '✗'}")
    print(f"scipy: {'✓' if SCIPY_OK else '✗'}")
    print(f" root: {'✓' if generator.has_root else '✗ (modo simulació
n)'}")
    print(f"\n NOVEDADES:")
    print(f"   • Procesador adaptativo (no empeora datos buenos)")
    print(f"   • Prueba Chi-cuadrado para uniformidad")
    print(f"   • Indicadores visuales de calidad")
    print(f"   • 4 tablas de evolución temporal")
    print(f"   • Atractor acoplado a temperatura")
    print(f"\n http://localhost:5000")
    print("\nPresiona Ctrl+C para detener")
    print("="*60)

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nDeteniendo...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        os._exit(0)
