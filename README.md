# aizawa_vonneumann_entropy
Generador caótico de entropía con post-procesamiento Von Neumann. Captura sensores térmicos en Android, visualiza el atractor de Aizawa y genera hashes SHA-2 con autocorrelación R(1) &lt; 0.2.
# Von Neumann Chaotic Analyzer

Analizador caotico que utiliza sensores termicos de CPU y PMIC en dispositivos Android como fuente de entropia fisica. Implementa un pipeline de post-procesamiento Von Neumann doble para eliminar correlaciones y generar bytes con alta calidad estadistica.

## Caracteristicas

- Captura de sensores termicos en tiempo real (thermal_zone7 y thermal_zone9)
- Visualizacion 3D del atractor de Aizawa modulado por temperatura
- Calculo de exponente de Lyapunov (λ ≈ 0.94, caos confirmado)
- Analisis de min-entropia por diferencia de muestras
- Pipeline de post-procesamiento: XOR + Von Neumann + XOR + Von Neumann + XOR
- Autocorrelacion post-procesada R(1) < 0.2
- Generacion de hashes SHA-256, SHA-384 y SHA-512
- Guardado automatico de graficas en /storage/emulated/0/Download/has1py/YYYYMMDD_HHMMSS/
- Interfaz web Flask con actualizacion en tiempo real

## Requisitos

- Dispositivo Android con root (para acceso a sensores termicos)
- Termux instalado
- Python 3.8+
- Dependencias: flask, numpy, matplotlib, scipy

## Instalacion

```bash
# En Termux, actualizar paquetes
pkg update && pkg upgrade

# Instalar dependencias del sistema
pkg install python python-pip

# Instalar librerias Python
pip install flask numpy matplotlib scipy

# Clonar repositorio
git clone https://github.com/tu-usuario/von-neumann-chaotic-analyzer.git
cd von-neumann-chaotic-analyzer

# Ejecutar (requiere root)
python von_aizawa.py
