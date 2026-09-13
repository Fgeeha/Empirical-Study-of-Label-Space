#!/bin/bash
# Collect edge deployment environment information from Raspberry Pi.
# Run on the Raspberry Pi (ssh rpi) to collect runtime versions.
# Output goes to results/edge/environment.txt.

OUTFILE="results/edge/environment.txt"
mkdir -p results/edge

echo "=== Edge Environment Dump ===" | tee "$OUTFILE"
echo "Collected: $(date)" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "--- Hardware ---" | tee -a "$OUTFILE"
cat /proc/cpuinfo | grep "Model\|Hardware\|Revision" | tee -a "$OUTFILE"
free -h | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "--- OS ---" | tee -a "$OUTFILE"
uname -a | tee -a "$OUTFILE"
cat /etc/os-release | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "--- Python ---" | tee -a "$OUTFILE"
python3 --version | tee -a "$OUTFILE"
python3 -c "import platform; print(platform.python_version())" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "--- TFLite Runtime ---" | tee -a "$OUTFILE"
python3 -c "import tflite_runtime; print('tflite_runtime:', tflite_runtime.__version__)" 2>/dev/null | tee -a "$OUTFILE" || \
python3 -c "import tensorflow as tf; print('tensorflow:', tf.__version__); print('TFLite:', tf.lite.__version__ if hasattr(tf.lite, '__version__') else 'N/A')" 2>/dev/null | tee -a "$OUTFILE" || \
echo "TFLite runtime version: not recorded" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "--- PyCoral / EdgeTPU ---" | tee -a "$OUTFILE"
python3 -c "from pycoral.utils import edgetpu; print('pycoral:', edgetpu.__version__)" 2>/dev/null | tee -a "$OUTFILE" || \
echo "pycoral version: not recorded" | tee -a "$OUTFILE"
dpkg -l libedgetpu1-std 2>/dev/null | grep libedgetpu | tee -a "$OUTFILE" || \
echo "libedgetpu version: not recorded" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "--- CPU Governor ---" | tee -a "$OUTFILE"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null | tee -a "$OUTFILE" || echo "CPU governor: not recorded" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "--- USB Mode ---" | tee -a "$OUTFILE"
lsusb 2>/dev/null | grep -i coral | tee -a "$OUTFILE" || echo "Coral USB: not detected" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "--- Installed packages (relevant) ---" | tee -a "$OUTFILE"
pip3 list 2>/dev/null | grep -iE "tensorflow|tflite|coral|pycoral|numpy|pillow|psutil" | tee -a "$OUTFILE"

echo "" | tee -a "$OUTFILE"
echo "=== Done ===" | tee -a "$OUTFILE"
echo "Saved to $OUTFILE"
