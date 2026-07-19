#!/bin/bash
# Installs the udev rule for a stable /dev/so101_follower symlink, and adds the
# current user to the dialout/audio groups needed for the arm's serial port and
# the USB headset. Log out/in for group membership to take effect.
set -euo pipefail
cd "$(dirname "$0")/.."

sudo cp scripts/udev/99-so101.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

sudo usermod -aG dialout "$USER"
sudo usermod -aG audio "$USER"

echo "Installed. Log out/in for group membership to take effect."
echo "Verify with: ls -la /dev/so101_follower"
