#!/bin/bash
# Reverse XFCE + VNC setup on Raspberry Pi OS Bookworm

set -e

echo "Stopping and disabling VNC service..."
VNC_USER=$(whoami)
sudo systemctl stop vncserver@$VNC_USER
sudo systemctl disable vncserver@$VNC_USER
sudo rm -f /etc/systemd/system/vncserver@$VNC_USER.service
sudo systemctl daemon-reload

echo "Removing VNC server packages..."
sudo apt purge -y tigervnc-standalone-server tigervnc-common
rm -rf ~/.vnc

echo "Disabling LightDM display manager..."
sudo systemctl stop lightdm
sudo systemctl disable lightdm

echo "Setting system to boot into console (text mode)..."
sudo systemctl set-default multi-user.target

echo "Removing XFCE packages..."
sudo apt purge -y xfce4 xfce4-goodies task-xfce-desktop

echo "Auto-remove unnecessary packages..."
sudo apt autoremove -y

echo "Optional: D-Bus cleanup skipped (not recommended, system depends on it)"

echo "Rebooting system..."
sudo reboot
