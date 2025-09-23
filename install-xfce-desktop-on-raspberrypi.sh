#!/bin/bash
# XFCE + VNC setup script for Raspberry Pi OS Bookworm

set -e

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing XFCE desktop environment..."
sudo apt install xfce4 xfce4-goodies -y

echo "Installing LightDM display manager..."
sudo apt install lightdm -y

echo "Setting LightDM as default display manager..."
sudo dpkg-reconfigure lightdm

echo "Installing D-Bus for session management..."
sudo apt install dbus-x11 -y

echo "Enabling and starting D-Bus service..."
sudo systemctl enable dbus
sudo systemctl start dbus

echo "Enabling and starting LightDM service..."
sudo systemctl enable lightdm
sudo systemctl start lightdm

echo "Setting system to boot into graphical target (GUI)..."
sudo systemctl set-default graphical.target

echo "Installing TigerVNC server..."
sudo apt install tigervnc-standalone-server tigervnc-common -y

# Configure VNC for the current user
VNC_USER=$(whoami)
VNC_HOME=$(eval echo "~$VNC_USER")
VNC_XSTARTUP="$VNC_HOME/.vnc/xstartup"

echo "Creating VNC startup configuration..."
mkdir -p "$VNC_HOME/.vnc"
cat > "$VNC_XSTARTUP" <<EOF
#!/bin/sh
xrdb \$HOME/.Xresources
startxfce4 &
EOF

chmod +x "$VNC_XSTARTUP"

echo "Setting VNC password (you will be prompted)..."
vncpasswd

echo "Creating systemd service for VNC server..."
VNC_SERVICE="/etc/systemd/system/vncserver@$VNC_USER.service"
sudo tee "$VNC_SERVICE" > /dev/null <<EOF
[Unit]
Description=Start TigerVNC server at startup
After=syslog.target network.target

[Service]
Type=forking
User=$VNC_USER
PAMName=login
PIDFile=$VNC_HOME/.vnc/%H:1.pid
ExecStartPre=-/usr/bin/vncserver -kill :1
ExecStart=/usr/bin/vncserver :1 -geometry 1280x720 -depth 24
ExecStop=/usr/bin/vncserver -kill :1

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Enabling VNC server to start at boot..."
sudo systemctl enable vncserver@$VNC_USER
sudo systemctl start vncserver@$VNC_USER

echo "Setup complete! Rebooting system..."
sudo reboot
