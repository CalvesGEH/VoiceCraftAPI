#!/bin/bash

info() {
    echo -e "[$(date '+%H:%M:%S')]\033[32m[INFO]\033[0m ${@}"
}
error() {
    echo -e "[$(date '+%H:%M:%S')]\033[31m[ERROR]\033[0m ${@}" 
}
warning() {
    echo -e "[$(date '+%H:%M:%S')]\033[33m[WARNING]\033[0m ${@}"
}

install_apt_packages() {
    # Install all packages from VoiceCraft README: https://github.com/jasonppy/VoiceCraft
    info "Installing apt packages..."
    sudo apt-get update
    sudo apt-get install -y git
    sudo apt-get install -y ffmpeg
    sudo apt-get install -y espeak-ng
    sudo apt-get install -y espeak espeak-data libespeak1 libespeak-dev
    sudo apt-get install -y festival*
    sudo apt-get install -y build-essential
    sudo apt-get install -y flac libasound2-dev libsndfile1-dev vorbis-tools
    sudo apt-get install -y libxml2-dev libxslt-dev zlib1g-dev
}

create_voicecraftapi_user() {
    if ! id voicecraftapi >/dev/null 2>&1; then
        info "Creating voicecraftapi user."
        sudo useradd -r -U -m -d /usr/share/voicecraftapi voicecraftapi
    fi
}

configure_systemd() {
    info "Creating systemd service."
    sudo cp /usr/share/voicecraftapi/VoiceCraftAPI/voicecraft-api.service /etc/systemd/system/voicecraft-api.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now voicecraft-api.service
}

info "Please input your password so that the script can use sudo to configure."
sudo -v
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

install_apt_packages

create_voicecraftapi_user

# Login as user
sudo -i -u voicecraftapi /bin/bash << EOF
cd ~/
git clone https://github.com/CalvesGEH/VoiceCraftAPI.git
cd ~/VoiceCraftAPI
./install_voicecraftapi.sh --skip-apt
EOF

configure_systemd