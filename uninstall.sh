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

delete_voicecraftapi_user() {
    if id voicecraftapi >/dev/null 2>&1; then
        info "Deleting voicecraftapi user."
        sudo userdel -r -f voicecraftapi
    else
        warning "voicecraftapi user does not exist. Skipping deleting user..."
    fi
}

configure_systemd() {
    info "Stopping and removing systemd service."
    sudo systemctl stop voicecraft-api.service
    sudo systemctl disable voicecraft-api.service
    sudo rm /etc/systemd/system/voicecraft-api.service
    sudo systemctl daemon-reload
}

info "Please input your password so that the script can use sudo to configure."
sudo -v
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

delete_voicecraftapi_user

configure_systemd