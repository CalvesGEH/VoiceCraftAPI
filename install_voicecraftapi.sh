######################
#      Variables     #
######################

# Folder path to script
VOICECRAFTAPI_PATH="$(dirname "$(readlink -f "${0}")")"
CONDA_PATH="${VOICECRAFTAPI_PATH}/conda"
CONDA_BINARY="${CONDA_PATH}/bin/conda"

######################
#     Functions     #
######################

info() {
    echo -e "[$(date '+%H:%M:%S')]\033[32m[INFO]\033[0m ${@}"
}
error() {
    echo -e "[$(date '+%H:%M:%S')]\033[31m[ERROR]\033[0m ${@}" 
}
warning() {
    echo -e "[$(date '+%H:%M:%S')]\033[33m[WARNING]\033[0m ${@}"
}

install_conda() {
    info "Installing Conda..."
    # Download the Conda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    # Install Conda
    bash miniconda.sh -b -p "${CONDA_PATH}"
    # Initialize Conda
    $CONDA_BINARY init
    # Cleanup
    rm miniconda.sh
    info "Conda installation completed."
}

install_api_packages() {
    info "Installing VoiceCraftAPI packages..."
    # Create a conda environment named voicecraftapi only if it doesn't exist
    if [ "$(${CONDA_BINARY} env list | grep voicecraftapi)" ]; then
        info "VoiceCraftAPI environment already exists. Updating..."
    else
        $CONDA_BINARY create -n voicecraftapi python=3.9.16 -y
    fi
    # Activate the environment
    $CONDA_BINARY activate voicecraftapi

    # Install all packages from VoiceCraft README: https://github.com/jasonppy/VoiceCraft
    info "Installing apt packages..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
    sudo apt-get install espeak-ng
    sudo apt-get install -y espeak espeak-data libespeak1 libespeak-dev
    sudo apt-get install -y festival*
    sudo apt-get install -y build-essential
    sudo apt-get install -y flac libasound2-dev libsndfile1-dev vorbis-tools
    sudo apt-get install -y libxml2-dev libxslt-dev zlib1g-dev

    info "Installing pip requirements..."
    pip install -r "${VOICECRAFTAPI_PATH}/requirements.txt"

    info "Installing mfa and mfa models..."
    $CONDA_BINARY install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
    mfa model download dictionary english_us_arpa
    mfa model download acoustic english_us_arpa

    info "Finished installing all VoiceCraftAPI packages."
}

# Call sudo once and then refresh it every 60s so that user only has to input it once.
info "Please input your password so that the script can use sudo to install system packages."
sudo -v
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# Enter the VoiceCraftAPI path
info "Entering VoiceCraftAPI folder: ${VOICECRAFTAPI_PATH}"
cd "${VOICECRAFTAPI_PATH}"

# If the conda binary doesn't exist, run install_conda()
if ! command -v $CONDA_BINARY &> /dev/null; then
    install_conda
else
    info "Conda already installed. Activating 'voicecraftapi' environment..."
    $CONDA_BINARY activate voicecraftapi
fi

# Install packages
install_api_packages

# Clone the VoiceCraft repository.
info "Cloning the VoiceCraft repository..."
git clone https://github.com/jasonppy/VoiceCraft