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
    # Source conda
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    # Init conda
    conda init
    # Cleanup
    rm miniconda.sh
    info "Conda installation completed."
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

install_api_packages() {
    info "Installing VoiceCraftAPI pip packages..."
    # Create a conda environment named voicecraftapiconda only if it doesn't exist
    if [ "$(conda env list | grep voicecraftapiconda)" ]; then
        info "VoiceCraftAPI environment already exists. Updating..."
    else
        conda create -n voicecraftapiconda python=3.9.16 -y
    fi
    # Activate the environment
    conda activate voicecraftapiconda

    info "Installing pip requirements..."
    pip install numpy==1.26.4 # Numpy is seperate otherwise 'aeneas' complains.
    pip install -r "${VOICECRAFTAPI_PATH}/requirements.txt"
    # Audiocraft needs to be installed seperately from requirement.txt as it seems like 'hydra' will complain about a missing 
    # config folder when installed through requirements.txt.
    pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft

    info "Installing mfa and mfa models..."
    conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
    mfa model download dictionary english_us_arpa
    mfa model download acoustic english_us_arpa

    info "Finished installing all VoiceCraftAPI packages."
}

for arg in "$@"; do
    if [ "$arg" == "--skip-apt" ]; then
        info "--skip-apt flag set. Skipping apt package installation."
        skip_apt_installation=true
    fi
done

# Call sudo once and then refresh it every 60s so that user only has to input it once.
info "Please input your password so that the script can use sudo to install system packages."
sudo -v
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# Log all paths
info "VOICECRAFTAPI_PATH: ${VOICECRAFTAPI_PATH}"
info "CONDA_PATH: ${CONDA_PATH}"
info "CONDA_BINARY: ${CONDA_BINARY}"

# Enter the VoiceCraftAPI path
info "Entering VoiceCraftAPI folder: ${VOICECRAFTAPI_PATH}"
cd "${VOICECRAFTAPI_PATH}"

# If the conda binary doesn't exist, run install_conda()
if ! command -v $CONDA_BINARY &> /dev/null; then
    install_conda
else
    info "Conda already installed. Activating 'voicecraftapiconda' environment..."
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate voicecraftapiconda
fi
source "${CONDA_PATH}/etc/profile.d/conda.sh"

# Install packages
if [ -z "${skip_apt_installation}" ]; then
    error "Installing pacakges"
    install_apt_packages
fi
install_api_packages

# Clone the VoiceCraft repository.
if [ -d "${VOICECRAFTAPI_PATH}/VoiceCraft" ]; then
    info "VoiceCraft repository already exists."
else
    info "Cloning the VoiceCraft repository..."
    git clone https://github.com/jasonppy/VoiceCraft
    cd "${VOICECRAFTAPI_PATH}/VoiceCraft"
    git reset --hard 57079c4
    cd "${VOICECRAFTAPI_PATH}"
fi