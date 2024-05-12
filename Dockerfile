# Use an official Ubuntu runtime as the base image
FROM ubuntu:latest

# Set the working directory in the container
WORKDIR /app

# Install necessary apt packages, including sudo
RUN apt-get update && apt-get install -y \
    git \
    wget \
    sudo

# Create a new user called "docker" and add it to the sudo group
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

# Copy the VoiceCraftAPI files to the working directory
COPY . .

# Make the install_voicecraftapi.sh and run_api.sh scripts executable
RUN chmod +x install_voicecraftapi.sh run_api.sh

# Allow the "docker" user to run sudo commands without a password prompt
RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to the "docker" user
USER docker

# Execute the install_voicecraftapi.sh script as the "docker" user
RUN sudo bash ./install_voicecraftapi.sh

# Expose the port on which the API will run (adjust if needed)
EXPOSE 8245

# Set the command to run the run_api.sh script as the "docker" user
CMD ["sudo", "bash", "./run_api.sh"]