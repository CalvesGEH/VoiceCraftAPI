# VoiceCraft API

This is a API for the extremely awesome [VoiceCraft](https://github.com/jasonppy/VoiceCraft) repository. The API allows you to generate voices from a single example .wav file and generate text using those voices.

## Installing

Currently, the API is only natively supported on Ubuntu/Debian systems, but that may change in the future if enough people want other distributions. Alternatively, the API can be run on Windows / macOS using Docker.

### Automatic

The VoiceCraft API can easily be installed (on ubuntu/debian) by running the following command:

```bash
curl -fsSL https://raw.github.com/CalvesGEH/VoiceCraftAPI/main/install.sh | sh
```

This will download and run the installation script which will create a new user `voicecraftapi` and configure the API, VoiceCraft and a systemd service that will run on startup. You need `SUDO` access to run this as it needs to install packages and create the systemd service.

### Manual/Development

The VoiceCraft API can also be built manually for either development or whatever other reason.

```bash
git clone https://github.com/CalvesGEH/VoiceCraftAPI.git
cd VoiceCraftAPI
./install_voicecraftapi.sh
# you can then run the api if desired
./run_api.sh
```

Once installed, you can also manually edit and install the systemd service.

### Docker
VoiceCraft API is compatible with Windows / macOS via Docker. Run the commands below to start the server:
```bash
docker build -t voicecraft-api .
docker run -p 8245:8245 --name voicecraft-api voicecraft-api
```

The API will be accessible on http://localhost:8245.

Note for Windows users: You may need to change `install_voicecraftapi.sh` and `run_api.sh` from a CRLF to an LF End of Line Sequence format to allow the Dockerfile to read these files correctly.

## Uninstalling

I have also provided a script which will uninstall VoiceCraftAPI except for the APT packages required, those can be manually uninstalled if you'd like.

```bash
curl -fsSL https://raw.github.com/CalvesGEH/VoiceCraftAPI/main/uninstall.sh | sh
```

## API Calls

The API only exposes a couple of endpoints and all of these endpoints can be tested/accessed from `http://<YOUR_SERVER_IP>:8245/docs`. This `/docs` endpoint also includes information about each endpoint and is usually a good place to start if you are confused.

### /newvoice

This endpoint creates a new voice from a given .wav audio file and optional configuration parameters. The .wav file should be following be ~6-12s and be 16000Hz. The name of the new voice will be exactly matched the name of the .wav file (`frank.wav` will create voice `frank`).

You can include a transcript but if one is not given, the API will automatically transcribe it for you. I've never not had it be correct but you can check the logs if you think it's wrong.

### /voicelist

This endpoint simply returns a list of all available voices for inference.

### /editvoice/{voice}

This endpoint will edit the saved inference parameters of a voice. Check out your server's `/docs` endpoint for available parameters.

### /generateaudio/{voice}

This endpoint will generate audio as the given voice using the `target_text` given. It will return a .wav audio file streaming response.

## Examples

It is extremely simple to use once the API is up and running. I start by finding a suitable voice clip for the character I want to clone and then edit the file to ensure it is a .wav at 16000Hz. Then, I navigate to `http://<YOUR_SERVER_IP>:8245/docs` and test the `/newvoice` endpoint directly in the browser, giving it my .wav file and then executing the request. Now that the voice is generated, I can make a request to the server to generate text.

```bash
curl --output generated.wav -d "target_text=It is crazy how easy it is to use VoiceCraft api!" http://<YOUR_SERVER_IP>:8245/generateaudio/<VOICE>
```