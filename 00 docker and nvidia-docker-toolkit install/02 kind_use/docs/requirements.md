# Requirements

## Hardware

- NVIDIA GPU (tested with RTX 3060)
- At least 8GB RAM
- 4 CPU cores minimum

## Software

- **Docker**: 20.10+ (for Kind)
- **NVIDIA Drivers**: 575.64.03+ (with CUDA 12.9)
- **Linux Kernel**: 5.4+ (for GPU passthrough)
- **curl/wget**: For downloads

## Network

- Ports 12741-12761 available
- Internet access for image pulls

## Permissions

- Sudo access for Kind installation
- Docker group membership or sudo for Docker commands

## Verification

Run `./validate.sh` to check prerequisites.