#!/bin/bash

# Settings
RAMDISK_DIR="/mnt/ramdisk"
RAMDISK_SIZE="10G"  # Adjust size

# Create mount point if it doesn't exist
mkdir -p "$RAMDISK_DIR"

# Mount tmpfs only if not already mounted
if ! mountpoint -q "$RAMDISK_DIR"; then
    mount -t tmpfs -o size=$RAMDISK_SIZE tmpfs "$RAMDISK_DIR"
    echo "RAM disk mounted at $RAMDISK_DIR with size $RAMDISK_SIZE"
else
    echo "RAM disk already mounted at $RAMDISK_DIR"
fi

#chmod +x mount_ramdisk.sh