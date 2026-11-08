#!/bin/bash
# Starts (or creates, on first run) the GCP VM and brings the app up.
# One command, so "forgot to turn it off" doesn't quietly eat your $50
# credit, and "forgot to turn it ON before the demo" doesn't either.
#
# Fill in these four values before first use:
set -e
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-resumematch-vm}"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"

EXISTS=$(gcloud compute instances list --project="$PROJECT_ID" --filter="name=$INSTANCE_NAME" --format="value(name)" || true)

if [ -z "$EXISTS" ]; then
  echo "Creating $INSTANCE_NAME (this only happens once)..."
  gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$GPU_TYPE,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu121 --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB
  echo "VM created. SSH in once, install nvidia-container-toolkit + docker-compose, clone this repo."
else
  echo "Starting existing instance $INSTANCE_NAME..."
  gcloud compute instances start "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE"
fi

echo ""
echo "Once it's up: ssh in, cd into the repo, and run:"
echo "  docker-compose -f docker-compose.gcp.yml up --build -d"
echo ""
echo "REMEMBER: run gcp/stop.sh the moment you're done. At ~\$0.50-0.55/hr,"
echo "\$50 of credit is roughly 90 hours — don't let this idle overnight."
