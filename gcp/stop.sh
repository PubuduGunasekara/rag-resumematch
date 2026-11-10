#!/bin/bash
# Stops the VM. Stopped = no compute billing (disk storage still bills,
# but that's cents/month, not dollars/hour). This is the other half of
# the cost guard for start.sh: run this every single time you're
# done, no exceptions.
set -e
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-resumematch-vm}"

gcloud compute instances stop "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE"
echo "Stopped. Verify in the console if you don't fully trust this output:"
echo "https://console.cloud.google.com/compute/instances?project=$PROJECT_ID"
