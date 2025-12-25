"""Download model weights."""
import os
import urllib.request

WEIGHTS_DIR = "weights"
MOBILE_SAM_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"

def download_mobile_sam():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = os.path.join(WEIGHTS_DIR, "mobile_sam.pt")
    
    if os.path.exists(path):
        print("mobile_sam.pt already exists")
        return
    
    print("Downloading MobileSAM weights...")
    urllib.request.urlretrieve(MOBILE_SAM_URL, path)
    print(f"Saved to {path}")

if __name__ == "__main__":
    download_mobile_sam()
    print("\nMiDAS weights will auto-download on first run via torch.hub")

