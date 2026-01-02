import zipfile
import os

def create_zip():
    files_to_include = [
        "joschekscaptions.py",
        "README.md",
        "requirements.txt",
        "LICENSE",
        "tooltips.json",
        "yolov8n-seg.pt",
        "jc32x32.png",
        "jc64x64.png"
    ]
    
    output_filename = "joschekscaptioner_v1.01.zip"
    
    try:
        with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in files_to_include:
                if os.path.exists(file):
                    print(f"Adding {file}...")
                    zf.write(file)
                else:
                    print(f"Warning: {file} not found")
        print(f"Successfully created {output_filename}")
    except Exception as e:
        print(f"Error creating zip: {e}")

if __name__ == "__main__":
    create_zip()
