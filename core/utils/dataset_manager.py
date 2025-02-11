import os


class VideoDatasetManager:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Given path '{self.folder_path}' dosn't exists.")
        
        self.video_files = self._load_video_files()

    
    def _load_video_files(self) -> list:
        return [
            os.path.join(self.folder_path, file)
            for file in os.listdir(self.folder_path)
            if file.endswith((".mp4", ".MP4"))
        ]
    

    def get_video_files(self) -> list:
        return self.video_files
    

    def describe(self) -> None:
        description = {
            "Number of files: ": len(self.video_files),
            "Examples: ": self.video_files[:5]}

        print(description)