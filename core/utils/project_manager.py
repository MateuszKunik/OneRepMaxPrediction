import os

class ProjectManager:
    def __init__(self, project_root: str = None):
        """
        opis
        """
        if project_root is None:
            file_path = os.path.abspath(__file__)
            self.project_root = find_project_root(file_path)

        else:
            self.project_root = project_root

        self.data_directory_path = self.get_data_directory_path()
        # self.configs_directory_path = self.get_configs_directory_path()

    def get_project_root(self) -> str:
        return self.project_root
    

    def get_configs_directory_path(self) -> str:
        return os.path.join(self.project_root, "configs")


    def get_data_directory_path(self) -> str:
        return os.path.join(self.project_root, "data")
    

    def get_raw_data_path(self) -> str:
        return os.path.join(self.data_directory_path, "01_raw")

    
    def get_intermediate_data_path(self) -> str:
        return os.path.join(self.data_directory_path, "02_intermediate")
    

    def get_primary_data_path(self) -> str:
        return os.path.join(self.data_directory_path, "03_primary")


    def get_model_data_path(self) -> str:
        return os.path.join(self.data_directory_path, "06_models")




def find_project_root(current_path: str) -> str:
    project_name = "Research"

    return os.path.join(current_path.split(project_name)[0], project_name)