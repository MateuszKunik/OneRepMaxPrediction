from .project_manager import ProjectManager
from .config_manager import ConfigManager

def setup_configs():
    project = ProjectManager()
    configs_directory_path = project.get_configs_directory_path()

    config = ConfigManager(configs_directory_path)

    pose_estimation = config.load_config("parameters_pose_estimation")
    data_processing =  config.load_config("parameters_data_processing")
    machine_learning = config.load_config("parameters_machine_learning")

    return pose_estimation, data_processing, machine_learning
