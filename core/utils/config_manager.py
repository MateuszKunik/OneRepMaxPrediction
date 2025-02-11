import os
import yaml

                                                                                
class ConfigManager:
    def __init__(self, configs_path: str = None):
        """
        opis klasy
        """
        self.configs_path = configs_path


    def load_config(self, config_name: str) -> dict:
        file_path = os.path.join(self.configs_path, f"{config_name}.yml")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Plik konfiguracyjny '{file_path}' nie istnieje.")
        
        with open(file_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
                return config if config is not None else {}
            
            except yaml.YAMLError as e:
                raise yaml.YAMLError(
                    f"Błąd parsowania pliku YAML: {file_path}\n{e}")