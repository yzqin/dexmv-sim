import os


def get_project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, "../../")
    return os.path.normpath(root_path)
