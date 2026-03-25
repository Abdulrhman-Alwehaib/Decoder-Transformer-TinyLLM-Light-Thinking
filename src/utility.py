import os
import yaml
from ensure import ensure_annotations
from pathlib import Path

@ensure_annotations
def readYaml(pathYaml: Path) -> dict:
    try:
        with open(pathYaml,"r") as y:
            objYamlContent = yaml.safe_load(y)
            return objYamlContent
    except Exception as e:
        raise e