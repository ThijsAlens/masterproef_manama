import os
import re

import dataset_creation.config as config

def main():
    for file in os.listdir(config.ROOT_DIRECTORY):
        if re.search(r'\._color.png$', file, re.IGNORECASE):
            img = file
            
            

if __name__ == "__main__":
    main()