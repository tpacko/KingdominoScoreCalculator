import re
from pathlib import Path


def main():
    # Get all files
    files = list(Path('.').glob('*.jpg'))

    # Find max number from gameX.jpg files
    game_files = [f for f in files if re.match(r'game\d+\.jpg', f.name)]
    max_num = max([int(re.search(r'game(\d+)\.jpg', f.name).group(1)) for f in game_files], default=0)

    # Get new files (IMG_*.jpg)
    new_files = sorted([f for f in files if f.name.startswith('IMG_')])

    # Rename
    for i, file in enumerate(new_files, start=1):
        new_name = f'game{max_num + i}.jpg'
        print(f'{file.name} -> {new_name}')
        file.rename(new_name)


if __name__ == '__main__':
    main()