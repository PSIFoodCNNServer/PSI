import json
from pathlib import Path

from Food import Food


BASE_DIR = Path(__file__).resolve().parent
FOOD_BASE_PATH = BASE_DIR / "foodBase.txt"


def load_food_base(path: Path = FOOD_BASE_PATH):
	if not path.exists():
		return {}

	with path.open("r", encoding="utf-8") as file:
		file_content = file.read()

	raw_data = json.loads(file_content)

	food_dictionary = {}
	for food_name, food_data in raw_data.items():
		food_dictionary[food_name] = Food.from_dict(food_name, food_data)

	return food_dictionary

