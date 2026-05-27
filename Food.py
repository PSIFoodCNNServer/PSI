from dataclasses import dataclass


@dataclass(slots=True)
class Food:
    name: str
    full_receipe: str
    kcal: int
    protein: int
    carbs: int
    fat: int

    @classmethod
    def from_dict(cls, name: str, data: dict):
        return cls(
            name=name,
            full_receipe=data.get("full_receipe", ""),
            kcal=int(data.get("kcal", 0)),
            protein=int(data.get("protein", 0)),
            carbs=int(data.get("carbs", 0)),
            fat=int(data.get("fat", 0)),
        )

    def to_dict(self):
        return {
            "name": self.name,
            "full_receipe": self.full_receipe,
            "kcal": self.kcal,
            "protein": self.protein,
            "carbs": self.carbs,
            "fat": self.fat,
        }

    def getKcal(self):
        return self.kcal

    def getProtein(self):
        return self.protein

    def getCarbs(self):
        return self.carbs

    def getFat(self):
        return self.fat

    def getReceipe(self):
        return self.full_receipe

    def getName(self):
        return self.name
