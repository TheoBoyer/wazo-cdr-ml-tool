from core.Variable import Binary

class DestnationContainsStar(Binary):
    def __init__(self):
        super().__init__("destination_contains_star", ["destination_extension"])

    def forward(self, data):
        data['destination_contains_star'] = data["destination_extension"].str.contains("\*").astype(float)
        return super().forward(data[['destination_contains_star']])