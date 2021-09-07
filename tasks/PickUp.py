from core.Classification import Classification
from core.CDR import CDR
from core.Variable import Category, Binary

from features.TimeOfDay import TimeOfDay
from features.DayOfWeek import DayOfWeek
from features.Hour import Hour
from features.DestnationContainsStar import DestnationContainsStar

class PickUp(Classification):
    def __init__(self):
        super(PickUp, self).__init__([
            Category("call_direction"),
            Category("destination_extension"),
            Category("destination_internal_context"),
            Category("destination_internal_extension"),
            Category("destination_line_id"),
            Category("requested_extension"),
            Category("requested_internal_extension"),
            Category("source_extension"),
            Category("source_internal_context"),
            Category("source_internal_extension"),
            Category("stack"),
            TimeOfDay(),
            DayOfWeek(),
            Hour(),
            DestnationContainsStar()
        ], [
            Binary("answered")
        ], CDR)