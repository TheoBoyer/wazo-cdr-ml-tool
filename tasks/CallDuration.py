from core.Regression import Regression
from core.CDR import CDR
from core.Variable import Category

from features.TimeOfDay import TimeOfDay
from features.DayOfWeek import DayOfWeek
from features.Hour import Hour
from features.DestnationContainsStar import DestnationContainsStar
from features.Duration import Duration

class CallDuration(Regression):
    def __init__(self):
        super(CallDuration, self).__init__([
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
            Duration()
        ], CDR)