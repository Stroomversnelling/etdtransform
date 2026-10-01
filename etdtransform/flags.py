"""The imputation stage's flag vocabulary.

``ImputeType`` records, per cell, what the fill found and what it did. It is
written into every dataset the stage produces, so it is the vocabulary a reader
needs rather than an implementation detail of the fill -- which is why it lives
here and not in the module that happens to write it.

``SIGNALS`` and ``ACTIONS`` are the categorisation, and the composites on the
class are the reference: nothing should keep a second copy of which flag is
which. ``DESCRIPTIONS`` is the one-line meaning per member, kept beside the
member it describes so the two are reviewed in the same change.

``etdmap.flags`` is the counterpart for the mapping stage.
"""

from enum import IntFlag, auto


class ImputeType(IntFlag):
    """What the fill found about a cell, and what it did to it.

    One value per cell, written into the impute-type column beside each interval
    column, so it travels with the data and is read long after the run.

    Two things are deliberately not written here. Which members are conditions
    and which are things done to a value is ``SIGNALS`` and ``ACTIONS`` below --
    read it from those, never from a member's position in this list, which is
    how two fill routes once ended up counted as conditions. What each member
    means is ``DESCRIPTIONS``, at the foot of this module. Restating either here
    would be a second copy that drifts.

    ``auto()`` ties a value to its declaration position, and the values are
    already written into datasets on disk. So a new member is **appended**, and
    no member is ever inserted, reordered or removed.
    """

    NONE = 0

    # --- signals: the condition the cell was in ---

    NEGATIVE_GAP_JUMP = auto()
    NEAR_ZERO_GAP_JUMP = auto()
    UNIFORM_FILL = auto()
    SCALED_FILL = auto()
    ZERO_END_VALUE = auto()
    POSITIVE_END_VALUE = auto()
    NO_END_VALUE = auto()

    THRESHOLD_ADJUSTED = auto()

    # --- actions: what imputation did about it ---

    ZERO_FILL = auto()
    UNSCALED_AVERAGE = auto()
    HOUSEHOLD_SCALED_AVERAGE = auto()
    NOT_FILLED = auto()

    # --- appended: signals ---

    ONLY_CLOSING_READING = auto()

    ONLY_OPENING_READING = auto()

    THRESHOLD_VIOLATED = auto()

    # --- appended: actions ---

    NULLED = auto()

    NULLED_POST_IMPUTATION = auto()

    # --- appended: signals ---

    NO_CUMULATIVE_VALUE = auto()

    # --- appended: actions ---

    CLIPPED = auto()

    CLIPPED_REDISTRIBUTION = auto()

    UNIFORM_RESIDUAL = auto()

    # --- appended: signals ---

    CLIPPED_AVERAGE = auto()

    BOUND_FROM_IMPUTED = auto()

    INFEASIBLE_BOUNDS_HARD = auto()

    INFEASIBLE_BOUNDS_SOFT = auto()

    INFEASIBLE_GAP = auto()

    DELIVERED_OUTSIDE_RULE_BOUND = auto()

    # --- composites ---

    SIGNALS = (
        NEGATIVE_GAP_JUMP
        | NEAR_ZERO_GAP_JUMP
        | ZERO_END_VALUE
        | POSITIVE_END_VALUE
        | NO_END_VALUE
        | THRESHOLD_ADJUSTED
        | ONLY_CLOSING_READING
        | ONLY_OPENING_READING
        | THRESHOLD_VIOLATED
        | NO_CUMULATIVE_VALUE
        | CLIPPED_AVERAGE
        | BOUND_FROM_IMPUTED
        | INFEASIBLE_BOUNDS_HARD
        | INFEASIBLE_BOUNDS_SOFT
        | INFEASIBLE_GAP
        | DELIVERED_OUTSIDE_RULE_BOUND
    )
    ACTIONS = (
        UNIFORM_FILL
        | SCALED_FILL
        | ZERO_FILL
        | UNSCALED_AVERAGE
        | HOUSEHOLD_SCALED_AVERAGE
        | NOT_FILLED
        | NULLED
        | NULLED_POST_IMPUTATION
        | CLIPPED
        | CLIPPED_REDISTRIBUTION
        | UNIFORM_RESIDUAL
    )

    FILLED = (
        UNIFORM_FILL
        | SCALED_FILL
        | ZERO_FILL
        | UNSCALED_AVERAGE
        | HOUSEHOLD_SCALED_AVERAGE
    )

    IMPUTED_VALUE = (
        FILLED
        | NULLED
        | NULLED_POST_IMPUTATION
        | CLIPPED
        | CLIPPED_REDISTRIBUTION
        | UNIFORM_RESIDUAL
    )

DESCRIPTIONS = {
    "NEGATIVE_GAP_JUMP": "The cumulative total fell across the gap.",
    "NEAR_ZERO_GAP_JUMP": "The cumulative total did not move across the gap.",
    "UNIFORM_FILL": "The gap's own jump, spread evenly over its cells.",
    "SCALED_FILL": "The project average, rescaled so the gap sums to its recorded jump.",
    "ZERO_END_VALUE": "No opening cumulative reading, and the closing one is zero.",
    "POSITIVE_END_VALUE": "No opening cumulative reading, and the closing one is positive.",
    "NO_END_VALUE": "An opening cumulative reading and no closing one.",
    "THRESHOLD_ADJUSTED": "DEPRECATED. Duplicates THRESHOLD_VIOLATED.",
    "ZERO_FILL": "A zero written into the cell.",
    "UNSCALED_AVERAGE": "The project average, unscaled.",
    "HOUSEHOLD_SCALED_AVERAGE": "The project average, multiplied by this household's own factor.",
    "NOT_FILLED": "A cell inside a gap that holds no value.",
    "ONLY_CLOSING_READING": "Only the closing cumulative reading bounds the gap.",
    "ONLY_OPENING_READING": "Only the opening cumulative reading bounds the gap.",
    "THRESHOLD_VIOLATED": "The value lies outside the bound it is judged against.",
    "NULLED": "A value imputation blanked.",
    "NULLED_POST_IMPUTATION": "An imputed value that was blanked.",
    "NO_CUMULATIVE_VALUE": "Neither end of the gap has a cumulative reading.",
    "CLIPPED": "A filled value moved to the edge of its narrowed bound.",
    "CLIPPED_REDISTRIBUTION": "A value moved to absorb another cell's clip.",
    "UNIFORM_RESIDUAL": (
        "Leftover volume spread evenly over cells the profile weights at "
        "zero."
    ),
    "CLIPPED_AVERAGE": (
        "The average used for the cell had been clipped into the cell's "
        "narrowed bound."
    ),
    "BOUND_FROM_IMPUTED": (
        "The cell's bound rests on a value this run filled rather than a "
        "measurement."
    ),
    "INFEASIBLE_BOUNDS_HARD": "The narrowed bound is empty using measurements alone.",
    "INFEASIBLE_BOUNDS_SOFT": (
        "The narrowed bound is empty, and a value this run filled contributes "
        "to it."
    ),
    "INFEASIBLE_GAP": (
        "The gap's measured total cannot be reconciled with its cells' "
        "bounds."
    ),
    "DELIVERED_OUTSIDE_RULE_BOUND": (
        "A delivered value outside the bound the rules narrow its row to, and "
        "inside the bound its own column declares."
    ),
}
