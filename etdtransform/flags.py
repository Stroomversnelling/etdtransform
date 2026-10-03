"""The per-cell flag vocabulary for the stages after mapping.

``ImputeType`` records, per cell, what the fill found and what it did, plus
what the calculated stage did to a derived cell. It is written into every
dataset those stages produce, so it is the vocabulary a reader needs rather
than an implementation detail of either -- which is why it lives here and not
in the module that happens to write it. The calculated stage shares this class
rather than declaring its own: three members do not justify a second
vocabulary a reader has to learn and a comparison has to decode separately.

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

    JITTER_CANDIDATE = auto()

    # --- appended: actions, written by the calculated stage ---

    CALCULATED = auto()

    # --- appended: signals, written by the calculated stage ---

    CALCULATED_THRESHOLD_VIOLATED = auto()

    CALCULATED_OUTSIDE_RULE_BOUND = auto()

    CALCULATED_RULE_BOUND_FROM_IMPUTED = auto()

    CALCULATED_UNDEFINED = auto()

    # --- appended: actions, written by the calculated stage ---

    CALCULATED_FROM_RULE_BOUND = auto()

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
        | JITTER_CANDIDATE
        | CALCULATED_THRESHOLD_VIOLATED
        | CALCULATED_OUTSIDE_RULE_BOUND
        | CALCULATED_RULE_BOUND_FROM_IMPUTED
        | CALCULATED_UNDEFINED
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
        | CALCULATED
        | CALCULATED_FROM_RULE_BOUND
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

    #: Every cell whose value the calculation stage produced, by either route:
    #: evaluating a catalog formula, or taking the single value the rules
    #: narrow its row to. The counterpart of IMPUTED_VALUE, and what a
    #: downstream filter tests to separate calculated values from delivered
    #: ones without needing to know which route produced each cell.
    CALCULATED_VALUE = (
        CALCULATED
        | CALCULATED_FROM_RULE_BOUND
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
    "JITTER_CANDIDATE": (
        "An empty bound whose contradiction is at metering resolution -- a "
        "candidate measurement-jitter conflict, marked for assessment."
    ),
    "CALCULATED": (
        "The value was produced by evaluating a catalog formula, not "
        "measured. Which formula, per household, is the calculated stage's "
        "derivation_plan and household_derivation sidecars."
    ),
    "CALCULATED_THRESHOLD_VIOLATED": (
        "A calculated value outside its own column's threshold, scaled to the "
        "cadence that column reports at."
    ),
    "CALCULATED_FROM_RULE_BOUND": (
        "The rules narrowed this row to a single value for the column -- the "
        "bound's lower and upper ends met -- and that value was written. No "
        "catalog formula reached the cell; the constraint system determined "
        "it from the other values in the row."
    ),
    "CALCULATED_UNDEFINED": (
        "Every input was present and the formula still could not produce a "
        "value -- a ratio whose denominator is zero. Distinct from a missing "
        "input: the data is there and the formula does not apply to the row."
    ),
    "CALCULATED_RULE_BOUND_FROM_IMPUTED": (
        "The rule bound this calculated cell was judged against rests partly "
        "on cells the fill wrote. Absent alongside "
        "CALCULATED_OUTSIDE_RULE_BOUND means the contradiction stands on "
        "measurements alone, which is the stronger signal."
    ),
    "CALCULATED_OUTSIDE_RULE_BOUND": (
        "A calculated value outside the bound the rules narrow its row to "
        "using the other values known in that row."
    ),
}
