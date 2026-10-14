# Decoder-tracking precheck

The prior secondary `regime_conditioned_external` observer is excluded from the
fresh confirmation protocol. Its implementation uses total raw state norm and
the fourth state slice (`message` in the current channel order), while earlier
description referred to log norm and gate mean. This discrepancy does not
affect the frozen cue6 decoder, identity, scale, orthogonal tracking, analytic
latest-cue, or conditional-shuffle measurements in the confirmation artifact.

Before any later regime-conditioned observer is interpreted, define its channel
slice and normalization in a separate frozen protocol and add replay tests.
