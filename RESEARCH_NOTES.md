# Research notes — availability and adjacent work

Date: 2026-08-16

## Supremus source code: not public

Searched for a code release accompanying arXiv:2007.10442. **None found.** The
paper is indexed on arXiv, Semantic Scholar, ADS, ResearchGate and DeepAI with
no linked repository, and the authors' affiliations (Minimal AI, FAIR) show no
corresponding public release.

This corroborates rather than resolves the project's blockers. The five
unresolved items in `HUNL_RIVER_VALUE_V1_MILESTONE.md` — private 1000-bucket
river artifact, integer-chip rounding of the Table-2 fractions, private RNG and
dataset ordering, the river-training update mode, and original weights — remain
closed by absence of the private code, exactly as the DeepStack-side blockers
are in `quant-trade`.

Status: **NEGATIVE FINDING, recorded so it is not re-searched blind.** It is not
proof of destruction; it is proof that no public copy was locatable on this
date.

## Adjacent work worth evaluating: predict EV, not CFV

Jeremiasz Wołosiuk, Maciej Świechowski, Jacek Mańdziuk,
*Don't Predict Counterfactual Values, Predict Expected Values Instead*,
AAAI-23. **Supplementary code is public:**
`https://github.com/jwolosiuk/dont-predict-cfvs-predict-evs-instead`

### The claim

A counterfactual value factorizes into two parts:

    CFV(state) = P(opponent reaches state) × EV(payoff in state)

The first factor is the opponent's reach probability, which is *already an
input* to the network — it is the opponent range. So a DCVN that regresses CFV
directly is spending capacity relearning a quantity the caller can compute
exactly. Their modification: have the network predict only the **EV** factor,
then multiply by the exactly-known reach. They report materially more accurate
CFV estimates from the same architecture.

### Why it is interesting here

The interface this project is bound to is fixed by the paper: 2001 inputs,
2000 outputs, 7×500 PReLU. The EV factorization does **not** change that
interface — the first 2000 inputs already *are* the two bucket ranges, so the
reach factor is available at the output stage without any new input. It is a
change of regression target and of the final multiply, sitting exactly where
`inverse_bucket_outputs` and `masked_card_huber_loss` already sit in
`hunl/value_training.py`.

That makes it unusually cheap to test: the frozen full-card targets under
`certification/hunl_river_value/` are raw chip CFVs, so an EV-target variant can
be derived from them without re-solving a single subgame — which is precisely
the property the V1 milestone was designed to preserve.

### Discipline

This is **third-party 2023 work, not Supremus.** It must not be folded into the
Supremus reconstruction any more than Supremus may be folded into the forensic
DeepStack baseline in `quant-trade`. If pursued, it belongs behind an explicit
flag as a separate `EV_FACTORED` profile, with its own certificate, and the
paper-faithful CFV-target path must stay the default and stay reproducible.

Status: **EVALUATED, NOT ADOPTED.** No code from that repository has been read
or copied into this project.

## Schmid correspondence

The direct email testimony from Martin Schmid (2026-08-16, recorded in
`quant-trade` at `FORENSICS/SCHMID_TESTIMONY_2026-08-16.md`) bears on the
DeepStack side — the existence of a skip/burn-in window in offline target
generation — and it was Schmid who recommended arXiv:2007.10442, the paper this
repository is built on. Its scope warning applies here too: the testimony does
not establish the private RNG, seed schedule, river bucketing, serialization, or
any exact numerical constant.
