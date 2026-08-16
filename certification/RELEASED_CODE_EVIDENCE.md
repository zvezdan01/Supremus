# Evidence extract — DeepStack-Leduc released code

Source: `github.com/lifrordi/DeepStack-Leduc`, commit
**`da416f9646725def43e668851593de13ead8b607`** (2017-04-05), released by Martin
Schmid, co-author of DeepStack.

## Why this is a citation, not a copy

That repository carries **no LICENCE file**, so it is all-rights-reserved. Its
code is therefore *not* vendored here. What follows is a factual extract —
file, line, and the finding — pinned to a commit so anyone can fetch the same
tree and check every claim. SHA-256 of each cited file is listed at the end.

For a project whose product is auditable provenance, a pinned citation is
stronger evidence than a copy would be, and it keeps the tree free of code we
cannot account for.

## Why this source outranks the alternatives

This is the closest public artifact to the original DeepStack: written by the
authors, for the algorithm as they implemented it. Everything else available —
Supremus (Zarick et al. 2020), the DEVN work (Wołosiuk et al. 2023) — is a
*reimplementation by people who also never had the original*. They are evidence
about their own systems, not about DeepStack.

Scope limit that applies throughout: this is the **Leduc** release. Where Leduc
and HUNL genuinely differ (deck, streets, bucket counts, pot scheme), a Leduc
finding constrains the *implementation idiom* of the authors, not the HUNL
constants.

---

## 1. CFR iterations and the skip window — CONFIRMS the project's anchor

`Source/Settings/arguments.lua:30,32`

```
params.cfr_iters = 1000
params.cfr_skip_iters = 500
```

with the comment at line 31: *"the number of preliminary CFR iterations which
DeepStack doesn't factor into the average strategy (included in cfr_iters)"*,
and `Source/Lookahead/lookahead.lua:354` dividing the accumulated CFVs by
`cfr_iters - cfr_skip_iters`.

So the skip window is real, it is 500 of 1000, and the average is over the last
500 only. This is the exact basis for the project's
`RELEASED-CODE-ANCHORED` verdict, and it corroborates Schmid's email
recollection that skip iterations were always used. **No change needed.**

## 2. Range generator odd-split is RANDOMIZED — conflicts with V2

`Source/DataGeneration/range_generator.lua:35-38`

```
if halfSize % 1 ~= 0 then
  halfSize = halfSize - 0.5
  halfSize = halfSize + torch.random(0,1)
end
```

When a sub-range has an odd card count, the middle card goes to one side or the
other **at random**.

The project's turn DataGenerator V2 replaced exactly this with a deterministic
`floor(|S|/2)` split, on the authority of the HUNL supplement's prose. That
remains a defensible reading of the HUNL text, but this pins the other side of
fork **C2** precisely: the authors' own released implementation randomizes.

**Action: the fork is now anchored on both sides and is cheaply testable.** Run
the generator both ways on identical seeds and measure the target delta in % of
pot. Until then, neither reading should be described as settled.

## 3. Equal-strength tie ordering — the U3 blocker, localized

`Source/DataGeneration/range_generator.lua:71`

```
_, order = non_coliding_strengths:sort()
```

The range is built over hands **sorted by strength**, then unsorted. The
project lists "equal-strength tie ordering inside `R(S,p)`" as unresolved.

This narrows it usefully: the ambiguity is not a missing algorithm, it is the
**tie behaviour of torch7's `sort`** on equal strengths. That is a bounded,
answerable question about a specific released function, not an unknowable
private detail. It has an unusually large blast radius, because equal-strength
hands are common and ties cross the recursive split boundaries.

## 4. Board sampler — CONFIRMS the project's implementation

`Source/DataGeneration/random_card_generator.lua:23-30`: rejection sampling,
`torch.random(1, card_count)` repeatedly, skipping cards already used.

This is exactly what `sample_river_board` in `hunl_datagen/river_datagen_v1.py`
does. **No change needed.**

## 5. Pot sampler — does NOT resolve the `[100,100)` question

`Source/DataGeneration/data_generation.lua:76-81`: pot sizes are drawn
**continuous uniform** on `[ante, stack - 0.1]`.

That is a different scheme from the HUNL supplement's binned sampler, so the
released code says nothing about the printed `[100,100)` interval. **Negative
finding — that blocker stays open.**

## 6. Pot convention is PER-PLAYER, not total pot — a live factor of 2

`Source/DataGeneration/data_generation.lua:84,106,111`

```
pot_size_features = random_pot_sizes * (1/arguments.stack)   -- line 84
current_node.bets = Tensor{pot_size, pot_size}               -- line 106
root_values:mul(1/pot_size)                                  -- line 111
```

`bets = {pot_size, pot_size}` means each player has committed `pot_size`, so the
**total** pot is `2 * pot_size`. Both the network's pot feature and the target
normalization use `pot_size` — the **per-player** amount.

This project uses the *total* pot for both, recorded as
`SUPREMUS_PAPER_LITERAL_TOTAL_POT_DIV_STARTING_STACK`, on the reading that the
Supremus paper's "current pot size" means the whole pot.

**These differ by a factor of 2 in the input feature and in the targets.** That
matters concretely: it rescales the loss, so our numbers and the paper's Table-1
values are not on the same axis until the convention is settled. Note
`RiverSolvedBatch` already exposes `targets_per_pot_half` alongside
`targets_per_total_pot`, so testing both costs nothing.

**Action: record as an open fork, and report the convention next to any loss.**

## 7. Zero-sum outer layer — CONFIRMS the project's architecture

`Source/Nn/net_builder.lua:65,67,77`: `nn.DotProduct()` of outputs with the
range half of the input, `nn.MulConstant(-0.5)`, then `nn.CAddTable()` onto the
feedforward output.

This is the project's bucket-space zero-sum correction, in the authors' own
code. Two consequences:

- the project's `DeepStackHUNLValueNet` is architecturally correct against
  released code, not merely against the paper's prose;
- it independently strengthens the decision recorded in `RESEARCH_NOTES.md` not
  to adopt DEVN, whose supplementary states zero-sum **cannot** be enforced on
  bucketed EVs and must move after inverse bucketing. The released design puts
  it in bucket space.

## 8. Optimizer and learning rate — CONFIRMS the project's choice

`Source/Training/train.lua:78,80` and `Source/Settings/arguments.lua:54`

```
local state = {learningRate = arguments.learning_rate}   -- 0.001
local optim_func = optim.adam
```

Plain **Adam at 1e-3**, batch 100 (`arguments.lua:36`).

This corrects an earlier note in this project. Wołosiuk et al. use AdamW at
3e-4, and that was previously flagged here as the only independent
hyperparameter data point. It is not the best one: the released DeepStack code
uses Adam at 1e-3, which is what `train_smoke_steps` already does. **The
project's existing choice is released-code-anchored; the third-party value is
not.** Vary it if the training curve stalls, but start from 1e-3.

## 9. Training loss is computed in BUCKET space

`Source/Nn/masked_huber_loss.lua`: `nn.SmoothL1Criterion`, masked to buckets
possible on the board.

The project computes masked Huber in **card space after inverse bucketing**.
That remains a deliberate, documented deviation — but it is a deviation from
released code, not only from unpublished private code, and should be described
that way.

## 10. The card→bucket target reduction IS published — supersedes a project claim

`Source/DataGeneration/data_generation.lua:116-120`

```
--translating values to nn targets
bucket_conversion:card_range_to_bucket_range(values[player], targets[...])
```

`card_range_to_bucket_range` is `bucket_range:mm(card_range, _range_matrix)`
(`Source/Nn/bucket_conversion.lua:51-53`) — a **sum** over the hands in each
bucket. It is applied here to *values*, not ranges: the same summation matrix is
reused.

So the reduction is: **bucket target = sum of the per-hand CFVs in that bucket**,
with the inverse direction (`bucket_value_to_card_value`, line 61-63) copying
the bucket value back to each member hand.

This directly contradicts the standing claim in `hunl/value_training.py`:

> "The private HUNL code did not reveal how card-level solved training targets
> were reduced to bucket-space targets. We therefore avoid inventing that
> reduction."

It was not revealed by the *HUNL* code, but the authors' released Leduc code
does show their reduction, and it is not an invention — it is summation.

**Action: this is the single highest-value item here.** The project's card-space
loss is still defensible and arguably better, but it should no longer be
justified by "the reduction is unknown". Update the docstring, and add a
bucket-space target path as a measurable alternative — the frozen raw chip CFVs
make it derivable with no re-solving.

---

## Net effect on the project's open questions

| item | before | after |
|---|---|---|
| skip window = 500 | released-code-anchored | **unchanged, citation pinned** |
| board sampler | reconstructed | **confirmed** |
| zero-sum in bucket space | paper-anchored | **confirmed in released code** |
| Adam @ 1e-3 | project choice | **released-code-anchored** |
| odd-split floor vs random | fork C2, prose vs code | **both sides pinned, testable** |
| equal-strength tie order | unresolved | **localized to torch7 `sort` semantics** |
| card→bucket target reduction | "unpublished, avoided" | **published: summation** |
| pot: total vs per-player | assumed total | **new fork, factor of 2** |
| `[100,100)` pot interval | unresolved | **unchanged — Leduc uses a different scheme** |

Two findings change what the project should do next: item 10 (the reduction is
published) and item 6 (the pot convention differs by a factor of 2, which
rescales every loss figure).

## Cited files, SHA-256 at commit `da416f96`

```
8c395f2886ca273e…  Source/Settings/arguments.lua
8245305c4448e6bf…  Source/DataGeneration/range_generator.lua
b7dcd00b427491e0…  Source/DataGeneration/random_card_generator.lua
42fa82ee90766a39…  Source/DataGeneration/data_generation.lua
c1a96384fbc12ea2…  Source/Nn/net_builder.lua
7a54e6b7801f9751…  Source/Nn/bucket_conversion.lua
2fc12634656c38e6…  Source/Nn/masked_huber_loss.lua
46100ff49a091355…  Source/Training/train.lua
8aab75e2fa3caffd…  Source/Lookahead/lookahead.lua
```

Reproduce with:

```
git clone https://github.com/lifrordi/DeepStack-Leduc
git -C DeepStack-Leduc checkout da416f9646725def43e668851593de13ead8b607
sha256sum DeepStack-Leduc/Source/Settings/arguments.lua   # etc.
```
