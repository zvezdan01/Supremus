"""In-repo restoration of the DS `datagen` package's RNG layer.

The original certified `datagen/` package lived in the (container-local)
DS workspace `/workspace/deepstack_leduc_v1.1-bitexact-certified` and was
lost with the 2026-08-14 container reclaim. Only the THRandom RNG layer
is required by the HUNL turn datagen; it is restored HERE, inside the
repository, so it survives infrastructure resets. Because the original
DS path is inserted before the repo root in `sys.path` by every frozen
harness, a restored real DS workspace would take precedence over this
package automatically.

Restoration provenance and BIT_EXACT re-certification:
`certification/hunl_g1/thrandom_oracle/` (untouched torch/torch7 @
814ea4a C oracle + frozen `certification/oracles/rng/` stream anchors,
incl. the Phase-2A recorded cross-anchor d85832ea…).
"""
