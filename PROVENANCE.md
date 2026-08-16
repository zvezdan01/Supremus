# PROVENANCE

Origin repository: `zvezdan01/quant-trade`, branch `claude/supresmus-jj7peq`
Origin commit: `7161f7c4bc6ab5a3a2373baade06143a2dcf4823`
Engine lineage: HUNL Golden Baseline v1 (`34a50560`)
Extraction date: 2026-08-16

This repository is an **extraction, not a rewrite**. Every file listed below was
copied byte-for-byte out of the origin repository. The SHA-256 values are the
authoritative link back to the certified originals: recomputing them here and in
`quant-trade` must give identical digests.

## Why the split exists

`quant-trade` holds the **forensic DeepStack reconstruction**: an attempt to
recover the original 2017 method exactly as published, where every deviation
from primary sources is a defect. Supremus (Zarick, Pellegrino, Brown, Banister,
arXiv:2007.10442) deliberately *changes* the algorithm after reimplementing
DeepStack — DCFR+ with delayed average weighting, a river value network, 4,000
iterations per player, a larger action abstraction, far more training data.

Those two goals contradict each other. Mixing them would let Supremus choices
leak into the forensic baseline and destroy its evidentiary value. Hence a
separate repository with its own history.

## Layer A — frozen certified core (copied, do not edit)

Exact full-card HUNL primitives, already certified in the origin repository
against untouched ACPC `game.c`, LP anchors and exhaustive recounts. These are
imported by Supremus but are **not** Supremus: treat them as read-only.

| file | sha256 |
|---|---|
| `hunl/__init__.py` | `1b4ca3edf7cd06992094a0e0274855e812d5f6370ac792d8d7f30da194d942c8` |
| `hunl/cards.py` | `122b44a8840225cda483abfaffd4e6b63ef162282402b8e2b4497af66865ceb6` |
| `hunl/evaluator.py` | `91964dda84b878a1be9a1cf3277af7b4ea86aee73c7c6e0226877dc2fdbe1b7b` |
| `hunl/blockers.py` | `28d6ee079712dc94f3bf1751b073c5695e84389ea4de258d3a0af5dbcc298eb0` |
| `hunl/showdown.py` | `ccf9f9d8fab61d7f9781065ff26def9aa0188610d8a2da3b4711b02ab33f1326` |
| `hunl/chance.py` | `16e93e7af12dcfe63a3333e68107f88b76823232c84bbd0129b4734730981d27` |
| `hunl/config.py` | `f9c25b2e6ef1adae5866003e94e255db232523e426793ab51551fdee53b46b40` |
| `hunl/tree.py` | `de3efb3582f87ed185549c16fe0a0ddfcb9bdb4b3e0fd18b0815f224f1153450` |
| `hunl/turn_tree.py` | `7a613b8199feea8c7f3ee9f5c2947ea894d077bc387ed95dd3d7d446d08a74da` |
| `hunl/river_terminal.py` | `11b43e59e62d875cae576024388ccd8c0fe7074518cf787e575e3f37ed6366f4` |
| `hunl/river_terminal_fast.py` | `62660bf1e31309229baf4e782815e5ca24ee1846ee969cb242d54d9963d92958` |
| `hunl/river_resolver.py` | `be314f007c1c737d85869255751f80b93ae44981e13f6604643e6ef02a2c5c27` |
| `hunl/turn_engine.py` | `77edf96246657c1fa25b4f55a62fbeeb92411f996dbacde39079fab875f15347` |
| `hunl_datagen/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `hunl_datagen/source_contract_v2.py` | `ea32875c776d2e468bf77ce5e4f08e42b281f76c4fb987c7181a22c497df5846` |
| `hunl_datagen/author_range_v2.py` | `fcec18c3c3ae5afdfb9c389ce8b53dd3ba4a4f6a4f326a76f932de154ee6e076` |
| `hunl_datagen/turn_datagen_v2.py` | `35e14cc6cf5baf497b72a9e86538c060111792c19dc2184afd3542ddcb681459` |
| `datagen/__init__.py` | `0b15598b982b5a0eef496d96fa372334f554f20a727f89b2a54bbc88f8a874e9` |
| `datagen/th_random.py` | `3a8c304d9d17b95f18d30e2d3929d5d679f785540510135012e3114ed9e47ad2` |

## Layer B — Supremus reconstruction (active work happens here)

| file | sha256 |
|---|---|
| `hunl/supremus_config.py` | `b95b18a0d0051216fba3a82e0e86dd1a83b987f23a8cea06ebd97e43b6057b61` |
| `hunl/river_dcfr_plus.py` | `eb204cfca4a81c9bdf14c6c10b5c6aec913ae08929ba824054e672b1b8d07c27` |
| `hunl_datagen/river_datagen_v1.py` | `f668954191830a547eda1958db2a5d69ec1f87c983d7437b4989895d5b7e78eb` |
| `certification/hunl_river_datagen/lp_supremus_river.py` | `18886b0a6fc8b33d56cbf7ffb219e85b3882537df92add21a784b834201b739c` |
| `certification/hunl_river_datagen/run_river_datagen_cert.py` | `7eebfecb427f7ff4165a6346fde9d547172f89c93c803497b467a3aa0a96282c` |

## Layer C — certification artifacts and primary source

| file | sha256 |
|---|---|
| `certification/hunl_river_datagen/HUNL_RIVER_DATAGEN_V1_CERT.json` | `b9aad154a180b0aa551fc5a84ee9a6903d2c6469b930cf31dfc54a9fbc02660d` |
| `certification/hunl_river_datagen/HUNL_RIVER_DATAGEN_V1_MILESTONE.md` | `0a0c34e984216e917b386af05b878b9a9e72261a43bfea5e7b7f4a423f3d1ec4` |
| `certification/hunl_river_datagen/HUNL_RIVER_4000_ANCHOR.json` | `4a742dc389daa059c18acb7b7584a9de50499fff41b5e1bf7101beab4b2040c7` |
| `certification/hunl_river_datagen/HUNL_RIVER_RANDOM_4000_ANCHOR.json` | `eb574f040dcd78e90ddd283ec1832ddbf74535b38c3c0b294b624d026f2b370b` |
| `certification/hunl_river_datagen/HUNL_RIVER_RANDOM_4000_ANCHOR.npz` | `f853aafa3d59d7b4db80acfb5a9a742204590cad2ce8f752e95b4af0fdfd6f89` |
| `certification/hunl_river_datagen/SHA256SUMS_RIVER_V1.txt` | `c9400a0c73ebba71e09d32c4c4f94cabf11d6f8d98b3f5d349f34e638406fcb8` |
| `third_party/papers/2007.10442v1.pdf` | `207887d411941e5ea6ab12910ddae3a38fad4d96b1393ec36d7e86eab9989f00` |

## What was deliberately left behind

`FORENSICS/` (author testimony, CPRG lineage, AIVAT, evidence tables), the
DeepStack certification gates `hunl_g1`, the AGT/Kuhn/CFR+ oracle infrastructure,
`third_party/CFR_plus`, the turn datagen pilot shards, and `CERTIFICATION_REPORT.md`
all stay in `quant-trade`. They are evidence about the *original* DeepStack and
have no bearing on the Supremus line.

`hunl/turn_engine.py` is the one deliberate borrow from that world: the paper
states Supremus generated its random training subgames "in a manner identical to
DeepStack", so the DeepStack-faithful subgame generator is a dependency of
Supremus datagen, not a contamination of it.
