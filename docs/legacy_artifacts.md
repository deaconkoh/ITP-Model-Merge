# Inherited artifact policy

Inherited checkpoints and results are preserved unchanged. They are evidence
of prior work, not canonical v2 artifacts.

| Artifact group | Permitted use | Scientific status |
|---|---|---|
| F11/P9 speed, carbon, priority specialists | Exploratory baseline | Retrain before claims |
| Exactly reconstructible F11/P9 soups | Merge arithmetic reference | Rerun source training and evaluation |
| Historical Task Arithmetic checkpoints | Arithmetic reference | No Task Arithmetic claim without shared-base evidence |
| F11/P8 specialists and merges | Legacy comparison | Excluded from canonical experiments |
| F10/P8 checkpoints | Historical reference | Legacy only |
| `ft_p045_200` checkpoint | Artifact preservation | Provenance insufficient |
| Job-priority models/results | Separate experimental branch | Must not enter operation-priority reporting |
| Historical test-driven sweeps and summaries | Exploratory evidence | Rerun using validation selection |
| Unknown-metric or unknown-dataset results | Preservation only | No scientific claims |

Legacy checkpoint filenames are not treated as metadata. Architecture must be
checked from tensor shapes and loading requires an explicit legacy feature
schema. Original files must not be rewritten to add metadata.

Generate a fresh checksummed inventory with
`tools/catalog_legacy_artifacts.py --out <new-catalog.json>`. The command refuses
to overwrite an existing catalog.
