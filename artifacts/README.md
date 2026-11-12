# Local API artifacts

`model/` stores trained snapshots and the current-model pointer;
`predictions/` stores one JSON per successful prediction. Generated contents
are ignored by Git; README files remain versioned.

Mount this directory at `/app/artifacts` when running Docker. Research outputs
continue to belong under `analysis/`. See `src/api/README.md` for usage.
