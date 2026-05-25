# Git LFS one-time setup (maintainer notes)

`datasets/mmlupro_complete.json` is 114 MB, over GitHub's 100 MB per-file hard limit. This repo uses Git LFS for that file only.

## First-time push (from a fresh local clone with the file as a regular blob)

If the `mmlupro_complete.json` file was committed before LFS was configured (as is the case for the very first commit of this repo), the file is still a regular Git blob and needs to be migrated:

```bash
# 1. Install LFS on your machine
brew install git-lfs        # macOS; use the appropriate package manager elsewhere
git lfs install             # one-time per-user config

# 2. Migrate the existing commit history so the large file moves to LFS
git lfs migrate import \
    --everything \
    --include="datasets/mmlupro_complete.json"

# 3. Push (initial push must include LFS objects)
git push -u origin main
```

`git lfs migrate import` rewrites history. Since this repo was a single initial commit at the time of LFS setup, this is harmless. If anyone has already cloned the repo without LFS, they will need to re-clone or do `git lfs pull` after the rewrite.

## Adding more LFS-tracked files later

Update `.gitattributes`:

```
datasets/some-new-large-file.json filter=lfs diff=lfs merge=lfs -text
```

Then `git add` the file as usual; LFS picks it up via the attribute.

## Sanity check

After push, `git lfs ls-files` from a fresh clone should show:

```
... * datasets/mmlupro_complete.json
```

If `cat datasets/mmlupro_complete.json | head -3` shows a `version https://git-lfs.github.com/spec/v1` pointer rather than JSON, you forgot `git lfs pull`.
