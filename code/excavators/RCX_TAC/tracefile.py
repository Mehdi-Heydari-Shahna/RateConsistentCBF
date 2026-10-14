"""Time traces of a run (``trace.npz``), also when the distributable parts hold them in several pieces.

A trace larger than a distributable part is packed as ``trace.npz``, ``trace.part2.npz``, ... (each an ordinary npz
archive with a share of the arrays).  ``load_trace`` returns the arrays of all pieces of a folder
as one read-only mapping, so that the report generator and the video script read a packed folder like the original.
"""
from pathlib import Path

import numpy as np


def piece_paths(folder):
    """Files of the trace of a run folder: ``trace.npz`` and the further pieces ``trace.part2.npz``, ... that belong to it.
    A piece whose arrays are all in ``trace.npz`` already is a leftover of an older, packed trace (the run was repeated
    in a folder that was unzipped from the parts) and is not part of the trace."""
    folder = Path(folder)
    main = folder / 'trace.npz'
    paths = [main]
    with np.load(main, allow_pickle=True) as z:
        have = set(z.files)
    for q in sorted(folder.glob('trace.part*.npz'), key=lambda q: int(q.name[len('trace.part'):-len('.npz')])):
        with np.load(q, allow_pickle=True) as z:
            keys = set(z.files)
        if keys <= have:
            continue                                         # stale piece of a trace that has been replaced
        if keys & have:
            raise ValueError(f'{q} holds arrays that are also in another piece of the trace in {folder}')
        have |= keys
        paths.append(q)
    return paths


class Trace:
    """Arrays of ``trace.npz`` and of its further pieces in the same folder (``files``, ``[key]``, ``in``)."""

    def __init__(self, folder):
        self._z = [np.load(q, allow_pickle=True) for q in piece_paths(folder)]
        self._where = {}
        for z in self._z:
            for k in z.files:
                self._where[k] = z
        self.files = list(self._where)

    def __getitem__(self, key):
        return self._where[key][key]

    def __contains__(self, key):
        return key in self._where

    def close(self):
        for z in self._z:
            z.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_trace(folder):
    """Trace of the run folder, or None if it has none."""
    return Trace(folder) if (Path(folder) / 'trace.npz').exists() else None


def split_npz(path, limit, keep=('time', 'c_time', 't')):
    """Bytes of npz archives that together hold the arrays of ``path``, each archive at most ``limit`` bytes: the arrays
    in their stored order, a new archive whenever the next array would exceed the limit (sizes of the compressed
    members of the original archive, with a reserve for the difference between two compressions).  Float64 arrays other
    than the time axes (``keep``) are stored as float32.  One archive is returned for a trace that fits."""
    import io
    import zipfile
    with zipfile.ZipFile(path) as zf:
        size = {i.filename[:-len('.npy')]: i.compress_size for i in zf.infolist()}
    groups, used = [[]], 0
    with np.load(path, allow_pickle=True) as z:
        for k in z.files:
            if groups[-1] and used + size[k] > 0.9 * limit:
                groups.append([])
                used = 0
            groups[-1].append(k)
            used += size[k]
        out = []
        for keys in groups:
            data = {k: (z[k].astype(np.float32) if z[k].dtype == np.float64 and k not in keep else z[k]) for k in keys}
            buf = io.BytesIO()
            np.savez_compressed(buf, **data)
            out.append(buf.getvalue())
    if any(len(b) > limit for b in out):
        raise ValueError(f'{path}: an array is larger than a part')
    return out
