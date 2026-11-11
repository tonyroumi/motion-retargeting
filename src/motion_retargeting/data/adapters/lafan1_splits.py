"""Published PAN LAFAN1/Dog clip membership (upstream Lafan1_and_dog/*_{train,test}.txt).

Only these selected actions/subjects belong to the experiment; other BVHs are ignored.
Sorting is deliberate: upstream os.listdir order is filesystem-dependent.
"""

from pathlib import Path


REFERENCE_SPLITS = {
    "lafan1": {
        "train": """
aiming1_subject4
aiming2_subject2
aiming2_subject3
aiming2_subject5
run1_subject2
run1_subject5
run2_subject4
sprint1_subject2
sprint1_subject4
walk1_subject2
walk1_subject5
walk2_subject3
walk2_subject4
walk3_subject2
walk3_subject3
walk3_subject4
walk3_subject5
""".split(),
        "test": """
aiming1_subject1
walk1_subject1
walk2_subject1
walk3_subject1
walk4_subject1
run2_subject1
""".split(),
    },
    "dog": {
        "train": """
D1_010_KAN01_002
D1_053_KAN01_002
D1_058_KAN01_001
D1_010_KAN01_003
D1_ex04_KAN02_004
D1_009_KAN01_002
D1_ex03_KAN02_003
D1_003_KAN01_001
D1_ex05_KAN02_001
D1_ex02_KAN02_001
D1_047z_KAN01_004
D1_ex06_KAN02_001
D1_ex01_KAN01_001
D1_071_KAN02_002
D1_086_KAN02_001
D1_010_KAN01_004
D1_073_KAN02_002
D1_ex03_KAN02_013
D1_ex03_KAN02_014
D1_059_KAN01_001
D1_ex03_KAN02_012
D1_047z_KAN01_005
D1_006_KAN01_001
D1_031_KAN01_001
D1_025_KAN01_001
D1_055_KAN01_001
D1_007_KAN01_001
D1_009_KAN01_001
D1_008_KAN01_001
D1_053_KAN01_001
D1_ex03_KAN02_001
D1_006_KAN01_002
D1_045_KAN01_001
D1_061z_KAN01_002
D1_ex04_KAN02_003
D1_049_KAN01_001
D1_ex04_KAN02_002
D1_061_KAN01_001
""".split(),
        "test": """
D1_001_KAN01_001
D1_004_KAN01_001
D1_005_KAN01_001
D1_008_KAN01_002
D1_010_KAN01_001
D1_013_KAN01_001
D1_047_KAN01_001
D1_047z_KAN01_002
D1_047z_KAN01_003
D1_057_KAN01_001
D1_061z_KAN01_003
D1_ex03_KAN02_006
D1_ex04_KAN02_001
""".split(),
    },
}


def reference_files(directory: Path, kind: str, split: str) -> list[Path]:
    """Resolve exact split membership, rejecting missing or ambiguous clip stems."""
    wanted = set(REFERENCE_SPLITS[kind][split])
    found = {}
    for path in sorted(directory.rglob("*.bvh")):
        if path.stem not in wanted:
            continue
        if path.stem in found:
            raise ValueError(f"{directory}: duplicate reference clip {path.stem}")
        found[path.stem] = path
    missing = wanted - found.keys()
    if missing:
        raise FileNotFoundError(f"{kind} {split}: missing reference clips under {directory}: {sorted(missing)}")
    return [found[stem] for stem in sorted(wanted)]
