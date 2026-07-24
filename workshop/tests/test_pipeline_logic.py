"""Unit tests for the workshop's pure logic (room filter, caption QC).

Run:  cd workshop && python -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s04_caption import looks_living_room


# ---- room-type filter (BLIP honest captions) --------------------------------

def test_keeps_obvious_living_rooms():
    assert looks_living_room("there is a large living room with two couches and a table")
    assert looks_living_room("a room with a sofa, coffee table and rug")


def test_drops_other_rooms():
    assert not looks_living_room("there is a bathroom with a toilet, sink and shower")
    assert not looks_living_room("a bedroom with a queen size bed and nightstands")
    assert not looks_living_room("a kitchen with a center island and bar stools")


def test_open_plan_living_room_kept():
    # has 'kitchen' but also 'living room' -> keep (open plan)
    assert looks_living_room("a living room with an open kitchen and a sofa")


def test_caption_qc_rebuilds_wrong_room():
    from s06_build_dataset import repair_caption

    cap, action = repair_caption(
        "lvngrm living room, a boho bedroom, queen size bed, nightstands",
        style="boho", room_known=True,
    )
    assert action == "rebuilt"
    assert "living room" in cap
    assert "bedroom" not in cap


def test_caption_qc_trims_runaway():
    from s06_build_dataset import repair_caption

    long_caption = ", ".join(["living room"] + [f"phrase{i}" for i in range(40)])
    cap, action = repair_caption(long_caption, style="modern", room_known=False)
    assert action in {"trimmed", "rebuilt"}
    assert len(cap.split(",")) <= 26


def test_caption_qc_passes_good_caption():
    from s06_build_dataset import repair_caption

    good = "lvngrm living room, a scandinavian living room, oak floor, grey sofa, plants"
    cap, action = repair_caption(good, style="scandinavian", room_known=True)
    assert action == "ok"
    assert cap == good
