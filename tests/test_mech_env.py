from __future__ import annotations

from pathlib import Path

from rl.mech_env import list_tasks


def test_list_tasks_matches_absolute_split_entries_by_task_name(tmp_path: Path) -> None:
    task = tmp_path / "rack_pinion_t00_rack_pinion_contact_stub_s20265610"
    task.mkdir()
    (task / "prompt.md").write_text("Repair the rack-pinion mechanism.")
    (task / "task.toml").write_text(
        'family = "rack_pinion"\n'
        'tier = "physics_contact"\n'
    )
    split = tmp_path / "train.txt"
    split.write_text(
        "/Users/source/machine/corl/runs/mechanism_repair_physics_final/tasks/"
        "rack_pinion_t00_rack_pinion_contact_stub_s20265610\n"
    )

    tasks = list_tasks(root=tmp_path, split_file=split)

    assert [task.task_id for task in tasks] == [
        "rack_pinion_t00_rack_pinion_contact_stub_s20265610"
    ]
