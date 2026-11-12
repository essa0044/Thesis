# Thesis: VLM-guided drone with a depth-camera guard

Backup of everything related to the MSc thesis (NeuroDrone): code, results, procurement lists, drafts and the
running report. Each machine's files are kept in their own folder so versions can be compared.

| Folder | What | From |
|---|---|---|
| `docs/Report_for_Thesis.md` | The running report (every decision, version, test and result). This is the copy to edit from now on. | Exported from the Claude Docs "Report for Thesis", rev 90, 12 Nov 2026 |
| `docs/Report_for_Thesis.docx` | Same report as Word, including the five charts that the Markdown export replaces with placeholders | same |
| `personal-pc/NeuroDrone-Thesis/` | Folder `Downloads\NeuroDrone-Thesis` on the personal laptop: early code (`neurodrone/`), first VLM test (`vlm_test/`), notes, hardware list, thesis plan and draft (state of mid October) | personal laptop |
| `personal-pc/downloads/` | Other thesis files from `Downloads`: procurement lists (Beschaffung, several versions), thesis draft, the simulation kit sent to WSL (`neurodrone_sim.tar.gz`, `setup_sim_ubuntu.sh`, `nd_update/` with the go-around change) and an older handoff zip | personal laptop |
| `personal-pc/wsl-neurodrone/` | The code that ran in Ubuntu 24.04 (WSL2) with PX4 + Gazebo, the four mission logs of 10 Nov (`mission_*.jsonl`) and the saved VLM frames | personal laptop, WSL (`~/neurodrone`, copied 12 Nov) |
| `workstation/` | (to come) The lab workstation's `NeuroDrone-Thesis` repository | lab workstation |

Not included: Python virtual environments, caches, large datasets (TartanAir, EvTTC, ODA), and the PX4 source.
