from __future__ import annotations


def render_v1_summary(receipt: dict) -> str:
    a=receipt['aggregate']
    verdict='PASS' if a['overall_pass'] else 'FAIL'
    diag=a.get('own_surface_medians',[None,None,None])
    margins=a.get('specificity_margins',[None,None,None])
    def f(x): return 'n/a' if x is None else f'{x:.3f}'
    return (
        '### GATGRILS v1 canonical 12-seed receipt\n\n'
        f'- **Overall:** {verdict}\n'
        f'- Task-valid seeds: **{a["task_valid_count"]}/12** (requires 9)\n'
        f'- Own-surface transplant medians — operator **{f(diag[0])}**, admission/phase **{f(diag[1])}**, publication/route **{f(diag[2])}**\n'
        f'- Specificity margins — operator **{f(margins[0])}**, phase **{f(margins[1])}**, route **{f(margins[2])}**\n'
        f'- Publication-clamp latent / release medians: **{f(a.get("publication_latent_median"))} / {f(a.get("publication_release_median"))}**\n'
        f'- Admission damage median: **{f(a.get("admission_damage_median"))}**\n'
        f'- Cycle-mean admission damage median: **{f(a.get("rhythmic_damage_median"))}**\n'
        '\nSecondary silent-ping, temporal-window, and gauge measurements are recorded in `results/gatgrils_v1.json` and do not rescue a failed primary gate.\n'
    )
