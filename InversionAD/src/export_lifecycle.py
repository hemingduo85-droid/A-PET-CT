def export_tracers_sequentially(
    *,
    tracers,
    groups,
    fixed_cases,
    load_model,
    render_case,
    release_model,
):
    records_by_group = {group: [] for group in groups}
    for tracer in tracers:
        bundle = load_model(tracer)
        try:
            for group in groups:
                patient, slice_id = fixed_cases[group][tracer]
                records_by_group[group].append(
                    render_case(bundle, group, tracer, patient, slice_id)
                )
        finally:
            release_model(bundle)
            del bundle
    return records_by_group
