import unittest


class ExportLifecycleTests(unittest.TestCase):
    def test_tracer_models_are_loaded_and_released_sequentially(self):
        try:
            from src.export_lifecycle import export_tracers_sequentially
        except ModuleNotFoundError:
            self.fail("src.export_lifecycle must provide sequential tracer export")

        active = set()
        peak_active = 0

        def load_model(tracer):
            nonlocal peak_active
            active.add(tracer)
            peak_active = max(peak_active, len(active))
            return tracer

        def render_case(bundle, group, tracer, patient, slice_id):
            self.assertEqual({tracer}, active)
            self.assertEqual(tracer, bundle)
            return (group, tracer, patient, slice_id)

        def release_model(bundle):
            active.remove(bundle)

        records = export_tracers_sequentially(
            tracers=("PSMA", "FDG"),
            groups=("small", "large"),
            fixed_cases={
                "small": {
                    "PSMA": ("psma-small", "1"),
                    "FDG": ("fdg-small", "2"),
                },
                "large": {
                    "PSMA": ("psma-large", "3"),
                    "FDG": ("fdg-large", "4"),
                },
            },
            load_model=load_model,
            render_case=render_case,
            release_model=release_model,
        )

        self.assertEqual(1, peak_active)
        self.assertEqual(set(), active)
        self.assertEqual(["PSMA", "FDG"], [row[1] for row in records["small"]])
        self.assertEqual(["PSMA", "FDG"], [row[1] for row in records["large"]])


if __name__ == "__main__":
    unittest.main()
