import math

from motion_retargeting.training.logger import MetricLogger, format_value


def pan_step_metrics() -> dict[str, float]:
    """ One PAN train step's metrics, keyed and ordered as `PANMethod.train_step` returns them. """
    domains = ("group_a", "group_b")
    routes = ("group_a_to_group_b", "group_b_to_group_a")
    generator = {}
    for domain in domains:
        generator.update({f"{domain}/reconstruction": 0.1834, f"{domain}/root": 0.0412, f"{domain}/position": 1.2731})
    for route in routes:
        generator.update({
            f"{route}/latent_cycle": 0.0921, f"{route}/position_cycle": 0.0045,
            f"{route}/end_effector": 2.4311, f"{route}/contact": 37.52,
        })
    for route in routes:
        generator[f"{route}/adversarial"] = 0.7012
    loss_grads = {f"loss_grad/{key}": 12.7 for key in generator}
    generator["total/generator"] = 81.33
    discriminator = {f"{domain}/discriminator": 0.4987 for domain in domains}
    discriminator["total/discriminator"] = 0.9974
    grads = {
        f"grad/{domain}/{component}": norm
        for domain in domains
        for component, norm in (("motion_encoder", 15.02), ("body_part_encoder", 3.1e-6), ("decoder", 1234.5))
    }
    grads.update({f"grad/{domain}/discriminator": 0.8 for domain in domains})
    return {**generator, **loss_grads, **discriminator, **grads}


def pan_validation_metrics() -> dict[str, float]:
    terms = (
        "retarget_latent_l1", "retarget_realism", "reconstruction_mse", "reconstruction_mpjpe_m",
        "reconstruction_squared_error_over_height2", "cycle_mpjpe_m", "cycle_squared_error_over_height2",
    )
    return {f"{domain}/val/{term}": 0.01 * (i + 1) for domain in ("group_a", "group_b") for i, term in enumerate(terms)}


def parse(table: str) -> dict[str, str]:
    """ Read a `format_metrics` table back into {key: value text}, failing on any unpaired line. """
    pairs: dict[str, str] = {}
    section = pending = None
    for line in table.splitlines():
        if not line.startswith("      "):
            assert pending is None, f"term {pending!r} has no value"
            section = line.strip()
            continue
        text = line.strip()
        if text.startswith("."):  # value line of a term too long for its row
            assert pending is not None, f"value line with no term: {line!r}"
            term, value = pending, text.lstrip(".").strip()
            pending = None
        elif " .." in text:
            term, _, rest = text.partition(" ..")
            value = rest.lstrip(".").strip()
        else:
            assert pending is None, f"term {pending!r} has no value"
            pending = text
            continue
        key = term if section == "metrics" else f"{section}/{term}"
        assert key not in pairs, f"{key} printed twice"
        pairs[key] = value
    assert pending is None, f"term {pending!r} has no value"
    return pairs


def test_every_pan_metric_paired_with_its_value():
    metrics = {**pan_step_metrics(), "grad/generator": 41.5}
    table = MetricLogger.format_metrics(metrics, width=80)

    assert parse(table) == {key: format_value(value) for key, value in metrics.items()}
    assert all(len(line) <= 80 for line in table.splitlines())
    # Keys of one group share one section, though PAN emits a route's adversarial term after the others.
    assert table.count("\n    group_a_to_group_b\n") == 1
    assert "      adversarial ........    0.7012" in table


def test_values_keep_magnitude_and_nonfinite_values_are_shown():
    metrics = {
        "grad/group_a/body_part_encoder": 3.1e-6,
        "grad/group_a/decoder": 1234.5,
        "grad/group_b/decoder": float("inf"),
        "grad/generator": float("nan"),
        "group_a/val/retarget_realism": -float("inf"),
        "total/generator": 0.0,
        "steps": 2e6,
    }
    pairs = parse(MetricLogger.format_metrics(metrics, width=80))

    assert pairs == {
        "grad/group_a/body_part_encoder": "3.100e-06",
        "grad/group_a/decoder": "1234.5000",
        "grad/group_b/decoder": "inf",
        "grad/generator": "nan",
        "group_a/val/retarget_realism": "-inf",
        "total/generator": "0.0000",
        "steps": "2.000e+06",
    }
    assert math.isnan(float(pairs["grad/generator"]))


def test_narrow_width_moves_long_terms_to_their_own_line_without_truncation():
    metrics = pan_validation_metrics()
    table = MetricLogger.format_metrics(metrics, width=40)
    lines = table.splitlines()

    assert parse(table) == {key: format_value(value) for key, value in metrics.items()}
    assert "      reconstruction_squared_error_over_height2" in lines
    for line in lines:
        # Only a term longer than the width itself may overflow it, and only on its own line.
        assert len(line) <= 40 or line.strip() in {"reconstruction_squared_error_over_height2", "cycle_squared_error_over_height2"}
    value_line_ends = {len(line) for line in lines if line.strip().startswith(".") or " .." in line}
    assert len(value_line_ends) == 1  # every value is right-aligned to the same column


def test_empty_and_ungrouped_metrics():
    assert MetricLogger.format_metrics({}) == "    (no metrics)"
    assert parse(MetricLogger.format_metrics({"lr": 1e-4, "group_a/root": 0.5}, width=80)) == {
        "lr": "1.000e-04", "group_a/root": "0.5000",
    }


def test_display_prints_header_then_table(capsys):
    logger = MetricLogger(use_tensorboard=False)
    metrics = pan_step_metrics()
    logger.display("  step=10", metrics)
    out = capsys.readouterr().out

    header, _, table = out.rstrip("\n").partition("\n")
    assert header == "  step=10"
    assert parse(table) == {key: format_value(value) for key, value in metrics.items()}

    MetricLogger(use_tensorboard=False, verbose=False).display("  step=10", metrics)
    assert capsys.readouterr().out == ""
