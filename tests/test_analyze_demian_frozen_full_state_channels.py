from development.analyze_demian_frozen_full_state_channels import channel_slices


def test_channel_slices_follow_demian_state_tuple_order():
    assert channel_slices(256) == {
        "fast": (0, 256), "slow": (256, 512), "control": (512, 544),
        "message": (544, 800), "carrier": (800, 1056), "gate": (1056, 1312),
    }
